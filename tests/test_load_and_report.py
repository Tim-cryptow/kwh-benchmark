import json
from pathlib import Path

import pytest

from kwh_bench import reference as ref
from kwh_bench.engines import MockEngine
from kwh_bench.load import PreparedPrompt, prepare_prompts, run_job, score_runs
from kwh_bench.lockfile import Canary, Lock, default_canary_ids
from kwh_bench.prompts import generate_prompts
from kwh_bench.report import build_report, evaluate_canary, report_hash, validate
from kwh_bench.runner import run_benchmark
from kwh_bench.verify import verify_report


@pytest.fixture
def fast_engine():
    return MockEngine(step_ms=0.05, prefill_ms_per_1k=0.1)


async def test_prepare_truncates_to_exact_prompt_tokens(fast_engine):
    async with fast_engine as e:
        prepared = await prepare_prompts(e, generate_prompts(count=4))
    assert all(len(p.token_ids) == ref.PROMPT_TOKENS for p in prepared)


async def test_prepare_rejects_short_prompt(fast_engine):
    from kwh_bench.prompts import Prompt
    async with fast_engine as e:
        with pytest.raises(ValueError, match="spec defect"):
            await prepare_prompts(e, [Prompt(id=0, text="too short")])


async def test_run_job_generates_exact_token_count(fast_engine):
    prepared = [PreparedPrompt(id=i, token_ids=[i] * ref.PROMPT_TOKENS) for i in range(64)]
    async with fast_engine as e:
        res = await run_job(e, prepared, concurrency=8, max_tokens=16, want_ids_for={3, 9})
    assert res.failures == 0
    assert res.generated_tokens == 64 * 16
    assert len(res.records) == 64
    ids_present = {r.prompt_id for r in res.records if r.token_ids is not None}
    assert ids_present == {3, 9}
    pct = res.percentiles()
    assert pct["ttft_p50_s"] is not None and pct["tpot_p95_s"] is not None


async def test_run_job_records_failures():
    class Flaky(MockEngine):
        async def complete(self, token_ids, max_tokens, want_token_ids=False):
            if token_ids[0] % 5 == 0:
                raise RuntimeError("boom")
            return await super().complete(token_ids, max_tokens, want_token_ids)

    prepared = [PreparedPrompt(id=i, token_ids=[i] * 8) for i in range(10)]
    async with Flaky(step_ms=0.05) as e:
        res = await run_job(e, prepared, concurrency=4, max_tokens=4)
    assert res.failures == 2
    assert res.generated_tokens == 8 * 4


def test_score_runs_median_and_stability():
    s = score_runs([40.0, 42.0, 41.0])
    assert s["median_job_seconds"] == 41.0
    assert s["units_per_hour"] == round(3600 / 41.0, 3)
    assert s["stability"] == round(2.0 / 41.0, 5)


def _locked(ref_lp=-1.0):
    return Lock(model_revision="abc", model_files={"a.safetensors": "0" * 64}, vllm_version="0.30.0",
                canaries=[Canary(i, list(range(ref.CANARY_TOKENS)), ref_lp) for i in range(ref.CANARY_COUNT)])


def test_canary_evaluation_logprob_delta():
    lock = _locked(ref_lp=-1.0)
    assert lock.is_locked
    scores = {i: -1.0 for i in range(ref.CANARY_COUNT)}
    scores[0] = -1.0 + 0.02            # numerical noise: passes
    scores[1] = -1.0 - 0.045           # inside tolerance: passes
    scores[2] = -1.6                   # different model/quant: fails
    scores[3] = None                   # scoring failed: fails
    c = evaluate_canary(scores, lock)
    assert c["passing"] == 6 and c["passed"] is True
    assert c["max_delta"] == ref.CANARY_MAX_LOGPROB_DELTA == 0.05
    assert c["results"][2]["pass"] is False and c["results"][3]["delta"] is None
    scores[4] = -0.5                   # a third failure -> below 6/8
    assert evaluate_canary(scores, lock)["passed"] is False


def test_awq_int4_control_fails_at_current_tolerance():
    """The 4-bit control measured on the 4090 (results/canary/) cleared rc.2's 0.10 on 5/8; it must fail now."""
    lock = _locked(ref_lp=-1.0)
    awq_deltas = [0.03582, 0.094, 0.05838, 0.09849, 0.05519, 0.14166, 0.16482, 0.21873]
    scores = {i: -1.0 - d for i, d in enumerate(awq_deltas)}
    c = evaluate_canary(scores, lock)
    assert c["passed"] is False and c["passing"] == 1


def test_verify_rejudges_canary_under_current_tolerance(tmp_path):
    """A report whose canary passed under a looser tolerance is rejected if it would not pass now."""
    from kwh_bench.verify import verify_report
    report = json.loads((Path(__file__).resolve().parent.parent / "results" / "rtx-4090-runpod.json").read_text())
    # Real certified report: all deltas are 0.0, passes at any tolerance.
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(report))
    ok, problems = verify_report(p)
    assert ok, problems
    # Same report, but pretend it was scored at 0.10 with the AWQ control's deltas and passed 5/8 there.
    awq = [0.03582, 0.094, 0.05838, 0.09849, 0.05519, 0.14166, 0.16482, 0.21873]
    for r, d in zip(report["canary"]["results"], awq):
        r["delta"] = d
        r["pass"] = d <= 0.10
    report["canary"]["passing"] = 5
    report["canary"]["max_delta"] = 0.10
    report["report_sha256"] = report_hash(report)
    p.write_text(json.dumps(report))
    ok, problems = verify_report(p)
    assert not ok and any("current tolerance" in x for x in problems), problems


def test_canary_none_when_unlocked():
    assert evaluate_canary({}, Lock()) is None


async def test_lock_and_run_roundtrip_on_mock(tmp_path, fast_engine):
    """Lock on the mock, then benchmark on the mock: every canary delta must be 0."""
    from kwh_bench.runner import score_canaries
    from kwh_bench.load import PreparedPrompt
    prepared = [PreparedPrompt(id=i, token_ids=[i] * 8) for i in range(ref.REQUESTS_PER_JOB)]
    async with fast_engine as e:
        canaries = []
        for pid in default_canary_ids():
            p = prepared[pid]
            c = await e.complete(p.token_ids, ref.CANARY_TOKENS, want_token_ids=True)
            lps = await e.score_continuation(p.token_ids, c.token_ids)
            canaries.append(Canary(pid, c.token_ids, sum(lps) / len(lps)))
        lock = Lock(model_revision="r", model_files={"w.safetensors": "0" * 64}, vllm_version="x", canaries=canaries)
        assert lock.is_locked
        scores = await score_canaries(e, prepared, lock, lambda s: None)
    c = evaluate_canary(scores, lock)
    assert c["passed"] and all(r["delta"] == 0 for r in c["results"])


def test_default_canary_ids_are_fixed_and_in_range():
    ids = default_canary_ids()
    assert len(ids) == ref.CANARY_COUNT == len(set(ids))
    assert all(0 <= i < ref.REQUESTS_PER_JOB for i in ids)
    assert ids == default_canary_ids()


async def test_full_mock_run_produces_valid_unverifiable_report(tmp_path, fast_engine):
    report = await run_benchmark(fast_engine, measured_jobs=3, log=lambda s: None)
    validate(report)
    assert report["certified"] is False
    assert any("mock" in r for r in report["certified_reasons"])
    # The committed reference/lock.json is complete; an unlocked checkout would add an "unlocked" reason.
    from kwh_bench.lockfile import load_lock
    assert load_lock().is_locked == (not any("unlocked" in r for r in report["certified_reasons"]))
    assert report["job"]["generated_tokens_per_job"] == ref.GENERATED_TOKENS_PER_JOB
    assert all(r["generated_tokens"] == ref.GENERATED_TOKENS_PER_JOB for r in report["runs"])
    assert report["report_sha256"] == report_hash(report)
    assert report["canary"] is None  # mock is not a certified engine

    p = tmp_path / "r.json"
    p.write_text(json.dumps(report))
    ok, problems = verify_report(p)
    assert ok, problems

    report["runs"][0]["job_seconds"] += 1.0
    p.write_text(json.dumps(report))
    ok, problems = verify_report(p)
    assert not ok and any("report_sha256" in x for x in problems)


def test_forbidden_flags_block_certification():
    from kwh_bench.engines.base import EngineInfo
    from kwh_bench.load import JobResult
    from kwh_bench.report import certification_reasons
    lock = _locked()
    info = EngineInfo(name="vllm", version="0.30.0", model_id=ref.MODEL_ID, model_revision="abc", launch_mode="subprocess",
                      launch_args=["--model", ref.MODEL_ID, "--max-num-seqs", "32", "--speculative-config", "{}"])
    runs = [JobResult(job_seconds=40.0, records=[], generated_tokens=ref.GENERATED_TOKENS_PER_JOB, failures=0) for _ in range(3)]
    score = score_runs([40.0, 40.0, 40.0])
    canary = {"passed": True, "passing": 8, "required": 6, "max_delta": ref.CANARY_MAX_LOGPROB_DELTA, "results": []}
    reasons = certification_reasons(info, runs, score, canary, lock)
    assert reasons == ["forbidden flag --speculative-config"]
    info.launch_args = ["--model", ref.MODEL_ID, "--max-num-seqs", "32"]
    assert certification_reasons(info, runs, score, canary, lock) == []
    info.launch_args = ["--model", ref.MODEL_ID, "--max-num-seqs", "64"]
    assert certification_reasons(info, runs, score, canary, lock) == ["--max-num-seqs 64 != 32"]
