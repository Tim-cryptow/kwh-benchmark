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


def test_canary_evaluation():
    from kwh_bench.load import RequestRecord
    exp = list(range(ref.CANARY_TOKENS))
    lock = Lock(canaries=[Canary(prompt_id=i, expected_token_ids=exp) for i in range(ref.CANARY_COUNT)])
    recs = []
    for i in range(ref.CANARY_COUNT):
        got = list(exp)
        if i < 2:                      # two canaries diverge badly (wrong model)
            got = [999] * ref.CANARY_TOKENS
        elif i == 2:                   # one canary drifts by 3 tokens (fp noise) -> still passes
            got[-3:] = [1, 2, 3]
        recs.append(RequestRecord(prompt_id=i, ttft_s=0.1, tpot_s=0.01, completion_tokens=256, token_ids=got))
    c = evaluate_canary(recs, lock)
    assert c["passing"] == 6 and c["passed"] is True
    recs[3].token_ids = [7] * ref.CANARY_TOKENS
    c = evaluate_canary(recs, lock)
    assert c["passing"] == 5 and c["passed"] is False


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
    assert any("unlocked" in r for r in report["certified_reasons"])
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
    lock = Lock(model_revision="abc", model_files={"a.safetensors": "0" * 64}, vllm_version="0.10.1",
                canaries=[Canary(i, list(range(ref.CANARY_TOKENS))) for i in range(ref.CANARY_COUNT)])
    assert lock.is_locked
    info = EngineInfo(name="vllm", version="0.10.1", model_id=ref.MODEL_ID, model_revision="abc", launch_mode="subprocess",
                      launch_args=["--model", ref.MODEL_ID, "--max-num-seqs", "32", "--speculative-config", "{}"])
    runs = [JobResult(job_seconds=40.0, records=[], generated_tokens=ref.GENERATED_TOKENS_PER_JOB, failures=0) for _ in range(3)]
    score = score_runs([40.0, 40.0, 40.0])
    canary = {"passed": True, "passing": 8, "required": 6, "results": []}
    reasons = certification_reasons(info, runs, score, canary, lock)
    assert reasons == ["forbidden flag --speculative-config"]
    info.launch_args = ["--model", ref.MODEL_ID, "--max-num-seqs", "32"]
    assert certification_reasons(info, runs, score, canary, lock) == []
    info.launch_args = ["--model", ref.MODEL_ID, "--max-num-seqs", "64"]
    assert certification_reasons(info, runs, score, canary, lock) == ["--max-num-seqs 64 != 32"]
