"""Orchestrates one benchmark session (SPEC.md §6) and the lock procedure (§9)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

from . import reference as ref
from .engines.base import Engine
from .hardware import HostContentionError, PowerSampler, preflight_gpu, probe
from .load import JobResult, PreparedPrompt, prepare_prompts, run_job
from .lockfile import Canary, Lock, default_canary_ids, load_lock, save_lock
from .prompts import canonical_prompts, load_prompt_file, sha256_text, to_jsonl
from .report import build_report, clock_problem, measure_clock, now_iso

Log = Callable[[str], None]


def _prompts(prompt_file: Optional[Path]):
    if prompt_file:
        return load_prompt_file(prompt_file)
    return canonical_prompts()


async def run_benchmark(
    engine: Engine,
    measured_jobs: int = ref.DEFAULT_MEASURED_JOBS,
    prompt_file: Optional[Path] = None,
    lock: Optional[Lock] = None,
    log: Log = lambda s: print(s, file=sys.stderr),
    concurrency: int = ref.CONCURRENCY,
    ignore_preflight: bool = False,
) -> dict:
    lock = lock or load_lock()
    started_at = now_iso()
    prompts = _prompts(prompt_file)
    prompt_hash = sha256_text(to_jsonl(prompts))

    log("probing hardware")
    hardware = probe()

    # SPEC.md §6 step 0: the GPU must be idle before anything is launched.
    preflight = preflight_gpu(seconds=ref.PREFLIGHT_SECONDS, hz=ref.PREFLIGHT_HZ,
                              max_util_pct=ref.PREFLIGHT_MAX_UTIL_PCT, min_free_fraction=ref.PREFLIGHT_MIN_FREE_FRACTION)
    if preflight["available"]:
        log(f"pre-flight: {preflight['free_fraction_min']:.0%} VRAM free, GPU {preflight['util_pct_mean']}% busy, "
            f"{preflight['power_w_mean']} W -> {'idle' if preflight['idle'] else 'NOT IDLE'}")
        if not preflight["idle"] and not ignore_preflight:
            raise HostContentionError(preflight)
    else:
        log("pre-flight: nvidia-smi unavailable, skipping")

    log(f"starting engine {engine.name}")
    async with engine:
        info = await engine.info()
        log(f"engine {info.name} {info.version or ''} model={info.model_id} mode={info.launch_mode}")

        log("tokenizing prompt set")
        prepared = await prepare_prompts(engine, prompts)

        log("warm-up job")
        warmup = await run_job(engine, prepared, concurrency=concurrency)
        log(f"  warm-up {warmup.job_seconds:.2f}s  {warmup.tokens_per_second:.0f} tok/s  failures={warmup.failures}")

        sampler = PowerSampler(hz=ref.POWER_SAMPLE_HZ)
        await sampler.start()
        runs: List[JobResult] = []
        # SPEC.md §6 (rc.7): the measured runs on two clocks, the jobs' timer and the wall clock.
        timer0, wall0 = time.perf_counter(), time.time()
        try:
            for i in range(measured_jobs):
                res = await run_job(engine, prepared, concurrency=concurrency)
                runs.append(res)
                log(f"  run {i + 1}/{measured_jobs}  {res.job_seconds:.2f}s  {res.tokens_per_second:.0f} tok/s  failures={res.failures}")
        finally:
            timer1, wall1 = time.perf_counter(), time.time()
            trace = await sampler.stop()
        clock = measure_clock(timer0, wall0, timer1, wall1)
        log(f"  clock: timer {clock['timer_seconds']:.1f}s, wall clock {clock['wall_seconds']:.1f}s"
            + (f"  -> {clock_problem(clock)}" if clock_problem(clock) else ""))

        # Canary: score the locked continuations (SPEC.md §7). Outside the timed jobs.
        canary_scores = {}
        if engine.name in ref.CERTIFIED_ENGINES and lock.is_locked:
            log("canary check")
            canary_scores = await score_canaries(engine, prepared, lock, log)

    return build_report(
        engine=info,
        hardware=hardware,
        warmup=warmup,
        runs=runs,
        power=trace.to_dict(),
        lock=lock,
        prompt_set_sha256=prompt_hash,
        started_at=started_at,
        canary_scores=canary_scores,
        preflight=preflight,
        clock=clock,
    )


async def score_canaries(engine: Engine, prepared: List[PreparedPrompt], lock: Lock, log: Log) -> dict:
    """Mean teacher-forced logprob per canary, one request at a time."""
    by_id = {p.id: p for p in prepared}
    out = {}
    for c in lock.canaries:
        p = by_id.get(c.prompt_id)
        if p is None:
            out[c.prompt_id] = None
            continue
        try:
            lps = await engine.score_continuation(p.token_ids, c.expected_token_ids)
            out[c.prompt_id] = sum(lps) / len(lps)
            log(f"  canary {c.prompt_id}: mean logprob {out[c.prompt_id]:+.4f} (reference {c.reference_mean_logprob:+.4f})")
        except NotImplementedError:
            out[c.prompt_id] = None
        except Exception as e:  # noqa: BLE001
            log(f"  canary {c.prompt_id}: error {type(e).__name__}: {e}")
            out[c.prompt_id] = None
    return out


# --- lock ---------------------------------------------------------------

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_model_dir(model_dir: Path) -> dict:
    files = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors") or name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            files[name] = _sha256_file(model_dir / name)
    if not any(k.endswith(".safetensors") for k in files):
        raise FileNotFoundError(f"no .safetensors in {model_dir}")
    return files


def resolve_hf_snapshot(model_id: str, revision: Optional[str]) -> tuple[Path, str]:
    """Locate the local HF cache snapshot for the checkpoint and return (dir, commit)."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required for `lock` (pip install huggingface_hub)") from e
    path = Path(snapshot_download(model_id, revision=revision, allow_patterns=["*.safetensors", "*.json"]))
    commit = path.name  # snapshots/<commit>
    return path, commit


async def lock_reference(
    engine: Engine,
    model_dir: Optional[Path],
    revision: Optional[str],
    docker_image: Optional[str],
    log: Log = lambda s: print(s, file=sys.stderr),
    out: Optional[Path] = None,
) -> Lock:
    """Run the certified engine once on the canary prompts and write kwh_bench/reference/lock.json."""
    if model_dir is None:
        model_dir, commit = resolve_hf_snapshot(ref.MODEL_ID, revision)
        revision = revision or commit
    log(f"hashing weights in {model_dir}")
    files = hash_model_dir(model_dir)

    prompts = canonical_prompts()
    ids = default_canary_ids()
    by_id = {p.id: p for p in prompts}

    async with engine:
        info = await engine.info()
        if info.name not in ref.CERTIFIED_ENGINES:
            raise ValueError("lock must run on a certified engine")
        prepared = await prepare_prompts(engine, [by_id[i] for i in ids])
        log("generating canary continuations (greedy) and scoring them")
        canaries: List[Canary] = []
        for p in prepared:
            c = await engine.complete(p.token_ids, ref.CANARY_TOKENS, want_token_ids=True)
            if not c.token_ids or len(c.token_ids) < ref.CANARY_TOKENS:
                raise RuntimeError(f"canary {p.id}: got {len(c.token_ids or [])} token ids, need {ref.CANARY_TOKENS}")
            expected = c.token_ids[: ref.CANARY_TOKENS]
            lps = await engine.score_continuation(p.token_ids, expected)
            mean_lp = sum(lps) / len(lps)
            log(f"  canary {p.id}: mean logprob {mean_lp:+.4f}")
            canaries.append(Canary(prompt_id=p.id, expected_token_ids=expected, reference_mean_logprob=round(mean_lp, 6)))

    lock = Lock(
        model_revision=revision,
        model_files=files,
        vllm_version=info.version,
        vllm_image=docker_image or info.extra.get("docker_image"),
        canaries=canaries,
        locked_at=now_iso(),
        locked_on=probe(),
    )
    digest = save_lock(lock, out) if out else save_lock(lock)
    log(f"wrote lock ({'complete' if lock.is_locked else 'INCOMPLETE'}) sha256={digest}")
    return lock


# --- canary-only run (calibration) ---------------------------------------

async def run_canary(engine: Engine, lock: Optional[Lock] = None, label: str = "", log: Log = lambda s: print(s, file=sys.stderr)) -> dict:
    """Score the locked canaries against `engine` without a benchmark run.

    Used to calibrate SPEC.md §7: run it against the reference model on other
    GPU architectures (must pass) and against wrong models — FP16, 4-bit, a
    different model — which must fail. Writes nothing itself; the CLI does.
    """
    from .lockfile import lock_sha256
    from .report import evaluate_canary

    lock = lock or load_lock()
    if not lock.is_locked:
        raise ValueError("kwh_bench/reference/lock.json is incomplete; nothing to score against")
    prompts = canonical_prompts()
    by_id = {p.id: p for p in prompts}
    ids = sorted(lock.canary_ids())
    started = now_iso()
    hardware = probe()
    async with engine:
        info = await engine.info()
        log(f"engine {info.name} {info.version or ''} model={info.model_id} mode={info.launch_mode}")
        prepared = await prepare_prompts(engine, [by_id[i] for i in ids])
        scores = await score_canaries(engine, prepared, lock, log)
    canary = evaluate_canary(scores, lock)
    return {
        "kind": "canary-calibration",
        "label": label or info.model_id,
        "spec": {"series": ref.SERIES, "spec_version": ref.SPEC_VERSION},
        "engine": {"name": info.name, "version": info.version, "model_id": info.model_id,
                   "model_revision": info.model_revision, "launch_mode": info.launch_mode,
                   "launch_args": list(info.launch_args)},
        "is_reference_model": info.model_id in (ref.MODEL_ID, *ref.MODEL_ALIASES),
        "hardware": hardware,
        "lock_sha256": lock_sha256(lock),
        "canary": canary,
        "started_at": started,
        "finished_at": now_iso(),
    }


def calibration_row(rec: dict) -> str:
    """One Markdown table row for results/canary-calibration.md."""
    gpu = (rec["hardware"].get("gpus") or [{"name": "no GPU"}])[0]
    cc = gpu.get("compute_capability")
    arch = f"{gpu['name']}" + (f" / sm{cc.replace('.', '')}" if cc else "")
    c = rec["canary"]
    deltas = [r["delta"] for r in c["results"]]
    known = [d for d in deltas if d is not None]
    if known:
        span = f"{min(known):.4f}–{max(known):.4f}" if min(known) != max(known) else f"{known[0]:.4f} ×{len(known)}"
    else:
        span = "n/a"
    from .report import canary_mean_delta
    mean = canary_mean_delta(c["results"])
    model = rec["engine"]["model_id"] + ("" if rec["is_reference_model"] else " (control)")
    expect = "pass" if rec["is_reference_model"] else "fail"
    verdict = "PASS" if c["passed"] else "FAIL"
    ok = "as expected" if (c["passed"] == rec["is_reference_model"]) else "UNEXPECTED"
    date = rec["finished_at"][:10]
    return (f"| {date} | {arch} | {rec['engine']['name']} {rec['engine']['version'] or ''} | {model} | {span} "
            f"| mean {'n/a' if mean is None else f'{mean:.4f}'} {verdict} | expected {expect}: {ok} |")
