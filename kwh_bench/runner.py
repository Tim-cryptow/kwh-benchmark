"""Orchestrates one benchmark session (SPEC.md §6) and the lock procedure (§9)."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

from . import reference as ref
from .engines.base import Engine
from .hardware import PowerSampler, probe
from .load import JobResult, prepare_prompts, run_job
from .lockfile import Canary, Lock, default_canary_ids, load_lock, save_lock
from .prompts import canonical_prompts, load_prompt_file, sha256_text, to_jsonl
from .report import build_report, now_iso

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
) -> dict:
    lock = lock or load_lock()
    started_at = now_iso()
    prompts = _prompts(prompt_file)
    prompt_hash = sha256_text(to_jsonl(prompts))

    log("probing hardware")
    hardware = probe()

    log(f"starting engine {engine.name}")
    async with engine:
        info = await engine.info()
        log(f"engine {info.name} {info.version or ''} model={info.model_id} mode={info.launch_mode}")

        log("tokenizing prompt set")
        prepared = await prepare_prompts(engine, prompts)

        canary_ids = lock.canary_ids() if engine.name in ref.CERTIFIED_ENGINES else set()

        log("warm-up job")
        warmup = await run_job(engine, prepared, concurrency=concurrency)
        log(f"  warm-up {warmup.job_seconds:.2f}s  {warmup.tokens_per_second:.0f} tok/s  failures={warmup.failures}")

        sampler = PowerSampler(hz=ref.POWER_SAMPLE_HZ)
        await sampler.start()
        runs: List[JobResult] = []
        try:
            for i in range(measured_jobs):
                res = await run_job(engine, prepared, concurrency=concurrency, want_ids_for=canary_ids if i == 0 else set())
                runs.append(res)
                log(f"  run {i + 1}/{measured_jobs}  {res.job_seconds:.2f}s  {res.tokens_per_second:.0f} tok/s  failures={res.failures}")
        finally:
            trace = await sampler.stop()

    return build_report(
        engine=info,
        hardware=hardware,
        warmup=warmup,
        runs=runs,
        power=trace.to_dict(),
        lock=lock,
        prompt_set_sha256=prompt_hash,
        started_at=started_at,
    )


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
    """Run the certified engine once on the canary prompts and write reference/lock.json."""
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
        log("generating canary continuations (greedy)")
        canaries: List[Canary] = []
        for p in prepared:
            c = await engine.complete(p.token_ids, ref.CANARY_TOKENS, want_token_ids=True)
            if not c.token_ids or len(c.token_ids) < ref.CANARY_TOKENS:
                raise RuntimeError(f"canary {p.id}: got {len(c.token_ids or [])} token ids, need {ref.CANARY_TOKENS}")
            canaries.append(Canary(prompt_id=p.id, expected_token_ids=c.token_ids[: ref.CANARY_TOKENS]))

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
