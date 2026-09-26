"""kwh-bench command line."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from . import reference as ref
from .engines import LlamaCppEngine, MockEngine, VLLMEngine
from .lockfile import LOCK_PATH, load_lock
from .prompts import canonical_prompts, sha256_text, to_jsonl, write_prompt_file
from .report import summarize, write_report
from .runner import lock_reference, run_benchmark
from .verify import verify_report

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_FILE = REPO_ROOT / "prompts" / "i1-prompts.jsonl"


def _log(s: str) -> None:
    click.echo(s, err=True)


@click.group()
@click.version_option(__version__, prog_name="kwh-bench")
def main():
    """kWh Exchange Grade I benchmark. One unit = one reference job (see SPEC.md)."""


@main.command()
@click.option("--engine", "engine_name", type=click.Choice(["vllm", "llamacpp", "mock"]), default="vllm", show_default=True)
@click.option("--runs", type=int, default=ref.DEFAULT_MEASURED_JOBS, show_default=True, help="Measured jobs after warm-up (min 3 for certification).")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Report path (default results/<timestamp>-<engine>.json).")
@click.option("--server-url", default=None, help="Attach to a running server instead of launching (result is uncertified).")
@click.option("--docker", "docker_image", default=None, help="Launch vLLM in this Docker image (e.g. vllm/vllm-openai:<tag>).")
@click.option("--revision", default=None, help="Checkpoint revision (defaults to reference/lock.json).")
@click.option("--port", type=int, default=None)
@click.option("--gguf", default=None, help="llama.cpp: path to the Q8_0 GGUF.")
@click.option("--hf-cache", default=None, help="Host HF cache dir to mount into the Docker container.")
@click.option("--engine-log", type=click.Path(path_type=Path), default=None, help="Write engine stdout/stderr here.")
@click.option("--prompt-file", type=click.Path(path_type=Path, exists=True), default=None, help="Use this prompt file (must match the pinned hash).")
@click.option("--mock-step-ms", type=float, default=0.5, hidden=True)
@click.option("--concurrency", type=int, default=ref.CONCURRENCY, hidden=True, help="Override for experiments only; any value but the spec's disqualifies the result.")
def run(engine_name, runs, out, server_url, docker_image, revision, port, gguf, hf_cache, engine_log, prompt_file, mock_step_ms, concurrency):
    """Run the Grade I benchmark and write a report."""
    lock = load_lock()
    if engine_name == "vllm":
        engine = VLLMEngine(
            revision=revision or lock.model_revision,
            server_url=server_url,
            docker_image=docker_image or (lock.vllm_image if not server_url else None),
            port=port or 8000,
            log_path=str(engine_log) if engine_log else None,
            hf_cache=hf_cache,
        )
    elif engine_name == "llamacpp":
        engine = LlamaCppEngine(gguf_path=gguf, server_url=server_url, port=port or 8080, log_path=str(engine_log) if engine_log else None)
    else:
        engine = MockEngine(step_ms=mock_step_ms)

    if concurrency != ref.CONCURRENCY:
        _log(f"WARNING: concurrency {concurrency} != spec {ref.CONCURRENCY}; result cannot be certified")

    try:
        report = asyncio.run(run_benchmark(engine, measured_jobs=runs, prompt_file=prompt_file, lock=lock, log=_log, concurrency=concurrency))
    except Exception as e:  # noqa: BLE001
        _log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)

    if concurrency != ref.CONCURRENCY:
        report["job"]["concurrency"] = concurrency
        report["certified"] = False
        report["certified_reasons"].append(f"concurrency {concurrency} != {ref.CONCURRENCY}")
        from .report import report_hash
        report["report_sha256"] = report_hash(report)

    if out is None:
        stamp = report["started_at"].replace(":", "").replace("-", "")
        out = REPO_ROOT / "results" / f"{stamp}-{engine_name}.json"
    write_report(report, out)
    click.echo(summarize(report))
    click.echo(f"report: {out}", err=True)


@main.command()
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_PROMPT_FILE, show_default=True)
@click.option("--check", is_flag=True, help="Only verify the generator reproduces the pinned hash.")
def prompts(out: Path, check: bool):
    """Generate (or check) the canonical I-1 prompt set."""
    if check:
        digest = sha256_text(to_jsonl(canonical_prompts()))
        click.echo(f"OK prompt set sha256 {digest}")
        return
    digest = write_prompt_file(out)
    status = "matches" if digest == ref.PROMPT_SET_SHA256 else "DOES NOT MATCH"
    click.echo(f"wrote {out}\nsha256 {digest} ({status} pinned PROMPT_SET_SHA256)")
    if digest != ref.PROMPT_SET_SHA256:
        sys.exit(1)


@main.command()
@click.argument("report", type=click.Path(path_type=Path, exists=True))
def verify(report: Path):
    """Validate a report: schema, hashes, recomputed score, certification claims."""
    ok, problems = verify_report(report)
    if ok:
        click.echo(f"OK {report}")
        return
    click.echo(f"FAIL {report}")
    for p in problems:
        click.echo(f"  - {p}")
    sys.exit(1)


@main.command()
@click.option("--model-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None, help="Local snapshot dir of the checkpoint (else resolved from the HF cache).")
@click.option("--revision", default=None, help="HF commit hash of the checkpoint.")
@click.option("--docker", "docker_image", default=None, help="Run vLLM in this image; recorded as the certified image.")
@click.option("--server-url", default=None, help="Use a running vLLM (must be launched with the pinned flags).")
@click.option("--port", type=int, default=8000)
@click.option("--engine-log", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=LOCK_PATH, show_default=True)
def lock(model_dir, revision, docker_image, server_url, port, engine_log, out):
    """Reference-node only: hash weights, record engine build, generate canaries -> reference/lock.json."""
    engine = VLLMEngine(revision=revision, server_url=server_url, docker_image=docker_image, port=port, log_path=str(engine_log) if engine_log else None)
    try:
        lk = asyncio.run(lock_reference(engine, model_dir, revision, docker_image, log=_log, out=out))
    except Exception as e:  # noqa: BLE001
        _log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)
    click.echo(json.dumps({"locked": lk.is_locked, "revision": lk.model_revision, "vllm": lk.vllm_version, "canaries": len(lk.canaries)}, indent=2))


@main.command()
def spec():
    """Print the pinned I-1 constants."""
    d = {k: getattr(ref, k) for k in dir(ref) if k.isupper()}
    click.echo(json.dumps(d, indent=2, default=str))


if __name__ == "__main__":
    main()
