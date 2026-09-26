"""Report assembly, canary evaluation, certification and hashing (SPEC.md §7-§8)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import jsonschema

from . import __version__
from . import reference as ref
from .engines.base import EngineInfo
from .load import JobResult, RequestRecord, score_runs
from .lockfile import Lock, canonical_json, lock_sha256

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "report.schema.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(report: dict) -> None:
    jsonschema.validate(report, load_schema())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def report_hash(report: dict) -> str:
    body = dict(report)
    body["report_sha256"] = None
    body["signature"] = None
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# --- canary -------------------------------------------------------------

def evaluate_canary(records: List[RequestRecord], lock: Lock) -> Optional[dict]:
    if not lock.canaries or any(len(c.expected_token_ids) < ref.CANARY_TOKENS for c in lock.canaries):
        return None  # unlocked: nothing to compare against
    by_id = {r.prompt_id: r for r in records}
    results = []
    for c in lock.canaries:
        r = by_id.get(c.prompt_id)
        got = (r.token_ids or []) if r and r.error is None else []
        n = min(ref.CANARY_TOKENS, len(c.expected_token_ids))
        matched = sum(1 for i in range(n) if i < len(got) and got[i] == c.expected_token_ids[i])
        results.append({"prompt_id": c.prompt_id, "matched": matched, "of": n, "pass": matched >= ref.CANARY_MIN_TOKEN_MATCH})
    passing = sum(1 for x in results if x["pass"])
    return {"passed": passing >= ref.CANARY_MIN_PASSING, "passing": passing, "required": ref.CANARY_MIN_PASSING, "results": results}


# --- certification ------------------------------------------------------

def certification_reasons(engine: EngineInfo, runs: List[JobResult], score: dict, canary: Optional[dict], lock: Lock) -> List[str]:
    reasons: List[str] = []
    if not lock.is_locked:
        reasons.append("unlocked: reference/lock.json is incomplete (spec is a release candidate)")
    if engine.name not in ref.CERTIFIED_ENGINES:
        reasons.append(ref.UNCERTIFIED_REASONS.get(engine.name, f"engine {engine.name} not certified for {ref.SERIES}"))
    if engine.launch_mode == "attached":
        reasons.append("attached: engine configuration not launched by kwh-bench, unverifiable")
    if engine.name == "vllm":
        if lock.vllm_version and engine.version and engine.version != lock.vllm_version:
            reasons.append(f"engine version {engine.version} != locked {lock.vllm_version}")
        if engine.model_id not in (ref.MODEL_ID, *ref.MODEL_ALIASES):
            reasons.append(f"model {engine.model_id} is not the reference checkpoint")
        if lock.model_revision and engine.model_revision and engine.model_revision != lock.model_revision:
            reasons.append(f"model revision {engine.model_revision} != locked {lock.model_revision}")
        args = " ".join(engine.launch_args)
        for flag in ref.VLLM_FORBIDDEN_FLAGS:
            if f" {flag}" in f" {args}":
                reasons.append(f"forbidden flag {flag}")
        if "--max-num-seqs" in engine.launch_args:
            v = engine.launch_args[engine.launch_args.index("--max-num-seqs") + 1]
            if v != str(ref.CONCURRENCY):
                reasons.append(f"--max-num-seqs {v} != {ref.CONCURRENCY}")
    if len(runs) < ref.MIN_MEASURED_JOBS:
        reasons.append(f"only {len(runs)} measured runs, minimum {ref.MIN_MEASURED_JOBS}")
    if any(r.failures for r in runs):
        reasons.append("request failures in measured runs")
    if any(r.generated_tokens != ref.GENERATED_TOKENS_PER_JOB for r in runs):
        reasons.append(f"generated tokens per job != {ref.GENERATED_TOKENS_PER_JOB}")
    if score.get("stability") is None or score["stability"] > ref.MAX_STABILITY:
        reasons.append(f"unstable: stability {score.get('stability')} > {ref.MAX_STABILITY}")
    if engine.name in ref.CERTIFIED_ENGINES:
        if canary is None:
            reasons.append("canary: not evaluated")
        elif not canary["passed"]:
            reasons.append(f"canary: {canary['passing']}/{len(canary['results'])} passed, need {canary['required']}")
    return reasons


# --- assembly -----------------------------------------------------------

def build_report(
    *,
    engine: EngineInfo,
    hardware: dict,
    warmup: JobResult,
    runs: List[JobResult],
    power: dict,
    lock: Lock,
    prompt_set_sha256: str,
    started_at: str,
    finished_at: Optional[str] = None,
) -> dict:
    score = score_runs([r.job_seconds for r in runs])
    mean_w = power.get("mean_watts")
    score["mean_power_watts"] = mean_w
    score["units_per_electric_kwh"] = (
        round(score["units_per_hour"] / (mean_w / 1000.0), 2) if (mean_w and score.get("units_per_hour")) else None
    )
    canary = evaluate_canary(runs[0].records, lock) if engine.name in ref.CERTIFIED_ENGINES else None
    reasons = certification_reasons(engine, runs, score, canary, lock)
    report = {
        "spec": {"series": ref.SERIES, "spec_version": ref.SPEC_VERSION, "bench_version": __version__},
        "job": {
            "requests": ref.REQUESTS_PER_JOB,
            "prompt_tokens": ref.PROMPT_TOKENS,
            "generated_tokens": ref.GENERATED_TOKENS,
            "generated_tokens_per_job": ref.GENERATED_TOKENS_PER_JOB,
            "concurrency": ref.CONCURRENCY,
            "sampling": dict(ref.SAMPLING),
        },
        "engine": {
            "name": engine.name,
            "version": engine.version,
            "model_id": engine.model_id,
            "model_revision": engine.model_revision,
            "launch_mode": engine.launch_mode,
            "launch_args": list(engine.launch_args),
            "server_url": engine.server_url,
            "certified_engine": engine.name in ref.CERTIFIED_ENGINES,
            "extra": {k: v for k, v in engine.extra.items() if v is not None},
        },
        "hardware": hardware,
        "warmup": warmup.to_dict(),
        "runs": [r.to_dict() for r in runs],
        "score": score,
        "canary": canary,
        "power": power,
        "certified": not reasons,
        "certified_reasons": reasons,
        "prompt_set_sha256": prompt_set_sha256,
        "lock_sha256": lock_sha256(lock) if lock.is_locked else None,
        "started_at": started_at,
        "finished_at": finished_at or now_iso(),
        "report_sha256": None,
        "signature": None,
    }
    report["report_sha256"] = report_hash(report)
    validate(report)
    return report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(report: dict) -> str:
    s = report["score"]
    gpu = (report["hardware"]["gpus"] or [{"name": "no GPU detected"}])[0]["name"]
    lines = [
        f"kWh Grade I  series {report['spec']['series']}  spec {report['spec']['spec_version']}",
        f"GPU: {gpu}   engine: {report['engine']['name']} {report['engine']['version'] or ''} ({report['engine']['launch_mode']})",
        f"units/hour: {s['units_per_hour']}   median job: {s['median_job_seconds']} s   stability: {s['stability']}",
    ]
    if s["mean_power_watts"] is not None:
        lines.append(f"mean power: {s['mean_power_watts']} W   units per electric kWh: {s['units_per_electric_kwh']}")
    r0 = report["runs"][0]
    lines.append(f"run 1: {r0['tokens_per_second']} tok/s   TTFT p50 {fmt_ms(r0['ttft_p50_s'])} p95 {fmt_ms(r0['ttft_p95_s'])}   TPOT p50 {fmt_ms(r0['tpot_p50_s'])} p95 {fmt_ms(r0['tpot_p95_s'])}")
    if report["canary"] is not None:
        c = report["canary"]
        lines.append(f"canary: {'PASS' if c['passed'] else 'FAIL'} ({c['passing']}/{len(c['results'])})")
    lines.append("certified: YES" if report["certified"] else "certified: NO\n  - " + "\n  - ".join(report["certified_reasons"]))
    lines.append(f"report sha256: {report['report_sha256']}")
    return "\n".join(lines)


def fmt_ms(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 1000:.1f}ms"
