"""`kwh-bench verify`: what the platform runs on ingest (SPEC.md §8)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import List, Tuple

import jsonschema

from . import reference as ref
from .lockfile import Lock, load_lock, lock_sha256
from .report import canary_mean_delta, load_schema, max_model_len_problem, report_hash


def verify_report(path: Path, lock: Lock | None = None) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, [f"unreadable: {e}"]

    try:
        jsonschema.validate(report, load_schema())
    except jsonschema.ValidationError as e:
        problems.append(f"schema: {e.message} at {'/'.join(str(p) for p in e.absolute_path)}")
        return False, problems

    if report["spec"]["series"] != ref.SERIES:
        problems.append(f"series {report['spec']['series']} is not {ref.SERIES}; verify with the matching kwh-bench major")

    # Hash integrity
    if report_hash(report) != report["report_sha256"]:
        problems.append("report_sha256 does not match content")
    if report["prompt_set_sha256"] != ref.PROMPT_SET_SHA256:
        problems.append("prompt_set_sha256 is not the pinned I-1 prompt set")

    # Job shape
    j = report["job"]
    for k, want in (("requests", ref.REQUESTS_PER_JOB), ("prompt_tokens", ref.PROMPT_TOKENS),
                    ("generated_tokens", ref.GENERATED_TOKENS), ("concurrency", ref.CONCURRENCY)):
        if j[k] != want:
            problems.append(f"job.{k} = {j[k]}, spec says {want}")

    # Recompute score
    secs = [r["job_seconds"] for r in report["runs"]]
    med = statistics.median(secs)
    uph = round(3600.0 / med, 3)
    if abs(uph - report["score"]["units_per_hour"]) > 0.01:
        problems.append(f"units_per_hour {report['score']['units_per_hour']} != recomputed {uph}")
    stab = round((max(secs) - min(secs)) / med, 5)
    if report["score"]["stability"] is None or abs(stab - report["score"]["stability"]) > 1e-4:
        problems.append(f"stability {report['score']['stability']} != recomputed {stab}")

    for i, r in enumerate(report["runs"]):
        tps = round(r["generated_tokens"] / r["job_seconds"], 2)
        if abs(tps - r["tokens_per_second"]) > 0.05:
            problems.append(f"runs[{i}].tokens_per_second {r['tokens_per_second']} != recomputed {tps}")

    # Certification claims
    if report["certified"]:
        lock = lock or load_lock()
        if not lock.is_locked:
            problems.append("report claims certification but the local lock is incomplete")
        elif report.get("lock_sha256") != lock_sha256(lock):
            problems.append("report lock_sha256 does not match the local kwh_bench/reference/lock.json")
        e = report["engine"]
        if e["name"] not in ref.CERTIFIED_ENGINES:
            problems.append(f"certified report from uncertified engine {e['name']}")
        if e["launch_mode"] == "attached":
            problems.append("certified report from attached engine")
        if e["model_id"] not in (ref.MODEL_ID, *ref.MODEL_ALIASES):
            problems.append(f"certified report on non-reference model {e['model_id']}")
        if lock.model_revision and e["model_revision"] != lock.model_revision:
            problems.append("certified report model revision != lock")
        if lock.vllm_version and e["version"] != lock.vllm_version:
            problems.append("certified report engine version != lock")
        args = " ".join(e["launch_args"])
        for flag in ref.VLLM_FORBIDDEN_FLAGS:
            if f" {flag}" in f" {args}":
                problems.append(f"certified report with forbidden flag {flag}")
        mml = max_model_len_problem(e["launch_args"])
        if mml:
            problems.append(f"certified report with {mml}")
        if len(report["runs"]) < ref.MIN_MEASURED_JOBS:
            problems.append("certified report with too few runs")
        if stab > ref.MAX_STABILITY:
            problems.append("certified report exceeds stability bound")
        if any(r["request_failures"] for r in report["runs"]):
            problems.append("certified report with request failures")
        if any(r["generated_tokens"] != ref.GENERATED_TOKENS_PER_JOB for r in report["runs"]):
            problems.append("certified report with wrong generated token count")
        c = report["canary"]
        if c is None or not c["passed"]:
            problems.append("certified report without passing canary")
        else:
            # Re-judge from the stored deltas under the rule and tolerance in force now
            # (SPEC.md §7): a report scored under an older rule or a looser tolerance
            # is accepted only if it would also pass the current one.
            mean = canary_mean_delta(c["results"])
            tol = ref.CANARY_MAX_LOGPROB_DELTA
            if mean is None:
                problems.append("canary: not every canary has a stored delta, so the current rule cannot pass it")
            elif mean > tol:
                problems.append(f"canary mean delta {mean} exceeds the current tolerance {tol} nats "
                                f"(report was scored under spec {report['spec']['spec_version']})")
        pf = report.get("preflight")
        if pf and pf.get("available") and pf.get("idle") is False:
            problems.append("certified report with a non-idle pre-flight (host contention)")
        if report["certified_reasons"]:
            problems.append("certified=true but certified_reasons is non-empty")

    return not problems, problems
