"""Load generator: executes one reference job (SPEC.md §3, §6).

Exactly CONCURRENCY requests are kept in flight until the queue of 256
prompts drains. Job wall time is from the dispatch of the first request to
the arrival of the last token of the last request, on this process's clock.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import reference as ref
from .engines.base import Completion, Engine


@dataclass
class PreparedPrompt:
    id: int
    token_ids: List[int]


@dataclass
class RequestRecord:
    prompt_id: int
    ttft_s: float
    tpot_s: Optional[float]
    completion_tokens: int
    token_ids: Optional[List[int]] = None
    error: Optional[str] = None


@dataclass
class JobResult:
    job_seconds: float
    records: List[RequestRecord]
    generated_tokens: int
    failures: int

    @property
    def job_seconds_rounded(self) -> float:
        """All derived numbers are computed from this so `verify` can reproduce them exactly."""
        return round(self.job_seconds, 4)

    @property
    def tokens_per_second(self) -> float:
        s = self.job_seconds_rounded
        return self.generated_tokens / s if s > 0 else 0.0

    def percentiles(self) -> Dict[str, Optional[float]]:
        ttfts = sorted(r.ttft_s for r in self.records if r.error is None)
        tpots = sorted(r.tpot_s for r in self.records if r.error is None and r.tpot_s is not None)
        return {
            "ttft_p50_s": _pct(ttfts, 50),
            "ttft_p95_s": _pct(ttfts, 95),
            "tpot_p50_s": _pct(tpots, 50),
            "tpot_p95_s": _pct(tpots, 95),
        }

    def to_dict(self) -> dict:
        d = {
            "job_seconds": self.job_seconds_rounded,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "request_failures": self.failures,
        }
        d.update({k: (round(v, 5) if v is not None else None) for k, v in self.percentiles().items()})
        return d


def _pct(sorted_vals: Sequence[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


async def prepare_prompts(engine: Engine, prompts, prompt_tokens: int = ref.PROMPT_TOKENS) -> List[PreparedPrompt]:
    """Tokenize with the engine and truncate to exactly `prompt_tokens` ids."""
    sem = asyncio.Semaphore(16)

    async def one(p) -> PreparedPrompt:
        async with sem:
            ids = await engine.tokenize(p.text)
        if len(ids) < prompt_tokens:
            raise ValueError(f"prompt {p.id} tokenizes to {len(ids)} < {prompt_tokens} tokens; spec defect")
        return PreparedPrompt(id=p.id, token_ids=ids[:prompt_tokens])

    return await asyncio.gather(*(one(p) for p in prompts))


async def run_job(
    engine: Engine,
    prepared: List[PreparedPrompt],
    concurrency: int = ref.CONCURRENCY,
    max_tokens: int = ref.GENERATED_TOKENS,
    want_ids_for: Optional[set] = None,
) -> JobResult:
    want_ids_for = want_ids_for or set()
    queue: asyncio.Queue = asyncio.Queue()
    for p in prepared:
        queue.put_nowait(p)
    records: List[RequestRecord] = []
    lock = asyncio.Lock()
    last_token_at = [0.0]

    async def worker():
        while True:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                c: Completion = await engine.complete(p.token_ids, max_tokens, want_token_ids=(p.id in want_ids_for))
                rec = RequestRecord(prompt_id=p.id, ttft_s=c.ttft, tpot_s=c.tpot,
                                    completion_tokens=c.completion_tokens, token_ids=c.token_ids)
                async with lock:
                    last_token_at[0] = max(last_token_at[0], c.last_token_at)
            except Exception as e:  # noqa: BLE001 - recorded, not swallowed
                rec = RequestRecord(prompt_id=p.id, ttft_s=0.0, tpot_s=None, completion_tokens=0, error=f"{type(e).__name__}: {e}")
            async with lock:
                records.append(rec)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    t1 = last_token_at[0] if last_token_at[0] > t0 else time.perf_counter()
    records.sort(key=lambda r: r.prompt_id)
    failures = sum(1 for r in records if r.error)
    generated = sum(r.completion_tokens for r in records)
    return JobResult(job_seconds=t1 - t0, records=records, generated_tokens=generated, failures=failures)


def score_runs(job_seconds: List[float]) -> dict:
    job_seconds = [round(s, 4) for s in job_seconds]
    med = statistics.median(job_seconds)
    stability = (max(job_seconds) - min(job_seconds)) / med if med > 0 else None
    return {
        "median_job_seconds": round(med, 4),
        "min_job_seconds": round(min(job_seconds), 4),
        "max_job_seconds": round(max(job_seconds), 4),
        "stability": round(stability, 5) if stability is not None else None,
        "units_per_hour": round(3600.0 / med, 3) if med > 0 else None,
    }
