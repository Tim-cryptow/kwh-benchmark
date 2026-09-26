"""Mock engine: no inference, simulated timing. For tests and CI.

It models a batch-decoding server: each decode step takes `step_ms` for the
whole batch (independent of how many requests are in flight, like a GPU
under a fixed batch), prefill costs `prefill_ms_per_1k` per 1k prompt
tokens. Rates it produces are meaningless as units and are marked
uncertified.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import List

from .base import Completion, Engine, EngineInfo
from .. import reference as ref


class MockEngine(Engine):
    name = "mock"

    def __init__(self, step_ms: float = 0.5, prefill_ms_per_1k: float = 2.0, vocab: int = 128256):
        self.step_ms = step_ms
        self.prefill_ms_per_1k = prefill_ms_per_1k
        self.vocab = vocab
        self._started = False

    async def start(self) -> None:
        await asyncio.sleep(0)
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def info(self) -> EngineInfo:
        return EngineInfo(
            name="mock",
            version="0",
            model_id=ref.MODEL_ID,
            model_revision=None,
            launch_mode="inprocess",
            launch_args=[f"step_ms={self.step_ms}", f"prefill_ms_per_1k={self.prefill_ms_per_1k}"],
        )

    async def tokenize(self, text: str) -> List[int]:
        # Roughly 1.3 tokens per word, deterministic, no real tokenizer.
        out: List[int] = []
        for w in text.split():
            h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "big")
            out.append(h % self.vocab)
            if len(w) > 6:
                out.append((h >> 7) % self.vocab)
        return out

    async def score_continuation(self, prompt_ids: List[int], continuation_ids: List[int]) -> List[float]:
        # Deterministic pseudo-logprobs so lock/run round-trips are exact in tests.
        out = []
        for i, t in enumerate(continuation_ids):
            h = int.from_bytes(hashlib.blake2b(f"{prompt_ids[:4]}|{i}|{t}".encode(), digest_size=4).digest(), "big")
            out.append(-(h % 3000) / 1000.0)
        await asyncio.sleep(0)
        return out

    async def complete(self, token_ids: List[int], max_tokens: int, want_token_ids: bool = False) -> Completion:
        assert self._started, "engine not started"
        started = time.perf_counter()
        await asyncio.sleep(self.prefill_ms_per_1k * len(token_ids) / 1000 / 1000)
        seed = int.from_bytes(hashlib.blake2b(str(token_ids[:8]).encode(), digest_size=8).digest(), "big")
        ids: List[int] = []
        first = last = None
        for _ in range(max_tokens):
            await asyncio.sleep(self.step_ms / 1000)
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            ids.append((seed >> 33) % self.vocab)
        return Completion(
            started_at=started,
            first_token_at=first,
            last_token_at=last,
            completion_tokens=len(ids),
            token_ids=ids if want_token_ids else None,
        )
