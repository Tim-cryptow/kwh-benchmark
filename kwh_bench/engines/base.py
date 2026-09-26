"""Engine interface. The load generator only sees this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EngineInfo:
    name: str                      # "vllm" | "llamacpp" | "mock"
    version: Optional[str]
    model_id: str
    model_revision: Optional[str]
    launch_mode: str               # "subprocess" | "docker" | "attached" | "inprocess"
    launch_args: List[str] = field(default_factory=list)
    server_url: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class Completion:
    """Timing and content of one streamed completion, on the client clock (perf_counter)."""
    started_at: float
    first_token_at: float
    last_token_at: float
    completion_tokens: int
    token_ids: Optional[List[int]] = None   # only when requested (canaries)

    @property
    def ttft(self) -> float:
        return self.first_token_at - self.started_at

    @property
    def tpot(self) -> Optional[float]:
        if self.completion_tokens < 2:
            return None
        return (self.last_token_at - self.first_token_at) / (self.completion_tokens - 1)


class Engine(ABC):
    """An inference server the benchmark can drive.

    All methods are coroutine-safe; the load generator calls `complete` from
    up to CONCURRENCY tasks at once.
    """

    name: str = "base"

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def info(self) -> EngineInfo: ...

    @abstractmethod
    async def tokenize(self, text: str) -> List[int]: ...

    @abstractmethod
    async def complete(self, token_ids: List[int], max_tokens: int, want_token_ids: bool = False) -> Completion:
        """Stream exactly `max_tokens` greedy tokens for a prompt given as token ids.

        Implementations must request greedy sampling and ignore EOS so the
        count is fixed (SPEC.md §3), and must record first/last token arrival
        with time.perf_counter().
        """
        ...

    async def score_continuation(self, prompt_ids: List[int], continuation_ids: List[int]) -> List[float]:
        """Teacher-forced per-token logprobs of `continuation_ids` given `prompt_ids`.

        Used by the canary check (SPEC.md §7). Certified engines must implement
        it; uncertified engines may raise NotImplementedError (canary is then
        reported as null).
        """
        raise NotImplementedError

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.stop()
