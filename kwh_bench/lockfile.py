"""kwh_bench/reference/lock.json: fields that need the real weights and engine build (SPEC.md §9)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import reference as ref

# Ships inside the package so a pip-installed kwh-bench (and anything that depends on
# it, such as the host client) sees the same lock as a checkout.
LOCK_PATH = Path(__file__).resolve().parent / "reference" / "lock.json"


@dataclass
class Canary:
    prompt_id: int
    expected_token_ids: List[int]                 # reference node's greedy continuation
    reference_mean_logprob: Optional[float] = None  # its mean per-token logprob, teacher-forced


@dataclass
class Lock:
    series: str = ref.SERIES
    spec_version: str = ref.SPEC_VERSION
    model_id: str = ref.MODEL_ID
    model_revision: Optional[str] = None
    model_files: Dict[str, str] = field(default_factory=dict)          # filename -> sha256
    vllm_version: Optional[str] = None
    vllm_image: Optional[str] = None
    canaries: List[Canary] = field(default_factory=list)
    locked_at: Optional[str] = None
    locked_on: Optional[dict] = None

    @property
    def is_locked(self) -> bool:
        return bool(
            self.model_revision and self.model_files and self.vllm_version
            and len(self.canaries) == ref.CANARY_COUNT
            and all(len(c.expected_token_ids) == ref.CANARY_TOKENS and c.reference_mean_logprob is not None
                    for c in self.canaries)
        )

    def canary_ids(self) -> set:
        return {c.prompt_id for c in self.canaries}

    def to_dict(self) -> dict:
        return {
            "series": self.series,
            "spec_version": self.spec_version,
            "model": {"id": self.model_id, "revision": self.model_revision, "files": self.model_files},
            "engine": {"vllm": {"version": self.vllm_version, "image": self.vllm_image}},
            "canaries": [
                {"prompt_id": c.prompt_id, "expected_token_ids": c.expected_token_ids,
                 "reference_mean_logprob": c.reference_mean_logprob}
                for c in self.canaries
            ],
            "locked_at": self.locked_at,
            "locked_on": self.locked_on,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lock":
        m = d.get("model", {})
        e = d.get("engine", {}).get("vllm", {})
        return cls(
            series=d.get("series", ref.SERIES),
            spec_version=d.get("spec_version", ref.SPEC_VERSION),
            model_id=m.get("id", ref.MODEL_ID),
            model_revision=m.get("revision"),
            model_files=m.get("files") or {},
            vllm_version=e.get("version"),
            vllm_image=e.get("image"),
            canaries=[
                Canary(int(c["prompt_id"]), [int(t) for t in c["expected_token_ids"]],
                       (float(c["reference_mean_logprob"]) if c.get("reference_mean_logprob") is not None else None))
                for c in d.get("canaries", [])
            ],
            locked_at=d.get("locked_at"),
            locked_on=d.get("locked_on"),
        )


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_lock(path: Path = LOCK_PATH) -> Lock:
    if not path.exists():
        return Lock()
    return Lock.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_lock(lock: Lock, path: Path = LOCK_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(lock.to_dict(), indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return lock_sha256(lock)


def lock_sha256(lock: Lock) -> str:
    return hashlib.sha256(canonical_json(lock.to_dict()).encode("utf-8")).hexdigest()


def default_canary_ids(count: int = ref.CANARY_COUNT, total: int = ref.REQUESTS_PER_JOB) -> List[int]:
    """Evenly spaced prompt ids; fixed for the series so the same prompts are canaries everywhere."""
    step = total // count
    return [i * step + step // 2 for i in range(count)]
