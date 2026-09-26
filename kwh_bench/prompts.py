"""Deterministic prompt set for series I-1 (SPEC.md §4).

The prompt set is generated from a seed and a fixed word list, never
downloaded. The canonical file is prompts/i1-prompts.jsonl; its SHA-256 is
pinned in reference.PROMPT_SET_SHA256.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from . import reference as ref
from .words import WORDS


@dataclass(frozen=True)
class Prompt:
    id: int
    text: str


def _sentence(rng: random.Random) -> str:
    n = rng.randint(6, 16)
    words = [rng.choice(WORDS) for _ in range(n)]
    words[0] = words[0].capitalize()
    # Occasional comma to vary tokenization a little; punctuation is cheap
    # to tokenize and keeps prompts looking like text rather than a word soup.
    if n > 9:
        words[rng.randint(3, n - 4)] += ","
    return " ".join(words) + rng.choice([".", ".", ".", "?", "!"])


def generate_prompts(
    count: int = ref.REQUESTS_PER_JOB,
    seed: int = ref.PROMPT_SEED,
    target_words: int = ref.PROMPT_TARGET_WORDS,
) -> List[Prompt]:
    """Return the canonical prompt list.

    Each prompt starts with a unique hex nonce so no two prompts share a
    prefix (defeats prefix caching even if an engine ignores the flag).
    """
    rng = random.Random(seed)
    prompts: List[Prompt] = []
    seen = set()
    for i in range(count):
        while True:
            nonce = f"{rng.getrandbits(4 * ref.PROMPT_NONCE_HEX_CHARS):0{ref.PROMPT_NONCE_HEX_CHARS}x}"
            if nonce not in seen:
                seen.add(nonce)
                break
        parts = [f"[{nonce}]"]
        n_words = 0
        while n_words < target_words:
            s = _sentence(rng)
            parts.append(s)
            n_words += len(s.split())
        prompts.append(Prompt(id=i, text=" ".join(parts)))
    return prompts


def to_jsonl(prompts: Iterable[Prompt]) -> str:
    """Canonical serialization: one compact JSON object per line, LF, trailing newline."""
    lines = [
        json.dumps({"id": p.id, "text": p.text}, ensure_ascii=False, separators=(",", ":"))
        for p in prompts
    ]
    return "\n".join(lines) + "\n"


def from_jsonl(text: str) -> List[Prompt]:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out.append(Prompt(id=int(o["id"]), text=str(o["text"])))
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_prompt_file(path: Path) -> str:
    """Generate and write the canonical file. Returns its SHA-256."""
    text = to_jsonl(generate_prompts())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_text(text)


def load_prompt_file(path: Path, verify: bool = True) -> List[Prompt]:
    text = path.read_text(encoding="utf-8")
    if verify:
        digest = sha256_text(text)
        if digest != ref.PROMPT_SET_SHA256:
            raise ValueError(
                f"prompt set hash mismatch: {digest} != pinned {ref.PROMPT_SET_SHA256}. "
                "Regenerate with `kwh-bench prompts` or check out the canonical file."
            )
    prompts = from_jsonl(text)
    if len(prompts) != ref.REQUESTS_PER_JOB:
        raise ValueError(f"expected {ref.REQUESTS_PER_JOB} prompts, got {len(prompts)}")
    return prompts


def canonical_prompts() -> List[Prompt]:
    """Generate in memory and check against the pinned hash."""
    text = to_jsonl(generate_prompts())
    digest = sha256_text(text)
    if digest != ref.PROMPT_SET_SHA256:
        raise ValueError(
            f"generated prompt set hash {digest} does not match pinned {ref.PROMPT_SET_SHA256}; "
            "the generator or word list has drifted from the series definition"
        )
    return from_jsonl(text)
