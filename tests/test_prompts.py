from pathlib import Path

import pytest

from kwh_bench import reference as ref
from kwh_bench.prompts import canonical_prompts, from_jsonl, generate_prompts, load_prompt_file, sha256_text, to_jsonl

REPO = Path(__file__).resolve().parent.parent


def test_generator_is_deterministic_and_pinned():
    a = to_jsonl(generate_prompts())
    b = to_jsonl(generate_prompts())
    assert a == b
    assert sha256_text(a) == ref.PROMPT_SET_SHA256


def test_prompt_count_and_unique_prefixes():
    ps = generate_prompts()
    assert len(ps) == ref.REQUESTS_PER_JOB
    nonces = {p.text.split("]")[0] for p in ps}
    assert len(nonces) == ref.REQUESTS_PER_JOB


def test_prompts_are_long_enough_for_512_tokens():
    # Llama 3 tokenizer averages > 1 token/word on English; 720+ words is a safe margin.
    for p in generate_prompts():
        assert len(p.text.split()) >= ref.PROMPT_TARGET_WORDS


def test_roundtrip_jsonl():
    ps = generate_prompts(count=5)
    assert from_jsonl(to_jsonl(ps)) == ps


def test_canonical_file_matches_generator():
    path = REPO / "prompts" / "i1-prompts.jsonl"
    assert path.exists(), "run `kwh-bench prompts`"
    assert load_prompt_file(path) == canonical_prompts()


def test_load_rejects_tampered_file(tmp_path):
    text = to_jsonl(generate_prompts())
    bad = tmp_path / "p.jsonl"
    bad.write_text(text.replace("Plateau", "Plateaux", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prompt_file(bad)


def test_lock_ships_inside_the_package():
    """A pip-installed kwh-bench (no checkout) must see the same lock as a checkout, or every
    report it produces is 'unlocked'. The lock therefore lives inside the package tree."""
    import kwh_bench
    from pathlib import Path
    from kwh_bench.lockfile import LOCK_PATH, load_lock
    pkg = Path(kwh_bench.__file__).resolve().parent
    assert LOCK_PATH.resolve().is_relative_to(pkg)
    assert LOCK_PATH.exists() and load_lock().is_locked
