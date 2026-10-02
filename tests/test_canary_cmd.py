from pathlib import Path

import pytest

from kwh_bench import reference as ref
from kwh_bench.engines import MockEngine
from kwh_bench.lockfile import load_lock
from kwh_bench.runner import calibration_row, run_canary


@pytest.fixture
def fast_engine():
    return MockEngine(step_ms=0.05, prefill_ms_per_1k=0.1)


async def test_run_canary_against_wrong_engine_fails(fast_engine):
    """The committed lock is the real I-1 lock; the mock is not that model, so it must fail."""
    lock = load_lock()
    assert lock.is_locked
    rec = await run_canary(fast_engine, lock=lock, label="mock-control", log=lambda s: None)
    assert rec["kind"] == "canary-calibration"
    assert rec["is_reference_model"] is True          # mock reports the reference model id...
    c = rec["canary"]
    assert len(c["results"]) == ref.CANARY_COUNT
    assert all(r["delta"] is not None for r in c["results"])
    assert c["passed"] is False                        # ...but its numbers are nothing like the lock's
    assert rec["lock_sha256"] == __import__("kwh_bench.lockfile", fromlist=["lock_sha256"]).lock_sha256(lock)


def test_calibration_row_shapes():
    rec = {
        "hardware": {"gpus": [{"name": "NVIDIA RTX A5000", "compute_capability": "8.6"}]},
        "engine": {"name": "vllm", "version": "0.30.0", "model_id": ref.MODEL_ID},
        "is_reference_model": True,
        "finished_at": "2026-09-26T07:16:32Z",
        "canary": {"passed": True, "passing": 8, "results": [{"delta": 0.0} for _ in range(8)]},
    }
    row = calibration_row(rec)
    assert row.startswith("| 2026-09-26 | NVIDIA RTX A5000 / sm86 | vllm 0.30.0 |")
    assert "0.0000 ×8" in row and "mean 0.0000 PASS" in row and "expected pass: as expected" in row

    rec["is_reference_model"] = False
    rec["engine"]["model_id"] = "meta-llama/Llama-3.1-8B-Instruct"
    rec["canary"] = {"passed": False, "passing": 1, "results": [{"delta": 0.21}, {"delta": 0.34}] + [{"delta": None}] * 6}
    row = calibration_row(rec)
    assert "(control)" in row and "0.2100–0.3400" in row and "mean n/a FAIL" in row and "expected fail: as expected" in row

    rec["canary"]["passed"] = True
    assert "UNEXPECTED" in calibration_row(rec)


def test_append_calibration_row_inserts_after_table(tmp_path):
    from kwh_bench.cli import _append_calibration_row
    md = tmp_path / "c.md"
    md.write_text("# T\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nPending:\n\n- x\n", encoding="utf-8")
    _append_calibration_row(md, "| 3 | 4 |")
    lines = md.read_text().splitlines()
    assert lines[4] == "| 1 | 2 |" and lines[5] == "| 3 | 4 |" and lines[6] == "" and lines[7] == "Pending:"
