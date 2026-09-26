import json

import pytest

from kwh_bench import hardware
from kwh_bench import reference as ref
from kwh_bench.engines import MockEngine
from kwh_bench.hardware import HostContentionError, preflight_gpu
from kwh_bench.runner import run_benchmark


def _fake_smi(monkeypatch, line):
    """Pretend nvidia-smi exists and answers every query with `line`."""
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(hardware, "_run", lambda argv, timeout=10.0: line)


def test_preflight_flags_the_contaminated_5090(monkeypatch):
    # Exactly what the RunPod community 5090 showed on 2026-09-26: 5695 MiB used, 100% busy, 400 W, nothing of ours running.
    _fake_smi(monkeypatch, "5695, 32607, 100, 399.99")
    pf = preflight_gpu(seconds=0.5, hz=2)
    assert pf["available"] and pf["idle"] is False
    assert pf["free_fraction_min"] == round(1 - 5695 / 32607, 4)
    assert any("VRAM free" in r for r in pf["reasons"]) and any("busy" in r for r in pf["reasons"])


def test_preflight_passes_an_idle_card(monkeypatch):
    # The A5000 before the lock: 0 MiB used, 0%, 27 W.
    _fake_smi(monkeypatch, "0, 24564, 0, 27.04")
    pf = preflight_gpu(seconds=0.5, hz=2)
    assert pf["idle"] is True and pf["reasons"] == [] and pf["power_w_mean"] == 27.0


def test_preflight_tolerates_driver_baseline(monkeypatch):
    # ~1.5% VRAM for a display/driver context and 2% utilization is still idle.
    _fake_smi(monkeypatch, "350, 24564, 2, 30")
    assert preflight_gpu(seconds=0.5, hz=2)["idle"] is True


def test_preflight_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    pf = preflight_gpu(seconds=0.5)
    assert pf["available"] is False and pf["idle"] is None


async def test_run_benchmark_refuses_a_busy_gpu(monkeypatch):
    _fake_smi(monkeypatch, "5695, 32607, 100, 399.99")
    monkeypatch.setattr(ref, "PREFLIGHT_SECONDS", 0.5)
    with pytest.raises(HostContentionError) as ei:
        await run_benchmark(MockEngine(step_ms=0.05), measured_jobs=3, log=lambda s: None)
    assert "host_contention" in str(ei.value)


async def test_ignore_preflight_yields_uncertified_report_with_reason(monkeypatch):
    _fake_smi(monkeypatch, "5695, 32607, 100, 399.99")
    monkeypatch.setattr(ref, "PREFLIGHT_SECONDS", 0.5)
    report = await run_benchmark(MockEngine(step_ms=0.05), measured_jobs=3, log=lambda s: None, ignore_preflight=True)
    assert report["preflight"]["idle"] is False
    assert report["certified"] is False
    assert any(r.startswith("host_contention") for r in report["certified_reasons"])


def test_verify_rejects_certified_report_with_busy_preflight(tmp_path):
    from kwh_bench.verify import verify_report
    from kwh_bench.report import report_hash
    # Take the real certified A5000 report, splice in a non-idle preflight, rehash, and verify must fail.
    from pathlib import Path
    r = json.loads((Path(__file__).resolve().parent.parent / "results" / "a5000-runpod.json").read_text())
    r["preflight"] = {"available": True, "idle": False, "reasons": ["host_contention: GPU 100% busy before launch, need <= 5%"]}
    r["report_sha256"] = report_hash(r)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r))
    ok, problems = verify_report(p)
    assert not ok and any("host contention" in x for x in problems)
