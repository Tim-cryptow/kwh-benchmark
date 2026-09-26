"""Hardware/software probe and GPU power sampler (SPEC.md §6, §8)."""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

from . import __version__

NVSMI_GPU_FIELDS = [
    "name", "memory.total", "driver_version", "pcie.link.gen.current",
    "pcie.link.width.current", "power.limit", "uuid", "compute_cap",
]


def _run(argv: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def probe_gpus() -> List[Dict[str, object]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", f"--query-gpu={','.join(NVSMI_GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for row in out.splitlines():
        vals = [v.strip() for v in row.split(",")]
        if len(vals) != len(NVSMI_GPU_FIELDS):
            continue
        rec = dict(zip(NVSMI_GPU_FIELDS, vals))
        gpus.append({
            "name": rec["name"],
            "vram_mib": _int(rec["memory.total"]),
            "driver_version": rec["driver_version"],
            "pcie_gen": _int(rec["pcie.link.gen.current"]),
            "pcie_width": _int(rec["pcie.link.width.current"]),
            "power_limit_w": _float(rec["power.limit"]),
            "uuid": rec["uuid"],
            "compute_capability": rec["compute_cap"],
        })
    return gpus


def probe_cuda_version() -> Optional[str]:
    out = _run(["nvidia-smi"])
    if out:
        for line in out.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].split("|")[0].strip()
    return None


def probe() -> Dict[str, object]:
    vm = psutil.virtual_memory()
    return {
        "gpus": probe_gpus(),
        "cuda_version": probe_cuda_version(),
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gib": round(vm.total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "kwh_bench": __version__,
        "hostname_hash": _hostname_hash(),
    }


def _hostname_hash() -> str:
    import hashlib
    return hashlib.sha256(platform.node().encode()).hexdigest()[:16]


def _int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except ValueError:
        return None


def _float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class PowerSample:
    t: float
    watts: float


@dataclass
class PowerTrace:
    samples: List[PowerSample] = field(default_factory=list)
    available: bool = False

    def mean_watts(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        return sum(s.watts for s in self.samples) / len(self.samples)

    def energy_wh(self) -> Optional[float]:
        """Trapezoidal integration of the trace."""
        if len(self.samples) < 2:
            return None
        e = 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            e += (a.watts + b.watts) / 2 * (b.t - a.t)
        return e / 3600.0

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "samples": len(self.samples),
            "mean_watts": round(self.mean_watts(), 1) if self.mean_watts() is not None else None,
            "energy_wh": round(self.energy_wh(), 3) if self.energy_wh() is not None else None,
        }


class PowerSampler:
    """Samples total GPU board power via nvidia-smi at a fixed rate."""

    def __init__(self, hz: float = 1.0, gpu_index: Optional[int] = None):
        self.period = 1.0 / hz
        self.gpu_index = gpu_index
        self.trace = PowerTrace()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def _read(self) -> Optional[float]:
        argv = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        if self.gpu_index is not None:
            argv += ["-i", str(self.gpu_index)]
        out = _run(argv, timeout=3.0)
        if not out:
            return None
        vals = [_float(v) for v in out.splitlines()]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            w = await loop.run_in_executor(None, self._read)
            if w is not None:
                self.trace.samples.append(PowerSample(t=time.monotonic(), watts=w))
                self.trace.available = True
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.period)
            except asyncio.TimeoutError:
                pass

    async def start(self):
        if not shutil.which("nvidia-smi"):
            self.trace.available = False
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> PowerTrace:
        self._stop.set()
        if self._task:
            await self._task
        return self.trace
