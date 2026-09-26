"""Subprocess / Docker launcher shared by the HTTP engines."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import List, Optional

import httpx


class ServerProcess:
    def __init__(self, argv: List[str], health_url: str, log_path: Optional[str] = None, env: Optional[dict] = None):
        self.argv = argv
        self.health_url = health_url
        self.log_path = log_path
        self.env = env
        self.proc: Optional[subprocess.Popen] = None
        self._log = None

    async def start(self, timeout_s: float = 900.0) -> None:
        if self._log is None and self.log_path:
            self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            self.argv,
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(self.env or {})},
            start_new_session=True,
        )
        await wait_healthy(self.health_url, timeout_s, self.proc)

    async def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(100):
                if self.proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            if self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        if self._log:
            self._log.close()
            self._log = None


async def wait_healthy(url: str, timeout_s: float, proc: Optional[subprocess.Popen] = None) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"engine process exited early with code {proc.returncode}")
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"engine did not become healthy at {url} within {timeout_s:.0f}s")


def which_or_module(binary: str, module: Optional[str] = None) -> List[str]:
    """Prefer a binary on PATH; fall back to `python -m module`."""
    path = shutil.which(binary)
    if path:
        return [path]
    if module:
        return [sys.executable, "-m", module]
    raise FileNotFoundError(f"{binary} not found on PATH")


def docker_argv(image: str, gpu: bool, ports: List[str], volumes: List[str], args: List[str], env: Optional[dict] = None) -> List[str]:
    argv = ["docker", "run", "--rm", "--init"]
    if gpu:
        argv += ["--gpus", "all"]
    argv += ["--ipc=host"]
    for p in ports:
        argv += ["-p", p]
    for v in volumes:
        argv += ["-v", v]
    for k, v in (env or {}).items():
        argv += ["-e", f"{k}={v}"]
    argv += [image] + args
    return argv
