"""llama.cpp server engine (uncertified for I-1: Q8_0 is weight-only 8-bit).

Endpoints used (llama-server native API):
  GET  /health
  GET  /props              -> build info
  POST /tokenize           {"content"} -> {"tokens": [...]}
  POST /completion         prompt as token ids, streamed, return_tokens=true
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

import httpx

from .base import Completion, Engine, EngineInfo
from .launcher import ServerProcess, wait_healthy, which_or_module
from .. import reference as ref


class LlamaCppEngine(Engine):
    name = "llamacpp"

    def __init__(
        self,
        gguf_path: Optional[str] = None,
        server_url: Optional[str] = None,
        port: int = 8080,
        extra_args: Optional[List[str]] = None,
        log_path: Optional[str] = None,
    ):
        self.gguf_path = gguf_path
        self.port = port
        self.extra_args = list(extra_args or [])
        self.log_path = log_path
        self.attached = server_url is not None
        self.base_url = (server_url or f"http://127.0.0.1:{port}").rstrip("/")
        self._proc: Optional[ServerProcess] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._launch_argv: List[str] = []

    def build_argv(self) -> List[str]:
        if not self.gguf_path:
            raise ValueError("--gguf is required to launch llama-server")
        head = which_or_module("llama-server")
        return head + ["-m", self.gguf_path, "--port", str(self.port), "--host", "127.0.0.1"] + list(ref.LLAMACPP_ARGS) + self.extra_args

    async def start(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(600.0, connect=10.0))
        if not self.attached:
            self._launch_argv = self.build_argv()
            self._proc = ServerProcess(self._launch_argv, f"{self.base_url}/health", log_path=self.log_path)
            await self._proc.start()
        else:
            await wait_healthy(f"{self.base_url}/health", timeout_s=30)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._proc:
            await self._proc.stop()
            self._proc = None

    async def info(self) -> EngineInfo:
        version = None
        model = self.gguf_path or "attached"
        try:
            r = await self._client.get("/props")
            if r.status_code == 200:
                props = r.json()
                version = str(props.get("build_info") or props.get("version") or "")
                model = props.get("model_path") or model
        except httpx.HTTPError:
            pass
        return EngineInfo(
            name="llamacpp",
            version=version or None,
            model_id=model,
            model_revision=None,
            launch_mode="attached" if self.attached else "subprocess",
            launch_args=list(self._launch_argv),
            server_url=self.base_url,
        )

    async def tokenize(self, text: str) -> List[int]:
        r = await self._client.post("/tokenize", json={"content": text, "add_special": False})
        r.raise_for_status()
        toks = r.json()["tokens"]
        return [t["id"] if isinstance(t, dict) else int(t) for t in toks]

    async def complete(self, token_ids: List[int], max_tokens: int, want_token_ids: bool = False) -> Completion:
        body = {
            "prompt": token_ids,
            "n_predict": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "seed": 0,
            "cache_prompt": False,
            "stream": True,
            "return_tokens": True,
        }
        ids: List[int] = []
        first = last = None
        n = 0
        started = time.perf_counter()
        async with self._client.stream("POST", "/completion", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                obj = json.loads(line[5:].strip())
                toks = obj.get("tokens")
                if toks:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    ids.extend(int(t) for t in toks)
                    n += len(toks)
                elif obj.get("content"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n += 1
                if obj.get("stop"):
                    timings = obj.get("timings") or {}
                    if timings.get("predicted_n"):
                        n = int(timings["predicted_n"])
                    break
        if first is None:
            raise RuntimeError("no tokens streamed")
        return Completion(started_at=started, first_token_at=first, last_token_at=last, completion_tokens=n,
                          token_ids=ids if want_token_ids else None)
