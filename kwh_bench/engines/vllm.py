"""vLLM engine (certified for I-1) driven over its OpenAI-compatible server.

Endpoints used:
  GET  /health
  GET  /version
  POST /tokenize          {"model", "prompt"} -> {"tokens": [...]}
  POST /v1/completions    prompt as token ids, streamed (SSE)

Token ids for canary requests come back via `logprobs=0` +
`return_tokens_as_token_ids=true` (tokens appear as "token_id:N" strings).
Non-canary requests carry no logprobs; their token count comes from the
final `usage` chunk (stream_options.include_usage).
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import httpx

from .base import Completion, Engine, EngineInfo
from .launcher import ServerProcess, docker_argv, which_or_module
from .. import reference as ref


class VLLMEngine(Engine):
    name = "vllm"

    def __init__(
        self,
        model: str = ref.MODEL_ID,
        revision: Optional[str] = None,
        server_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        port: int = 8000,
        extra_args: Optional[List[str]] = None,
        log_path: Optional[str] = None,
        hf_cache: Optional[str] = None,
    ):
        self.model = model
        self.revision = revision
        self.port = port
        self.docker_image = docker_image
        self.extra_args = list(extra_args or [])
        self.log_path = log_path
        self.hf_cache = hf_cache
        self.attached = server_url is not None
        self.base_url = (server_url or f"http://127.0.0.1:{port}").rstrip("/")
        self._proc: Optional[ServerProcess] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._launch_argv: List[str] = []
        self._served_model: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def _server_args(self) -> List[str]:
        """Flags common to every launch form (model is added per form: positional for
        `vllm serve` and the Docker image entrypoint, `--model` for the module form)."""
        args = list(ref.VLLM_ARGS) + ["--port", str(self.port), "--host", "0.0.0.0"]
        if self.revision:
            args += ["--revision", self.revision]
        return args + self.extra_args

    def build_argv(self) -> List[str]:
        if self.docker_image:
            # vllm/vllm-openai's entrypoint is `vllm serve`; the model is positional.
            volumes = []
            if self.hf_cache:
                volumes.append(f"{self.hf_cache}:/root/.cache/huggingface")
            env = {}
            if os.environ.get("HF_TOKEN"):
                env["HF_TOKEN"] = os.environ["HF_TOKEN"]
            return docker_argv(self.docker_image, gpu=True, ports=[f"{self.port}:{self.port}"], volumes=volumes,
                               args=[self.model] + self._server_args(), env=env)
        try:
            head = which_or_module("vllm")
            return head + ["serve", self.model] + self._server_args()
        except FileNotFoundError:
            return which_or_module("vllm", "vllm.entrypoints.openai.api_server") + ["--model", self.model] + self._server_args()

    async def start(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(600.0, connect=10.0))
        if not self.attached:
            self._launch_argv = self.build_argv()
            self._proc = ServerProcess(self._launch_argv, f"{self.base_url}/health", log_path=self.log_path)
            await self._proc.start()
        else:
            from .launcher import wait_healthy
            await wait_healthy(f"{self.base_url}/health", timeout_s=30)
        # Resolve served model name (may differ from the HF id when --served-model-name is used).
        r = await self._client.get("/v1/models")
        r.raise_for_status()
        models = r.json().get("data", [])
        self._served_model = models[0]["id"] if models else self.model

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._proc:
            await self._proc.stop()
            self._proc = None

    async def info(self) -> EngineInfo:
        version = None
        try:
            r = await self._client.get("/version")
            if r.status_code == 200:
                version = r.json().get("version")
        except httpx.HTTPError:
            pass
        mode = "attached" if self.attached else ("docker" if self.docker_image else "subprocess")
        return EngineInfo(
            name="vllm",
            version=version,
            # What the server actually serves, not what we expected. Matters for attached servers.
            model_id=self._served_model or self.model,
            model_revision=self.revision,
            launch_mode=mode,
            launch_args=list(self._launch_argv),
            server_url=self.base_url,
            extra={"served_model": self._served_model, "docker_image": self.docker_image},
        )

    # -- inference -----------------------------------------------------

    async def tokenize(self, text: str) -> List[int]:
        r = await self._client.post("/tokenize", json={"model": self._served_model, "prompt": text, "add_special_tokens": False})
        r.raise_for_status()
        return list(r.json()["tokens"])

    async def score_continuation(self, prompt_ids: List[int], continuation_ids: List[int]) -> List[float]:
        """Per-token logprobs of `continuation_ids` via vLLM's `prompt_logprobs`.

        The full sequence is sent as the prompt; `prompt_logprobs=0` returns
        the logprob of each actual prompt token. One token is generated and
        discarded (vLLM requires max_tokens >= 1).
        """
        body = {
            "model": self._served_model,
            "prompt": list(prompt_ids) + list(continuation_ids),
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
            "stream": False,
        }
        r = await self._client.post("/v1/completions", json=body)
        r.raise_for_status()
        plp = r.json()["choices"][0].get("prompt_logprobs")
        if not plp:
            raise RuntimeError("engine returned no prompt_logprobs")
        tail = plp[-len(continuation_ids):]
        out: List[float] = []
        for tok, entry in zip(continuation_ids, tail):
            if entry is None:
                raise RuntimeError("missing prompt_logprobs entry")
            item = entry.get(str(tok)) or entry.get(tok)
            if item is None:
                # prompt_logprobs=0 returns only the actual token; take it.
                item = next(iter(entry.values()))
            out.append(float(item["logprob"] if isinstance(item, dict) else item))
        return out

    async def complete(self, token_ids: List[int], max_tokens: int, want_token_ids: bool = False) -> Completion:
        body = {
            "model": self._served_model,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "skip_special_tokens": False,
        }
        if want_token_ids:
            body["logprobs"] = 0
            body["return_tokens_as_token_ids"] = True

        ids: List[int] = []
        n_tokens = 0
        usage_tokens: Optional[int] = None
        first = last = None
        started = time.perf_counter()
        async with self._client.stream("POST", "/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                choices = obj.get("choices") or []
                if choices:
                    now = time.perf_counter()
                    ch = choices[0]
                    lp = ch.get("logprobs")
                    if lp and lp.get("tokens"):
                        toks = lp["tokens"]
                        for t in toks:
                            if isinstance(t, str) and t.startswith("token_id:"):
                                ids.append(int(t.split(":", 1)[1]))
                        n_tokens += len(toks)
                        got = len(toks) > 0
                    else:
                        got = bool(ch.get("text")) or ch.get("finish_reason") is None
                        if ch.get("text"):
                            n_tokens += 1  # lower bound; usage is authoritative
                    if got and first is None:
                        first = now
                    if got:
                        last = now
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
        if first is None:
            raise RuntimeError("no tokens streamed")
        completion_tokens = usage_tokens if usage_tokens is not None else (len(ids) if ids else n_tokens)
        return Completion(
            started_at=started,
            first_token_at=first,
            last_token_at=last,
            completion_tokens=completion_tokens,
            token_ids=ids if want_token_ids else None,
        )
