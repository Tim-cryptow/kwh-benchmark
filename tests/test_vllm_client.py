"""Drive VLLMEngine against a tiny in-process fake of the vLLM OpenAI server.

Checks the request shape (token-id prompt, greedy, min=max tokens, ignore_eos)
and the SSE parsing paths (usage-only and token_id logprobs).
"""

import asyncio
import json

import httpx
import pytest

from kwh_bench import reference as ref
from kwh_bench.engines.vllm import VLLMEngine

SEEN = {}


async def fake_app(scope, receive, send):
    assert scope["type"] == "http"
    path = scope["path"]
    body = b""
    while True:
        msg = await receive()
        body += msg.get("body", b"")
        if not msg.get("more_body"):
            break

    async def respond(status, payload, ctype=b"application/json"):
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", ctype)]})
        await send({"type": "http.response.body", "body": payload})

    if path == "/health":
        return await respond(200, b"")
    if path == "/version":
        return await respond(200, json.dumps({"version": "0.10.1"}).encode())
    if path == "/v1/models":
        return await respond(200, json.dumps({"data": [{"id": ref.MODEL_ID}]}).encode())
    if path == "/tokenize":
        req = json.loads(body)
        SEEN["tokenize"] = req
        toks = list(range(len(req["prompt"].split()) * 2))
        return await respond(200, json.dumps({"tokens": toks, "count": len(toks)}).encode())
    if path == "/v1/completions":
        req = json.loads(body)
        SEEN["completion"] = req
        n = req["max_tokens"]
        if not req.get("stream"):
            # teacher-forced scoring: prompt_logprobs for every prompt token
            plp = [None] + [{str(t): {"logprob": -0.001 * t, "rank": 1}} for t in req["prompt"][1:]]
            payload = {"choices": [{"index": 0, "text": "x", "prompt_logprobs": plp}], "usage": {"completion_tokens": 1}}
            return await respond(200, json.dumps(payload).encode())
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        for i in range(n):
            chunk = {"choices": [{"index": 0, "text": f"t{i}", "finish_reason": None}]}
            if req.get("logprobs") is not None and req.get("return_tokens_as_token_ids"):
                chunk["choices"][0]["logprobs"] = {"tokens": [f"token_id:{1000 + i}"], "token_logprobs": [0.0]}
            await send({"type": "http.response.body", "body": f"data: {json.dumps(chunk)}\n\n".encode(), "more_body": True})
        usage = {"choices": [], "usage": {"prompt_tokens": len(req["prompt"]), "completion_tokens": n}}
        await send({"type": "http.response.body", "body": f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode(), "more_body": True})
        await send({"type": "http.response.body", "body": b""})
        return
    await respond(404, b"{}")


@pytest.fixture
async def engine(monkeypatch):
    e = VLLMEngine(server_url="http://fake")
    transport = httpx.ASGITransport(app=fake_app)

    async def fake_start():
        e._client = httpx.AsyncClient(transport=transport, base_url="http://fake")
        e._served_model = ref.MODEL_ID

    monkeypatch.setattr(e, "start", fake_start)
    await e.start()
    yield e
    await e.stop()


async def test_tokenize_uses_served_model(engine):
    toks = await engine.tokenize("a b c")
    assert toks == list(range(6))
    assert SEEN["tokenize"]["model"] == ref.MODEL_ID
    assert SEEN["tokenize"]["add_special_tokens"] is False


async def test_completion_request_shape_and_usage_count(engine):
    c = await engine.complete([1, 2, 3], 5)
    req = SEEN["completion"]
    assert req["prompt"] == [1, 2, 3]
    assert req["max_tokens"] == req["min_tokens"] == 5
    assert req["ignore_eos"] is True and req["temperature"] == 0.0 and req["seed"] == 0
    assert req["stream"] is True and req["stream_options"]["include_usage"] is True
    assert "logprobs" not in req
    assert c.completion_tokens == 5 and c.token_ids is None
    assert c.first_token_at >= c.started_at and c.last_token_at >= c.first_token_at
    assert c.tpot is not None


async def test_completion_token_ids_via_logprobs(engine):
    c = await engine.complete([1, 2, 3], 4, want_token_ids=True)
    req = SEEN["completion"]
    assert req["logprobs"] == 0 and req["return_tokens_as_token_ids"] is True
    assert c.token_ids == [1000, 1001, 1002, 1003]
    assert c.completion_tokens == 4


async def test_info_reports_version(engine):
    info = await engine.info()
    assert info.name == "vllm" and info.version == "0.10.1" and info.launch_mode == "attached"


def test_build_argv_has_pinned_flags():
    e = VLLMEngine(revision="deadbeef", port=8123)
    try:
        argv = e.build_argv()
    except FileNotFoundError:
        pytest.skip("no vllm binary or module on PATH")
    joined = " ".join(argv)
    assert f"serve {ref.MODEL_ID}" in joined or f"--model {ref.MODEL_ID}" in joined
    assert "--revision deadbeef" in joined
    assert "--max-num-seqs 32" in joined and "--no-enable-prefix-caching" in joined
    assert "--port 8123" in joined
    assert "--max-model-len 1024" in joined                      # the default


def test_build_argv_carries_the_chosen_context_length():
    e = VLLMEngine(docker_image="vllm/vllm-openai:test", max_model_len=8192)
    argv = e.build_argv()
    assert argv[argv.index("--max-model-len") + 1] == "8192" and argv.count("--max-model-len") == 1


def test_build_docker_argv():
    e = VLLMEngine(docker_image="vllm/vllm-openai:test", hf_cache="/hf")
    argv = e.build_argv()
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in argv and "vllm/vllm-openai:test" in argv
    assert "/hf:/root/.cache/huggingface" in argv
    # model is the first arg after the image (the image entrypoint is `vllm serve`)
    assert argv[argv.index("vllm/vllm-openai:test") + 1] == ref.MODEL_ID
    assert "--max-num-seqs" in argv and "--no-enable-prefix-caching" in argv
    assert "--disable-log-requests" not in argv


async def test_score_continuation_reads_prompt_logprobs(engine):
    lps = await engine.score_continuation([1, 2, 3], [10, 20])
    req = SEEN["completion"]
    assert req["prompt"] == [1, 2, 3, 10, 20] and req["prompt_logprobs"] == 0 and req["max_tokens"] == 1
    assert lps == [-0.01, -0.02]
