#!/usr/bin/env bash
# Launch the certified vLLM server in Docker with the pinned I-1 flags, then
# run the benchmark against it. Equivalent to `kwh-bench run --engine vllm --docker <image>`,
# kept as a plain script so the flags are visible without reading Python.
#
# Usage: scripts/run_vllm_docker.sh [image] [runs]
#   HF_TOKEN must be set if the checkpoint requires license acceptance.
set -euo pipefail

IMAGE="${1:-$(python3 -c 'import json;print(json.load(open("kwh_bench/reference/lock.json"))["engine"]["vllm"]["image"] or "vllm/vllm-openai:latest")')}"
RUNS="${2:-3}"
MODEL="RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
REVISION="$(python3 -c 'import json;print(json.load(open("kwh_bench/reference/lock.json"))["model"]["revision"] or "")')"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "image:    $IMAGE"
echo "model:    $MODEL${REVISION:+@$REVISION}"
echo "hf cache: $HF_CACHE"
[ "$IMAGE" = "vllm/vllm-openai:latest" ] && echo "WARNING: lock.json has no pinned image; result will be uncertified" >&2

exec kwh-bench run --engine vllm --docker "$IMAGE" --hf-cache "$HF_CACHE" --runs "$RUNS" \
  ${REVISION:+--revision "$REVISION"} --engine-log results/vllm-server.log
