#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod pod (Runpod PyTorch template, any 24GB+ NVIDIA card).
# Pods cannot run Docker, so this uses the pip path: pinned vLLM + this repo, then lock (optional) + run.
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-benchmark/main/scripts/runpod.sh) [--lock] [--runs N]
#
# Set HF_TOKEN if the checkpoint ever becomes gated. Output: results/<gpu>-runpod.json in the clone.
# Disk: 40 GB is enough for the benchmark and one 4-bit control; the BF16 control needs >= 60 GB.
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
REVISION="${REVISION:-024e24cbe4153670f747383ea3265d0fb197c727}"
REPO="${REPO:-https://github.com/Tim-cryptow/kwh-benchmark}"
DO_LOCK=0; RUNS=3
while [ $# -gt 0 ]; do case "$1" in --lock) DO_LOCK=1;; --runs) RUNS="$2"; shift;; esac; shift; done

cd /workspace
[ -d kwh-benchmark ] || git clone -q "$REPO" kwh-benchmark
cd kwh-benchmark && git pull -q

# The PyPI vLLM wheel is built against CUDA 13 and fails at import ("driver too old") on hosts whose
# driver only supports CUDA 12.x (e.g. driver 570 = CUDA 12.8, common on community 4090s). vLLM publishes
# a cu129 build of the same version on its own index; same engine, same flags, same lock.
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
if [ -n "$DRIVER_CUDA" ] && [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  echo "driver supports CUDA ${DRIVER_CUDA}; installing vllm==${VLLM_VERSION}+cu129"
  pip install -q uv
  uv pip install --system --break-system-packages -q "vllm==${VLLM_VERSION}+cu129" huggingface_hub \
    --torch-backend=cu129 --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/cu129" \
    --index-strategy unsafe-best-match
else
  pip install -q "vllm==${VLLM_VERSION}" huggingface_hub
fi
pip install -q -e .
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr '[:upper:]' '[:lower:]' | sed 's/nvidia-//; s/geforce-//')"
mkdir -p results
if [ "$DO_LOCK" = 1 ]; then
  kwh-bench lock --revision "$REVISION" --engine-log "results/vllm-lock-${GPU}.log"
fi
kwh-bench run --engine vllm --runs "$RUNS" --engine-log "results/vllm-run-${GPU}.log" --out "results/${GPU}-runpod.json"
kwh-bench verify "results/${GPU}-runpod.json"
