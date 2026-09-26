#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod pod (Runpod PyTorch template, any 24GB+ NVIDIA card).
# Pods cannot run Docker, so this uses the pip path: pinned vLLM + this repo, then lock (optional) + run.
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-benchmark/main/scripts/runpod.sh) [--lock] [--runs N]
#
# Set HF_TOKEN if the checkpoint ever becomes gated. Output: results/<gpu>-runpod.json in the clone.
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
REVISION="${REVISION:-024e24cbe4153670f747383ea3265d0fb197c727}"
REPO="${REPO:-https://github.com/Tim-cryptow/kwh-benchmark}"
DO_LOCK=0; RUNS=3
while [ $# -gt 0 ]; do case "$1" in --lock) DO_LOCK=1;; --runs) RUNS="$2"; shift;; esac; shift; done

cd /workspace
[ -d kwh-benchmark ] || git clone -q "$REPO" kwh-benchmark
cd kwh-benchmark && git pull -q
pip install -q "vllm==${VLLM_VERSION}" huggingface_hub && pip install -q -e .
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr '[:upper:]' '[:lower:]' | sed 's/nvidia-//; s/geforce-//')"
mkdir -p results
if [ "$DO_LOCK" = 1 ]; then
  kwh-bench lock --revision "$REVISION" --engine-log "results/vllm-lock-${GPU}.log"
fi
kwh-bench run --engine vllm --runs "$RUNS" --engine-log "results/vllm-run-${GPU}.log" --out "results/${GPU}-runpod.json"
kwh-bench verify "results/${GPU}-runpod.json"
