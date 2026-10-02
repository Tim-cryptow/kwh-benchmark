# kWh Grade I Benchmark

**One kWh unit is one execution of the Grade I reference job.** This repo defines that job and ships the tool that measures how many of them a rig completes per hour. It is build step 1 of the kWh Exchange; nothing downstream exists until this is public and locked.

- **SPEC.md** — the normative definition of unit series `I-1`: model, quantization, prompt set, token count, concurrency, scoring, canary, report.
- **VERSIONING.md** — what changes the unit (a new series) versus what does not, and the reference-model deprecation policy.
- **kwh_bench/** — the benchmark tool (`kwh-bench`).
- **kwh_bench/reference/lock.json** — weight hashes, engine build and canary expectations. Filled by `kwh-bench lock` on the reference node (RTX A5000, 2026-09-26).
- **results/** — the published units/hour table (RTX 4090: 100.56, RTX 3090: 78.29, RTX A5000: 65.10, A40: 60.19) and the canary calibration evidence.

## The unit in one table

| | |
| --- | --- |
| Model | Llama 3.1 8B Instruct, W8A8 INT8 (`RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8`) |
| Job | 256 requests × (512 prompt tokens → 256 generated tokens), 32 in flight, greedy, EOS ignored |
| Work per unit | 65,536 generated tokens (+131,072 prompt tokens of prefill) |
| Score | `units_per_hour = 3600 / median(job wall seconds over 3 runs)` after 1 warm-up |
| Certified engine | vLLM OpenAI server, pinned flags, single GPU |

## Run it

Requirements: Linux (or Windows via WSL2), an NVIDIA GPU with ≥ 16 GB VRAM (24 GB class is the target), Docker with the NVIDIA container toolkit **or** a local vLLM install, Python ≥ 3.10.

```bash
pip install -e .
kwh-bench prompts --check           # generator reproduces the pinned prompt-set hash

# Certified path: kwh-bench launches vLLM itself with the pinned flags
kwh-bench run --engine vllm                          # local vllm on PATH
kwh-bench run --engine vllm --max-model-len 8192     # certify at the context you will serve (1024-8192)
kwh-bench run --engine vllm --docker vllm/vllm-openai:<tag from kwh_bench/reference/lock.json>
scripts/run_vllm_docker.sh                           # same thing, flags visible in shell

# Exploratory (uncertified) paths
kwh-bench run --engine vllm --server-url http://127.0.0.1:8000   # attach to your own server
kwh-bench run --engine llamacpp --gguf ./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
kwh-bench run --engine mock                                     # no GPU, CI

kwh-bench verify results/<report>.json               # what the platform runs on ingest

# Canary calibration (SPEC.md §7): score the locked canaries without a full run
kwh-bench canary --model meta-llama/Llama-3.1-8B-Instruct --label llama31-8b-bf16 --append   # must FAIL
kwh-bench canary --server-url http://127.0.0.1:8000 --label my-rig                             # must PASS on the reference model
```

The checkpoint is gated behind the Llama 3.1 Community License; log in with `huggingface-cli login` (or set `HF_TOKEN`) before the first run. The repo redistributes no weights.

A run takes a few minutes on a 4090-class card: model load, one warm-up job, three measured jobs. Output is a JSON report (schema in `kwh_bench/schema/`) and a summary. This one is the certified RTX 4090 run in `results/`:

```
kWh Grade I  series I-1  spec 1.0.0-rc.2
GPU: NVIDIA GeForce RTX 4090   engine: vllm 0.30.0 (subprocess)
units/hour: 100.561   median job: 35.7992 s   stability: 0.00283
mean power: 319.4 W   units per electric kWh: 314.84
run 1: 1830.66 tok/s   TTFT p50 263.3ms p95 496.7ms   TPOT p50 16.5ms p95 17.0ms
canary: PASS (8/8)
certified: YES
report sha256: e517fb699eef65c72b5bbf65f3f1e457ff616a85e721ed49ec574f0eb2d4e1e7
```

## RunPod (or any pod that can't run Docker)

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-benchmark/main/scripts/runpod.sh) --lock   # reference node
bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-benchmark/main/scripts/runpod.sh)          # every other card
```

Installs the pinned vLLM with pip, clones this repo, runs lock (with `--lock`) and the benchmark, and verifies the report. ~10 minutes on a 24GB card including the weights download.

Two things learned on community pods, both handled by the script: hosts whose driver only supports CUDA 12.x (driver 570 = 12.8 is common) cannot import the PyPI vLLM wheel, so the script installs vLLM's own cu129 build of the same version there; and 40 GB of container disk fits the benchmark plus one 4-bit control but not the BF16 control, so deploy with ≥ 60 GB for that. The pre-flight check (SPEC.md §6 step 0) refuses a host another tenant is already using; pick a different one rather than `--ignore-preflight`.

## What "certified" means

A report is certified when every condition in SPEC.md §5–§7 holds: launched by `kwh-bench` on the certified engine at the locked version and checkpoint revision, no forbidden flags, a context length from 1024 to 8192, ≥ 3 measured runs within the stability bound, exactly 65,536 tokens per job with no request failures, and the canary check passed (the mean of the eight canaries' deltas within 0.05 nats). `certified_reasons` lists every failing condition when it is false. Uncertified numbers are still useful; they are just not a rate the exchange will mint against.

## Status: release candidate (rc.6)

The lock is complete and reports certify. Left before `v1.0.0` (SPEC.md §9):

- One more consumer card in `results/` (RTX 5090; the 3090 and 4090 are in).
- The BF16 negative control in `results/canary-calibration.md` (needs a pod with ≥ 60 GB disk). The 4-bit control is in and fails at the current tolerance.

`kwh_bench/reference/lock.json` was produced once, on the reference node, with `kwh-bench lock`; it does not change for the life of series I-1.

## Layout

```
SPEC.md  VERSIONING.md  CHANGELOG.md
kwh_bench/
  reference.py      pinned constants for I-1 (the numbers)
  prompts.py        deterministic prompt set + hash
  load.py           the reference job: 32-wide load generator, timing, percentiles
  engines/          vllm (certified), llamacpp (uncertified), mock; subprocess/docker launcher
  hardware.py       GPU/CPU/OS probe, 1 Hz power sampler -> units per electric kWh
  report.py         canary, certification rules, report assembly + sha256
  verify.py         ingest-side validation
  runner.py         run + lock orchestration
  cli.py
  schema/report.schema.json
prompts/i1-prompts.jsonl
kwh_bench/reference/lock.json
results/
```

## License

Apache-2.0 for everything in this repository. Model weights are not included and carry their own license.
