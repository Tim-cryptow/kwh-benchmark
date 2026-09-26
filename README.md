# kWh Grade I Benchmark

**One kWh unit is one execution of the Grade I reference job.** This repo defines that job and ships the tool that measures how many of them a rig completes per hour. It is build step 1 of the kWh Exchange; nothing downstream exists until this is public and locked.

- **SPEC.md** — the normative definition of unit series `I-1`: model, quantization, prompt set, token count, concurrency, scoring, canary, report.
- **VERSIONING.md** — what changes the unit (a new series) versus what does not, and the reference-model deprecation policy.
- **kwh_bench/** — the benchmark tool (`kwh-bench`).
- **reference/lock.json** — weight hashes, engine build and canary expectations. Filled by `kwh-bench lock` on the reference node; incomplete while the spec is a release candidate.
- **results/** — the published units/hour table.

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
kwh-bench run --engine vllm --docker vllm/vllm-openai:<tag from reference/lock.json>
scripts/run_vllm_docker.sh                           # same thing, flags visible in shell

# Exploratory (uncertified) paths
kwh-bench run --engine vllm --server-url http://127.0.0.1:8000   # attach to your own server
kwh-bench run --engine llamacpp --gguf ./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
kwh-bench run --engine mock                                     # no GPU, CI

kwh-bench verify results/<report>.json               # what the platform runs on ingest
```

The checkpoint is gated behind the Llama 3.1 Community License; log in with `huggingface-cli login` (or set `HF_TOKEN`) before the first run. The repo redistributes no weights.

A run takes a few minutes on a 4090-class card: model load, one warm-up job, three measured jobs. Output is a JSON report (schema in `kwh_bench/schema/`) and a summary like:

```
kWh Grade I  series I-1  spec 1.0.0
GPU: NVIDIA GeForce RTX 4090   engine: vllm 0.x.y (docker)
units/hour: 91.2   median job: 39.47 s   stability: 0.021
mean power: 312.4 W   units per electric kWh: 292.0
run 1: 1660 tok/s   TTFT p50 410ms p95 1.2s   TPOT p50 17.9ms p95 19.4ms
canary: PASS (8/8)
certified: YES
```

(Illustrative numbers. The real ones go in `results/`.)

## RunPod (or any pod that can't run Docker)

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-benchmark/main/scripts/runpod.sh) --lock   # reference node
bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-benchmark/main/scripts/runpod.sh)          # every other card
```

Installs the pinned vLLM with pip, clones this repo, runs lock (with `--lock`) and the benchmark, and verifies the report. ~10 minutes on a 24GB card including the weights download.

## What "certified" means

A report is certified when every condition in SPEC.md §5–§7 holds: launched by `kwh-bench` on the certified engine at the locked version and checkpoint revision, no forbidden flags, ≥ 3 measured runs within the stability bound, exactly 65,536 tokens per job with no request failures, and the canary check passed. `certified_reasons` lists every failing condition when it is false. Uncertified numbers are still useful; they are just not a rate the exchange will mint against.

## Status: release candidate

Until `reference/lock.json` is complete every report is `certified: false` with reason `unlocked`. Completing it is a one-time step on a machine with the weights and a GPU:

```bash
pip install huggingface_hub
kwh-bench lock --docker vllm/vllm-openai:<tag>       # hashes weights, records build, generates canaries
git add reference/lock.json && git commit -m "Lock I-1 reference"
```

Then run the benchmark on at least three consumer cards, commit the reports to `results/`, and tag `v1.0.0`. See SPEC.md §9.

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
reference/lock.json
results/
```

## License

Apache-2.0 for everything in this repository. Model weights are not included and carry their own license.
