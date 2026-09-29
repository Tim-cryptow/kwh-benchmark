# Results

The published units/hour table for series I-1. One JSON report per run, as written by `kwh-bench run`, plus this table.

Reports here must pass `kwh-bench verify`. Only certified reports enter the table; uncertified reports live in `results/uncertified/` for reference.

| GPU | Cloud | Engine | units/hour | median job (s) | stability | mean W | units per electric kWh | TPOT p50 | certified | report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4090 (24 GB, Ada) | RunPod Community | vLLM 0.30.0 | **100.56** | 35.80 | 0.0028 | 319.4 | 314.8 | 16.5 ms | yes | [rtx-4090-runpod.json](rtx-4090-runpod.json) |
| NVIDIA RTX A5000 (24 GB, Ampere) | RunPod Secure CA-MTL-1 | vLLM 0.30.0 | **65.10** | 55.30 | 0.0047 | 228.7 | 284.6 | 23.2 ms | yes | [a5000-runpod.json](a5000-runpod.json) |

The A5000 is also the reference node for the I-1 lock (`reference/lock.json`, locked 2026-09-26T07:12Z). The 4090 is the first consumer card in the table and the first cross-architecture datapoint: its canaries scored the locked continuations to the same five decimals as the Ampere reference node (delta 0.0000 on all eight).

Target for 1.0.0: at least three consumer cards (RTX 3090, RTX 4090, RTX 5090). The 4090 row is in; 3090 and 5090 rows follow as RunPod community capacity allows.

## Reading the table

- **units/hour** is the rate the exchange would mint against: `3600 / median(job seconds)`.
- **stability** is `(max − min) / median` over the three measured runs; certification requires ≤ 0.10.
- **units per electric kWh** is `units/hour ÷ (mean W / 1000)`: how many units one kilowatt-hour of electricity buys on that rig. The host dashboard puts this next to the host's power tariff.
- **TPOT p50** is median time per output token at concurrency 32, i.e. the per-stream speed a buyer sees (~43 tok/s on the A5000, ~61 tok/s on the 4090).

## Field notes

- **2026-09-26, RunPod Community RTX 5090 (pod `gruz0z76gmyc8y`).** Deployed for the second table row. Before anything of ours ran, `nvidia-smi` showed 5,695 MiB in use, 100% utilization and 400 W, sustained, with no process visible inside the container: another tenant or a leftover workload on the host. vLLM refused to start (25.3 of 31.4 GiB free, 28.2 needed). The pod was stopped after ten minutes. This is the case the primer's liveness and slashing design exists for, and it is why `kwh-bench run` now samples the GPU before launching and refuses with `host_contention` (SPEC.md §6 step 0). The evidence file for a refused run lands in `results/uncertified/<timestamp>-preflight-failed.json`.
- **2026-09-29, RunPod Community RTX 4090 (pod `utnxnk5xx77hav`).** Pre-flight idle (1 MiB used, 0% util, 14.6 W). Two things cost time before the run and are now handled by `scripts/runpod.sh`:
  - The host driver was 570.195.03 (CUDA 12.8). The PyPI `vllm==0.30.0` wheel is built against CUDA 13 and fails at import with *driver too old*. The fix is the cu129 build of the same version from vLLM's own wheel index, installed with uv (`--torch-backend=cu129`); the script now checks `nvidia-smi`'s CUDA version and does this automatically. Same engine version, same flags, and the report still verifies against the lock. The alternative is to filter RunPod hosts by CUDA ≥ 13.0 at deploy time.
  - 40 GB of container disk is enough for the W8A8 checkpoint plus one 4-bit control, not for the BF16 control (16 GB of weights on top of the HF cache). Deploy with ≥ 60 GB when the BF16 control is on the plan.
