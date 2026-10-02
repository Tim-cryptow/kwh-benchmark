# Changelog

## 1.0.0-rc.6 — 2026-10-02

- **Context length is the host's choice, 1024 to 8192** (SPEC.md §5). `kwh-bench run --max-model-len N`, default 1024; the value is in the report's launch args and anything outside the range, or unset, does not certify. The job, and so the unit, are unchanged: two attached A40 runs in the same session ran the reference job at 60.230 units/hour at 1024 and 60.228 at 8192 (`results/uncertified/a40-attached-mml1024.json`, `a40-attached-mml8192.json`). A host serves at the value it certified with, so buyers' requests can be up to 8,192 tokens instead of 1,024. Settles the host client's D8.
- **The canary check judges the mean** (SPEC.md §7). The run passes when all eight canaries were scored and the mean of their deltas is within 0.05 nats; rc.3–rc.5 required six of eight canaries each within 0.05. Per canary, honest noise and the nearest wrong model now overlap (the A40 reaches 0.048, the 4-bit control starts at 0.036); their means are 0.023 and 0.108. Reports gain `canary.rule` and `canary.mean_delta`, and `verify` re-judges every certified report under the new rule: all four pass. Settles the benchmark half of the host client's D9.
- A40 certified row: **60.19 units/hour**, stability 0.0016, 296 W, 203 units per electric kWh (`results/a40-runpod.json`), produced through the host client and carrying its signature. The first certified card with nonzero canary deltas (0.005–0.048); evidence in `results/canary-calibration.md`.

## 1.0.0-rc.5 — 2026-10-02

- `--watermark-config` joins the forbidden flags (SPEC.md §5). vLLM 0.30.0 can apply a text watermark (a keyed Gumbel sampler) when the engine is launched with a watermark config; it is off by default and changes no work, but it changes which token is chosen. The host client verifies delivered greedy outputs by scoring them under the reference model, and a watermarked engine's outputs would fail that check while being honestly produced. Not to be confused with `--watermark`, the scheduler's KV-cache headroom, which stays allowed.

## 1.0.0-rc.4 — 2026-09-30

- RTX 3090 certified row: **78.29 units/hour**, stability 0.0199, 347 W, 225 units per electric kWh, canary 8/8 at delta 0.0000 (`results/rtx-3090-runpod.json`). Produced through the host client and carries its signature; two of the three consumer cards for 1.0.0 are now in.
- `reference/lock.json` moved to `kwh_bench/reference/lock.json` and ships as package data. A pip-installed kwh-bench (`pip install git+https://github.com/Tim-cryptow/kwh-benchmark`) previously resolved the lock relative to a repo root that does not exist in site-packages, got an empty lock, and marked every report `unlocked`. The host client depends on kwh-bench as a library, so this had to go first. No spec change; certified reports and their `lock_sha256` are unaffected.

## 1.0.0-rc.3 — 2026-09-29

- Canary tolerance tightened from 0.10 to 0.05 nats (SPEC.md §7). Evidence in `results/canary-calibration.md`: the reference model scores the locked continuations at delta 0.0000 on both Ampere (A5000) and Ada (4090), while a 4-bit AWQ quantization of the same weights lands at 0.036–0.219 and cleared 0.10 on five of eight canaries, one short of passing the run. At 0.05 it fails 7/8.
- `kwh-bench verify` re-judges a certified report's canaries from the stored deltas under the current tolerance, so reports scored under an older, looser tolerance are accepted only if they would pass now. Both certified reports in `results/` do.
- First consumer-card row: RTX 4090, **100.56 units/hour**, stability 0.0028, 319.4 W mean, 314.8 units per electric kWh, canary 8/8 at delta 0.0000 (`results/rtx-4090-runpod.json`). First cross-architecture datapoint against the Ampere lock.
- First negative control: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` on the same 4090 (`results/canary/llama31-8b-awq-int4-rtx4090.json`).
- `scripts/runpod.sh` installs vLLM's cu129 build when the host driver only supports CUDA 12.x (the PyPI wheel needs CUDA 13 and fails at import with *driver too old*); prints the resolved vLLM/torch versions before running. Disk guidance: ≥ 60 GB for the BF16 control.
- SPEC.md §9 now lists the calibration file among the conditions for tagging 1.0.0.

## 1.0.0-rc.2 — 2026-09-26

- Pre-flight check (SPEC.md §6 step 0): `kwh-bench run` samples the GPU for 5 s before launching the engine and refuses with `host_contention` unless ≥ 95% of VRAM is free and mean utilization is ≤ 5%. `--ignore-preflight` runs anyway, uncertified. Reports carry the `preflight` sample; `verify` rejects certified reports whose pre-flight was not idle. Prompted by a RunPod community RTX 5090 that arrived with a foreign process holding 5.7 GB at 100%/400 W (see results/README.md, Field notes).

- `kwh-bench canary`: score the locked canaries against a running server (`--server-url`) or a launched model (`--model`, for negative controls) without a benchmark run; writes `results/canary/<label>.json` and, with `--append`, a row in `results/canary-calibration.md`. Records whether the outcome matched expectation (reference model must pass, anything else must fail).
- vLLM engine reports the model the server actually serves (`/v1/models`), so an attached server serving the wrong model is recorded as such.

- Canary check redesigned (SPEC.md §7). rc.1 compared greedy token IDs and failed 6/8 on the reference node itself: word-salad prompts have near-flat next-token distributions and the lock (batch 1) and benchmark (batch 32) take different kernel paths. The host now *scores* the locked continuation under teacher forcing and must land within 0.10 nats of the reference mean log-probability. Canaries are scored after the measured runs, so the timed job no longer carries any logprob requests.
- `reference/lock.json` is now complete: locked on an RTX A5000 (RunPod), vLLM 0.30.0, checkpoint 024e24c, 8 canaries with reference log-probabilities.
- First certified I-1 report: RTX A5000, 65.10 units/hour, stability 0.0047, canary 8/8 with delta 0.0000 (`results/a5000-runpod.json`). rc.1's uncertified run of the same card (65.03 units/hour) is kept in `results/uncertified/`.
- `results/canary-calibration.md` tracks the tolerance evidence required for 1.0.0.
- Engine interface: `score_continuation(prompt_ids, continuation_ids)`; vLLM implements it via `prompt_logprobs`.

## 1.0.0-rc.1 — 2026-09-26

First public release candidate of the Grade I unit (series I-1).

- SPEC.md: reference model (Llama 3.1 8B Instruct, W8A8), job shape (256 × 512→256, concurrency 32), scoring (median of 3, stability ≤ 10%), canary check, report contract.
- VERSIONING.md: series vs spec vs tool versions; reference-model deprecation policy.
- `kwh-bench run` (vllm certified, llamacpp and mock uncertified), `prompts`, `verify`, `lock`, `spec`.
- Deterministic prompt set pinned by SHA-256.
- Report schema with recomputable score and content hash.

Not yet: `reference/lock.json` is unlocked (weight hashes, engine build, canary expectations pending the reference node), and `results/` has no table. Both are required for 1.0.0.
