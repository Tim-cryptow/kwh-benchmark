# Changelog

## 1.0.0-rc.2 — 2026-09-26

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
