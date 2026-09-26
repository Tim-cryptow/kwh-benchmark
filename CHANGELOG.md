# Changelog

## 1.0.0-rc.2 — 2026-09-26

- Canary check redesigned (SPEC.md §7). rc.1 compared greedy token IDs and failed 6/8 on the reference node itself: word-salad prompts have near-flat next-token distributions and the lock (batch 1) and benchmark (batch 32) take different kernel paths. The host now *scores* the locked continuation under teacher forcing and must land within 0.10 nats of the reference mean log-probability. Canaries are scored after the measured runs, so the timed job no longer carries any logprob requests.
- First lock produced on RunPod (RTX A5000, vLLM 0.30.0, checkpoint 024e24c). First uncertified full run: 65.0 units/hour, stability 0.004.
- Engine interface: `score_continuation(prompt_ids, continuation_ids)`; vLLM implements it via `prompt_logprobs`.

## 1.0.0-rc.1 — 2026-09-26

First public release candidate of the Grade I unit (series I-1).

- SPEC.md: reference model (Llama 3.1 8B Instruct, W8A8), job shape (256 × 512→256, concurrency 32), scoring (median of 3, stability ≤ 10%), canary check, report contract.
- VERSIONING.md: series vs spec vs tool versions; reference-model deprecation policy.
- `kwh-bench run` (vllm certified, llamacpp and mock uncertified), `prompts`, `verify`, `lock`, `spec`.
- Deterministic prompt set pinned by SHA-256.
- Report schema with recomputable score and content hash.

Not yet: `reference/lock.json` is unlocked (weight hashes, engine build, canary expectations pending the reference node), and `results/` has no table. Both are required for 1.0.0.
