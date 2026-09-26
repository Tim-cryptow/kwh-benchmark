# Changelog

## 1.0.0-rc.1 — 2026-09-26

First public release candidate of the Grade I unit (series I-1).

- SPEC.md: reference model (Llama 3.1 8B Instruct, W8A8), job shape (256 × 512→256, concurrency 32), scoring (median of 3, stability ≤ 10%), canary check, report contract.
- VERSIONING.md: series vs spec vs tool versions; reference-model deprecation policy.
- `kwh-bench run` (vllm certified, llamacpp and mock uncertified), `prompts`, `verify`, `lock`, `spec`.
- Deterministic prompt set pinned by SHA-256.
- Report schema with recomputable score and content hash.

Not yet: `reference/lock.json` is unlocked (weight hashes, engine build, canary expectations pending the reference node), and `results/` has no table. Both are required for 1.0.0.
