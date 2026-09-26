# Canary calibration (SPEC.md §7)

The canary tolerance is 0.10 nats on the mean teacher-forced log-probability of each locked 32-token continuation. This file records the evidence for that number. It must, by 1.0.0, show (a) every certified card inside the tolerance and (b) at least one wrong-model negative control outside it.

| Date | GPU / arch | Engine | Model served | Canary deltas (nats) | Pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-26 | RTX A5000 / Ampere sm86 | vLLM 0.30.0 | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | Same node as the lock. Sequential lock scoring vs post-run scoring: bit-identical. |

Each row comes from `kwh-bench canary … --append`, which writes the full record to `results/canary/<label>.json`.

Pending:

- **Cross-architecture**: Ada (4090), Blackwell (5090), Ampere consumer (3090). Expect deltas in the 0.00–0.03 range from kernel differences.
- **Negative controls** (must fail), each one `kwh-bench canary --model <id> --label <name> --append` on any 24 GB pod: `meta-llama/Llama-3.1-8B-Instruct` at BF16 (gated; `unsloth/Llama-3.1-8B-Instruct` is an ungated mirror of the same weights) (same weights, no quantization; expected delta ~0.05–0.2 — if it passes, tighten the tolerance or accept that FP16 serving is a superset of the unit), a 4-bit variant such as `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (expected > 0.2), and a different model of similar size such as `Qwen/Qwen2.5-7B-Instruct` (different tokenizer, so the locked token ids are meaningless to it; expected ≫ 0.5).
- If any negative control lands inside 0.10, the tolerance drops before 1.0.0 (a minor spec bump); if a certified card lands outside, the tolerance rises. Record both here.

## Why rc.1's canary failed on its own reference node

rc.1 compared greedy token IDs (28 of 32 positions had to match). On the same A5000, same engine, same weights, the lock (batch size 1) and the benchmark (batch size 32) matched 4, 14, 20, 17, 32, 32, 2, 12 tokens. The prompts are random word sequences, so next-token distributions are nearly flat, the top-1 margin is tiny, and batch-dependent GEMM reduction order flips it; one flip and the rest diverges. Scoring a fixed continuation has no such cliff: a flipped argmax changes one token's logprob by a hair and nothing downstream.
