# Canary calibration (SPEC.md §7)

The canary tolerance is **0.05 nats** on the mean teacher-forced log-probability of each locked 32-token continuation (rc.3; it was 0.10 in rc.2). This file records the evidence for that number. It must, by 1.0.0, show (a) every certified card inside the tolerance and (b) at least one wrong-model negative control outside it.

| Date | GPU / arch | Engine | Model served | Canary deltas (nats) | Pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-26 | RTX A5000 / Ampere sm86 | vLLM 0.30.0 | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | Same node as the lock. Sequential lock scoring vs post-run scoring: bit-identical. |
| 2026-09-29 | RTX 4090 / Ada sm89 | vLLM 0.30.0 (cu129 build) | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | First cross-architecture card. Every mean logprob matched the Ampere reference to five decimals ([rtx-4090-runpod.json](rtx-4090-runpod.json), canary block). |
| 2026-09-30 | RTX 3090 / Ampere sm86 | vLLM 0.30.0 (cu129 build) | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | Consumer Ampere, different host and driver from the A5000 lock node ([rtx-3090-runpod.json](rtx-3090-runpod.json)). |
| 2026-09-29 | RTX 4090 / Ada sm89 | vLLM 0.30.0 (cu129 build) | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (control) | 0.0358–0.2187 | 5/8 at 0.10 → **FAIL**; 1/8 at 0.05 | Expected fail, as expected, but only 3 of 8 canaries cleared 0.10 (0.1417, 0.1648, 0.2187). Five sat between 0.036 and 0.099. Tolerance tightened to 0.05. Record: [canary/llama31-8b-awq-int4-rtx4090.json](canary/llama31-8b-awq-int4-rtx4090.json). |

Each row comes from `kwh-bench canary … --append` (or the canary block of a certified report), which writes the full record to `results/canary/<label>.json`. The deltas are the evidence; the pass count is at the tolerance in force when the row was recorded, and `kwh-bench verify` re-derives pass/fail from the stored deltas under the current tolerance.

## Why 0.05

- Three certified cards on two architectures (Ampere ×2, Ada) score the locked continuations to the same five decimals: the measured cross-hardware noise so far is 0.0000, not the "hundredths of a nat" the rc.2 text allowed for. There is room under 0.05 for a real kernel difference (Blackwell, a different vLLM point release) without admitting a wrong model.
- The nearest wrong model tested, a 4-bit AWQ quantization of the same weights, moves the mean by 0.036–0.219 nats. At 0.10 it fails only 3/8 canaries and passes the run's 6-of-8 rule by one canary's margin; at 0.05 it fails 7/8. A guard that a 4-bit substitute nearly clears is not a guard.
- 0.05 stays above anything a well-behaved reference-model host has produced. If a certified card ever lands above it, the tolerance rises and this file says why.

Pending:

- **Cross-architecture**: Blackwell (5090). Expect 0.00–0.02.
- **Negative controls** still owed: `meta-llama/Llama-3.1-8B-Instruct` at BF16 (gated; `unsloth/Llama-3.1-8B-Instruct` is an ungated mirror of the same weights; needs a pod with ≥ 60 GB disk) — the closest possible wrong model, same weights and no quantization; if it passes at 0.05, the tolerance drops again or FP16 serving is accepted as a superset of the unit. And a different model of similar size such as `Qwen/Qwen2.5-7B-Instruct` (different tokenizer, expected ≫ 0.5). Each is `kwh-bench canary --model <id> --label <name> --append` on any 24 GB pod.
- If any negative control lands inside 0.05, the tolerance drops before 1.0.0 (a minor spec bump); if a certified card lands outside, the tolerance rises. Record both here.

## Why rc.1's canary failed on its own reference node

rc.1 compared greedy token IDs (28 of 32 positions had to match). On the same A5000, same engine, same weights, the lock (batch size 1) and the benchmark (batch size 32) matched 4, 14, 20, 17, 32, 32, 2, 12 tokens. The prompts are random word sequences, so next-token distributions are nearly flat, the top-1 margin is tiny, and batch-dependent GEMM reduction order flips it; one flip and the rest diverges. Scoring a fixed continuation has no such cliff: a flipped argmax changes one token's logprob by a hair and nothing downstream.
