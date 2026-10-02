# Canary calibration (SPEC.md §7)

The canary tolerance is **0.05 nats** on the mean teacher-forced log-probability of each locked 32-token continuation (rc.3; it was 0.10 in rc.2). This file records the evidence for that number. It must, by 1.0.0, show (a) every certified card inside the tolerance and (b) at least one wrong-model negative control outside it.

| Date | GPU / arch | Engine | Model served | Canary deltas (nats) | Pass | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-26 | RTX A5000 / Ampere sm86 | vLLM 0.30.0 | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | Same node as the lock. Sequential lock scoring vs post-run scoring: bit-identical. |
| 2026-09-29 | RTX 4090 / Ada sm89 | vLLM 0.30.0 (cu129 build) | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | First cross-architecture card. Every mean logprob matched the Ampere reference to five decimals ([rtx-4090-runpod.json](rtx-4090-runpod.json), canary block). |
| 2026-09-30 | RTX 3090 / Ampere sm86 | vLLM 0.30.0 (cu129 build) | reference (W8A8, 024e24c) | 0.0000 ×8 | 8/8 | Consumer Ampere, different host and driver from the A5000 lock node ([rtx-3090-runpod.json](rtx-3090-runpod.json)). |
| 2026-10-02 | A40 / Ampere sm86 | vLLM 0.30.0 (PyPI, CUDA 13) | reference (W8A8, 024e24c) | 0.0052–0.0479 | 8/8 | **First nonzero honest deltas**, on the lock node's architecture and engine build. Canary 112 at 0.0479, 0.0021 inside the tolerance ([a40-runpod.json](a40-runpod.json)). An attached rerun in the same session reproduced them exactly at `--max-model-len 1024` and moved all eight at 8192: 0.0016–0.0345 ([uncertified/a40-attached-mml8192.json](uncertified/a40-attached-mml8192.json)). |
| 2026-09-29 | RTX 4090 / Ada sm89 | vLLM 0.30.0 (cu129 build) | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (control) | 0.0358–0.2187 | 5/8 at 0.10 → **FAIL**; 1/8 at 0.05 | Expected fail, as expected, but only 3 of 8 canaries cleared 0.10 (0.1417, 0.1648, 0.2187). Five sat between 0.036 and 0.099. Tolerance tightened to 0.05. Record: [canary/llama31-8b-awq-int4-rtx4090.json](canary/llama31-8b-awq-int4-rtx4090.json). |
| 2026-10-02 | A40 / Ampere sm86 | vLLM 0.30.0 (PyPI, CUDA 13) | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (control) | 0.1646 (canary 208 only) | **FAIL** | One canary, served as a kwh-host liveness challenge during its M2 run. The 4090 control scored the same canary at 0.1648: the substitute's distance is the same on both cards, while the honest deltas differ by card. |

Each row comes from `kwh-bench canary … --append` (or the canary block of a certified report), which writes the full record to `results/canary/<label>.json`. The deltas are the evidence; the pass count is at the tolerance in force when the row was recorded, and `kwh-bench verify` re-derives pass/fail from the stored deltas under the current tolerance.

## Why 0.05

- Three of four certified cards (A5000, 3090, 4090: Ampere and Ada) score the locked continuations to the same five decimals. The fourth, an A40, does not: 0.005–0.048, on the same architecture and engine build as the lock node, and a different 0.002–0.035 when only `--max-model-len` changes. Honest noise is real, depends on the card and the launch configuration, and on the A40 reaches 96% of the tolerance.
- The nearest wrong model tested, a 4-bit AWQ quantization of the same weights, moves the mean by 0.036–0.219 nats. At 0.10 it fails only 3/8 canaries and passes the run's 6-of-8 rule by one canary's margin; at 0.05 it fails 7/8. A guard that a 4-bit substitute nearly clears is not a guard.
- So canary by canary the ranges now overlap: the A40's worst honest delta (0.048) is above the substitute's best (0.036). The run-level rule still separates them (A40 8/8, substitute 1/8), and the substitute's distance does not depend on the card (canary 208: 0.1648 on the 4090, 0.1646 on the A40) where honest noise does. 0.05 holds, by 0.002. If a certified card lands above it, the tolerance rises or the rule changes (next item), and this file says why.

**Candidate for the next rc: judge the mean, not the count.** Average the eight deltas and certify when the mean is within 0.05. On the data so far the A40 averages 0.023 (0.013 at 8192) and the AWQ control 0.108 (on the 4090; the canary it shares with the A40 says the card does not matter for it): about a 2× margin on each side of 0.05, where the per-canary rule leaves 0.002 on the honest side. One noisy canary would no longer decide a run. The host client's liveness challenge would carry several fresh continuations and be judged the same way (kwh-host HOST-CLIENT.md, D9).

Pending:

- **Cross-architecture**: Blackwell (5090). After the A40, nothing from 0.00 to 0.05 would be surprising; the architecture alone does not predict it.
- **A second A40**, on another host, to tell whether 0.048 belongs to the card model or to that machine.
- **Negative controls** still owed: `meta-llama/Llama-3.1-8B-Instruct` at BF16 (gated; `unsloth/Llama-3.1-8B-Instruct` is an ungated mirror of the same weights; needs a pod with ≥ 60 GB disk) — the closest possible wrong model, same weights and no quantization; if it passes at 0.05, the tolerance drops again or FP16 serving is accepted as a superset of the unit. And a different model of similar size such as `Qwen/Qwen2.5-7B-Instruct` (different tokenizer, expected ≫ 0.5). Each is `kwh-bench canary --model <id> --label <name> --append` on any 24 GB pod.
- If any negative control lands inside 0.05, the tolerance drops before 1.0.0 (a minor spec bump); if a certified card lands outside, the tolerance rises. Record both here.

## Why rc.1's canary failed on its own reference node

rc.1 compared greedy token IDs (28 of 32 positions had to match). On the same A5000, same engine, same weights, the lock (batch size 1) and the benchmark (batch size 32) matched 4, 14, 20, 17, 32, 32, 2, 12 tokens. The prompts are random word sequences, so next-token distributions are nearly flat, the top-1 margin is tiny, and batch-dependent GEMM reduction order flips it; one flip and the rest diverges. Scoring a fixed continuation has no such cliff: a flipped argmax changes one token's logprob by a hair and nothing downstream.
