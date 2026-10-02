# Canary calibration (SPEC.md §7)

A canary's delta is the gap, in nats, between the host's and the reference's mean teacher-forced log-probability of a locked 32-token continuation. Since rc.6 a run passes when all eight canaries are scored and **the mean of the eight deltas is within 0.05 nats**. rc.3–rc.5 applied 0.05 to each canary and required six of eight; rc.2 did the same at 0.10. This file records the evidence. It must, by 1.0.0, show (a) every certified card inside the tolerance and (b) at least one wrong-model negative control outside it.

| Date | GPU / arch | Engine | Model served | Canary deltas (nats) | Mean (verdict) | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-26 | RTX A5000 / Ampere sm86 | vLLM 0.30.0 | reference (W8A8, 024e24c) | 0.0000 ×8 | 0.0000 (pass) | Same node as the lock. Sequential lock scoring vs post-run scoring: bit-identical. |
| 2026-09-29 | RTX 4090 / Ada sm89 | vLLM 0.30.0 (cu129 build) | reference (W8A8, 024e24c) | 0.0000 ×8 | 0.0000 (pass) | First cross-architecture card. Every mean logprob matched the Ampere reference to five decimals ([rtx-4090-runpod.json](rtx-4090-runpod.json), canary block). |
| 2026-09-30 | RTX 3090 / Ampere sm86 | vLLM 0.30.0 (cu129 build) | reference (W8A8, 024e24c) | 0.0000 ×8 | 0.0000 (pass) | Consumer Ampere, different host and driver from the A5000 lock node ([rtx-3090-runpod.json](rtx-3090-runpod.json)). |
| 2026-10-02 | A40 / Ampere sm86 | vLLM 0.30.0 (PyPI, CUDA 13) | reference (W8A8, 024e24c) | 0.0052–0.0479 | 0.0229 (pass) | **First nonzero honest deltas**, on the lock node's architecture and engine build. Canary 112 at 0.0479, 0.0021 inside the old per-canary tolerance ([a40-runpod.json](a40-runpod.json)). An attached rerun in the same session reproduced them exactly at `--max-model-len 1024` and moved all eight at 8192: 0.0016–0.0345, mean 0.0134 ([uncertified/a40-attached-mml8192.json](uncertified/a40-attached-mml8192.json)). |
| 2026-09-29 | RTX 4090 / Ada sm89 | vLLM 0.30.0 (cu129 build) | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (control) | 0.0358–0.2187 | 0.1084 (**FAIL**) | Expected fail. Under rc.2 (six of eight canaries within 0.10) it failed by one canary: three were above 0.10 (0.1417, 0.1648, 0.2187), five sat between 0.036 and 0.099. rc.3 tightened the tolerance to 0.05, where one canary of eight was within it. Record: [canary/llama31-8b-awq-int4-rtx4090.json](canary/llama31-8b-awq-int4-rtx4090.json). |
| 2026-10-02 | A40 / Ampere sm86 | vLLM 0.30.0 (PyPI, CUDA 13) | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (control) | 0.1646 (canary 208 only) | — (**FAIL**) | One canary, served as a kwh-host liveness challenge during its M2 run. The 4090 control scored the same canary at 0.1648: the substitute's distance is the same on both cards, while the honest deltas differ by card. |

Each row comes from `kwh-bench canary … --append` (or the canary block of a certified report), which writes the full record to `results/canary/<label>.json`. The deltas are the evidence; `kwh-bench verify` re-derives the verdict from the stored deltas under the rule and tolerance in force.

## Why the mean, and why 0.05

- Three of four certified cards (A5000, 3090, 4090: Ampere and Ada) score the locked continuations to the same five decimals. The fourth, an A40, does not: 0.005–0.048, on the same architecture and engine build as the lock node, and a different 0.002–0.035 when only `--max-model-len` changes. Honest noise is real, depends on the card and the launch configuration, and on one A40 canary reaches 96% of the tolerance.
- The nearest wrong model tested, a 4-bit AWQ quantization of the same weights, moves each canary by 0.036–0.219 nats. At 0.10 per canary, five of its eight canaries were within tolerance, one short of passing the old six-of-eight rule; at 0.05, one of eight was. A guard that a 4-bit substitute nearly clears is not a guard.
- So canary by canary the ranges overlap: the A40's worst honest delta (0.048) is above the substitute's best (0.036), and the per-canary rule left the honest side 0.002 of room. The means do not overlap: the A40 averages 0.023 (0.013 at 8192), the AWQ control 0.108, about 2× either side of 0.05. Hence rc.6: judge the mean, and one noisy canary no longer decides a run. The substitute's distance does not depend on the card (canary 208: 0.1648 on the 4090, 0.1646 on the A40) where honest noise does, so the mean of a wrong model should stay far out on any card.
- The host client's liveness challenges are judged the same way: four fresh continuations per challenge, passing on their mean delta (kwh-host HOST-CLIENT.md §4, D9).

Pending:

- **Cross-architecture**: Blackwell (5090). After the A40, nothing from 0.00 to 0.05 would be surprising; the architecture alone does not predict it.
- **A second A40**, on another host, to tell whether 0.048 belongs to the card model or to that machine.
- **Negative controls** still owed: `meta-llama/Llama-3.1-8B-Instruct` at BF16 (gated; `unsloth/Llama-3.1-8B-Instruct` is an ungated mirror of the same weights; needs a pod with ≥ 60 GB disk) — the closest possible wrong model, same weights and no quantization; if it passes at 0.05, the tolerance drops again or FP16 serving is accepted as a superset of the unit. And a different model of similar size such as `Qwen/Qwen2.5-7B-Instruct` (different tokenizer, expected ≫ 0.5). Each is `kwh-bench canary --model <id> --label <name> --append` on any 24 GB pod.
- If any negative control's mean lands inside 0.05, the tolerance drops before 1.0.0 (a minor spec bump); if a certified card's mean lands outside, the tolerance rises. Record both here.

## Why rc.1's canary failed on its own reference node

rc.1 compared greedy token IDs (28 of 32 positions had to match). On the same A5000, same engine, same weights, the lock (batch size 1) and the benchmark (batch size 32) matched 4, 14, 20, 17, 32, 32, 2, 12 tokens. The prompts are random word sequences, so next-token distributions are nearly flat, the top-1 margin is tiny, and batch-dependent GEMM reduction order flips it; one flip and the rest diverges. Scoring a fixed continuation has no such cliff: a flipped argmax changes one token's logprob by a hair and nothing downstream.
