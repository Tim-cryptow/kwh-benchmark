# kWh Grade I Unit Specification

**Spec version:** 1.0.0-rc.6 (becomes 1.0.0 at lock, see §9)
**Unit series:** `I-1`
**Status:** Release candidate. Every number in this document is fixed except the fields listed in §9 (weight hashes, canary expectations, exact engine build), which are filled in by `kwh-bench lock` on the reference node before 1.0.0 is tagged.

---

## 1. What a unit is

One kWh Grade I unit (series I-1) is **one execution of the Grade I reference job** defined in this document: a fixed batch of inference requests against a fixed open-weight model at a fixed quantization, producing a fixed number of generated tokens under fixed sampling and concurrency conditions.

The unit is a quantity of *work*, not of time and not of hardware. A host's **rate** is how many reference jobs its rig completes per hour under this spec. The rate belongs to the rig and can change; the unit belongs to the series and cannot.

Anything that changes the amount or kind of work in the reference job creates a **new series** (`I-2`, `I-3`, ...). It never mutates `I-1`. See VERSIONING.md.

## 2. Reference model

| Field | Value |
| --- | --- |
| Base model | Llama 3.1 8B Instruct (Meta) |
| Quantization | INT8 weights, INT8 activations (W8A8), per-channel weight scales, dynamic per-token activation scales |
| Reference checkpoint | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8` on Hugging Face (formerly published under `neuralmagic/`) |
| Checkpoint revision | pinned by commit hash in `kwh_bench/reference/lock.json` (§9) |
| Weight manifest | SHA-256 of every `*.safetensors` file and of `config.json`, `tokenizer.json`, `tokenizer_config.json`, recorded in `kwh_bench/reference/lock.json` |
| Tokenizer | the checkpoint's own (Llama 3.1 tokenizer, 128,256 vocabulary) |
| License | Llama 3.1 Community License. Hosts accept it when they pull the weights. The benchmark repo redistributes no weights. |

**Why this model.** It is the most widely benchmarked 8B-class open-weight instruct model, it has a maintained W8A8 checkpoint that runs on INT8 tensor cores in the certified engine, and its license permits commercial serving. The deprecation policy (VERSIONING.md §4) covers what happens when it ages out.

**Why W8A8 and not weight-only INT8.** Weight-only schemes dequantize to FP16 for the matmul; the work is then FP16 compute with INT8 memory traffic. W8A8 keeps both the memory traffic and the arithmetic at 8 bits, which is what the Grade I definition ("INT8, memory-bandwidth weighted") means. It also gives one unambiguous kernel path for the certified engine.

## 3. Reference job

| Parameter | Value | Note |
| --- | --- | --- |
| Requests per job | **256** | |
| Prompt length | **512 tokens** exactly | after tokenization with the reference tokenizer; see §4 |
| Generated tokens per request | **256** exactly | `max_tokens = min_tokens = 256`, EOS ignored |
| Generated tokens per job | **65,536** | 256 × 256. This is the quantity of work in one unit. |
| Prompt tokens per job | 131,072 | prefill work, part of the unit |
| Concurrency | **32** requests in flight | the load generator keeps exactly 32 open until the queue drains |
| Sampling | greedy (`temperature = 0`), `top_p = 1`, `top_k` disabled, no repetition penalties, `seed = 0` | greedy makes outputs reproducible for the canary check |
| Stop conditions | none | EOS is ignored so the token count is fixed |
| Prefix caching | **disabled** in the engine | prompts also share no common prefix, so an engine that ignores the flag gains nothing |
| Speculative decoding | **not permitted** | it changes the work done per token |
| Chat template | **not applied** | prompts are raw token sequences fed to the completion endpoint. The reference job measures the model, not a chat wrapper. |
| Context length | engine `max_model_len` from 1024 to 8192, the host's choice (§5) | 512 + 256 = 768 fits in any of them |

A job is complete when all 256 requests have returned all 256 tokens. **Job wall time** is measured from the dispatch of the first request to the receipt of the last token of the last request, on the load generator's clock.

### Why these numbers

- **65,536 generated tokens** puts a 4090-class card at an estimated 80–100 units per hour and a unit at roughly one US cent when priced against public per-token rates for the same model. Round numbers at human scale. (Estimates. The first published units/hour table in `results/` replaces them.)
- **512 in / 256 out** is a realistic ratio for agent and RAG traffic, and it makes prefill a meaningful (about 20–30%) but not dominant share of the work, so the unit rewards both memory bandwidth and compute.
- **Concurrency 32** is a deliberate compromise and the most consequential choice in the spec. Higher concurrency inflates throughput on large cards but degrades per-request latency; the rate would then measure a capacity that the router could not deliver at the latency buyers see. Fixing concurrency ties the rate to a deliverable service profile. A 24GB card fits 32 × 768 tokens of FP16 KV cache (about 3.2GB) plus the ~8.5GB model with room to spare. Larger cards mint more units per hour through faster per-step decode, not through larger batches. A future series may add a high-batch profile; it will be a different unit.

## 4. Prompt set

The prompt set is **generated, not downloaded**. `kwh-bench prompts` produces it deterministically from the constants in `kwh_bench/reference.py`:

1. A fixed word list of common English words ships in the package (`kwh_bench/words.py`).
2. A seeded generator (`seed = 0x6B5768` — "kWh" in ASCII) produces 256 prompts. Each prompt begins with a unique 8-character hexadecimal nonce (so no two prompts share a prefix), followed by pseudo-random sentences from the word list, long enough to exceed 512 tokens under the reference tokenizer.
3. The canonical prompt file is `prompts/i1-prompts.jsonl`, one object per line: `{"id": <int>, "text": <string>}`. Its SHA-256 is pinned in `kwh_bench/reference.py` (`PROMPT_SET_SHA256`) and checked at run time. Text is UTF-8, LF line endings, no trailing whitespace.

At run time, each prompt is tokenized with the reference tokenizer (via the engine's `/tokenize` endpoint, so the tool needs no local copy of the tokenizer) and **truncated to exactly 512 token IDs**. The token IDs, not the text, are what is sent to the engine. This makes the prompt work exactly 512 tokens regardless of how a given text tokenizes. A prompt that tokenizes to fewer than 512 tokens is a spec defect and fails the run; the generator over-produces to make this impossible with the reference tokenizer.

## 5. Certified engine

Rates are **certified** only when produced by an engine in the certified set for the series, launched by `kwh-bench` with the pinned configuration. Other engines produce **uncertified** rates (still reported, marked `certified: false`), which are useful for exploration and for hardware the certified engine does not support.

| | Certified for I-1 | Uncertified |
| --- | --- | --- |
| Engine | vLLM, OpenAI-compatible server, version pinned in `kwh_bench/reference/lock.json` | llama.cpp server with a Q8_0 GGUF of the same base model (weight-only 8-bit; different numerical work, so not the same unit) |
| Launched by | `kwh-bench run --engine vllm` (subprocess or Docker, pinned flags) | `kwh-bench run --engine llamacpp` |
| Attach to an existing server | `--server-url` accepted, result marked uncertified (configuration unverifiable) | same |

### Pinned vLLM launch configuration

```
--model RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8
--revision <lock.json: model.revision>
--dtype auto
--max-model-len <1024 to 8192, the host's choice; recorded in the report>
--max-num-seqs 32
--no-enable-prefix-caching
--seed 0
--gpu-memory-utilization 0.90
```

Forbidden for certified runs: `--speculative-config` (any), `--enable-prefix-caching`, `--quantization` other than the checkpoint's own (`compressed-tensors`), tensor parallelism > 1 (I-1 is a single-device unit), any `--max-num-seqs` other than 32, and `--watermark-config` (any). A text watermark changes which token is sampled, so a watermarked engine's outputs stop matching the reference model's choices and cannot be verified against it.

**Context length (rc.6).** The reference job needs 768 tokens, but a host serves buyers with the engine it certified, so the context length it benchmarks at is the longest request it can take. Any value from 1024 to 8192 certifies (`kwh-bench run --max-model-len N`, default 1024). It does not change the work or the rate: vLLM's batching defaults do not depend on it, and on an A40 the reference job ran at 60.230 units/hour at 1024 and 60.228 at 8192, in the same session. It can move the canary deltas (§7), which is why the host serves at the value it certified with. Outside the range, or unset, a report does not certify.

Everything else (CUDA graphs, attention backend, chunked prefill) is the engine's default and is the host's to optimize within the pinned version. Optimizations that do less work per token are what the forbidden list excludes; optimizations that do the same work faster are the point.

The Docker path uses the official `vllm/vllm-openai` image at the pinned tag; `scripts/run_vllm_docker.sh` applies the same flags. On Windows the certified path is WSL2 + Docker, which is also the host client's sandbox in build step 2.

## 6. Measurement procedure

`kwh-bench run` performs, in order:

0. **Pre-flight.** Sample the GPU for 5 seconds before launching anything. The GPU must be idle: at least 95% of VRAM free at every sample and mean utilization at or below 5%. Otherwise the run is refused with reason `host_contention` and nothing is launched (`--ignore-preflight` runs anyway; the result is uncertified). A rig with another process on the GPU cannot produce a rate that means anything, and in practice the certified engine cannot allocate its 0.90 share on it.
1. **Probe** the hardware and software environment (§8).
2. **Launch** the engine with the pinned configuration (or attach, uncertified).
3. **Prepare** the prompt set: verify `PROMPT_SET_SHA256`, tokenize, truncate to 512 IDs.
4. **Warm-up:** one complete reference job, discarded. This absorbs model load, CUDA graph capture, kernel autotuning and memory allocation.
5. **Measured runs:** `N` complete reference jobs back to back, default `N = 3`, minimum 3 for a certified result. Between runs there is no pause; a rig that throttles thermally shows it here. The measured runs are timed on two clocks (rc.7): the timer the jobs are timed with, and the wall clock.
6. **Power sampling** at 1 Hz throughout the measured runs, when the platform exposes it (`nvidia-smi` power draw), integrated to watt-hours.
7. **Canary check** (§7): score the locked continuations, one request at a time, after the measured runs.
8. **Report** (§8), schema-validated, hashed.

### Score

- `job_seconds[i]` = wall time of measured run *i*.
- **`units_per_hour = 3600 / median(job_seconds)`**. The median is robust to one bad run; the mean is not.
- A certified result requires the pre-flight to have been idle (step 0) when the platform can measure it.
- `stability = (max − min) / median` over measured runs. A certified result requires `stability ≤ 0.10`. Above that, the result is reported with `certified: false` and reason `unstable`; the host should fix cooling or background load and rerun.
- **The two clocks must agree** (rc.7): over the measured runs, `|timer − wall| ≤ max(1% of wall, 2 s)`. Otherwise `certified: false` with reason `clock`. Every rate is work divided by time, so a machine whose timer runs slow overstates units per hour, and understates energy, which overstates units per kWh. Under WSL2 on a Windows laptop the timer ran about 5% slow, and the wall clock was pulled back to Windows' time by a jump of about 2 seconds every half minute (the host client's `results/wsl-windows11-2026-10-06`). The 2-second floor keeps one such jump from failing a short run. A host whose clocks disagree fixes the machine's clock and reruns; the rate is not corrected for it.
- Per-request latency is recorded per run: time to first token (TTFT) and time per output token (TPOT, i.e. `(last_token_time − first_token_time) / 255`), with p50 and p95 across the 256 requests. These are reported for bucketing and routing; I-1 imposes no latency SLO on certification. A later spec version may.
- `units_per_electric_kwh = units_per_hour / (mean_power_watts / 1000)` when power sampling succeeded; otherwise `null`. This is the number the host dashboard puts next to the host's electricity price.

## 7. Canary check

The canary check is a light, in-benchmark guard that the engine is serving the reference model rather than something smaller or more aggressively quantized. It is not the delivery verification layer (build step 3); it is a sanity check.

The host does **not** have to reproduce the reference node's output. Greedy argmax is not stable across batch shapes, kernels or GPU generations: on near-tie tokens it flips, and after one flip the sequences diverge. rc.1 tried exact token matching and failed its own reference machine. Instead the host **scores** the reference continuation:

- 8 of the 256 prompts (IDs listed in `kwh_bench/reference/lock.json`) are canaries.
- At lock time, for each canary, the reference node records the first 32 token IDs of its greedy continuation **and** the mean per-token log-probability it assigns to those 32 tokens under teacher forcing (`prompt_logprobs` on prompt + continuation).
- At benchmark time, after the measured runs, the host computes the same teacher-forced mean log-probability for the same 32 tokens, one request at a time. A canary's **delta** is `|host − reference|` in nats.
- The run **passes** the canary check if all eight canaries were scored and the **mean of their deltas is ≤ 0.05** nats (rc.6; rc.3–rc.5 required six of eight canaries each within 0.05). Failure is reported as `canary: {passed: false}` and the result is `certified: false` with reason `canary`. Each canary's own `pass` (its delta within 0.05) is kept in the report for reading; it does not decide the run.
- The report stores every canary's delta. `kwh-bench verify` re-derives pass/fail from the stored deltas under the tolerance in force, so a report scored under an earlier, looser tolerance is accepted only if it would also pass the current one.

Why this works: the locked continuation is the reference model's own greedy path, and no other model finds those exact tokens as likely. Serving a different model, a different checkpoint or a coarser quantization moves the mean log-probability over 32 tokens by tenths of a nat; the kernel, card and configuration differences between certified hosts have moved it by 0.0000 on three cards (Ampere and Ada) and by up to 0.048 on a fourth (an A40, Ampere). The tolerance is calibrated during the rc phase: it must accept every certified card in `results/` and reject an FP16 or 4-bit variant of the base model. rc.2 used 0.10 and a 4-bit AWQ variant cleared it on five of eight canaries; rc.3 moved to 0.05, which that variant fails 7/8. The A40 then showed that per canary the two overlap: its worst honest canary (0.048) is above the variant's best (0.036). Their means do not overlap (0.023 against 0.108), so rc.6 judges the mean: one noisy canary no longer decides a run, and the margin is about 2× on either side of 0.05. Calibration runs are recorded in `results/canary-calibration.md`.

Canary scoring happens outside the timed jobs and adds no work to the reference job. Uncertified engines skip the canary check and report `canary: null`.

## 8. Report

`kwh-bench run` writes one JSON document validated against `kwh_bench/schema/report.schema.json`. Required top-level content:

| Field | Content |
| --- | --- |
| `spec` | `{series: "I-1", spec_version, bench_version}` |
| `job` | the §3 parameters as actually executed (requests, prompt tokens, generated tokens, concurrency) |
| `engine` | name, version, certified flag, launch mode, full launch args, model id + revision, uncertified reason if any |
| `hardware` | GPU name, VRAM, driver, CUDA version, PCIe generation/width where available, CPU model, RAM, OS, kernel, Python |
| `preflight` | the step-0 sample: VRAM free fraction, mean/max utilization, mean power, `idle` verdict and reasons |
| `runs` | per measured run: job_seconds, tokens_per_second, TTFT p50/p95, TPOT p50/p95, request failures |
| `score` | `units_per_hour`, `median_job_seconds`, `stability`, `units_per_electric_kwh`, `mean_power_watts` |
| `canary` | passed flag, per-canary host/reference mean log-probabilities and deltas, or `null` |
| `clock` | the measured runs on both clocks: `timer_seconds`, `wall_seconds`, `drift_pct` (rc.7; absent from earlier reports) |
| `certified` | boolean; `certified_reasons` lists every failing condition when false |
| `prompt_set_sha256` | must equal `PROMPT_SET_SHA256` |
| `started_at`, `finished_at` | UTC ISO-8601 |
| `report_sha256` | SHA-256 of the canonical JSON of the report with this field and `signature` set to null |
| `signature` | reserved, `null` in step 1; the host client (step 2) signs `report_sha256` with the host key |

`kwh-bench verify <report.json>` re-validates the schema, recomputes `units_per_hour`, `stability` and `report_sha256`, checks the prompt-set hash and, for reports claiming certification, checks every certified-run condition. The clock condition applies to reports made under rc.7 or later, which must carry a `clock`; earlier reports predate it. It is what the platform runs on ingest.

## 9. Lock procedure (rc → 1.0.0)

Fields that can only be produced with the actual weights and the actual engine build are filled by `kwh-bench lock`, run once on the reference node. It writes `kwh_bench/reference/lock.json`:

- `model.revision` — the Hugging Face commit hash of the checkpoint used.
- `model.files` — SHA-256 of each weight and tokenizer file.
- `engine.vllm.version` and `engine.vllm.image` — the exact certified build.
- `canaries[]` — the 8 canary prompt IDs, their 32-token reference continuations, and the reference mean log-probability of each.
- `locked_at`, `locked_on` (hardware of the reference node).

Tagging `v1.0.0` requires a committed `kwh_bench/reference/lock.json` with no null fields, a `results/` table from at least three consumer cards produced by that build, and a `results/canary-calibration.md` showing every certified card inside the §7 tolerance and at least one wrong-model control outside it. Until the lock is complete, every report carries `certified: false` with reason `unlocked`. Until the tag, the spec is `1.0.0-rc.N` and the §7 tolerance may still move; reports keep their deltas, so `verify` re-judges them under whatever tolerance is current.

## 10. What the spec does not decide

Out of scope for this document, decided by the platform on top of it:

- Score **bucketing** granularity. The report gives the platform `units_per_hour`, latency percentiles and stability; how many buckets Grade I has is a market-structure decision (primer §11).
- **Stake sizing** per unit.
- **Re-benchmark cadence** and score decay. The report has everything a decay rule needs; the rule lives in the host client and verifier.
- **Units per hour → units minted.** Minting is the host client's job (step 2) and depends on liveness, not on this spec.
