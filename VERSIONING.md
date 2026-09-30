# Versioning Policy

Three things carry versions and they move independently.

## 1. Unit series (`I-1`, `I-2`, ...)

A series identifies **the work**. It changes when, and only when, anything in SPEC.md §2–§5 that alters the amount or kind of work in the reference job changes:

- reference model or checkpoint revision
- quantization scheme
- prompt set (content, count or length)
- generated tokens per request or per job
- sampling parameters
- concurrency
- the certified engine's *forbidden list* (anything that changes work per token)

A new series is a **new unit**. Units of series I-1 are never converted into I-2 units; each series has its own order book if both are live, and a series is retired by ceasing to mint it, not by redefining it. Old reports stay valid forever against their own series.

What does **not** create a new series: a new certified engine version that does the same work faster, a new hardware probe field, a report schema addition, a bug fix in the load generator that does not change the job.

## 2. Spec version (`1.0.0`, semver)

The spec version tracks the *document and its reference data*.

- **Major** — new series. `2.0.0` defines `I-2`.
- **Minor** — backward-compatible change within the series: a new certified engine or engine version, a new report field, a tightened certification rule that only *removes* results from the certified set (e.g. adding a latency SLO). Reports from an older minor remain valid I-1 reports.
- **Patch** — clarifications, typo fixes, no change to any number or rule.

Pre-release: `1.0.0-rc.N` until `kwh_bench/reference/lock.json` is complete (SPEC.md §9).

## 3. Benchmark tool version (`kwh-bench x.y.z`)

The Python package follows semver on its own. Its major version equals the spec major it implements (kwh-bench 1.x implements spec 1.x / series I-1). A tool release states the spec version it implements in `kwh_bench/reference.py`, and every report carries both.

## 4. Reference model deprecation

The reference model will age out. The policy:

1. **Announce** a successor series (`I-2`) with its own SPEC at least 90 days before I-1 minting stops. Both series run in parallel during the overlap; hosts benchmark against both.
2. **Trigger** for starting the process, any of: the checkpoint is removed or relicensed upstream; the certified engine drops support for the quantization format; the model falls out of the top tier of open-weight 8B models by common benchmarks for two consecutive quarters (the platform publishes its yardstick when this is invoked); or the platform's own demand for the reference model falls below a published floor.
3. **Weights are mirrored** by the platform at lock time so an upstream removal cannot strand I-1 before the overlap ends. The mirror is hash-verified against `kwh_bench/reference/lock.json`.
4. **After minting stops**, I-1 units already minted expire on their own 72-hour clock. Nothing is converted.

## 5. Changing this file

Edits to VERSIONING.md are patch-level unless they change the definition of a series boundary, which is itself a minor bump with a note in CHANGELOG.md.
