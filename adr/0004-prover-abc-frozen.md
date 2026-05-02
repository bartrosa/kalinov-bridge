# ADR-0004 — Prover ABC frozen

## Status

Accepted

## Context

Downstream components (oracle loop, benchmarks, mining) need a single stable
contract for invoking verification backends. Lean and other provers differ in
CLI shape and error formats, but upper layers should depend only on a shared
interface so we can swap implementations without rewriting orchestration.

## Decision

- Define a **prover-agnostic** abstract base class `Prover` with exactly four
  operations:
  1. **`compile`** — fast syntax/type smoke check of an artifact.
  2. **`check`** — full verification that an artifact discharges an obligation.
  3. **`extract_obligations`** — map an interpreted spec into a sequence of
     `ProofObligation` values.
  4. **`parse_error`** — normalize backend text into `StructuredError` rows.
- **No additional ABC methods in this ADR** (no cancel, version, or health
  hooks). Extensions ship as new optional protocols or additive methods only
  when a concrete consumer requires them.
- Ship **`NullProver` permanently** as the deterministic in-process backend for
  unit tests, CI smoke checks, and harness development. It is not a temporary
  stub; it remains the reference implementation of the contract’s semantics.
- Use **`StructuredError`** (severity, message, optional location, optional
  code) instead of raw strings in results so UIs and telemetry can classify
  failures without regex scraping.
- **Telemetry:** each `compile`, `check`, and `extract_obligations` call on a
  concrete prover must emit one JSON line to `runs/<run_id>/prover_calls.jsonl`
  when a `RunContext` is active (`backend`, `operation`, timing, pass/fail,
  diagnostic counts, obligation name when applicable).

## Consequences

- Upper layers import **`kalinov.provers`** types and `Prover` only; no direct
  imports of Lean-specific modules from orchestration code.
- Adding a new backend means implementing the four methods plus honest logging;
  the ABC stays small and reviewable.
- `StructuredError` forces adapters to surface machine-readable codes where the
  backend provides them, while still allowing free-form `message` text.
