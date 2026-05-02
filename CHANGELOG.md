# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Python package moved to `src/kalinov_bridge/` (src layout).

### Added

- `kalinov.bridges.forthel_lean`: ForTheL → Lean translation pipeline (`translate_step`, `translate_spec`, telemetry to `forthel_translations.jsonl`, 32 KiB cap on captured Naproche output).
- `kalinov check --prover lean4 --no-forthel` to disable the ForTheL bridge; default Lean CLI passes ForTheL claims through Naproche (`--lean` after the temp `.ftl`) before `LeanProver.check`, deduped with `extract_obligations` when scenarios are tagged `@lean`.
- `LeanProver` adapter using a vendored Lean 4 runtime project with mathlib.
- Lean error output parser into structured diagnostics.
- `kalinov check --prover lean4` CLI.
- Dedicated CI job for Lean integration tests, gated behind elan setup.
- ADR-0005: Lean toolchain choice and version pinning.
- `Prover` ABC with `compile`, `check`, `extract_obligations`, `parse_error`.
- `NullProver` deterministic backend for testing (modes: always_ok, always_fail, fail_after_n).
- `kalinov check` CLI subcommand.
- Per-call prover telemetry written to `prover_calls.jsonl` on the active RunContext.
- ADR-0004: Prover ABC frozen.
- `MathTexInterpreter` — extracts inline and display LaTeX math from steps.
- `ForTheLInterpreter` — degradation-tolerant bridge to a local Naproche binary, when present.
- ADR-0001b: CNL-agnostic step body.
- Gherkin frontend: parser (`kalinov.gherkin.parse_feature_file`, `parse_feature_text`) and typed AST.
- Pluggable step interpretation: `StepInterpreter` ABC, `InterpreterChain`, `RawInterpreter`.
- Five example `.feature` files under `examples/`.
- `experiments/` directory for ad-hoc scripts (see `experiments/README.md`).
- `lean/` Lake workspace with **mathlib** (`kalinov_bridge` package, `KalinovBridge` library) and CI job using **lean-action** + Mathlib cache.
- **Makefile** (`make check`, `make run-demo`, …) delegating to `uv` and `lake`.
- Minimal **demo runner**: mock LLM (`by sorry` → `by trivial`), `lake build`, restore `lean/KalinovBridge/Scratch.lean`, write `artifacts/.../results.jsonl`; CLI `kalinov-bridge run-demo`.
- **`experiments/hello_e2e.py`** — same E2E path as a runnable hello-world script.
- Runner **artifacts** now include `*.patched.lean` (verified source) and `*.original.lean` (pre-run snapshot) beside `results.jsonl`.

## [0.1.0] - 2025-03-19

### Added

- OSS scaffold: README, CONTRIBUTING, Code of Conduct, GitHub issue/PR templates, SECURITY.
- Python package layout (`kalinov_bridge`), `pyproject.toml` with uv dev workflow.
- CI workflow: Ruff, Mypy, Pytest on Ubuntu.
- Initial `docs/` stub pointing to architecture notes on GitHub.

## [0.0.1] — 2026-05-01

### Added

- Project bootstrap: `pyproject.toml` (uv), CI, ADRs 0001–0003, telemetry & cost primitives.

[Unreleased]: https://github.com/bartrosa/kalinov-bridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bartrosa/kalinov-bridge/releases/tag/v0.1.0
[0.0.1]: https://github.com/bartrosa/kalinov-bridge/releases/tag/v0.0.1
