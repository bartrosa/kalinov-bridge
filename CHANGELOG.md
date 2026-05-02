# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Python package moved to `src/kalinov_bridge/` (src layout).

### Added

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
