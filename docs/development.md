# Development

## Prerequisites

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/)
- **Git** with hooks (optional but recommended):

  ```bash
  git config core.hooksPath .githooks
  ```

## Python

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest
```

## Layout

- `src/kalinov_bridge/` — importable Python package (orchestration will land here).
- `tests/` — Pytest.
- `experiments/` — ad-hoc scripts and exploratory runs (not part of the package; see `experiments/README.md`).
- `lean/` — *planned*: Lake project for benchmark tasks (`lake build` as verifier).

## Lean (planned)

When `lean/` exists:

1. Install Lean 4 / Elan per [Lean 4 manual](https://lean-lang.org/lean4/doc/setup.html).
2. Pin toolchain via `lean-toolchain` and mathlib via Lake (see project `README` inside `lean/`).
3. Expect long first builds; CI will cache when the job is added.

## Optional: git branch aliases

If you use a global helper script (e.g. `git feat slug` → `feat/slug` from `main`), document it in your own `~/.gitconfig`; this repo only requires sensible branch names and Conventional Commits.
