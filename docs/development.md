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

### Make (optional)

If you have GNU Make:

```bash
make help
make check        # Python checks + lake build in lean/
make run-demo     # mock LLM patches Scratch.lean, runs lake build, writes artifacts/
```

Same steps as CI for Python; `make run-demo` requires a working `lake` on your PATH.

End-to-end smoke (also as a script under `experiments/`):

```bash
uv run python experiments/hello_e2e.py
```

Each run writes under `artifacts/…/`:

- **`Scratch.patched.lean`** — source after the mock / LLM (what `lake build` checked; safe to copy into your own Lean file).
- **`Scratch.original.lean`** — tree state before the run.
- **`results.jsonl`**, **`lake_stderr.txt`** — metadata and logs.

## Layout

- `src/kalinov_bridge/` — importable Python package (orchestration will land here).
- `tests/` — Pytest.
- `experiments/` — ad-hoc scripts and exploratory runs (not part of the package; see `experiments/README.md`).
- `lean/` — Lake + **mathlib** workspace (`lake build` is the verifier). See [`lean/README.md`](../lean/README.md). The demo runner edits `lean/KalinovBridge/Scratch.lean` briefly (then restores it).

## Lean / Lake

1. Install [Elan](https://github.com/leanprover/elan) (Lean version manager).
2. From repo root:

   ```bash
   cd lean
   lake exe cache get   # recommended; uses Mathlib precompiled cache when available
   lake build
   ```

3. First clone can take a while while dependencies download; CI uses **lean-action** caching on `lean/.lake` and Mathlib’s `lake exe cache get`.

Toolchain is pinned in `lean/lean-toolchain`; do not hand-edit without matching Mathlib’s expectations (see `lean/lakefile.toml`).

## Optional: git branch aliases

If you use a global helper script (e.g. `git feat slug` → `feat/slug` from `main`), document it in your own `~/.gitconfig`; this repo only requires sensible branch names and Conventional Commits.
