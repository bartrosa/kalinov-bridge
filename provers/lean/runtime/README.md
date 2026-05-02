# Kalinov Lean prover runtime

Vendored Lean 4 + Mathlib project used by `LeanProver` (`lake build`, `lake env lean`).

## Prerequisites

- [elan](https://github.com/leanprover/elan) provides `lean` and `lake` on your `PATH`.

## Local build

```bash
cd provers/lean/runtime
lake build
```

The first build compiles Mathlib and may take a long time.

## Bumping Lean / Mathlib

1. Edit `lean-toolchain` to the desired `leanprover/lean4:vX.Y.Z` tag.
2. Edit `lakefile.toml`: set `[[require]]` `rev` to the Mathlib git tag or SHA that matches that Lean release (see [mathlib4](https://github.com/leanprover-community/mathlib4) CI / releases).
3. Run `lake update` in this directory.
4. Commit `lake-manifest.json` and `lakefile.toml` / `lean-toolchain` together.
5. Run `lake build` and `uv run pytest -m lean`.
