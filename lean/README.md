# Lean 4 workspace (`kalinov_bridge`)

Lake package depending on **mathlib** (pin in `lakefile.toml` / `lake-manifest.json`). This folder is the formal side of the bridge: benchmarks and verified proofs will live under `KalinovBridge/`.

## Commands

From repository root:

```bash
cd lean
lake exe cache get   # optional; speeds up first build when Mathlib cache hits
lake build
```

Toolchain is pinned in `lean-toolchain` (Elan reads it automatically).

## Layout

| Path | Role |
|------|------|
| `lakefile.toml` | Package name, Mathlib `require`, default library target |
| `KalinovBridge/` | Project modules (e.g. `Basic.lean` with a `sorry` scaffold for future LLM fills) |
| `lake-manifest.json` | Locked dependency revisions — commit changes with care |
