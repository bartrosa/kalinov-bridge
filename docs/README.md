# Documentation

Project is in **early scaffold**; deeper design docs will grow here.

| Doc | Purpose |
|-----|---------|
| [development.md](development.md) | Local setup, hooks, `src/` package, `experiments/`, `lean/` + Lake |
| [`lean/README.md`](../lean/README.md) | Lake + mathlib commands and layout |
| *Architecture / ADRs* | To be added under `docs/adr/` once the Lean benchmark and public I/O spec stabilize |

## Glossary (preview)

- **Task** — one benchmark item: Lean sources + metadata (e.g. theorem with `sorry`).
- **Run** — single orchestrator execution (model + parameters + outcome).
- **Snapshot** — saved generated `.lean` / logs for a run.
