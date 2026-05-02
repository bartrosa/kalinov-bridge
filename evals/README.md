# Evaluations (`evals/`)

## YAML gotcha: quote `null` provers

In experiment YAML, write `provers: ["null"]` — a bare `null` is parsed as YAML
`~` and becomes the string `"None"` when stringified.

## Running a suite

From the repository root, with a `kalinov.config.yaml` that defines your LLM providers:

```bash
kalinov eval --suite evals/suites/smoke.yaml --prover null \
  --provider my_provider --out ./reports/smoke_run
```

Use `--config-file experiments/cross_model_smoke.yaml` for multi-provider matrices defined in YAML.

Reports are written to `--out` (default when using a config file comes from that file): `report.json` (canonical) and `report.md` (tables for papers and PRs). JSON round-trips for analysis; Markdown is generated only from JSON-shaped data.

## Adding a task

1. Copy or author a `.feature` file under `evals/tasks/` (copies stay stable when `examples/` changes).
2. Add an entry to a suite YAML under `evals/suites/`:

```yaml
tasks:
  - id: my_task
    file: ../tasks/my_task.feature
    expected: either        # solved | gave_up | either
    tags: [algebra]
```

Paths in `file:` are resolved relative to the suite file directory.

### `expected:` semantics

| Value     | Meaning |
|-----------|---------|
| `solved`  | Harness treats mismatches as regressions (`matched_expected`). |
| `gave_up` | Majority of obligations should end in `GAVE_UP` (rare; for negative tests). |
| `either`  | Record outcomes only; `matched_expected` is always true. |

Use short, lowercase tag names (domain or proof style).

## Reproducibility

Identical results (to the cent) require:

- `temperature: 0` in the oracle config
- LLM response cache in `read_only` mode with a fixed `--cache-dir`
- The same `evals/` suite and task files
- The same committed `src/kalinov/cost/pricing.yaml` (SHA-256 is embedded in `report.json`)

Seeds are carried in `EvalConfig` for future adapter hooks; combine with the above for deterministic replays.
