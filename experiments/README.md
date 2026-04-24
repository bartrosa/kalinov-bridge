# Experiments

Ad-hoc scripts and exploratory runs live here. They are **not** part of the importable `kalinov_bridge` package and are not required to pass the same quality bar as library code.

`experiments/` is excluded from **Ruff** in `pyproject.toml` so quick scripts do not block CI; run Ruff manually on a file if you want lint there.

Run from the repo root with the project env:

```bash
uv sync --group dev
uv run python experiments/your_script.py
```

Promote stable code into `src/kalinov_bridge/` (and tests under `tests/`) when it matures.
