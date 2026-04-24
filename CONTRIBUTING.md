# Contributing

Thanks for helping improve **kalinov-bridge**.

## Workflow

- **Default branch:** `main` should stay green (CI passing).
- **Branches:** short-lived feature branches, PR into `main` (GitHub Flow).
- **Naming:** prefix + kebab-case slug, e.g. `feat/benchmark-runner`, `fix/ci-cache`, `docs/contributing`. Optional: a global shell script + `git config alias.feat='!bash ~/.git-newbranch.sh feat'` (or similar) so `git feat my-slug` creates `feat/my-slug` from `main`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) — enforced locally via `.githooks/commit-msg` when hooks are enabled:

  ```bash
  git config core.hooksPath .githooks
  ```

  Examples: `feat(ci): add uv cache`, `docs: link development guide`.

- **PRs:** keep them small, describe *what* and *why*, link an issue when it exists.

## Local checks (must match CI)

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest
```

Equivalent shortcut: `make python-check` (and `make check` includes `lake build` in `lean/`).

## Python environment

Use **uv** (`uv sync --group dev`). Python **3.12+** is supported.

## Lean / Lake

When the `lean/` benchmark lands, `docs/development.md` will describe how to pin `lean-toolchain` and run `lake build`. Until then, Python-only contributions are enough.

## Code style

- **Ruff** for lint + format; **Mypy** strict on `kalinov_bridge` (tests relaxed).
- No filler comments; public functions need type hints.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reports.
