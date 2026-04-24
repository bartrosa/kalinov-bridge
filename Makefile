# Convenience targets — CI remains the source of truth (see .github/workflows/ci.yml).
UV ?= uv

.PHONY: help sync lint typecheck test python-check lean-check check run-demo

help:
	@echo "Targets:"
	@echo "  make sync          uv sync --group dev"
	@echo "  make lint          ruff check + format --check"
	@echo "  make typecheck     mypy"
	@echo "  make test          pytest"
	@echo "  make python-check  lint + typecheck + test"
	@echo "  make lean-check    lake build in lean/"
	@echo "  make check         python-check + lean-check"
	@echo "  make run-demo      mock LLM + lake build + artifacts (see Scratch.lean)"

sync:
	$(UV) sync --group dev

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy .

test:
	$(UV) run pytest

python-check: lint typecheck test

lean-check:
	cd lean && lake build

check: python-check lean-check

run-demo:
	$(UV) run kalinov-bridge run-demo
