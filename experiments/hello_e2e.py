#!/usr/bin/env python3
"""
Hello-world end-to-end: mock "LLM" patches ``lean/KalinovBridge/Scratch.lean``,
runs ``lake build`` in ``lean/``, restores the file, writes ``artifacts/...``.

Run from repository root::

    uv run python experiments/hello_e2e.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

# Repo root on sys.path when using ``uv run python …`` after ``uv sync``.
from kalinov_bridge.lean_build import run_lake_build
from kalinov_bridge.mock_llm import fill_proof
from kalinov_bridge.pipeline import run_demo_cycle
from kalinov_bridge.repo import find_repo_root


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = find_repo_root(here)
    scratch = repo_root / "lean" / "KalinovBridge" / "Scratch.lean"
    lean_dir = repo_root / "lean"
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    artifacts = repo_root / "artifacts" / f"hello-e2e-{stamp}-{uuid.uuid4().hex[:8]}"

    print("kalinov-bridge hello E2E")
    print(f"  repo_root     = {repo_root}")
    print(f"  scratch_file  = {scratch}")
    print(f"  lean_dir      = {lean_dir}")
    print(f"  artifacts_dir = {artifacts}")
    print()

    result = run_demo_cycle(
        scratch_file=scratch,
        lean_dir=lean_dir,
        artifacts_dir=artifacts,
        fill_proof=fill_proof,
        lake_build=run_lake_build,
        task_name="hello-e2e",
    )

    print(f"  success      = {result.success}")
    print(f"  returncode   = {result.returncode}")
    print(f"  duration_ms  = {result.duration_ms:.1f}")
    print(f"  results.jsonl   = {artifacts / 'results.jsonl'}")
    print(f"  patched Lean    = {artifacts / result.patched_lean_name}")
    print(f"  original snapshot = {artifacts / result.original_lean_name}")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
