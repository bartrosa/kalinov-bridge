"""Orchestration: apply LLM output, verify with Lean, record results."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a single runner cycle."""

    task: str
    success: bool
    returncode: int
    duration_ms: float
    stderr_tail: str
    artifacts_dir: str
    patched_lean_name: str
    original_lean_name: str


def _stderr_tail(stderr: str, *, max_chars: int = 4000) -> str:
    if len(stderr) <= max_chars:
        return stderr
    return stderr[-max_chars:]


def run_demo_cycle(
    *,
    scratch_file: Path,
    lean_dir: Path,
    artifacts_dir: Path,
    fill_proof: Callable[[str], str],
    lake_build: Callable[[Path], subprocess.CompletedProcess[str]],
    task_name: str = "run-demo",
) -> RunResult:
    """
    Read *scratch_file*, replace body via *fill_proof*, run *lake_build*(*lean_dir*), restore file.

    Writes under *artifacts_dir*:

    - ``<Stem>.patched<suffix>`` — Lean source **after** the mock / LLM (what was verified).
    - ``<Stem>.original<suffix>`` — committed source before the run (for diff / audit).
    - ``results.jsonl`` — one JSON object (includes filenames above).
    - ``lake_stderr.txt`` — stderr from ``lake build``.
    """
    scratch_file = scratch_file.resolve()
    lean_dir = lean_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    patched_name = f"{scratch_file.stem}.patched{scratch_file.suffix}"
    original_name = f"{scratch_file.stem}.original{scratch_file.suffix}"
    patched_path = artifacts_dir / patched_name
    original_path = artifacts_dir / original_name

    original = scratch_file.read_text(encoding="utf-8")
    patched_source = fill_proof(original)
    original_path.write_text(original, encoding="utf-8")
    patched_path.write_text(patched_source, encoding="utf-8")

    t0 = time.perf_counter()
    proc: subprocess.CompletedProcess[str]
    try:
        scratch_file.write_text(patched_source, encoding="utf-8")
        proc = lake_build(lean_dir)
    finally:
        scratch_file.write_text(original, encoding="utf-8")

    duration_ms = (time.perf_counter() - t0) * 1000.0
    stderr = proc.stderr or ""
    (artifacts_dir / "lake_stderr.txt").write_text(stderr, encoding="utf-8")

    success = proc.returncode == 0
    result = RunResult(
        task=task_name,
        success=success,
        returncode=proc.returncode,
        duration_ms=duration_ms,
        stderr_tail=_stderr_tail(stderr),
        artifacts_dir=str(artifacts_dir),
        patched_lean_name=patched_name,
        original_lean_name=original_name,
    )
    line: dict[str, Any] = {**asdict(result)}
    (artifacts_dir / "results.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    return result
