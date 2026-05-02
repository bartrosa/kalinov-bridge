"""Subprocess helpers for Lake / Lean with timeouts."""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kalinov.provers.base import ProofArtifact


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


def run_lake(
    lake_exe: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> SubprocessResult:
    """Run ``lake <args>`` under *cwd* with a hard timeout."""
    cmd = [str(lake_exe), *args]
    t0 = time.perf_counter_ns()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
    return SubprocessResult(
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration_ms=elapsed_ms,
    )


def write_artifact_to_runtime(
    artifact: ProofArtifact,
    *,
    runtime_root: Path,
) -> Path:
    """Write ``artifact.body`` to ``KalinovBridge/Tmp_<hash>.lean``."""
    body_bytes = artifact.body.encode("utf-8")
    digest = hashlib.sha256(body_bytes).hexdigest()[:16]
    rel_dir = runtime_root / "KalinovBridge"
    rel_dir.mkdir(parents=True, exist_ok=True)
    path = rel_dir / f"Tmp_{digest}.lean"
    path.write_text(artifact.body, encoding="utf-8")
    return path
