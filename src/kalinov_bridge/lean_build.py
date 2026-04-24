"""Invoke Lake from Python."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final


def run_lake_build(
    lean_project_dir: Path,
    *,
    timeout_sec: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``lake build`` in *lean_project_dir* (must contain ``lakefile.toml``)."""
    lean_project_dir = lean_project_dir.resolve()
    if not (lean_project_dir / "lakefile.toml").is_file():
        msg = f"Not a Lake project directory: {lean_project_dir}"
        raise FileNotFoundError(msg)
    cmd: Final[tuple[str, ...]] = ("lake", "build")
    return subprocess.run(
        cmd,
        cwd=lean_project_dir,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
