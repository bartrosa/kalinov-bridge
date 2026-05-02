"""Detect elan-provided ``lake`` / ``lean`` executables."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kalinov.provers.errors import ProverError

_ELIAN_HINT = "Install elan: curl https://elan.lean-lang.org/elan-init.sh -sSf | sh"


class ToolchainNotFoundError(ProverError):
    """Raised when ``elan``, ``lake``, or ``lean`` is not available."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ToolchainInfo:
    elan_path: Path
    lake_path: Path
    lean_version: str
    lake_version: str


def _version_output(cmd: list[str], *, timeout: float) -> str:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def detect_toolchain() -> ToolchainInfo:
    """Locate ``elan`` / ``lake`` / ``lean`` on ``PATH`` and read versions."""
    elan = shutil.which("elan")
    lake = shutil.which("lake")
    lean = shutil.which("lean")
    missing = [name for name, path in (("elan", elan), ("lake", lake), ("lean", lean)) if not path]
    if missing:
        raise ToolchainNotFoundError(
            f"missing tools on PATH: {', '.join(missing)}. {_ELIAN_HINT}",
        )
    assert elan is not None and lake is not None and lean is not None
    lake_ver = _version_output([lake, "--version"], timeout=5.0).strip().splitlines()[0]
    lean_ver = _version_output([lean, "--version"], timeout=5.0).strip().splitlines()[0]
    return ToolchainInfo(
        elan_path=Path(elan),
        lake_path=Path(lake),
        lean_version=lean_ver,
        lake_version=lake_ver,
    )


def runtime_project_root() -> Path:
    """Absolute path to ``provers/lean/runtime`` next to the repo root."""
    here = Path(__file__).resolve()
    # src/kalinov/provers/lean/toolchain.py -> parents[3] == repo root
    repo_root = here.parents[3]
    return repo_root / "provers" / "lean" / "runtime"
