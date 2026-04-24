"""Locate repository root (directory containing this project's pyproject.toml)."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents from *start* (default: cwd) for ``pyproject.toml`` naming this package."""
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        manifest = directory / "pyproject.toml"
        if not manifest.is_file():
            continue
        text = manifest.read_text(encoding="utf-8")
        if 'name = "kalinov-bridge"' in text:
            return directory
    msg = "Could not find kalinov-bridge repo root (pyproject.toml with name kalinov-bridge)."
    raise FileNotFoundError(msg)
