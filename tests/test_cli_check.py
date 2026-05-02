"""Tests for ``kalinov check``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_check(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kalinov.cli", "check", *args],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_with_existing_example(tmp_path: Path) -> None:
    feat = Path("examples/pythagoras.feature")
    res = _run_check(str(feat), "--runs-dir", str(tmp_path))
    assert res.returncode == 0
    assert "OK " in res.stdout
    assert "run_id=" in res.stdout


def test_check_always_fail_returns_1(tmp_path: Path) -> None:
    feat = Path("examples/pythagoras.feature")
    res = _run_check(
        str(feat),
        "--runs-dir",
        str(tmp_path),
        "--mode",
        "always_fail",
    )
    assert res.returncode == 1
    assert "FAIL" in res.stdout


def test_check_creates_telemetry(tmp_path: Path) -> None:
    feat = Path("examples/pythagoras.feature")
    res = _run_check(str(feat), "--runs-dir", str(tmp_path))
    assert res.returncode == 0
    runs = list(tmp_path.iterdir())
    assert runs
    jsonl = runs[0] / "prover_calls.jsonl"
    assert jsonl.is_file()
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    json.loads(lines[0])


def test_check_missing_file_exit_2(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.feature"
    res = _run_check(str(missing), "--runs-dir", str(tmp_path))
    assert res.returncode == 2
