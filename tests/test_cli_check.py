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


def test_check_non_utf8_feature_does_not_crash(tmp_path: Path) -> None:
    """Regression: a non-UTF-8 ``.feature`` mid-run must not crash the CLI.

    ``parse_feature_file`` calls ``Path.read_text(encoding="utf-8")``, which
    raises :class:`UnicodeDecodeError` (a ``ValueError`` subclass — *not* a
    :class:`GherkinParseError`) on the first non-UTF-8 byte. Pre-fix only
    ``GherkinParseError`` was caught in :func:`run_check_programmatic`'s
    per-file loop, so the decoding error propagated out as an uncaught
    Python exception, replacing the documented exit code 2 (parse failure)
    with an exit-1 traceback and skipping every later file in ``paths``.

    This also reproduced the same crash inside the MCP server's ``check``
    tool handler, which severed the connection for every other client.
    """
    bad = tmp_path / "bad.feature"
    bad.write_bytes(
        b"# language: en\nFeature: bad\n  Scenario: oh \xff no\n    Then 1=1\n",
    )
    good = Path("examples/pythagoras.feature")

    res = _run_check(str(good), str(bad), "--runs-dir", str(tmp_path))
    assert res.returncode == 2, (
        "non-UTF-8 file should be reported as a parse failure (exit 2), "
        f"not crash with a traceback. stderr:\n{res.stderr}"
    )
    assert "could not decode" in res.stderr.lower() or "utf-8" in res.stderr.lower(), (
        f"stderr must explain the decoding failure; got: {res.stderr!r}"
    )
    assert "Traceback" not in res.stderr, (
        f"CLI must not emit a Python traceback for a malformed feature file; got: {res.stderr!r}"
    )
