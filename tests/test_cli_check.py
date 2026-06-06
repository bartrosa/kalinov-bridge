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


def test_check_continues_past_non_utf8_feature(tmp_path: Path) -> None:
    """A non-UTF-8 file mid-list must not crash ``kalinov check``.

    Pre-fix ``parse_feature_file``'s ``read_text(encoding="utf-8")``
    raised :class:`UnicodeDecodeError` (a ``ValueError`` subclass — not a
    :class:`GherkinParseError`) on the first stray ``\\xff`` byte. The
    exception escaped the for-path loop in ``run_check_programmatic``,
    skipped the run-summary print, and surfaced as a Python stack trace
    on stderr. The good and trailing files in the same invocation never
    produced their per-obligation ``OK``/``FAIL`` lines.
    """
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: G\n  Scenario: GoodScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.feature"
    bad.write_bytes(
        b"# language: en\nFeature: bad\n  Scenario: \xff oops\n    Then nope\n",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: T\n  Scenario: ThirdScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    res = _run_check(
        str(good),
        str(bad),
        str(third),
        "--runs-dir",
        str(runs),
    )
    # Parse failures yield exit code 2.
    assert res.returncode == 2, (
        f"expected exit 2 (parse_failed); got {res.returncode}, stderr={res.stderr!r}"
    )
    # Both good files must have produced an OK line and the run summary
    # must still be printed even though the middle file was un-decodable.
    assert "GoodScenario" in res.stdout
    assert "ThirdScenario" in res.stdout
    assert "summary:" in res.stdout, (
        f"summary line must still be emitted across the failure; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )


def test_check_continues_past_missing_feature(tmp_path: Path) -> None:
    """A vanished feature file mid-list must not crash ``kalinov check``.

    Trigger: a CI-friendly invocation lists files that may be cleaned up
    by a parallel job between the CLI's pre-flight ``is_file()`` check and
    the per-path parse. Pre-fix the resulting :class:`FileNotFoundError`
    (an :class:`OSError` subclass) escaped the for-path loop and skipped
    the summary print for the surviving files.
    """
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: G\n  Scenario: GoodScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: T\n  Scenario: ThirdScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    # Stage a feature path, then remove the file *after* recording the
    # path on the command line. The CLI's pre-flight ``is_file()``
    # validates *before* the loop; this simulates the file vanishing
    # between validation and per-path parse.
    vanish = tmp_path / "vanish.feature"
    vanish.write_text(
        "# language: en\nFeature: V\n  Scenario: V\n    Then $1=1$\n",
        encoding="utf-8",
    )
    # We deliberately do not delete it here because ``_run_check`` already
    # passes through argparse's pre-flight; instead we drive the
    # programmatic API directly so the ``is_file()`` happens at parse time
    # only.
    from kalinov.cli_core import run_check_programmatic
    from kalinov.provers import NullProver, NullProverConfig, NullProverMode

    vanish.unlink()  # vanished after path recording
    runs = tmp_path / "runs"
    prover = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))
    res = run_check_programmatic(
        prover,
        [good.resolve(), vanish.resolve(), third.resolve()],
        "null",
        runs,
        forthel_bridge=False,
        echo=False,
    )
    assert res.parse_failed is True, (
        "missing file must surface in parse_failed=True (drives exit 2)"
    )
    names = [r.obligation_name for r in res.results]
    assert any("GoodScenario" in n for n in names), (
        f"good.feature must produce a result; got {names!r}"
    )
    assert any("ThirdScenario" in n for n in names), (
        f"third.feature must be processed after the missing one; "
        f"got {names!r} (pre-fix FileNotFoundError aborted the loop)"
    )
