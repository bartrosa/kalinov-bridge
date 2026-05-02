"""CLI tests for `kalinov check --prover lean4` + ForTheL bridge (mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from kalinov.bridges.forthel_lean import TranslationOutcome, TranslationOutcomeKind
from kalinov.cli import main
from kalinov.provers import CheckResult
from kalinov.provers.lean import LeanProver
from kalinov.provers.lean.toolchain import ToolchainInfo

_FAKE_TC = ToolchainInfo(
    elan_path=Path("/fake/elan"),
    lake_path=Path("/fake/lake"),
    lean_version="lean test",
    lake_version="lake test",
)


def _patched_naproche_for_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.interpreters.forthel.shutil.which", lambda _n: "/fake/naproche")

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr("kalinov.interpreters.forthel.subprocess.run", _run)


def _check_always_ok(self: LeanProver, artifact: Any) -> CheckResult:
    return CheckResult(
        ok=True,
        duration_ms=0,
        diagnostics=(),
        obligation=artifact.obligation,
        raw_output="",
    )


def _write(
    path: Path,
    *,
    lean_tag: bool,
    forthel: bool,
) -> Path:
    tag = "  @lean\n" if lean_tag else ""
    body = "[ForTheL] Theorem. trivial." if forthel else "Then 1 = 1"
    text = f"Feature: F\n{tag}  Scenario: S\n    Then {body}\n"
    p = path / "t.feature"
    p.write_text(text, encoding="utf-8")
    return p


def test_no_forthel_does_not_call_translate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patched_naproche_for_interpreter(monkeypatch)
    monkeypatch.setattr("kalinov.cli.detect_toolchain", lambda: _FAKE_TC)
    monkeypatch.setattr(LeanProver, "check", _check_always_ok)

    def _boom(*_a: Any, **_k: Any) -> TranslationOutcome:
        msg = "translate_step should not run with --no-forthel"
        raise AssertionError(msg)

    monkeypatch.setattr("kalinov.cli.translate_step", _boom)

    feat = _write(tmp_path, lean_tag=True, forthel=True)
    rc = main(
        [
            "check",
            "--prover",
            "lean4",
            "--no-forthel",
            str(feat),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert rc == 0


def test_forthel_bridge_runs_without_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patched_naproche_for_interpreter(monkeypatch)
    monkeypatch.setattr("kalinov.cli.detect_toolchain", lambda: _FAKE_TC)
    monkeypatch.setattr(LeanProver, "check", _check_always_ok)

    calls: list[Any] = []

    def _translate(step: Any, **kw: Any) -> TranslationOutcome:
        calls.append(step)
        return TranslationOutcome(
            kind=TranslationOutcomeKind.OK,
            lean_source="#check True",
            raw_output="",
            duration_ms=1,
            diagnostic=None,
        )

    monkeypatch.setattr("kalinov.cli.translate_step", _translate)

    feat = _write(tmp_path, lean_tag=False, forthel=True)
    rc = main(["check", "--prover", "lean4", str(feat), "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
    assert len(calls) == 1


def test_skip_line_on_translation_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patched_naproche_for_interpreter(monkeypatch)
    monkeypatch.setattr("kalinov.cli.detect_toolchain", lambda: _FAKE_TC)
    monkeypatch.setattr(LeanProver, "check", _check_always_ok)

    monkeypatch.setattr(
        "kalinov.cli.translate_step",
        lambda *_a, **_k: TranslationOutcome(
            kind=TranslationOutcomeKind.SKIPPED,
            lean_source=None,
            raw_output="",
            duration_ms=0,
            diagnostic="skipped for test",
        ),
    )

    feat = _write(tmp_path, lean_tag=True, forthel=True)
    rc = main(["check", "--prover", "lean4", str(feat), "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP FORTHEL" in out


def test_failed_translation_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patched_naproche_for_interpreter(monkeypatch)
    monkeypatch.setattr("kalinov.cli.detect_toolchain", lambda: _FAKE_TC)
    monkeypatch.setattr(LeanProver, "check", _check_always_ok)

    monkeypatch.setattr(
        "kalinov.cli.translate_step",
        lambda *_a, **_k: TranslationOutcome(
            kind=TranslationOutcomeKind.FAILED,
            lean_source=None,
            raw_output="e",
            duration_ms=0,
            diagnostic="naproche failed",
        ),
    )

    feat = _write(tmp_path, lean_tag=False, forthel=True)
    rc = main(["check", "--prover", "lean4", str(feat), "--runs-dir", str(tmp_path / "runs")])
    assert rc == 1


def test_dedup_lean_obligation_after_bridge_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patched_naproche_for_interpreter(monkeypatch)
    monkeypatch.setattr("kalinov.cli.detect_toolchain", lambda: _FAKE_TC)

    checks: list[Any] = []

    def _check(self: LeanProver, artifact: Any) -> CheckResult:
        checks.append(artifact)
        return _check_always_ok(self, artifact)

    monkeypatch.setattr(LeanProver, "check", _check)

    monkeypatch.setattr(
        "kalinov.cli.translate_step",
        lambda *_a, **_k: TranslationOutcome(
            kind=TranslationOutcomeKind.OK,
            lean_source="#check True",
            raw_output="",
            duration_ms=0,
            diagnostic=None,
        ),
    )

    feat = _write(tmp_path, lean_tag=True, forthel=True)
    rc = main(["check", "--prover", "lean4", str(feat), "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
    assert len(checks) == 1


def test_null_prover_does_not_call_translate_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _translate(*_a: Any, **_k: Any) -> TranslationOutcome:
        raise AssertionError("translate_step must not run with --prover null")

    monkeypatch.setattr("kalinov.cli.translate_step", _translate)

    feat = _write(tmp_path, lean_tag=True, forthel=True)
    rc = main(["check", "--prover", "null", str(feat), "--runs-dir", str(tmp_path / "runs")])
    assert rc == 0
