"""Unit tests for ForTheL → Lean bridge (no real Naproche / Lean)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from kalinov.bridges import (
    ForTheLToLeanError,
    TranslationConfig,
    TranslationOutcomeKind,
    translate_spec,
    translate_step,
)
from kalinov.bridges.forthel_lean import bridge_translate_calls, reset_bridge_translate_calls
from kalinov.gherkin import parse_feature_text
from kalinov.gherkin.ast import Location, Step
from kalinov.interpreters.base import InterpretedStep
from kalinov.provers.base import SpecDocument

_LOC = Location(line=1, column=1)


def _step(keyword: str, text: str) -> Step:
    return Step(
        keyword=keyword,
        text=text,
        doc_string=None,
        data_table=None,
        location=_LOC,
    )


def _raw_claim() -> InterpretedStep:
    return InterpretedStep(
        original=_step("Then ", "[ForTheL] Theorem. trivial."),
        kind="claim",
        payload={"raw_input": "Theorem. trivial.", "kind_detail": "parsed_only"},
        interpreter_name="forthel",
    )


def _skipped_upstream() -> InterpretedStep:
    return InterpretedStep(
        original=_step("Then ", "[ForTheL] x"),
        kind="skipped",
        payload={"reason": "binary_not_found", "raw_input": "x"},
        interpreter_name="forthel",
    )


def _non_forthel() -> InterpretedStep:
    return InterpretedStep(
        original=_step("Then ", "plain"),
        kind="claim",
        payload={},
        interpreter_name="raw",
    )


@pytest.fixture(autouse=True)
def _reset_bridge_counter() -> Iterator[None]:
    reset_bridge_translate_calls()
    yield
    reset_bridge_translate_calls()


def test_raises_for_non_forthel_step() -> None:
    with pytest.raises(ForTheLToLeanError, match="ForTheLInterpreter"):
        translate_step(_non_forthel())


def test_raises_for_unsupported_kind() -> None:
    st = InterpretedStep(
        original=_step("Given ", "x"),
        kind="unknown",
        payload={},
        interpreter_name="forthel",
    )
    with pytest.raises(ForTheLToLeanError, match="unsupported kind"):
        translate_step(st)


def test_skipped_upstream_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = MagicMock()
    monkeypatch.setattr("kalinov.bridges.forthel_lean.subprocess.run", spy)
    out = translate_step(_skipped_upstream())
    assert out.kind == TranslationOutcomeKind.SKIPPED
    assert "upstream skipped" in (out.diagnostic or "")
    spy.assert_not_called()


def test_skipped_when_naproche_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: None)
    out = translate_step(_raw_claim())
    assert out.kind == TranslationOutcomeKind.SKIPPED
    assert "naproche not found" in (out.diagnostic or "").lower()


def test_ok_when_naproche_emits_lean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: "/fake/naproche")
    lean = "theorem bridge_ok : True := trivial"

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=lean + "\n",
            stderr="",
        )

    monkeypatch.setattr("kalinov.bridges.forthel_lean.subprocess.run", _run)
    out = translate_step(_raw_claim())
    assert out.kind == TranslationOutcomeKind.OK
    assert out.lean_source == lean.strip()


def test_failed_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: "/fake/naproche")

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="",
            stderr="bad",
        )

    monkeypatch.setattr("kalinov.bridges.forthel_lean.subprocess.run", _run)
    out = translate_step(_raw_claim())
    assert out.kind == TranslationOutcomeKind.FAILED
    assert "code 2" in (out.diagnostic or "")


def test_failed_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: "/fake/naproche")

    def _run(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="naproche", timeout=1.0, output=None, stderr=None)

    monkeypatch.setattr("kalinov.bridges.forthel_lean.subprocess.run", _run)
    out = translate_step(_raw_claim())
    assert out.kind == TranslationOutcomeKind.FAILED
    assert "timed out" in (out.diagnostic or "").lower()


def test_translate_spec_walk_mismatch_short() -> None:
    feature = parse_feature_text("Feature: F\n  Scenario: S\n    Then x\n")
    spec = SpecDocument(feature_file=feature, interpreted_steps=())
    with pytest.raises(ForTheLToLeanError, match="shorter"):
        translate_spec(spec)


def test_translate_spec_filters_non_forthel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: None)
    text = """
Feature: F
  Scenario: One
    Then plain step
    Then [ForTheL] Theorem. x.
""".strip()
    ff = parse_feature_text(text)
    from kalinov.interpreters import InterpreterChain, MathTexInterpreter, RawInterpreter
    from kalinov.interpreters.forthel import ForTheLInterpreter

    chain = InterpreterChain(
        [MathTexInterpreter(), ForTheLInterpreter(), RawInterpreter()],
    )
    steps: list[InterpretedStep] = []
    for sc in ff.feature.scenarios:
        for st in sc.steps:
            steps.append(chain.interpret(st))
    spec = SpecDocument(feature_file=ff, interpreted_steps=tuple(steps))
    pairs = translate_spec(spec)
    assert len(pairs) == 1
    assert pairs[0][0].interpreter_name == "forthel"


def test_telemetry_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: "/fake/naproche")

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="theorem t : True := trivial\n",
            stderr="",
        )

    monkeypatch.setattr("kalinov.bridges.forthel_lean.subprocess.run", _run)

    from kalinov.telemetry import start_run

    with start_run(runs_dir=tmp_path) as run:
        translate_step(_raw_claim())

    log = run.run_dir / "forthel_translations.jsonl"
    assert log.is_file()
    line = log.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["outcome_kind"] == "ok"
    assert "step_keyword" in rec
    assert rec["naproche_exit_code"] == 0


def test_output_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: "/fake/naproche")
    cap = 32 * 1024
    huge = "x" * (cap + 1000)

    def _run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=huge, stderr="")

    monkeypatch.setattr("kalinov.bridges.forthel_lean.subprocess.run", _run)
    out = translate_step(_raw_claim())
    assert len(out.raw_output) == cap


def test_bridge_counter_increments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kalinov.bridges.forthel_lean.shutil.which", lambda _n: None)
    assert bridge_translate_calls() == 0
    translate_step(_raw_claim())
    assert bridge_translate_calls() == 1


def test_default_config_naproche_args() -> None:
    cfg = TranslationConfig()
    assert cfg.naproche_args == ("--lean",)
