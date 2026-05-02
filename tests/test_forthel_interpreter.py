"""Tests for ForTheLInterpreter."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from kalinov.gherkin.ast import DocString, Location, Step
from kalinov.interpreters.forthel import (
    ForTheLBackendStatus,
    ForTheLConfig,
    ForTheLInterpreter,
)

_LOC = Location(line=1, column=1)


def _step(
    keyword: str,
    text: str,
    *,
    doc_string: DocString | None = None,
) -> Step:
    return Step(
        keyword=keyword,
        text=text,
        doc_string=doc_string,
        data_table=None,
        location=_LOC,
    )


def test_no_marker_returns_none() -> None:
    interp = ForTheLInterpreter()
    step = _step("Then ", "Just plain English.")
    assert interp.interpret(step, {}) is None


def test_marker_in_text_recognized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: None,
    )
    interp = ForTheLInterpreter()
    step = _step("Then ", "[ForTheL] Signature. Let n denote a natural number.")
    out = interp.interpret(step, {})
    assert out is not None
    assert out.payload["raw_input"] == "Signature. Let n denote a natural number."


def test_doc_string_ftl_recognized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: None,
    )
    doc = DocString(content="Theorem. x = x.", content_type="ftl", location=_LOC)
    step = _step("Then ", "see below", doc_string=doc)
    out = ForTheLInterpreter().interpret(step, {})
    assert out is not None
    assert out.payload["raw_input"] == "Theorem. x = x."


def test_context_language_forthel_recognized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: None,
    )
    step = _step("Then ", "Any placeholder text.")
    out = ForTheLInterpreter().interpret(step, {"language": "forthel"})
    assert out is not None
    assert out.payload["raw_input"] == step.text


def test_skipped_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kalinov.interpreters.forthel.shutil.which", lambda _name: None)
    interp = ForTheLInterpreter()
    step = _step("Then ", "[ForTheL] foo")
    out = interp.interpret(step, {})
    assert out is not None
    assert out.kind == "skipped"
    assert out.payload["reason"] == "binary_not_found"
    assert out.payload["raw_input"] == "foo"


def test_skipped_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="naproche", timeout=1.0)

    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: "/bin/naproche",
    )
    monkeypatch.setattr("kalinov.interpreters.forthel.subprocess.run", _boom)
    interp = ForTheLInterpreter()
    step = _step("Then ", "[ForTheL] bar")
    out = interp.interpret(step, {})
    assert out is not None
    assert out.kind == "skipped"
    assert out.payload["reason"] == "timeout"


def test_success_path_returns_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(
        cmd: list[str | Any],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: "/bin/naproche",
    )
    monkeypatch.setattr("kalinov.interpreters.forthel.subprocess.run", _fake_run)
    interp = ForTheLInterpreter()
    step = _step("Then ", "[ForTheL] baz")
    out = interp.interpret(step, {})
    assert out is not None
    assert out.kind == "claim"
    assert out.payload["exit_code"] == 0
    assert "ok" in out.payload["backend_output"]


def test_disabled_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: "/bin/naproche",
    )
    interp = ForTheLInterpreter(ForTheLConfig(enabled=False))
    step = _step("Then ", "[ForTheL] quux")
    out = interp.interpret(step, {})
    assert out is not None
    assert out.kind == "skipped"
    assert out.payload["reason"] == "disabled"
    assert out.payload["raw_input"] == "quux"


def test_status_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _which(_name: str) -> str | None:
        nonlocal calls
        calls += 1
        return "/bin/naproche"

    monkeypatch.setattr("kalinov.interpreters.forthel.shutil.which", _which)
    interp = ForTheLInterpreter()
    assert interp.status() == ForTheLBackendStatus.AVAILABLE
    assert interp.status() == ForTheLBackendStatus.AVAILABLE
    assert calls == 1


def test_output_truncated_at_16kb(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = "x" * 100_000

    def _fake_run(
        cmd: list[str | Any],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=huge, stderr="")

    monkeypatch.setattr(
        "kalinov.interpreters.forthel.shutil.which",
        lambda _name: "/bin/naproche",
    )
    monkeypatch.setattr("kalinov.interpreters.forthel.subprocess.run", _fake_run)
    interp = ForTheLInterpreter()
    step = _step("Then ", "[ForTheL] tiny")
    out = interp.interpret(step, {})
    assert out is not None
    assert len(out.payload["backend_output"]) <= 16 * 1024
