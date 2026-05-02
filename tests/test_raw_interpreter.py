"""Tests for RawInterpreter and InterpreterChain."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from kalinov.gherkin.ast import Location, Step
from kalinov.interpreters import (
    InterpretedStep,
    InterpreterChain,
    NoMatchingInterpreterError,
    RawInterpreter,
    StepInterpreter,
)


def _dummy_step(text: str = "hello") -> Step:
    loc = Location(line=1, column=1)
    return Step(
        keyword="Given ",
        text=text,
        doc_string=None,
        data_table=None,
        location=loc,
    )


def test_raw_always_matches() -> None:
    raw = RawInterpreter()
    out = raw.interpret(_dummy_step(), {})
    assert out is not None
    assert out.kind == "raw"


def test_raw_payload_contains_step_text() -> None:
    raw = RawInterpreter()
    step = _dummy_step("Σ k")
    out = raw.interpret(step, {})
    assert out.payload["text"] == "Σ k"


class _AlwaysNone(StepInterpreter):
    name: ClassVar[str] = "none"

    def interpret(self, step: Step, context: Mapping[str, Any]) -> InterpretedStep | None:
        return None


class _AlwaysHit(StepInterpreter):
    name: ClassVar[str] = "hit"

    def interpret(self, step: Step, context: Mapping[str, Any]) -> InterpretedStep | None:
        return InterpretedStep(
            original=step,
            kind="hit",
            payload={"ok": True},
            interpreter_name=self.name,
        )


def test_chain_uses_first_matching_interpreter() -> None:
    step = _dummy_step()
    chain = InterpreterChain([_AlwaysNone(), _AlwaysHit()])
    out = chain.interpret(step, {})
    assert out.interpreter_name == "hit"
    assert out.kind == "hit"


def test_chain_with_only_raw_always_succeeds() -> None:
    chain = InterpreterChain([RawInterpreter()])
    out = chain.interpret(_dummy_step(), {})
    assert out.kind == "raw"


def test_chain_raises_when_no_match() -> None:
    chain = InterpreterChain([_AlwaysNone()])
    with pytest.raises(NoMatchingInterpreterError):
        chain.interpret(_dummy_step(), {})
