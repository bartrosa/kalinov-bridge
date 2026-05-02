"""Fallback interpreter that preserves step text verbatim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from kalinov.gherkin.ast import Step
from kalinov.interpreters.base import InterpretedStep, StepInterpreter


class RawInterpreter(StepInterpreter):
    """Always matches; wraps opaque step text for downstream tooling."""

    name: ClassVar[str] = "raw"

    def interpret(self, step: Step, context: Mapping[str, Any]) -> InterpretedStep:
        _ = context
        return InterpretedStep(
            original=step,
            kind="raw",
            payload={"text": step.text},
            interpreter_name=self.name,
        )
