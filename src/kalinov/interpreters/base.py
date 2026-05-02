"""Pluggable interpretation of Gherkin steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from kalinov.gherkin.ast import Step


@dataclass(frozen=True, slots=True)
class InterpretedStep:
    """Structured output from a :class:`StepInterpreter`."""

    original: Step
    kind: str
    payload: Mapping[str, Any]
    interpreter_name: str


class StepInterpreter(ABC):
    """Interpreter for a single Gherkin step; chain multiple for layered semantics."""

    name: ClassVar[str]

    @abstractmethod
    def interpret(self, step: Step, context: Mapping[str, Any]) -> InterpretedStep | None:
        """Return a result, or ``None`` to defer to the next interpreter in the chain."""


class NoMatchingInterpreterError(Exception):
    """Raised when no interpreter in the chain handled the step."""


class InterpreterChain:
    """Chain-of-responsibility over interpreters (first non-``None`` wins)."""

    def __init__(self, interpreters: Sequence[StepInterpreter]) -> None:
        self._interpreters = tuple(interpreters)

    def interpret(
        self,
        step: Step,
        context: Mapping[str, Any] | None = None,
    ) -> InterpretedStep:
        ctx: Mapping[str, Any] = context if context is not None else {}
        for interp in self._interpreters:
            out = interp.interpret(step, ctx)
            if out is not None:
                return out
        raise NoMatchingInterpreterError("No interpreter matched the step.")
