"""ForTheL-shaped steps bridged to a local Naproche binary (optional, graceful skip)."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from kalinov.gherkin.ast import Step
from kalinov.interpreters.base import InterpretedStep, StepInterpreter

_OUTPUT_LIMIT = 16 * 1024
_UNSET: Any = object()


class ForTheLBackendStatus(StrEnum):
    AVAILABLE = "available"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ForTheLConfig:
    enabled: bool = True
    binary_name: str = "naproche"
    timeout_seconds: float = 5.0
    extra_args: tuple[str, ...] = ()


_FORTHEL_PREFIX = re.compile(r"^\[ForTheL\]\s*", re.IGNORECASE)


def _extract_raw_input(step: Step, context: Mapping[str, Any]) -> str | None:
    m = _FORTHEL_PREFIX.match(step.text)
    if m:
        return step.text[m.end() :].lstrip()
    if step.doc_string is not None and step.doc_string.content_type == "ftl":
        return step.doc_string.content
    if context.get("language") == "forthel":
        return step.text
    return None


def _truncate_output(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


class ForTheLInterpreter(StepInterpreter):
    """Bridges ForTheL-tagged step content to a local Naproche binary."""

    name: ClassVar[str] = "forthel"

    def __init__(self, config: ForTheLConfig | None = None) -> None:
        self._config = config or ForTheLConfig()
        self._which_cache: str | None | Any = _UNSET

    def status(self) -> ForTheLBackendStatus:
        if not self._config.enabled:
            return ForTheLBackendStatus.DISABLED
        resolved = self._resolve_binary()
        return ForTheLBackendStatus.AVAILABLE if resolved else ForTheLBackendStatus.NOT_FOUND

    def _resolve_binary(self) -> str | None:
        if self._which_cache is _UNSET:
            self._which_cache = shutil.which(self._config.binary_name)
        return self._which_cache

    def interpret(self, step: Step, context: Mapping[str, Any]) -> InterpretedStep | None:
        raw = _extract_raw_input(step, context)
        if raw is None:
            return None
        try:
            if not self._config.enabled:
                return self._skipped(step, reason="disabled", raw_input=raw)
            binary_path = self._resolve_binary()
            if not binary_path:
                return self._skipped(step, reason="binary_not_found", raw_input=raw)
            return self._invoke(step, raw, binary_path)
        except subprocess.TimeoutExpired:
            return self._skipped(step, reason="timeout", raw_input=raw)
        except OSError:
            return self._skipped(step, reason="binary_not_found", raw_input=raw)

    def _skipped(
        self,
        step: Step,
        *,
        reason: str,
        raw_input: str,
    ) -> InterpretedStep:
        payload: dict[str, Any] = {
            "kind_detail": "skipped",
            "raw_input": raw_input,
            "reason": reason,
        }
        return InterpretedStep(
            original=step,
            kind="skipped",
            payload=payload,
            interpreter_name=self.name,
        )

    def _invoke(self, step: Step, raw_input: str, binary_path: str) -> InterpretedStep:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ftl",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(raw_input)
            tmp_path = tmp.name
        proc: subprocess.CompletedProcess[str]
        try:
            proc = subprocess.run(
                [binary_path, tmp_path, *self._config.extra_args],
                check=False,
                text=True,
                capture_output=True,
                timeout=self._config.timeout_seconds,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        combined = (proc.stdout or "") + (proc.stderr or "")
        detail = "verified" if proc.returncode == 0 else "parsed_only"
        payload: dict[str, Any] = {
            "kind_detail": detail,
            "raw_input": raw_input,
            "backend_output": _truncate_output(combined),
            "exit_code": proc.returncode,
        }
        return InterpretedStep(
            original=step,
            kind="claim",
            payload=payload,
            interpreter_name=self.name,
        )
