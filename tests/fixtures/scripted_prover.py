"""Configurable :class:`~kalinov.provers.base.Prover` for oracle loop tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

from kalinov.provers.base import (
    CheckResult,
    CompileResult,
    ProofArtifact,
    ProofObligation,
    Prover,
    SpecDocument,
)
from kalinov.provers.errors import StructuredError
from kalinov.provers.telemetry import log_prover_call


@dataclass
class ScriptedProver(Prover):
    """Each iteration uses ``rounds[i]`` as ``(compile_ok, check_ok)``."""

    backend_name: ClassVar[str] = "scripted"
    language: ClassVar[str] = "null"
    rounds: list[tuple[bool, bool]] = field(default_factory=list)
    compile_fail_message: str = "scripted compile failure"
    check_fail_message: str = "scripted check failure"
    _round_idx: int = 0

    def compile(self, artifact: ProofArtifact) -> CompileResult:
        t0 = time.perf_counter_ns()
        c_ok, _ = self.rounds[self._round_idx]
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        log_prover_call(
            backend=self.backend_name,
            operation="compile",
            obligation_name=artifact.obligation.name,
            ok=c_ok,
            duration_ms=elapsed_ms,
            diagnostic_count=0 if c_ok else 1,
        )
        diags: tuple[StructuredError, ...] = ()
        if not c_ok:
            diags = (
                StructuredError(
                    severity="error",
                    message=self.compile_fail_message,
                    file=None,
                    line=None,
                    column=None,
                    code="scripted_compile",
                ),
            )
        result = CompileResult(
            ok=c_ok,
            duration_ms=1,
            diagnostics=diags,
            raw_output="",
        )
        if not c_ok:
            self._round_idx += 1
        return result

    def check(self, artifact: ProofArtifact) -> CheckResult:
        t0 = time.perf_counter_ns()
        _, k_ok = self.rounds[self._round_idx]
        self._round_idx += 1
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        log_prover_call(
            backend=self.backend_name,
            operation="check",
            obligation_name=artifact.obligation.name,
            ok=k_ok,
            duration_ms=elapsed_ms,
            diagnostic_count=0 if k_ok else 1,
        )
        diags: tuple[StructuredError, ...] = ()
        if not k_ok:
            diags = (
                StructuredError(
                    severity="error",
                    message=self.check_fail_message,
                    file=None,
                    line=None,
                    column=None,
                    code="scripted_check",
                ),
            )
        return CheckResult(
            ok=k_ok,
            duration_ms=1,
            diagnostics=diags,
            obligation=artifact.obligation,
            raw_output="",
        )

    def extract_obligations(self, spec: SpecDocument) -> tuple[ProofObligation, ...]:
        return ()

    def parse_error(self, raw: str) -> tuple[StructuredError, ...]:
        return (
            StructuredError(
                severity="error",
                message=raw,
                file=None,
                line=None,
                column=None,
                code=None,
            ),
        )


__all__ = ["ScriptedProver"]
