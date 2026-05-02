"""Deterministic in-memory prover for tests and smoke tooling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from kalinov.provers.base import (
    CheckResult,
    CompileResult,
    ProofArtifact,
    ProofObligation,
    Prover,
    SpecDocument,
)
from kalinov.provers.errors import ProverError, StructuredError
from kalinov.provers.telemetry import log_prover_call


class NullProverMode(StrEnum):
    ALWAYS_OK = "always_ok"
    ALWAYS_FAIL = "always_fail"
    FAIL_AFTER_N = "fail_after_n"


@dataclass(frozen=True, slots=True)
class NullProverConfig:
    mode: NullProverMode = NullProverMode.ALWAYS_OK
    fail_after: int = 0
    fixed_duration_ms: int = 1


class NullProver(Prover):
    """In-memory deterministic prover."""

    backend_name: ClassVar[str] = "null"
    language: ClassVar[str] = "null"

    def __init__(self, config: NullProverConfig | None = None) -> None:
        self._config = config or NullProverConfig()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def _eval_ok(self) -> bool:
        if self._config.mode == NullProverMode.ALWAYS_OK:
            return True
        if self._config.mode == NullProverMode.ALWAYS_FAIL:
            return False
        return self._call_count < self._config.fail_after

    def _fail_diagnostics(self) -> tuple[StructuredError, ...]:
        err = StructuredError(
            severity="error",
            message="null prover forced failure",
            file=None,
            line=None,
            column=None,
            code="null_forced_fail",
        )
        return (err,)

    def compile(self, artifact: ProofArtifact) -> CompileResult:
        t0 = time.perf_counter_ns()
        ok = self._eval_ok()
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        diag_count = 0 if ok else 1
        log_prover_call(
            backend=self.backend_name,
            operation="compile",
            obligation_name=artifact.obligation.name,
            ok=ok,
            duration_ms=elapsed_ms,
            diagnostic_count=diag_count,
        )
        self._call_count += 1
        return CompileResult(
            ok=ok,
            duration_ms=self._config.fixed_duration_ms,
            diagnostics=() if ok else self._fail_diagnostics(),
            raw_output="",
        )

    def check(self, artifact: ProofArtifact) -> CheckResult:
        t0 = time.perf_counter_ns()
        ok = self._eval_ok()
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        diag_count = 0 if ok else 1
        log_prover_call(
            backend=self.backend_name,
            operation="check",
            obligation_name=artifact.obligation.name,
            ok=ok,
            duration_ms=elapsed_ms,
            diagnostic_count=diag_count,
        )
        self._call_count += 1
        return CheckResult(
            ok=ok,
            duration_ms=self._config.fixed_duration_ms,
            diagnostics=() if ok else self._fail_diagnostics(),
            obligation=artifact.obligation,
            raw_output="",
        )

    def extract_obligations(self, spec: SpecDocument) -> tuple[ProofObligation, ...]:
        t0 = time.perf_counter_ns()
        out: list[ProofObligation] = []
        it = iter(spec.interpreted_steps)
        for scenario in spec.feature_file.feature.scenarios:
            for step_idx, _ in enumerate(scenario.steps):
                try:
                    interp = next(it)
                except StopIteration as exc:
                    raise ProverError(
                        "interpreted_steps shorter than scenario step walk",
                    ) from exc
                if interp.kind == "claim":
                    out.append(
                        ProofObligation(
                            name=f"{scenario.name}#{step_idx}",
                            statement=interp.original.text,
                            hypotheses=(),
                            metadata={},
                        ),
                    )
        try:
            next(it)
        except StopIteration:
            pass
        else:
            raise ProverError("interpreted_steps longer than scenario step walk")
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        log_prover_call(
            backend=self.backend_name,
            operation="extract_obligations",
            obligation_name=None,
            ok=True,
            duration_ms=elapsed_ms,
            diagnostic_count=0,
        )
        return tuple(out)

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
