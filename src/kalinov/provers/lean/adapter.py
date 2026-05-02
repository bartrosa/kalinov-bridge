"""Lean 4 :class:`Prover` implementation."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
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
from kalinov.provers.lean.error_parser import parse_lean_output
from kalinov.provers.lean.runtime import run_lake, write_artifact_to_runtime
from kalinov.provers.lean.toolchain import ToolchainInfo, detect_toolchain, runtime_project_root
from kalinov.provers.telemetry import log_prover_call


def _has_lean_tag(tags: tuple[str, ...]) -> bool:
    return any(t.strip().lower() == "@lean" for t in tags)


def _timeout_diagnostic(operation: str) -> StructuredError:
    return StructuredError(
        severity="error",
        message=f"{operation} timed out",
        file=None,
        line=None,
        column=None,
        code="timeout",
    )


@dataclass(frozen=True, slots=True)
class LeanProverConfig:
    compile_timeout_seconds: float = 60.0
    check_timeout_seconds: float = 120.0
    runtime_root_override: Path | None = None


class LeanProver(Prover):
    """Lean 4 backend using the vendored Lake workspace."""

    backend_name: ClassVar[str] = "lean4"
    language: ClassVar[str] = "lean4"

    def __init__(
        self,
        config: LeanProverConfig | None = None,
        *,
        toolchain: ToolchainInfo | None = None,
    ) -> None:
        self._config = config or LeanProverConfig()
        self._tc = toolchain or detect_toolchain()
        root = self._config.runtime_root_override or runtime_project_root()
        self._runtime_root = root.resolve()

    def compile(self, artifact: ProofArtifact) -> CompileResult:
        if artifact.language != self.language:
            return CompileResult(
                ok=False,
                duration_ms=0,
                diagnostics=(
                    StructuredError(
                        severity="error",
                        message=f"expected artifact.language={self.language!r}",
                        file=None,
                        line=None,
                        column=None,
                        code="language_mismatch",
                    ),
                ),
                raw_output="",
            )
        t0 = time.perf_counter_ns()
        path = write_artifact_to_runtime(artifact, runtime_root=self._runtime_root)
        rel = path.relative_to(self._runtime_root)
        try:
            r = run_lake(
                self._tc.lake_path,
                ["env", "lean", str(rel)],
                cwd=self._runtime_root,
                timeout_seconds=self._config.compile_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.perf_counter_ns() - t0) / 1_000_000)
            timeout_diags = (_timeout_diagnostic("compile"),)
            log_prover_call(
                backend=self.backend_name,
                operation="compile",
                obligation_name=artifact.obligation.name,
                ok=False,
                duration_ms=elapsed,
                diagnostic_count=len(timeout_diags),
            )
            return CompileResult(
                ok=False,
                duration_ms=elapsed,
                diagnostics=timeout_diags,
                raw_output="",
            )
        finally:
            path.unlink(missing_ok=True)

        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        raw = r.stdout + r.stderr
        parsed_diags = parse_lean_output(raw)
        has_err = any(d.severity == "error" for d in parsed_diags)
        ok = r.exit_code == 0 and not has_err
        log_prover_call(
            backend=self.backend_name,
            operation="compile",
            obligation_name=artifact.obligation.name,
            ok=ok,
            duration_ms=elapsed_ms,
            diagnostic_count=len(parsed_diags),
        )
        return CompileResult(
            ok=ok,
            duration_ms=r.duration_ms,
            diagnostics=parsed_diags,
            raw_output=raw,
        )

    def check(self, artifact: ProofArtifact) -> CheckResult:
        if artifact.language != self.language:
            err = StructuredError(
                severity="error",
                message=f"expected artifact.language={self.language!r}",
                file=None,
                line=None,
                column=None,
                code="language_mismatch",
            )
            return CheckResult(
                ok=False,
                duration_ms=0,
                diagnostics=(err,),
                obligation=artifact.obligation,
                raw_output="",
            )
        t0 = time.perf_counter_ns()
        path = write_artifact_to_runtime(artifact, runtime_root=self._runtime_root)
        rel = path.relative_to(self._runtime_root)
        try:
            r = run_lake(
                self._tc.lake_path,
                ["build", str(rel)],
                cwd=self._runtime_root,
                timeout_seconds=self._config.check_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            elapsed = int((time.perf_counter_ns() - t0) / 1_000_000)
            timeout_diags = (_timeout_diagnostic("check"),)
            log_prover_call(
                backend=self.backend_name,
                operation="check",
                obligation_name=artifact.obligation.name,
                ok=False,
                duration_ms=elapsed,
                diagnostic_count=len(timeout_diags),
            )
            return CheckResult(
                ok=False,
                duration_ms=elapsed,
                diagnostics=timeout_diags,
                obligation=artifact.obligation,
                raw_output="",
            )
        finally:
            path.unlink(missing_ok=True)

        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        raw = r.stdout + r.stderr
        parsed_diags = parse_lean_output(raw)
        has_err = any(d.severity == "error" for d in parsed_diags)
        ok = r.exit_code == 0 and not has_err
        log_prover_call(
            backend=self.backend_name,
            operation="check",
            obligation_name=artifact.obligation.name,
            ok=ok,
            duration_ms=elapsed_ms,
            diagnostic_count=len(parsed_diags),
        )
        return CheckResult(
            ok=ok,
            duration_ms=r.duration_ms,
            diagnostics=parsed_diags,
            obligation=artifact.obligation,
            raw_output=raw,
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
                if not _has_lean_tag(scenario.tags):
                    continue
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
        return parse_lean_output(raw)
