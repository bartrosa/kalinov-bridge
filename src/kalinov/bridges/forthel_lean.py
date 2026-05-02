"""Translate Naproche/ForTheL interpreted steps into Lean source for :class:`LeanProver`."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from kalinov.interpreters.base import InterpretedStep
from kalinov.provers.base import SpecDocument
from kalinov.telemetry.context import active_run
from kalinov.telemetry.jsonl import append_jsonl_record

_OUTPUT_CAP = 32 * 1024

_BRIDGE_INVOCATIONS = 0


class ForTheLToLeanError(Exception):
    """Caller misuse or unsupported translation input."""


class TranslationOutcomeKind(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TranslationConfig:
    naproche_binary: str = "naproche"
    naproche_args: tuple[str, ...] = ("--lean",)
    """Arguments after the temporary ``.ftl`` path (see CHANGELOG / Naproche release notes)."""
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class TranslationOutcome:
    kind: TranslationOutcomeKind
    lean_source: str | None
    raw_output: str
    duration_ms: int
    diagnostic: str | None


def bridge_translate_calls() -> int:
    """Test helper: number of :func:`translate_step` invocations (see tests)."""
    return _BRIDGE_INVOCATIONS


def reset_bridge_translate_calls() -> None:
    """Test helper: reset :func:`bridge_translate_calls` counter."""
    global _BRIDGE_INVOCATIONS  # noqa: PLW0603
    _BRIDGE_INVOCATIONS = 0


def _truncate(text: str, limit: int = _OUTPUT_CAP) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _timeout_process_streams(exc: subprocess.TimeoutExpired) -> str:
    chunks: list[str] = []
    for raw in (exc.stdout, exc.stderr):
        if raw is None:
            chunks.append("")
        elif isinstance(raw, bytes):
            chunks.append(raw.decode("utf-8", errors="replace"))
        else:
            chunks.append(raw)
    return "".join(chunks)


def _forthel_raw_source(step: InterpretedStep) -> str:
    payload: Mapping[str, Any] = step.payload
    raw = payload.get("raw_input")
    if isinstance(raw, str) and raw.strip():
        return raw
    return step.original.text


def _log_translation(
    *,
    step_keyword: str,
    outcome_kind: TranslationOutcomeKind,
    duration_ms: int,
    naproche_exit_code: int | None,
    output_size_bytes: int,
) -> None:
    ctx = active_run()
    if ctx is None:
        return
    path = ctx.run_dir / "forthel_translations.jsonl"
    append_jsonl_record(
        path,
        {
            "step_keyword": step_keyword,
            "outcome_kind": outcome_kind.value,
            "duration_ms": duration_ms,
            "naproche_exit_code": naproche_exit_code,
            "output_size_bytes": output_size_bytes,
        },
    )


def translate_step(
    step: InterpretedStep,
    *,
    config: TranslationConfig | None = None,
) -> TranslationOutcome:
    """Emit Lean source from a ForTheL-recognized interpreted step via Naproche."""
    global _BRIDGE_INVOCATIONS  # noqa: PLW0603
    cfg = config or TranslationConfig()

    if step.interpreter_name != "forthel":
        raise ForTheLToLeanError("step must come from ForTheLInterpreter")
    if step.kind not in {"claim", "skipped"}:
        raise ForTheLToLeanError(f"unsupported kind {step.kind!r}")

    _BRIDGE_INVOCATIONS += 1

    kw = step.original.keyword
    t0 = time.perf_counter_ns()

    if step.kind == "skipped":
        reason = str(step.payload.get("reason", "skipped"))
        elapsed = int((time.perf_counter_ns() - t0) / 1_000_000)
        out = TranslationOutcome(
            kind=TranslationOutcomeKind.SKIPPED,
            lean_source=None,
            raw_output="",
            duration_ms=elapsed,
            diagnostic=f"upstream skipped ({reason})",
        )
        _log_translation(
            step_keyword=kw,
            outcome_kind=out.kind,
            duration_ms=elapsed,
            naproche_exit_code=None,
            output_size_bytes=0,
        )
        return out

    nap = shutil.which(cfg.naproche_binary)
    if nap is None:
        elapsed = int((time.perf_counter_ns() - t0) / 1_000_000)
        out = TranslationOutcome(
            kind=TranslationOutcomeKind.SKIPPED,
            lean_source=None,
            raw_output="",
            duration_ms=elapsed,
            diagnostic="naproche not found on PATH",
        )
        _log_translation(
            step_keyword=kw,
            outcome_kind=out.kind,
            duration_ms=elapsed,
            naproche_exit_code=None,
            output_size_bytes=0,
        )
        return out

    raw_src = _forthel_raw_source(step)
    tmp_path: Path | None = None
    exit_code: int | None = None
    nap_outcome: TranslationOutcome

    try:
        fd, tmp_path_str = tempfile.mkstemp(suffix=".ftl", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(raw_src)
        tmp_path = Path(tmp_path_str)
        proc = subprocess.run(
            [nap, str(tmp_path), *cfg.naproche_args],
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
            check=False,
        )
        exit_code = proc.returncode
        combined_out = _truncate((proc.stdout or "") + (proc.stderr or ""))
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        lean_stdout = (proc.stdout or "").strip()

        if exit_code == 0 and lean_stdout:
            nap_outcome = TranslationOutcome(
                kind=TranslationOutcomeKind.OK,
                lean_source=lean_stdout,
                raw_output=combined_out,
                duration_ms=elapsed_ms,
                diagnostic=None,
            )
        elif exit_code != 0:
            nap_outcome = TranslationOutcome(
                kind=TranslationOutcomeKind.FAILED,
                lean_source=None,
                raw_output=combined_out,
                duration_ms=elapsed_ms,
                diagnostic=f"naproche exited with code {exit_code}",
            )
        else:
            nap_outcome = TranslationOutcome(
                kind=TranslationOutcomeKind.FAILED,
                lean_source=None,
                raw_output=combined_out,
                duration_ms=elapsed_ms,
                diagnostic="naproche produced no Lean output on stdout",
            )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        exit_code = None
        partial = _truncate(_timeout_process_streams(exc))
        nap_outcome = TranslationOutcome(
            kind=TranslationOutcomeKind.FAILED,
            lean_source=None,
            raw_output=partial,
            duration_ms=elapsed_ms,
            diagnostic="naproche subprocess timed out",
        )
    except OSError as exc:
        elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        exit_code = None
        nap_outcome = TranslationOutcome(
            kind=TranslationOutcomeKind.FAILED,
            lean_source=None,
            raw_output="",
            duration_ms=elapsed_ms,
            diagnostic=f"naproche translation I/O error: {exc}",
        )
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    _log_translation(
        step_keyword=kw,
        outcome_kind=nap_outcome.kind,
        duration_ms=nap_outcome.duration_ms,
        naproche_exit_code=exit_code,
        output_size_bytes=len(nap_outcome.raw_output.encode("utf-8")),
    )
    return nap_outcome


def translate_spec(
    spec: SpecDocument,
    *,
    config: TranslationConfig | None = None,
) -> tuple[tuple[InterpretedStep, TranslationOutcome], ...]:
    """Run :func:`translate_step` on every ForTheL-recognized step."""
    pairs: list[tuple[InterpretedStep, TranslationOutcome]] = []
    it = iter(spec.interpreted_steps)
    for scenario in spec.feature_file.feature.scenarios:
        for _step in scenario.steps:
            try:
                interp = next(it)
            except StopIteration as exc:
                raise ForTheLToLeanError(
                    "interpreted_steps shorter than feature step walk",
                ) from exc
            if interp.interpreter_name != "forthel":
                continue
            pairs.append((interp, translate_step(interp, config=config)))
    try:
        next(it)
    except StopIteration:
        pass
    else:
        raise ForTheLToLeanError("interpreted_steps longer than feature step walk")
    return tuple(pairs)
