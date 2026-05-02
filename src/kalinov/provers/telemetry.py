"""One-line-per-call telemetry for prover adapters."""

from __future__ import annotations

from pathlib import Path

from ulid import ULID

from kalinov.telemetry.context import active_run
from kalinov.telemetry.jsonl import append_jsonl_record

_LAST_PROVER_COMPILE_CALL_ID: str | None = None
_LAST_PROVER_CHECK_CALL_ID: str | None = None


def log_prover_call(
    *,
    backend: str,
    operation: str,
    obligation_name: str | None,
    ok: bool,
    duration_ms: int,
    diagnostic_count: int,
) -> str | None:
    """Append one JSON object to ``runs/<run_id>/prover_calls.jsonl`` when a run is active.

    Returns the per-call ``call_id`` (ULID string) when a row was written.
    """
    global _LAST_PROVER_COMPILE_CALL_ID, _LAST_PROVER_CHECK_CALL_ID
    ctx = active_run()
    if ctx is None:
        return None
    path: Path = ctx.run_dir / "prover_calls.jsonl"
    call_id = str(ULID())
    if operation == "compile":
        _LAST_PROVER_COMPILE_CALL_ID = call_id
    elif operation == "check":
        _LAST_PROVER_CHECK_CALL_ID = call_id
    append_jsonl_record(
        path,
        {
            "call_id": call_id,
            "backend": backend,
            "operation": operation,
            "obligation_name": obligation_name,
            "ok": ok,
            "duration_ms": duration_ms,
            "diagnostic_count": diagnostic_count,
        },
    )
    return call_id


def take_last_prover_compile_call_id() -> str | None:
    return _LAST_PROVER_COMPILE_CALL_ID


def take_last_prover_check_call_id() -> str | None:
    return _LAST_PROVER_CHECK_CALL_ID


__all__ = [
    "log_prover_call",
    "take_last_prover_check_call_id",
    "take_last_prover_compile_call_id",
]
