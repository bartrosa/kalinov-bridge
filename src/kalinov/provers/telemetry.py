"""One-line-per-call telemetry for prover adapters."""

from __future__ import annotations

from pathlib import Path

from kalinov.telemetry.context import active_run
from kalinov.telemetry.jsonl import append_jsonl_record


def log_prover_call(
    *,
    backend: str,
    operation: str,
    obligation_name: str | None,
    ok: bool,
    duration_ms: int,
    diagnostic_count: int,
) -> None:
    """Append one JSON object to ``runs/<run_id>/prover_calls.jsonl`` when a run is active."""
    ctx = active_run()
    if ctx is None:
        return
    path: Path = ctx.run_dir / "prover_calls.jsonl"
    append_jsonl_record(
        path,
        {
            "backend": backend,
            "operation": operation,
            "obligation_name": obligation_name,
            "ok": ok,
            "duration_ms": duration_ms,
            "diagnostic_count": diagnostic_count,
        },
    )
