"""Append-only ``oracle_loop.jsonl`` on the active run."""

from __future__ import annotations

import json
import time
from pathlib import Path

from kalinov.telemetry.context import active_run
from kalinov.telemetry.jsonl import append_jsonl_record


def log_iteration(
    *,
    obligation_name: str,
    iteration: int,
    outcome_so_far: str,
    duration_ms: int,
    llm_call_id: str | None,
    prover_call_id: str | None,
    cost_usd: str | None,
) -> None:
    """Write one JSON line to ``runs/<run_id>/oracle_loop.jsonl`` when a run is active."""
    ctx = active_run()
    if ctx is None:
        return
    path: Path = ctx.run_dir / "oracle_loop.jsonl"
    record = {
        "ts_ms": int(time.time() * 1000),
        "obligation_name": obligation_name,
        "iteration": iteration,
        "outcome_so_far": outcome_so_far,
        "duration_ms": duration_ms,
        "llm_call_id": llm_call_id,
        "prover_call_id": prover_call_id,
        "cost_usd": cost_usd,
    }
    append_jsonl_record(path, record)


def save_transcript_json(*, obligation_name: str, payload: object) -> Path | None:
    """Write ``transcripts/<safe_name>.json`` under the active run directory."""
    ctx = active_run()
    if ctx is None:
        return None
    safe = obligation_name.replace("/", "_").replace("\\", "_")
    tdir = ctx.run_dir / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f"{safe}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


__all__ = ["log_iteration", "save_transcript_json"]
