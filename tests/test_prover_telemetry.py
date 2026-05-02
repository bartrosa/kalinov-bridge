"""Tests for prover call telemetry."""

from __future__ import annotations

import contextvars
import json
import threading
from pathlib import Path

from kalinov.provers import NullProver, ProofArtifact, ProofObligation
from kalinov.provers.telemetry import log_prover_call
from kalinov.telemetry import start_run


def test_logs_to_active_run(tmp_path: Path) -> None:
    obl = ProofObligation(name="goal_a", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    with start_run(runs_dir=tmp_path) as run:
        NullProver().check(artifact=art)
        log_path = run.run_dir / "prover_calls.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    row = json.loads(lines[-1])
    assert row["backend"] == "null"
    assert row["operation"] == "check"
    assert row["ok"] is True
    assert "duration_ms" in row


def test_no_run_no_log(tmp_path: Path) -> None:
    log_prover_call(
        backend="null",
        operation="check",
        obligation_name=None,
        ok=True,
        duration_ms=1,
        diagnostic_count=0,
    )
    assert not any(tmp_path.iterdir())


def test_log_includes_obligation_name(tmp_path: Path) -> None:
    obl = ProofObligation(name="named_goal", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    with start_run(runs_dir=tmp_path) as run:
        NullProver().check(artifact=art)
        body = (run.run_dir / "prover_calls.jsonl").read_text(encoding="utf-8")
    row = json.loads(body.strip().splitlines()[-1])
    assert row["obligation_name"] == "named_goal"


def test_concurrent_calls_no_interleave(tmp_path: Path) -> None:
    obl = ProofObligation(name="o", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    p = NullProver()
    n = 12

    with start_run(runs_dir=tmp_path) as run:
        threads: list[threading.Thread] = []
        for _ in range(n):
            ctx = contextvars.copy_context()

            def worker(ctx: contextvars.Context = ctx) -> None:
                ctx.run(lambda: p.check(artifact=art))

            threads.append(threading.Thread(target=worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        raw = (run.run_dir / "prover_calls.jsonl").read_text(encoding="utf-8")

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(lines) == n
    for ln in lines:
        json.loads(ln)
