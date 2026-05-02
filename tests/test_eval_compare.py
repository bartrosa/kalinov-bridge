"""Compare helper tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalinov.eval.compare import compare_runs
from kalinov.eval.matrix import EvalConfig
from kalinov.eval.runner import RunResult, TaskResult
from kalinov.eval.suite import Suite, Task, TaskExpected
from kalinov.oracle.strategy import OracleOutcome, OracleOutcomeKind
from kalinov.provers.base import ProofArtifact, ProofObligation


def _tr(task_id: str, kind: OracleOutcomeKind) -> TaskResult:
    obl = ProofObligation(name="n", statement="s", hypotheses=())
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    oc = OracleOutcome(
        obligation=obl,
        kind=kind,
        attempts=(),
        final_artifact=art,
        total_cost_usd=Decimal("0"),
        diagnostic=None,
    )
    return TaskResult(
        task=Task(id=task_id, file=Path("f.feature"), expected=TaskExpected.EITHER, tags=()),
        config_label="c",
        obligations_total=1,
        obligations_solved=1 if kind is OracleOutcomeKind.SOLVED else 0,
        outcomes=(oc,),
        total_cost_usd=Decimal("0"),
        total_tokens=__import__("kalinov.cost.models", fromlist=["TokenUsage"]).TokenUsage(),
        duration_ms=1,
        matched_expected=True,
        telemetry_run_id="x",
    )


def _run(label: str, trs: tuple[TaskResult, ...]) -> RunResult:
    suite = Suite(suite_id="s", description="", tasks=tuple(tr.task for tr in trs))
    oc = __import__("kalinov.oracle.strategy", fromlist=["OracleConfig"]).OracleConfig()
    cfg = EvalConfig(
        prover_name="null",
        provider_name="p",
        model=None,
        seed=0,
        oracle=oc,
        label=label,
    )
    return RunResult(
        suite=suite,
        config=cfg,
        task_results=trs,
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
    )


def test_compare_highlights_differing_task() -> None:
    a = _run("a", (_tr("t1", OracleOutcomeKind.SOLVED),))
    b = _run("b", (_tr("t1", OracleOutcomeKind.GAVE_UP),))
    md = compare_runs(a, b)
    assert "t1" in md
    assert "solved" in md.lower() or "gave_up" in md.lower()
