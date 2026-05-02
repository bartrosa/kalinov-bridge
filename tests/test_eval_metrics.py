"""Metrics aggregation tests."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalinov.cost.models import TokenUsage
from kalinov.eval.matrix import EvalConfig
from kalinov.eval.metrics import aggregate
from kalinov.eval.runner import RunResult, TaskResult
from kalinov.eval.suite import Suite, Task, TaskExpected
from kalinov.oracle.strategy import OracleAttempt, OracleOutcome, OracleOutcomeKind
from kalinov.provers.base import ProofArtifact, ProofObligation


def _obl(name: str = "g") -> ProofObligation:
    return ProofObligation(name=name, statement="s", hypotheses=())


def _art(obl: ProofObligation) -> ProofArtifact:
    return ProofArtifact(obligation=obl, body="", language="null", metadata={})


def _solved_outcome(attempts_n: int) -> OracleOutcome:
    obl = _obl()
    attempts = tuple(
        OracleAttempt(
            iteration=i,
            artifact=_art(obl),
            check_result=None,
            cost=None,
            duration_ms=1,
        )
        for i in range(attempts_n)
    )
    return OracleOutcome(
        obligation=obl,
        kind=OracleOutcomeKind.SOLVED,
        attempts=attempts,
        final_artifact=_art(obl),
        total_cost_usd=Decimal("0"),
        diagnostic=None,
    )


def _rr_one_task(
    *,
    suite_id: str,
    label: str,
    outcome: OracleOutcome,
) -> RunResult:
    t = Task(
        id="t1",
        file=Path("x.feature"),
        expected=TaskExpected.EITHER,
        tags=(),
    )
    suite = Suite(suite_id=suite_id, description="", tasks=(t,))
    tr = TaskResult(
        task=t,
        config_label=label,
        obligations_total=1,
        obligations_solved=1 if outcome.kind is OracleOutcomeKind.SOLVED else 0,
        outcomes=(outcome,),
        total_cost_usd=Decimal("0.01"),
        total_tokens=TokenUsage(input=1, output=1),
        duration_ms=10,
        matched_expected=True,
        telemetry_run_id="r1",
    )
    cfg = EvalConfig(
        prover_name="null",
        provider_name="p",
        model=None,
        seed=0,
        oracle=__import__("kalinov.oracle.strategy", fromlist=["OracleConfig"]).OracleConfig(),
        label=label,
    )
    return RunResult(
        suite=suite,
        config=cfg,
        task_results=(tr,),
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
    )


def test_aggregate_exact_fields_and_median() -> None:
    o1 = _solved_outcome(1)
    o2 = _solved_outcome(3)
    o3 = _solved_outcome(5)
    rr = RunResult(
        suite=Suite(suite_id="s", description="", tasks=()),
        config=EvalConfig(
            prover_name="null",
            provider_name="p",
            model=None,
            seed=0,
            oracle=__import__("kalinov.oracle.strategy", fromlist=["OracleConfig"]).OracleConfig(),
            label="c1",
        ),
        task_results=(
            TaskResult(
                task=Task(id="a", file=Path("a.feature"), expected=TaskExpected.EITHER, tags=()),
                config_label="c1",
                obligations_total=1,
                obligations_solved=1,
                outcomes=(o1,),
                total_cost_usd=Decimal("0"),
                total_tokens=TokenUsage(),
                duration_ms=1,
                matched_expected=True,
                telemetry_run_id="x",
            ),
            TaskResult(
                task=Task(id="b", file=Path("b.feature"), expected=TaskExpected.EITHER, tags=()),
                config_label="c1",
                obligations_total=1,
                obligations_solved=1,
                outcomes=(o2,),
                total_cost_usd=Decimal("0"),
                total_tokens=TokenUsage(),
                duration_ms=1,
                matched_expected=True,
                telemetry_run_id="y",
            ),
            TaskResult(
                task=Task(id="c", file=Path("c.feature"), expected=TaskExpected.EITHER, tags=()),
                config_label="c1",
                obligations_total=1,
                obligations_solved=1,
                outcomes=(o3,),
                total_cost_usd=Decimal("0"),
                total_tokens=TokenUsage(),
                duration_ms=1,
                matched_expected=True,
                telemetry_run_id="z",
            ),
        ),
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
    )
    (m,) = aggregate((rr,))
    assert m.tasks_total == 3
    assert m.obligations_total == 3
    assert m.obligations_solved == 3
    assert m.median_attempts_to_solve == statistics.median([1, 3, 5])
