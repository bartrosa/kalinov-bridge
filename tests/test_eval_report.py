"""Report JSON / Markdown tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalinov.cost.models import TokenUsage
from kalinov.eval.matrix import EvalConfig
from kalinov.eval.report import build_report_payload, render_json, render_markdown
from kalinov.eval.runner import RunResult, TaskResult
from kalinov.eval.suite import Suite, Task, TaskExpected
from kalinov.oracle.strategy import OracleAttempt, OracleOutcome, OracleOutcomeKind
from kalinov.provers.base import ProofArtifact, ProofObligation


def _minimal_run_result() -> RunResult:
    obl = ProofObligation(name="n", statement="s", hypotheses=())
    art = ProofArtifact(obligation=obl, body="x", language="null", metadata={})
    oc = OracleOutcome(
        obligation=obl,
        kind=OracleOutcomeKind.SOLVED,
        attempts=(
            OracleAttempt(
                iteration=0,
                artifact=art,
                check_result=None,
                cost=None,
                duration_ms=1,
            ),
        ),
        final_artifact=art,
        total_cost_usd=Decimal("0.1"),
        diagnostic=None,
    )
    tr = TaskResult(
        task=Task(
            id="task_a",
            file=Path("/tmp/a.feature"),
            expected=TaskExpected.EITHER,
            tags=("t",),
        ),
        config_label="cfg",
        obligations_total=1,
        obligations_solved=1,
        outcomes=(oc,),
        total_cost_usd=Decimal("0.1"),
        total_tokens=TokenUsage(input=2, output=3),
        duration_ms=5,
        matched_expected=True,
        telemetry_run_id="runxyz",
    )
    suite = Suite(suite_id="su", description="d", tasks=(tr.task,))
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model="gpt-4o",
        seed=1,
        oracle=__import__("kalinov.oracle.strategy", fromlist=["OracleConfig"]).OracleConfig(),
        label="cfg",
    )
    return RunResult(
        suite=suite,
        config=cfg,
        task_results=(tr,),
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC),
    )


def test_json_roundtrip_equivalence() -> None:
    rr = _minimal_run_result()
    s1 = render_json((rr,))
    d1 = json.loads(s1)
    d2 = json.loads(render_json((rr,)))
    assert d1 == d2
    assert "aggregate_metrics" in d1
    assert "pricing_yaml_sha256" in d1


def test_build_report_payload_structure() -> None:
    p = build_report_payload((_minimal_run_result(),))
    json.dumps(p)  # serializable
    assert p["report_version"] == 1


def test_markdown_contains_configs_tasks_no_bad_floats() -> None:
    md = render_markdown((_minimal_run_result(),))
    assert "cfg" in md
    assert "task_a" in md
    assert "nan" not in md.lower()
    assert "inf" not in md.lower()
    # median may be nan in aggregates if no solved attempts — here we have one solved
    for line in md.splitlines():
        if "median" in line.lower():
            assert "nan" not in line.lower()
