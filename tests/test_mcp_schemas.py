"""MCP Pydantic schema tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kalinov.mcp.schemas import (
    CheckRequest,
    EvalResponse,
    SolveOutcomeSummary,
    SolveRequest,
)


def test_solve_request_decimal_strings_only() -> None:
    SolveRequest(feature_path="a.feature", provider="p", max_cost_usd="1.00")
    with pytest.raises(ValidationError):
        SolveRequest(feature_path="a.feature", provider="p", max_cost_usd=1.0)  # type: ignore[arg-type]


def test_check_request_defaults() -> None:
    c = CheckRequest(feature_path="x.feature")
    assert c.prover == "null"
    assert c.no_forthel is False
    assert c.null_mode == "always_ok"
    assert c.null_fail_after == 0


def test_outcome_kind_literal_validates() -> None:
    SolveOutcomeSummary(
        obligation_name="n",
        kind="solved",
        iterations=1,
        total_cost_usd="0",
        final_artifact=None,
        diagnostic=None,
    )
    with pytest.raises(ValidationError):
        SolveOutcomeSummary(
            obligation_name="n",
            kind="not_a_kind",  # type: ignore[arg-type]
            iterations=1,
            total_cost_usd="0",
            final_artifact=None,
            diagnostic=None,
        )


def test_eval_response_round_trips_via_json() -> None:
    r = EvalResponse(
        run_ids=["a", "b"],
        report_paths={"json": "/out/report.json"},
        summary_markdown="# ok",
        total_cost_usd="1.50",
    )
    dumped = r.model_dump_json()
    assert EvalResponse.model_validate(json.loads(dumped)) == r
