"""Telemetry JSONL tests."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from kalinov.cost.models import CostBreakdown, TokenUsage
from kalinov.llm.telemetry import extras_summary_from, log_llm_call
from kalinov.telemetry import start_run


def test_logs_success_line(tmp_path: Path) -> None:
    with start_run(runs_dir=tmp_path) as run:
        log_llm_call(
            provider="anthropic",
            model_id_resolved="claude-x",
            usage=TokenUsage(input=3, output=4),
            cost=CostBreakdown(
                total_usd=Decimal("0.01"),
                input_usd=Decimal("0.005"),
                output_usd=Decimal("0.005"),
                reasoning_usd=Decimal("0"),
                cache_read_usd=Decimal("0"),
                cache_write_usd=Decimal("0"),
                pricing_source="catalogue",
            ),
            latency_ms=12,
            cache_hit=False,
            error_code=None,
            extras_summary={"extended_thinking_budget_tokens": 1024},
        )
        log_path = run.run_dir / "llm_calls.jsonl"
        assert log_path.is_file()
        row = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
        assert row["provider"] == "anthropic"
        assert row["cache_hit"] is False
        assert row["error_code"] is None
        assert row["extras_summary"]["extended_thinking_budget_tokens"] == 1024


def test_extras_summary_filters() -> None:
    s = extras_summary_from(
        {"extended_thinking_budget_tokens": 1, "noise": 2, "reasoning_effort": "high"},
    )
    assert "noise" not in s
    assert s["reasoning_effort"] == "high"


def test_error_line_written(tmp_path: Path) -> None:
    with start_run(runs_dir=tmp_path) as run:
        log_llm_call(
            provider="openai",
            model_id_resolved="gpt-4o",
            usage=TokenUsage(),
            cost=CostBreakdown(
                total_usd=Decimal("0"),
                input_usd=Decimal("0"),
                output_usd=Decimal("0"),
                reasoning_usd=Decimal("0"),
                cache_read_usd=Decimal("0"),
                cache_write_usd=Decimal("0"),
                pricing_source="catalogue",
            ),
            latency_ms=5,
            cache_hit=False,
            error_code="rate_limit",
            extras_summary={},
        )
        row = json.loads(
            (run.run_dir / "llm_calls.jsonl").read_text(encoding="utf-8").strip(),
        )
        assert row["error_code"] == "rate_limit"
