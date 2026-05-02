"""Tests for MCP tool wrappers."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

from kalinov.cli_core import (
    CheckObligationResult,
    CheckProgrammaticResult,
    SolveOutcomeEntry,
    SolveProgrammaticResult,
)
from kalinov.eval.cli_impl import EvalProgrammaticResult
from kalinov.eval.report import pricing_yaml_sha256
from kalinov.llm.base import BudgetExceededError
from kalinov.mcp.runtime import MCPServerConfig
from kalinov.mcp.schemas import (
    CheckRequest,
    CostReportRequest,
    EvalRequest,
    MineRequest,
    SolveRequest,
)
from kalinov.mcp.tools import (
    tool_check,
    tool_cost_report,
    tool_eval,
    tool_mine,
    tool_solve,
)


def _cfg(tmp_path: Path) -> MCPServerConfig:
    return MCPServerConfig(
        transport="stdio",
        runs_dir=tmp_path / "runs",
        kalinov_config_path=None,
    )


def test_tool_solve_happy_path(tmp_path: Path) -> None:
    feat = tmp_path / "x.feature"
    feat.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then 1=1\n",
        encoding="utf-8",
    )
    fake = SolveProgrammaticResult(
        run_id="a1b2c3d4e5f6",
        runs_dir=tmp_path / "runs",
        outcomes=(
            SolveOutcomeEntry(
                obligation_name="S#0",
                kind="solved",
                iterations=2,
                total_cost_usd="0.01",
                final_artifact="theorem h : True := trivial",
                diagnostic=None,
            ),
        ),
        total_cost_usd="0.01",
        duration_ms=10,
        parse_failed=False,
        obligations_total=1,
        obligations_solved=1,
    )

    async def run() -> None:
        with patch(
            "kalinov.mcp.tools.run_solve_programmatic",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            req = SolveRequest(feature_path=str(feat), provider="p")
            out = await tool_solve(req, _cfg(tmp_path))
        assert out.ok is True
        assert out.run_id == "a1b2c3d4e5f6"
        assert out.outcomes[0].kind == "solved"
        assert out.total_cost_usd == "0.01"
        assert isinstance(out.total_cost_usd, str)

    asyncio.run(run())


def test_tool_solve_budget_exceeded(tmp_path: Path) -> None:
    async def run() -> None:
        with patch(
            "kalinov.mcp.tools.run_solve_programmatic",
            new_callable=AsyncMock,
            side_effect=BudgetExceededError(provider="p", message="over budget"),
        ):
            p = tmp_path / "f.feature"
            p.write_text(
                "# language: en\nFeature: F\n  Scenario: S\n    Then 1=1\n",
                encoding="utf-8",
            )
            out = await tool_solve(
                SolveRequest(feature_path=str(p), provider="p"),
                _cfg(tmp_path),
            )
        assert out.ok is False
        assert out.error
        assert "budget" in out.error.lower()

    asyncio.run(run())


def test_tool_solve_truncates_long_fields(tmp_path: Path) -> None:
    long_d = "x" * 40000
    fake = SolveProgrammaticResult(
        run_id="a1b2c3d4e5f6",
        runs_dir=tmp_path / "runs",
        outcomes=(
            SolveOutcomeEntry(
                obligation_name="S#0",
                kind="llm_error",
                iterations=0,
                total_cost_usd="0",
                final_artifact=long_d,
                diagnostic=long_d,
            ),
        ),
        total_cost_usd="0",
        duration_ms=1,
        parse_failed=False,
        obligations_total=1,
        obligations_solved=0,
    )

    async def run() -> None:
        with patch(
            "kalinov.mcp.tools.run_solve_programmatic",
            new_callable=AsyncMock,
            return_value=fake,
        ):
            p = tmp_path / "f.feature"
            p.write_text(
                "# language: en\nFeature: F\n  Scenario: S\n    Then 1=1\n",
                encoding="utf-8",
            )
            out = await tool_solve(
                SolveRequest(feature_path=str(p), provider="p"),
                _cfg(tmp_path),
            )
        assert out.ok is True
        d = out.outcomes[0].diagnostic
        assert d is not None
        assert len(d) <= 32768

    asyncio.run(run())


def test_tool_check_with_null_prover(tmp_path: Path) -> None:
    feat = tmp_path / "t.feature"
    feat.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    res = CheckProgrammaticResult(
        run_id="deadbeefcafe",
        runs_dir=tmp_path / "runs",
        results=(
            CheckObligationResult("S#0", True, (), 1),
            CheckObligationResult("S#1", False, ("bad",), 2),
        ),
        parse_failed=False,
        total_ok=1,
        total_fail=1,
        total_obligations=2,
    )

    async def run() -> None:
        with patch(
            "kalinov.mcp.tools.run_check_programmatic",
            return_value=res,
        ):
            out = await tool_check(CheckRequest(feature_path=str(feat)), _cfg(tmp_path))
        assert out.ok is True
        assert out.run_id == "deadbeefcafe"
        assert out.results[0].ok is True
        assert out.results[1].ok is False

    asyncio.run(run())


def test_tool_eval_runs_smoke_suite(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("suite: {}\n", encoding="utf-8")
    er = EvalProgrammaticResult(
        run_ids=("r1", "r2"),
        report_paths={"json": str(tmp_path / "out.json"), "md": str(tmp_path / "out.md")},
        summary_markdown="## summary",
        total_cost_usd="3.00",
        hard_failure=False,
    )

    async def run() -> None:
        with patch(
            "kalinov.mcp.tools.run_eval_programmatic",
            new_callable=AsyncMock,
            return_value=er,
        ):
            out = await tool_eval(
                EvalRequest(suite_path=str(suite), providers=["p1"]),
                _cfg(tmp_path),
            )
        assert out.ok is True
        assert out.run_ids == ["r1", "r2"]
        assert "json" in out.report_paths
        assert out.summary_markdown.startswith("##")

    asyncio.run(run())


def test_tool_mine_requires_network(tmp_path: Path) -> None:
    async def run() -> None:
        out = await tool_mine(
            MineRequest(query="q", network=False),
            _cfg(tmp_path),
        )
        assert out.ok is False
        assert out.error
        assert "network" in out.error.lower()

    asyncio.run(run())


def test_tool_cost_report_grouped_by_provider(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for rid, prov in (
        ("111111111111", "anthropic"),
        ("222222222222", "openai"),
    ):
        d = runs / rid
        d.mkdir(parents=True)
        line = (
            '{"provider":"'
            + prov
            + '","model_id_resolved":"m","cost_usd":"1.0","usage":{"input":1,"output":1},'
            '"ts_ms":1000}\n'
        )
        (d / "llm_calls.jsonl").write_text(line * 3, encoding="utf-8")

    async def run() -> None:
        out = await tool_cost_report(
            CostReportRequest(group_by="provider"),
            MCPServerConfig(transport="stdio", runs_dir=runs),
        )
        assert out.ok is True
        assert out.pricing_snapshot_sha == pricing_yaml_sha256()
        assert Decimal(out.total_usd) == Decimal("6")
        keys = {row["key"] for row in out.breakdown}
        assert keys == {"anthropic", "openai"}

    asyncio.run(run())
