"""Thin MCP tool wrappers over :mod:`kalinov.cli_core` and eval/mining APIs."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from kalinov.cli_core import (
    ClientConfigError,
    check_exit_code,
    run_check_programmatic,
    run_solve_programmatic,
)
from kalinov.eval.cli_impl import EvalProgrammaticResult, run_eval_programmatic
from kalinov.eval.matrix import ConfigMatrix
from kalinov.eval.report import pricing_yaml_sha256
from kalinov.eval.suite import SuiteError
from kalinov.gherkin.errors import GherkinParseError
from kalinov.llm.base import BudgetExceededError, LLMError
from kalinov.llm.budget import Budget
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.config import ConfigError
from kalinov.llm.run_report import GroupBy, aggregate_run_directories, discover_run_dirs
from kalinov.mcp.runtime import MCPServerConfig
from kalinov.mcp.schemas import (
    CheckRequest,
    CheckResponse,
    CheckResultEntry,
    CostReportRequest,
    CostReportResponse,
    EvalRequest,
    EvalResponse,
    MineRequest,
    MineResponse,
    SolveOutcomeSummary,
    SolveRequest,
    SolveResponse,
)
from kalinov.mining import MiningConfig, MiningError, mine
from kalinov.provers import NullProver, NullProverConfig, NullProverMode
from kalinov.provers.base import Prover
from kalinov.provers.errors import ProverError
from kalinov.provers.lean import LeanProver, ToolchainNotFoundError, detect_toolchain
from kalinov.telemetry import start_run

_MAX_TEXT = 32768


def _trunc(text: str | None, limit: int = _MAX_TEXT) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _matrix_from_eval_req(req: EvalRequest) -> ConfigMatrix:
    from kalinov.oracle.strategy import OracleConfig

    providers = tuple((p, None) for p in req.providers)
    seeds = tuple(req.seeds) if req.seeds else (42,)
    oc = OracleConfig(max_repair_attempts=req.max_repair_attempts)
    return ConfigMatrix(
        provers=(req.prover,),
        providers=providers,
        seeds=seeds,
        oracle_configs=(oc,),
    )


async def tool_solve(req: SolveRequest, cfg: MCPServerConfig) -> SolveResponse:
    path = Path(req.feature_path)
    if not path.is_file():
        return SolveResponse(ok=False, error=f"file not found: {req.feature_path}")

    cache: LLMCache | None = None
    if cfg.cache_dir is not None:
        cache = LLMCache(Path(cfg.cache_dir), mode=CacheMode(cfg.cache_mode))

    max_cost = req.max_cost_usd if req.max_cost_usd is not None else cfg.default_max_cost_usd

    try:
        res = await run_solve_programmatic(
            paths=[path.resolve()],
            runs_dir=cfg.runs_dir.resolve(),
            prover_name=req.prover,
            provider=req.provider,
            model=req.model,
            llm_config_path=cfg.kalinov_config_path,
            cache=cache,
            max_repair_attempts=req.max_repair_attempts,
            max_tokens=2048,
            temperature=req.temperature,
            save_transcripts=req.save_transcripts,
            max_cost_usd=max_cost,
            echo=False,
        )
    except ConfigError as exc:
        return SolveResponse(ok=False, error=f"config: {exc}")
    except ClientConfigError as exc:
        return SolveResponse(ok=False, error=f"client config: {exc}")
    except ToolchainNotFoundError as exc:
        return SolveResponse(ok=False, error=str(exc))
    except GherkinParseError as exc:
        return SolveResponse(ok=False, error=f"parse: {exc}")
    except (LLMError, BudgetExceededError) as exc:
        return SolveResponse(ok=False, error=str(exc))
    except ProverError as exc:
        return SolveResponse(ok=False, error=f"prover: {exc}")

    outcomes: list[SolveOutcomeSummary] = []
    for o in res.outcomes:
        kind_s = o.kind
        outcomes.append(
            SolveOutcomeSummary(
                obligation_name=o.obligation_name,
                kind=kind_s,  # type: ignore[arg-type]
                iterations=o.iterations,
                total_cost_usd=o.total_cost_usd,
                final_artifact=_trunc(o.final_artifact),
                diagnostic=_trunc(o.diagnostic),
            ),
        )

    return SolveResponse(
        ok=True,
        run_id=res.run_id,
        runs_dir=str(res.runs_dir),
        outcomes=outcomes,
        total_cost_usd=res.total_cost_usd,
        duration_ms=res.duration_ms,
    )


async def tool_check(req: CheckRequest, cfg: MCPServerConfig) -> CheckResponse:
    path = Path(req.feature_path)
    if not path.is_file():
        return CheckResponse(ok=False, error=f"file not found: {req.feature_path}")

    mode_map = {
        "always_ok": NullProverMode.ALWAYS_OK,
        "always_fail": NullProverMode.ALWAYS_FAIL,
        "fail_after_n": NullProverMode.FAIL_AFTER_N,
    }
    prover: Prover
    if req.prover == "null":
        prover = NullProver(
            NullProverConfig(mode=mode_map[req.null_mode], fail_after=req.null_fail_after),
        )
        try:
            r = run_check_programmatic(
                prover,
                [path.resolve()],
                "null",
                cfg.runs_dir,
                forthel_bridge=False,
                echo=False,
            )
        except GherkinParseError as exc:
            return CheckResponse(ok=False, error=f"parse: {exc}")
    else:
        try:
            tc = detect_toolchain()
        except ToolchainNotFoundError as exc:
            return CheckResponse(ok=False, error=str(exc))
        prover = LeanProver(toolchain=tc)
        try:
            r = run_check_programmatic(
                prover,
                [path.resolve()],
                "lean4",
                cfg.runs_dir,
                forthel_bridge=not req.no_forthel,
                echo=False,
            )
        except GherkinParseError as exc:
            return CheckResponse(ok=False, error=f"parse: {exc}")

    rows = [
        CheckResultEntry(
            obligation_name=x.obligation_name,
            ok=x.ok,
            diagnostics=[_trunc(d, 500) or "" for d in x.diagnostics],
            duration_ms=x.duration_ms,
        )
        for x in r.results
    ]
    _ = check_exit_code(r)
    return CheckResponse(ok=True, run_id=r.run_id, results=rows)


async def tool_eval(req: EvalRequest, cfg: MCPServerConfig) -> EvalResponse:
    if not req.providers:
        return EvalResponse(ok=False, error="providers list must be non-empty")
    suite = Path(req.suite_path)
    if not suite.is_file():
        return EvalResponse(ok=False, error=f"suite not found: {req.suite_path}")

    out_dir = Path(req.out_dir or cfg.runs_dir / "eval_reports").resolve()
    budget = (
        Budget(max_cost_usd=Decimal(str(req.max_cost_usd)))
        if req.max_cost_usd is not None
        else None
    )
    cache: LLMCache | None = None
    if cfg.cache_dir is not None:
        cache = LLMCache(Path(cfg.cache_dir), mode=CacheMode(cfg.cache_mode))

    matrix = _matrix_from_eval_req(req)

    try:
        er: EvalProgrammaticResult = await run_eval_programmatic(
            suite.resolve(),
            matrix,
            llm_config_path=cfg.kalinov_config_path,
            cache=cache,
            budget=budget,
            runs_dir=cfg.runs_dir.resolve(),
            out_dir=out_dir,
            formats=("json", "md"),
        )
    except (SuiteError, RuntimeError, ConfigError) as exc:
        return EvalResponse(ok=False, error=str(exc))

    return EvalResponse(
        ok=True,
        run_ids=list(er.run_ids),
        report_paths=er.report_paths,
        summary_markdown=_trunc(er.summary_markdown) or "",
        total_cost_usd=er.total_cost_usd,
    )


async def tool_mine(req: MineRequest, cfg: MCPServerConfig) -> MineResponse:
    if not req.network:
        return MineResponse(
            ok=False,
            error="Mining requires network=true (same policy as CLI --network).",
        )
    mcfg = MiningConfig(
        source_name=req.source,
        query=req.query,
        limit=req.limit,
        extractor_name="heuristic",
        out_dir=Path(req.out_dir),
        feature_name="Mined claims",
        network_enabled=True,
    )

    async def inner() -> tuple[str, tuple[Path, ...]]:
        with start_run(runs_dir=cfg.runs_dir) as run:
            paths = await mine(mcfg)
            return run.run_id, paths

    try:
        run_id, paths = await inner()
    except MiningError as exc:
        return MineResponse(ok=False, error=str(exc))

    total = 0
    ep: list[str] = []
    for p in paths:
        ep.append(str(p))
        text = Path(p).read_text(encoding="utf-8")
        total += sum(1 for line in text.splitlines() if line.lstrip().startswith("Scenario:"))

    return MineResponse(ok=True, run_id=run_id, emitted_paths=ep, candidate_total=total)


async def tool_cost_report(req: CostReportRequest, cfg: MCPServerConfig) -> CostReportResponse:
    runs_root = Path(req.runs_dir or cfg.runs_dir).resolve()
    dirs = discover_run_dirs(runs_dir=runs_root, run_id=req.run_id)
    if not dirs:
        return CostReportResponse(
            ok=False,
            error="no matching runs",
            pricing_snapshot_sha=pricing_yaml_sha256(),
        )

    gb: GroupBy = req.group_by
    payload = aggregate_run_directories(dirs, group_by=gb)
    totals = payload["totals"]
    groups = payload.get("groups") or []
    breakdown = []
    for g in groups:
        breakdown.append(dict(g))
    tok = {
        "input": int(totals.get("input_tokens", 0)),
        "output": int(totals.get("output_tokens", 0)),
        "reasoning": int(totals.get("reasoning_tokens", 0)),
        "cache_read": int(totals.get("cache_read_tokens", 0)),
        "cache_write": int(totals.get("cache_write_tokens", 0)),
    }
    return CostReportResponse(
        ok=True,
        total_usd=str(totals.get("total_usd", "0")),
        total_tokens=tok,
        breakdown=breakdown,
        pricing_snapshot_sha=pricing_yaml_sha256(),
    )


__all__ = [
    "tool_check",
    "tool_cost_report",
    "tool_eval",
    "tool_mine",
    "tool_solve",
]
