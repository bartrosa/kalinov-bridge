"""Implementation of ``kalinov eval`` (invoked from :mod:`kalinov.cli`)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.eval.experiment import ExperimentError, load_experiment
from kalinov.eval.matrix import ConfigMatrix
from kalinov.eval.report import render_markdown, write_report
from kalinov.eval.runner import EvalRunner, RunResult
from kalinov.eval.suite import SuiteError, load_suite
from kalinov.llm.budget import Budget
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.config import ConfigError
from kalinov.llm.config import load_config as load_llm_config
from kalinov.oracle.strategy import OracleConfig, OracleOutcomeKind
from kalinov.provers.lean import ToolchainNotFoundError


def _build_matrix_from_args(args: argparse.Namespace) -> ConfigMatrix:
    if not args.provider:
        raise ValueError("at least one --provider is required without --config-file")
    providers = tuple((p, args.model) for p in args.provider)
    seeds = tuple(args.seed) if args.seed else (42,)
    oc = OracleConfig(
        max_repair_attempts=args.max_repair_attempts if args.max_repair_attempts is not None else 3,
        max_tokens_per_call=args.max_tokens if args.max_tokens is not None else 2048,
        temperature=args.temperature if args.temperature is not None else 0.0,
    )
    return ConfigMatrix(
        provers=(args.prover,),
        providers=providers,
        seeds=seeds,
        oracle_configs=(oc,),
    )


def _apply_oracle_cli(matrix: ConfigMatrix, args: argparse.Namespace) -> ConfigMatrix:
    if args.max_repair_attempts is None and args.max_tokens is None and args.temperature is None:
        return matrix
    new_configs: list[OracleConfig] = []
    for oc in matrix.oracle_configs:
        new_configs.append(
            OracleConfig(
                strategy=oc.strategy,
                max_repair_attempts=(
                    args.max_repair_attempts
                    if args.max_repair_attempts is not None
                    else oc.max_repair_attempts
                ),
                max_tokens_per_call=(
                    args.max_tokens if args.max_tokens is not None else oc.max_tokens_per_call
                ),
                temperature=args.temperature if args.temperature is not None else oc.temperature,
                extras=oc.extras,
                save_transcripts=oc.save_transcripts,
            ),
        )
    return ConfigMatrix(
        provers=matrix.provers,
        providers=matrix.providers,
        seeds=matrix.seeds,
        oracle_configs=tuple(new_configs),
    )


def _merge_experiment_cli(
    expr_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, ConfigMatrix, Budget | None, Path]:
    spec = load_experiment(expr_path)
    matrix = spec.matrix
    if args.prover:
        matrix = ConfigMatrix(
            provers=(args.prover,),
            providers=matrix.providers,
            seeds=matrix.seeds,
            oracle_configs=matrix.oracle_configs,
        )
    if args.provider:
        matrix = ConfigMatrix(
            provers=matrix.provers,
            providers=tuple((p, args.model) for p in args.provider),
            seeds=matrix.seeds,
            oracle_configs=matrix.oracle_configs,
        )
    if args.seed:
        matrix = ConfigMatrix(
            provers=matrix.provers,
            providers=matrix.providers,
            seeds=tuple(args.seed),
            oracle_configs=matrix.oracle_configs,
        )
    matrix = _apply_oracle_cli(matrix, args)

    budget = spec.budget
    if args.max_cost_usd is not None:
        budget = Budget(max_cost_usd=Decimal(str(args.max_cost_usd)))

    out_dir = Path(args.out).resolve() if args.out else spec.out_dir
    return spec.suite_path, matrix, budget, out_dir


async def _eval_async(
    suite_path: Path,
    matrix: ConfigMatrix,
    *,
    llm_config_path: Path | None,
    cache: LLMCache | None,
    budget: Budget | None,
    runs_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
    silent: bool = False,
) -> tuple[bool, list[RunResult], dict[str, Path], str]:
    """Return hard_failure, results, report paths, markdown summary."""
    try:
        llm_cfg = load_llm_config(llm_config_path)
    except ConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    suite = load_suite(suite_path)
    pricing = load_default_catalogue()
    configs = matrix.expand()

    results: list[RunResult] = []
    hard_failure = False

    for cfg in configs:
        if cfg.provider_name not in llm_cfg.providers:
            raise RuntimeError(f"unknown provider {cfg.provider_name!r}")

        runner = EvalRunner(
            kalinov_config=llm_cfg,
            pricing=pricing,
            cache=cache,
            budget=budget,
            runs_dir=runs_dir,
        )
        try:
            rr = await runner.run(suite, cfg)
        except ToolchainNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        results.append(rr)
        for tr in rr.task_results:
            for o in tr.outcomes:
                if o.kind in (OracleOutcomeKind.LLM_ERROR, OracleOutcomeKind.BUDGET_EXCEEDED):
                    hard_failure = True

    written = dict(write_report(results, out_dir=out_dir, formats=formats))
    md = render_markdown(results)
    if not silent:
        print(md, end="")
    return hard_failure, results, written, md


@dataclass(frozen=True, slots=True)
class EvalProgrammaticResult:
    run_ids: tuple[str, ...]
    report_paths: dict[str, str]
    summary_markdown: str
    total_cost_usd: str
    hard_failure: bool


async def run_eval_programmatic(
    suite_path: Path,
    matrix: ConfigMatrix,
    *,
    llm_config_path: Path | None,
    cache: LLMCache | None,
    budget: Budget | None,
    runs_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
) -> EvalProgrammaticResult:
    """Like ``run_eval`` but returns structured data (no stdout)."""
    hf, results, written, md = await _eval_async(
        suite_path,
        matrix,
        llm_config_path=llm_config_path,
        cache=cache,
        budget=budget,
        runs_dir=runs_dir,
        out_dir=out_dir,
        formats=formats,
        silent=True,
    )
    seen: set[str] = set()
    run_ids_list: list[str] = []
    for rr in results:
        for tr in rr.task_results:
            rid = tr.telemetry_run_id
            if rid not in seen:
                seen.add(rid)
                run_ids_list.append(rid)
    total = sum((tr.total_cost_usd for rr in results for tr in rr.task_results), Decimal("0"))
    return EvalProgrammaticResult(
        run_ids=tuple(run_ids_list),
        report_paths={k: str(v) for k, v in written.items()},
        summary_markdown=md,
        total_cost_usd=str(total),
        hard_failure=hf,
    )


def run_eval(args: argparse.Namespace) -> int:
    """Synchronous entry for argparse."""
    cache: LLMCache | None = None
    if args.cache_mode != "off" and args.cache_dir is None:
        print("error: --cache-dir is required when --cache-mode is not off", file=sys.stderr)
        return 2
    if args.cache_mode != "off" and args.cache_dir is not None:
        cache = LLMCache(Path(args.cache_dir), mode=CacheMode(args.cache_mode))

    llm_path = getattr(args, "llm_config", None)
    runs_dir = Path(args.runs_dir).resolve()

    try:
        if args.config_file:
            suite_path, matrix, budget, out_dir = _merge_experiment_cli(
                Path(args.config_file),
                args,
            )
        else:
            if not args.suite:
                print("error: --suite is required without --config-file", file=sys.stderr)
                return 2
            if not args.out:
                print("error: --out is required without --config-file", file=sys.stderr)
                return 2
            suite_path = Path(args.suite).resolve()
            matrix = _build_matrix_from_args(args)
            budget = (
                Budget(max_cost_usd=Decimal(str(args.max_cost_usd)))
                if args.max_cost_usd is not None
                else None
            )
            out_dir = Path(args.out).resolve()

        fmt_parts = [x.strip() for x in args.format.split(",") if x.strip()]
        formats = tuple(f for f in fmt_parts if f in ("json", "md"))
        if not formats:
            formats = ("json", "md")

        hard_fail, _, _, _ = asyncio.run(
            _eval_async(
                suite_path,
                matrix,
                llm_config_path=llm_path,
                cache=cache,
                budget=budget,
                runs_dir=runs_dir,
                out_dir=out_dir,
                formats=formats,
                silent=False,
            ),
        )
        return 1 if hard_fail else 0
    except (SuiteError, ExperimentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["EvalProgrammaticResult", "run_eval", "run_eval_programmatic"]
