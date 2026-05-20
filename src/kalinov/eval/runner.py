"""Sequential eval runner: one `EvalConfig` and one `Suite` from this package.

Tasks run one after another (single-threaded). Each task uses its own
:class:`~kalinov.telemetry.context.RunContext` so telemetry stays isolated.
Parallel eval across tasks/configs is intentionally deferred.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import TokenUsage
from kalinov.eval.matrix import EvalConfig
from kalinov.eval.suite import Suite, Task, TaskExpected
from kalinov.gherkin import parse_feature_file
from kalinov.gherkin.errors import GherkinParseError
from kalinov.interpreters import (
    ForTheLInterpreter,
    InterpreterChain,
    MathTexInterpreter,
    RawInterpreter,
)
from kalinov.interpreters.base import InterpretedStep
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
from kalinov.llm.cache import LLMCache
from kalinov.llm.config import KalinovConfig
from kalinov.llm.factory import make_client
from kalinov.llm.telemetry import token_usage_from_json
from kalinov.oracle import OracleLoop
from kalinov.oracle.strategy import OracleOutcome, OracleOutcomeKind
from kalinov.provers import NullProver, NullProverConfig, NullProverMode
from kalinov.provers.base import ProofObligation, Prover, SpecDocument
from kalinov.provers.errors import ProverError
from kalinov.provers.lean import LeanProver, ToolchainNotFoundError, detect_toolchain
from kalinov.telemetry import start_run


def _interpret_chain() -> InterpreterChain:
    return InterpreterChain(
        [
            MathTexInterpreter(),
            ForTheLInterpreter(),
            RawInterpreter(),
        ],
    )


def _token_usage_from_llm_jsonl(path: Path) -> TokenUsage:
    if not path.is_file():
        return TokenUsage()
    total = TokenUsage()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        u = row.get("usage")
        if isinstance(u, dict):
            chunk = token_usage_from_json(u)
            total = TokenUsage(
                input=total.input + chunk.input,
                output=total.output + chunk.output,
                reasoning=total.reasoning + chunk.reasoning,
                cache_read=total.cache_read + chunk.cache_read,
                cache_write=total.cache_write + chunk.cache_write,
            )
    return total


def _make_prover(name: str) -> Prover:
    if name == "null":
        return NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))
    if name == "lean4":
        tc = detect_toolchain()
        return LeanProver(toolchain=tc)
    raise ValueError(f"unknown prover {name!r}")


def _outcomes_match_expected(
    expected: TaskExpected,
    outcomes: tuple[OracleOutcome, ...],
) -> bool:
    if expected is TaskExpected.EITHER:
        return True
    kinds = [o.kind for o in outcomes]
    if not kinds:
        return True
    if expected is TaskExpected.SOLVED:
        return all(k is OracleOutcomeKind.SOLVED for k in kinds)
    if expected is TaskExpected.GAVE_UP:
        n = len(kinds)
        gave = sum(1 for k in kinds if k is OracleOutcomeKind.GAVE_UP)
        return gave > n / 2
    return False


@dataclass(frozen=True, slots=True)
class TaskResult:
    task: Task
    config_label: str
    obligations_total: int
    obligations_solved: int
    outcomes: tuple[OracleOutcome, ...]
    total_cost_usd: Decimal
    total_tokens: TokenUsage
    duration_ms: int
    matched_expected: bool
    telemetry_run_id: str


@dataclass(frozen=True, slots=True)
class RunResult:
    suite: Suite
    config: EvalConfig
    task_results: tuple[TaskResult, ...]
    started_at: datetime
    ended_at: datetime


class EvalRunner:
    """Run one :class:`EvalConfig` across one :class:`Suite`."""

    def __init__(
        self,
        *,
        kalinov_config: KalinovConfig,
        pricing: PricingCatalogue,
        cache: LLMCache | None = None,
        budget: Budget | None = None,
        guard: BudgetGuard | None = None,
        runs_dir: Path = Path("runs"),
    ) -> None:
        """``guard`` overrides ``budget`` when set: callers that run multiple
        ``EvalRunner`` instances (e.g. a matrix expansion in
        :func:`kalinov.eval.cli_impl._eval_async`) must construct a single
        :class:`BudgetGuard` once and pass it to every runner so the budget
        cap applies cumulatively across configs. When only ``budget`` is
        supplied the runner builds an isolated guard, which is correct for
        single-runner callers but leaks across configs."""
        self._kalinov_config = kalinov_config
        self._pricing = pricing
        self._cache = cache
        self._budget_template = budget
        self._external_guard = guard
        self._runs_dir = Path(runs_dir).resolve()

    async def run(self, suite: Suite, config: EvalConfig) -> RunResult:
        """Execute the suite. Each task file runs under a fresh active run context."""
        started = datetime.now(tz=UTC)
        prov_entry = self._kalinov_config.providers[config.provider_name]
        model = config.model or prov_entry.default_model

        try:
            prover = _make_prover(config.prover_name)
        except ToolchainNotFoundError:
            raise
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        client = make_client(
            config.provider_name,
            config=self._kalinov_config,
            cache=self._cache,
            pricing=self._pricing,
        )

        oracle_loop = OracleLoop(
            prover=prover,
            llm=client,
            model=model,
            config=config.oracle,
            catalogue=self._pricing,
        )

        chain = _interpret_chain()
        task_results: list[TaskResult] = []

        # Resolve the BudgetGuard once for the whole run. ``--max-cost-usd``
        # (and the YAML ``budget:`` block) is a cap on cumulative spend
        # across every task and obligation. An externally-supplied guard
        # (passed by ``_eval_async``) additionally shares that cap across
        # every config in a matrix run, matching ``kalinov solve`` semantics.
        # Re-creating the guard per task — or per config — would silently let
        # total spend grow as ``cap × tasks × configs``, defeating the cap.
        shared_guard: BudgetGuard | None = self._external_guard
        if shared_guard is None and self._budget_template is not None:
            shared_guard = BudgetGuard(self._budget_template)

        for task in suite.tasks:
            t0 = time.perf_counter_ns()
            with start_run(runs_dir=self._runs_dir) as run:
                set_budget_guard(shared_guard)
                outcomes: list[OracleOutcome] = []
                obligations_solved = 0
                sum_usd = Decimal("0")
                obligations: tuple[ProofObligation, ...] = ()
                try:
                    try:
                        ff = parse_feature_file(task.file)
                    except GherkinParseError as exc:
                        raise RuntimeError(f"parse error in {task.file}: {exc}") from exc

                    interpreted: list[InterpretedStep] = []
                    for scenario in ff.feature.scenarios:
                        for step in scenario.steps:
                            interpreted.append(chain.interpret(step))

                    spec = SpecDocument(
                        feature_file=ff,
                        interpreted_steps=tuple(interpreted),
                    )
                    try:
                        obligations = prover.extract_obligations(spec)
                    except ProverError as exc:
                        msg = f"extract_obligations failed for {task.id}: {exc}"
                        raise RuntimeError(msg) from exc

                    for obl in obligations:
                        out = await oracle_loop.run(obl)
                        outcomes.append(out)
                        sum_usd += out.total_cost_usd
                        if out.kind is OracleOutcomeKind.SOLVED:
                            obligations_solved += 1

                    manifest = {
                        "run_id": run.run_id,
                        "total_cost_usd": str(sum_usd),
                        "obligations_total": len(obligations),
                        "obligations_solved": obligations_solved,
                        "eval_task_id": task.id,
                        "eval_config_label": config.label,
                    }
                    (run.run_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                finally:
                    set_budget_guard(None)

            elapsed_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
            tokens = _token_usage_from_llm_jsonl(run.run_dir / "llm_calls.jsonl")
            task_results.append(
                TaskResult(
                    task=task,
                    config_label=config.label,
                    obligations_total=len(outcomes),
                    obligations_solved=obligations_solved,
                    outcomes=tuple(outcomes),
                    total_cost_usd=sum_usd,
                    total_tokens=tokens,
                    duration_ms=elapsed_ms,
                    matched_expected=_outcomes_match_expected(task.expected, tuple(outcomes)),
                    telemetry_run_id=run.run_id,
                ),
            )

        ended = datetime.now(tz=UTC)
        return RunResult(
            suite=suite,
            config=config,
            task_results=tuple(task_results),
            started_at=started,
            ended_at=ended,
        )


__all__ = [
    "EvalRunner",
    "RunResult",
    "TaskResult",
]
