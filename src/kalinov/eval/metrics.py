"""Pure aggregation of eval :class:`~kalinov.eval.runner.RunResult` rows."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from kalinov.cost.models import TokenUsage
from kalinov.eval.runner import RunResult, TaskResult
from kalinov.oracle.strategy import OracleOutcomeKind


def _add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input=a.input + b.input,
        output=a.output + b.output,
        reasoning=a.reasoning + b.reasoning,
        cache_read=a.cache_read + b.cache_read,
        cache_write=a.cache_write + b.cache_write,
    )


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    config_label: str
    suite_id: str
    tasks_total: int
    tasks_solved: int
    tasks_gave_up: int
    tasks_errored: int
    obligations_total: int
    obligations_solved: int
    success_rate: float
    total_cost_usd: Decimal
    total_tokens: TokenUsage
    median_attempts_to_solve: float
    error_breakdown: Mapping[str, int]


def _categorize_task(tr: TaskResult) -> str:
    err_kinds = {
        OracleOutcomeKind.LLM_ERROR,
        OracleOutcomeKind.PROVER_ERROR,
        OracleOutcomeKind.BUDGET_EXCEEDED,
    }
    if any(o.kind in err_kinds for o in tr.outcomes):
        return "errored"
    if not tr.outcomes and tr.obligations_total == 0:
        return "solved"
    if tr.outcomes and all(o.kind is OracleOutcomeKind.SOLVED for o in tr.outcomes):
        return "solved"
    return "gave_up"


def aggregate(results: Sequence[RunResult]) -> tuple[AggregateMetrics, ...]:
    """Compute one :class:`AggregateMetrics` per run result (one row per config × suite run)."""
    out: list[AggregateMetrics] = []
    for rr in results:
        breakdown: Counter[str] = Counter()
        obligations_total = 0
        obligations_solved = 0
        sum_cost = Decimal("0")
        sum_tokens = TokenUsage()
        solved_attempt_counts: list[int] = []

        tasks_solved = 0
        tasks_gave_up = 0
        tasks_errored = 0

        for tr in rr.task_results:
            obligations_total += tr.obligations_total
            obligations_solved += tr.obligations_solved
            sum_cost += tr.total_cost_usd
            sum_tokens = _add_usage(sum_tokens, tr.total_tokens)

            for o in tr.outcomes:
                breakdown[o.kind.value] += 1
                if o.kind is OracleOutcomeKind.SOLVED:
                    solved_attempt_counts.append(len(o.attempts))

            cat = _categorize_task(tr)
            if cat == "errored":
                tasks_errored += 1
            elif cat == "solved":
                tasks_solved += 1
            else:
                tasks_gave_up += 1

        success_rate = (
            obligations_solved / obligations_total if obligations_total > 0 else float("nan")
        )

        median_attempts = (
            float(statistics.median(solved_attempt_counts))
            if solved_attempt_counts
            else float("nan")
        )

        out.append(
            AggregateMetrics(
                config_label=rr.config.label,
                suite_id=rr.suite.suite_id,
                tasks_total=len(rr.task_results),
                tasks_solved=tasks_solved,
                tasks_gave_up=tasks_gave_up,
                tasks_errored=tasks_errored,
                obligations_total=obligations_total,
                obligations_solved=obligations_solved,
                success_rate=success_rate,
                total_cost_usd=sum_cost,
                total_tokens=sum_tokens,
                median_attempts_to_solve=median_attempts,
                error_breakdown=dict(breakdown),
            ),
        )
    return tuple(out)


def is_finite_number(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


__all__ = ["AggregateMetrics", "aggregate", "is_finite_number"]
