"""Budget guard tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from kalinov.cost.models import CostBreakdown, TokenUsage
from kalinov.llm.base import BudgetExceededError
from kalinov.llm.budget import Budget, BudgetGuard


def _cost(usd: str, pricing_source: str = "catalogue") -> CostBreakdown:
    z = Decimal("0")
    d = Decimal(usd)
    return CostBreakdown(
        total_usd=d,
        input_usd=d,
        output_usd=z,
        reasoning_usd=z,
        cache_read_usd=z,
        cache_write_usd=z,
        pricing_source=pricing_source,
    )


def test_max_calls_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    g = BudgetGuard(Budget(max_calls=1))
    g.record(cost=_cost("0"), usage=TokenUsage(input=1, output=1), provider="anthropic")
    with pytest.raises(BudgetExceededError):
        g.record(cost=_cost("0"), usage=TokenUsage(input=1, output=1), provider="anthropic")


def test_max_cost_threshold() -> None:
    g = BudgetGuard(Budget(max_cost_usd=Decimal("1.00")))
    g.record(cost=_cost("0.50"), usage=TokenUsage(input=1), provider="openai")
    with pytest.raises(BudgetExceededError):
        g.record(cost=_cost("0.60"), usage=TokenUsage(input=1), provider="openai")


def test_max_tokens_total() -> None:
    g = BudgetGuard(Budget(max_total_tokens=5))
    g.record(cost=_cost("0"), usage=TokenUsage(input=3, output=2), provider="gemini")
    with pytest.raises(BudgetExceededError):
        g.record(cost=_cost("0"), usage=TokenUsage(input=1), provider="gemini")


def test_state_snapshot() -> None:
    g = BudgetGuard(Budget(max_cost_usd=Decimal("10")))
    g.record(cost=_cost("1.5"), usage=TokenUsage(input=2, output=3), provider="p")
    s = g.state
    assert s.calls == 1
    assert s.spent_usd == Decimal("1.5")
    assert s.total_tokens == 5


def test_unknown_pricing_with_cost_cap_refuses() -> None:
    """A max_cost_usd cap must not be silently bypassed by unknown-priced calls.

    Without this check, a single call whose model isn't in pricing.yaml records
    $0 against the guard and the run keeps spending real money past the cap.
    """
    g = BudgetGuard(Budget(max_cost_usd=Decimal("1.00")))
    with pytest.raises(BudgetExceededError, match="no pricing entry"):
        g.record(
            cost=_cost("0", pricing_source="unknown"),
            usage=TokenUsage(input=1000, output=1000),
            provider="openai",
        )
    # The refused call must NOT be counted (no state corruption).
    s = g.state
    assert s.calls == 0
    assert s.spent_usd == Decimal("0")
    assert s.total_tokens == 0


def test_unknown_pricing_without_cost_cap_is_allowed() -> None:
    """When no max_cost_usd is configured, unknown pricing is fine."""
    g = BudgetGuard(Budget(max_total_tokens=100, max_calls=10))
    g.record(
        cost=_cost("0", pricing_source="unknown"),
        usage=TokenUsage(input=2, output=3),
        provider="openai",
    )
    s = g.state
    assert s.calls == 1
    assert s.total_tokens == 5


def test_cost_cap_overrun_carries_attempted_cost_on_error() -> None:
    """When ``record`` raises because the latest billable call pushed the
    cumulative spend over ``max_cost_usd``, the resulting
    :class:`BudgetExceededError` must carry that call's ``total_usd`` on
    ``attempted_cost_usd`` so callers can include it in per-task /
    per-obligation cost aggregates. Without this, the summary surface
    that the user reads (eval report, solve summary) drops the exact
    overrun call's cost and looks like the run stayed within budget.
    """
    g = BudgetGuard(Budget(max_cost_usd=Decimal("1.00")))
    g.record(cost=_cost("0.50"), usage=TokenUsage(input=1), provider="openai")
    with pytest.raises(BudgetExceededError) as ei:
        g.record(cost=_cost("0.60"), usage=TokenUsage(input=1), provider="openai")
    assert ei.value.attempted_cost_usd == Decimal("0.60")


def test_token_cap_overrun_carries_attempted_cost_on_error() -> None:
    g = BudgetGuard(Budget(max_total_tokens=5))
    g.record(cost=_cost("0.10"), usage=TokenUsage(input=3, output=2), provider="gemini")
    with pytest.raises(BudgetExceededError) as ei:
        g.record(cost=_cost("0.25"), usage=TokenUsage(input=1), provider="gemini")
    assert ei.value.attempted_cost_usd == Decimal("0.25")


def test_call_cap_overrun_carries_attempted_cost_on_error() -> None:
    g = BudgetGuard(Budget(max_calls=1))
    g.record(cost=_cost("0.01"), usage=TokenUsage(input=1, output=1), provider="anthropic")
    with pytest.raises(BudgetExceededError) as ei:
        g.record(cost=_cost("0.33"), usage=TokenUsage(input=1, output=1), provider="anthropic")
    assert ei.value.attempted_cost_usd == Decimal("0.33")


def test_unknown_pricing_refusal_has_no_attempted_cost() -> None:
    """The "unknown pricing" refusal happens before any state is mutated;
    no real cost can be attributed to the refused call, so
    ``attempted_cost_usd`` must stay ``None`` (oracle loop will skip adding)."""
    g = BudgetGuard(Budget(max_cost_usd=Decimal("1.00")))
    with pytest.raises(BudgetExceededError) as ei:
        g.record(
            cost=_cost("0", pricing_source="unknown"),
            usage=TokenUsage(input=1, output=1),
            provider="openai",
        )
    assert ei.value.attempted_cost_usd is None
