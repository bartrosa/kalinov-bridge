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


def test_ensure_not_exceeded_passes_when_under_caps() -> None:
    g = BudgetGuard(Budget(max_cost_usd=Decimal("1.00"), max_total_tokens=100, max_calls=5))
    g.ensure_not_exceeded(provider="openai")  # should not raise on a fresh guard
    g.record(cost=_cost("0.10"), usage=TokenUsage(input=1, output=1), provider="openai")
    g.ensure_not_exceeded(provider="openai")  # still well under the cap


def test_ensure_not_exceeded_blocks_after_cost_cap_tripped() -> None:
    g = BudgetGuard(Budget(max_cost_usd=Decimal("1.00")))
    g.record(cost=_cost("0.60"), usage=TokenUsage(input=1), provider="openai")
    with pytest.raises(BudgetExceededError):
        g.record(cost=_cost("0.60"), usage=TokenUsage(input=1), provider="openai")
    with pytest.raises(BudgetExceededError, match="already exceeded"):
        g.ensure_not_exceeded(provider="openai")


def test_ensure_not_exceeded_blocks_after_max_calls_reached() -> None:
    g = BudgetGuard(Budget(max_calls=2))
    g.record(cost=_cost("0"), usage=TokenUsage(input=1), provider="openai")
    g.record(cost=_cost("0"), usage=TokenUsage(input=1), provider="openai")
    # ``record`` only raises on the call that would exceed; ``ensure_not_exceeded``
    # is the pre-check, so it must refuse the NEXT call too.
    with pytest.raises(BudgetExceededError, match="exhausted"):
        g.ensure_not_exceeded(provider="openai")


def test_ensure_not_exceeded_blocks_after_token_cap_tripped() -> None:
    g = BudgetGuard(Budget(max_total_tokens=5))
    with pytest.raises(BudgetExceededError):
        g.record(cost=_cost("0"), usage=TokenUsage(input=10, output=0), provider="openai")
    with pytest.raises(BudgetExceededError, match="already exceeded"):
        g.ensure_not_exceeded(provider="openai")
