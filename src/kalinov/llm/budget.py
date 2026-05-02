"""Per-run LLM budget tracking (USD, tokens, call count)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal

from kalinov.cost.models import CostBreakdown, TokenUsage
from kalinov.llm.base import BudgetExceededError


@dataclass(frozen=True, slots=True)
class Budget:
    max_cost_usd: Decimal | None = None
    max_total_tokens: int | None = None
    max_calls: int | None = None


@dataclass(frozen=True, slots=True)
class BudgetState:
    spent_usd: Decimal
    total_tokens: int
    calls: int


class BudgetGuard:
    """Thread-safe cumulative limits checked after each recorded call."""

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._spent = Decimal("0")
        self._tokens = 0
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> BudgetState:
        with self._lock:
            return BudgetState(
                spent_usd=self._spent,
                total_tokens=self._tokens,
                calls=self._calls,
            )

    def record(self, *, cost: CostBreakdown, usage: TokenUsage, provider: str) -> None:
        """Apply a completed non-cached call; raise if any limit is exceeded."""
        with self._lock:
            self._spent += cost.total_usd
            self._tokens += usage.total_all()
            self._calls += 1

            b = self._budget
            if b.max_cost_usd is not None and self._spent > b.max_cost_usd:
                raise BudgetExceededError(
                    provider=provider,
                    message=f"budget max_cost_usd exceeded ({self._spent} > {b.max_cost_usd})",
                )
            if b.max_total_tokens is not None and self._tokens > b.max_total_tokens:
                raise BudgetExceededError(
                    provider=provider,
                    message=(
                        f"budget max_total_tokens exceeded ({self._tokens} > {b.max_total_tokens})"
                    ),
                )
            if b.max_calls is not None and self._calls > b.max_calls:
                raise BudgetExceededError(
                    provider=provider,
                    message=f"budget max_calls exceeded ({self._calls} > {b.max_calls})",
                )


__all__ = ["Budget", "BudgetGuard", "BudgetState"]
