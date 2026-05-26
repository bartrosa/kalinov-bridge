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
            b = self._budget
            # When max_cost_usd is configured but the call lacks a pricing entry,
            # estimate_cost returns total_usd=0 + pricing_source="unknown". Adding
            # those zeros to self._spent would let the run accumulate unbounded
            # real-money spend without ever tripping the cap. Refuse the call
            # instead — the user can either add a pricing.yaml row or unset the
            # cap.
            if b.max_cost_usd is not None and cost.pricing_source == "unknown":
                raise BudgetExceededError(
                    provider=provider,
                    message=(
                        "refusing to silently bypass max_cost_usd="
                        f"{b.max_cost_usd}: no pricing entry for this model "
                        f"(provider={provider}). Add it to pricing.yaml or "
                        "unset max_cost_usd."
                    ),
                )

            self._spent += cost.total_usd
            self._tokens += usage.total_all()
            self._calls += 1

            # The call has already happened and been billed by the provider.
            # When we raise below, attach ``cost.total_usd`` so the caller
            # (oracle loop, eval runner) can include this overrun in its
            # per-obligation / per-task ``total_cost_usd``. Without this,
            # the user-visible spend summary silently drops the cost of the
            # call that tripped the cap, masking the actual overrun.
            attempted = cost.total_usd

            if b.max_cost_usd is not None and self._spent > b.max_cost_usd:
                raise BudgetExceededError(
                    provider=provider,
                    message=f"budget max_cost_usd exceeded ({self._spent} > {b.max_cost_usd})",
                    attempted_cost_usd=attempted,
                )
            if b.max_total_tokens is not None and self._tokens > b.max_total_tokens:
                raise BudgetExceededError(
                    provider=provider,
                    message=(
                        f"budget max_total_tokens exceeded ({self._tokens} > {b.max_total_tokens})"
                    ),
                    attempted_cost_usd=attempted,
                )
            if b.max_calls is not None and self._calls > b.max_calls:
                raise BudgetExceededError(
                    provider=provider,
                    message=f"budget max_calls exceeded ({self._calls} > {b.max_calls})",
                    attempted_cost_usd=attempted,
                )


__all__ = ["Budget", "BudgetGuard", "BudgetState"]
