"""ContextVar binding :class:`BudgetGuard` to the active execution scope."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kalinov.llm.budget import BudgetGuard

_active_guard: ContextVar[BudgetGuard | None] = ContextVar("kalinov_budget_guard", default=None)


def active_budget_guard() -> BudgetGuard | None:
    return _active_guard.get()


def set_budget_guard(guard: BudgetGuard | None) -> None:
    _active_guard.set(guard)
