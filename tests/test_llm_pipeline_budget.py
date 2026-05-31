"""Regression: budget enforcement must not drop telemetry/cache for the
successful provider call that pushed the run over budget.

Pre-fix, ``run_completion`` recorded the cost against the active
:class:`BudgetGuard` *before* writing the ``llm_calls.jsonl`` row and
populating the response cache. When that call was the one that exceeded
the budget, the resulting ``BudgetExceededError`` aborted the
log/cache work, so:

  * ``kalinov cost report`` would show $0 spent for a run where the
    provider actually billed real money, and
  * a retry with a larger budget would re-bill the same prompt because
    the cache was still empty.

Both are ``data loss`` flavoured outcomes; this test pins the new
ordering (log + cache, then ``guard.record``).
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import BudgetExceededError, Completion, Message
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.pipeline import run_completion
from kalinov.telemetry.context import start_run


def _completion() -> Completion:
    return Completion(
        text="hi",
        usage=TokenUsage(input=1000, output=1000),
        model_id_resolved="gpt-4o",
        raw_response={"id": "abc"},
        cache_hit=False,
    )


def test_budget_exceeded_still_logs_call_and_writes_cache(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    cache = LLMCache(cache_dir, mode=CacheMode.READ_WRITE)
    catalogue = load_default_catalogue()
    messages = [Message(role="user", content="please")]

    with start_run(runs_dir=runs_dir) as run:
        guard = BudgetGuard(Budget(max_cost_usd=Decimal("0.0001")))
        set_budget_guard(guard)
        try:
            with pytest.raises(BudgetExceededError):
                run_completion(
                    provider_catalog_key="openai",
                    provider_label="openai",
                    model_alias="gpt-4o",
                    messages=messages,
                    max_tokens=1024,
                    temperature=None,
                    stop=None,
                    extras=None,
                    cache=cache,
                    catalogue=catalogue,
                    uncached=_completion,
                )

            llm_calls = run.run_dir / "llm_calls.jsonl"
            assert llm_calls.is_file(), (
                "successful provider call missing from llm_calls.jsonl after "
                "BudgetExceededError; cost report would under-report spend"
            )
            rows = [json.loads(line) for line in llm_calls.read_text().splitlines() if line]
            assert len(rows) == 1
            row = rows[0]
            assert row["error_code"] is None
            assert row["cache_hit"] is False
            assert Decimal(row["cost_usd"]) > 0

            cached_files = list(cache_dir.rglob("*.json"))
            assert cached_files, (
                "cache not populated after a successful (paid) provider call "
                "tripped the budget; retry would re-bill the same prompt"
            )
        finally:
            set_budget_guard(None)


def test_subsequent_call_after_budget_exceeded_does_not_bill_provider(
    tmp_path: Path,
) -> None:
    """Once the cumulative budget has been tripped, ``run_completion`` must
    refuse to invoke the provider (``uncached()``) for any further call.

    Pre-fix, ``run_completion`` always called the provider first and only
    checked the budget afterwards via ``guard.record``. That meant every
    obligation / task / matrix config that ran AFTER the first over-budget
    call still hit the provider once (real money) before being rejected,
    silently turning a $X overage into roughly ``N × per-call cost`` of
    additional spend across the rest of the run.

    Post-fix, ``run_completion`` consults ``BudgetGuard.ensure_not_exceeded``
    before any billable work and aborts cleanly.
    """
    runs_dir = tmp_path / "runs"
    catalogue = load_default_catalogue()
    messages = [Message(role="user", content="please")]

    provider_call_count = 0

    def counting_uncached() -> Completion:
        nonlocal provider_call_count
        provider_call_count += 1
        return _completion()

    with start_run(runs_dir=runs_dir):
        guard = BudgetGuard(Budget(max_cost_usd=Decimal("0.0001")))
        set_budget_guard(guard)
        try:
            with pytest.raises(BudgetExceededError):
                run_completion(
                    provider_catalog_key="openai",
                    provider_label="openai",
                    model_alias="gpt-4o",
                    messages=messages,
                    max_tokens=1024,
                    temperature=None,
                    stop=None,
                    extras=None,
                    cache=None,
                    catalogue=catalogue,
                    uncached=counting_uncached,
                )
            assert provider_call_count == 1, (
                "first call must invoke the provider so we can detect the overage"
            )

            with pytest.raises(BudgetExceededError, match="already exceeded"):
                run_completion(
                    provider_catalog_key="openai",
                    provider_label="openai",
                    model_alias="gpt-4o",
                    messages=messages,
                    max_tokens=1024,
                    temperature=None,
                    stop=None,
                    extras=None,
                    cache=None,
                    catalogue=catalogue,
                    uncached=counting_uncached,
                )
            assert provider_call_count == 1, (
                "second call must NOT invoke the provider once the cumulative "
                f"budget is already exceeded; uncached() ran {provider_call_count} times"
            )
        finally:
            set_budget_guard(None)


def test_cache_hits_still_served_after_budget_exceeded(tmp_path: Path) -> None:
    """A cache hit costs $0 and must still be served once the budget is tripped
    (the pre-check only gates billable provider work, not free cache reads)."""
    runs_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    cache = LLMCache(cache_dir, mode=CacheMode.READ_WRITE)
    catalogue = load_default_catalogue()
    messages = [Message(role="user", content="please")]

    provider_call_count = 0

    def counting_uncached() -> Completion:
        nonlocal provider_call_count
        provider_call_count += 1
        return _completion()

    with start_run(runs_dir=runs_dir):
        guard = BudgetGuard(Budget(max_cost_usd=Decimal("0.0001")))
        set_budget_guard(guard)
        try:
            with pytest.raises(BudgetExceededError):
                run_completion(
                    provider_catalog_key="openai",
                    provider_label="openai",
                    model_alias="gpt-4o",
                    messages=messages,
                    max_tokens=1024,
                    temperature=None,
                    stop=None,
                    extras=None,
                    cache=cache,
                    catalogue=catalogue,
                    uncached=counting_uncached,
                )
            assert provider_call_count == 1

            result = run_completion(
                provider_catalog_key="openai",
                provider_label="openai",
                model_alias="gpt-4o",
                messages=messages,
                max_tokens=1024,
                temperature=None,
                stop=None,
                extras=None,
                cache=cache,
                catalogue=catalogue,
                uncached=counting_uncached,
            )
            assert result.cache_hit is True
            assert result.text == "hi"
            assert provider_call_count == 1, (
                "cache hit must not invoke the provider; the pre-check should "
                "only gate billable work"
            )
        finally:
            set_budget_guard(None)
