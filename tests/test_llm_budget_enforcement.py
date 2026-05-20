"""End-to-end budget enforcement through ``run_completion``.

These tests pin two related correctness properties for ``--max-cost-usd``:

* The pipeline must look up pricing using the user-supplied ``model_alias``
  when the provider's ``model_id_resolved`` string isn't in ``pricing.yaml``
  (OpenAI / Anthropic frequently return date-versioned ids).
* When no pricing entry can be found for an aliased call **and** a
  ``max_cost_usd`` cap is configured, ``BudgetGuard`` must refuse the call
  instead of silently treating it as $0.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import BudgetExceededError, Message
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
from kalinov.llm.providers.openai_client import OpenAIClient


def _openai_resp(model: str, prompt: int = 1_000_000, completion: int = 0) -> SimpleNamespace:
    """Fabricate an OpenAI SDK response shape sufficient for OpenAIClient."""
    msg = SimpleNamespace(content="hi")
    ch = SimpleNamespace(message=msg, finish_reason="stop")
    pd = SimpleNamespace(cached_tokens=0)
    cd = SimpleNamespace(reasoning_tokens=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=pd,
        completion_tokens_details=cd,
    )
    return SimpleNamespace(model=model, choices=[ch], usage=usage)


def test_pipeline_falls_back_to_model_alias_for_versioned_resolved_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI returns ``gpt-4o-2024-08-06``; pricing.yaml only lists ``gpt-4o``.

    Pre-fix: ``estimate_cost(model_id="gpt-4o-2024-08-06")`` → ``unknown``
    → ``total_usd=0`` → BudgetGuard accumulates nothing → cap never trips.
    Post-fix: alias fallback resolves to the ``gpt-4o`` row and the
    real (priced) cost is recorded.
    """
    cat = load_default_catalogue()
    client = OpenAIClient(api_key="k", catalogue=cat)
    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        lambda **kw: _openai_resp("gpt-4o-2024-08-06"),
    )

    # gpt-4o input pricing = $2.50 / 1M, so 1M input tokens = $2.50 > $0.10 cap.
    guard = BudgetGuard(Budget(max_cost_usd=Decimal("0.10")))
    set_budget_guard(guard)
    try:
        with pytest.raises(BudgetExceededError, match="max_cost_usd exceeded"):
            client.complete(
                messages=[Message(role="user", content="x")],
                model="gpt-4o",
                max_tokens=8,
                temperature=0.0,
                stop=None,
                extras=None,
            )
    finally:
        set_budget_guard(None)

    state = guard.state
    assert state.calls == 1
    assert state.spent_usd == Decimal("2.50")


def test_pipeline_refuses_unknown_model_when_cost_cap_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entirely unpriced model (e.g. ``gpt-4o-mini``) must not silently
    bypass ``max_cost_usd``: the guard refuses the call so the run halts
    instead of being billed without bound.
    """
    cat = load_default_catalogue()
    client = OpenAIClient(api_key="k", catalogue=cat)
    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        lambda **kw: _openai_resp("gpt-4o-mini-2024-07-18"),
    )

    guard = BudgetGuard(Budget(max_cost_usd=Decimal("5.00")))
    set_budget_guard(guard)
    try:
        with pytest.raises(BudgetExceededError, match="no pricing entry"):
            client.complete(
                messages=[Message(role="user", content="x")],
                model="gpt-4o-mini",
                max_tokens=8,
                temperature=0.0,
                stop=None,
                extras=None,
            )
    finally:
        set_budget_guard(None)

    # The refused call must not have been counted against the guard.
    state = guard.state
    assert state.calls == 0
    assert state.spent_usd == Decimal("0")


def test_pipeline_allows_unknown_model_without_cost_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``max_cost_usd``, an unpriced model still succeeds (callers
    that don't ask for monetary enforcement aren't disrupted).
    """
    cat = load_default_catalogue()
    client = OpenAIClient(api_key="k", catalogue=cat)
    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        lambda **kw: _openai_resp("gpt-4o-mini-2024-07-18"),
    )

    guard = BudgetGuard(Budget(max_calls=10))
    set_budget_guard(guard)
    try:
        out = client.complete(
            messages=[Message(role="user", content="x")],
            model="gpt-4o-mini",
            max_tokens=8,
            temperature=0.0,
            stop=None,
            extras=None,
        )
    finally:
        set_budget_guard(None)

    assert out.text == "hi"
    assert guard.state.calls == 1
