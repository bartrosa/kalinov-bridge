"""Gemini adapter tests (mocked client)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import BudgetExceededError, LLMError, Message
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
from kalinov.llm.providers.gemini_client import GeminiClient, _flatten_messages


def test_message_flattening() -> None:
    t = _flatten_messages(
        [Message(role="user", content="1"), Message(role="assistant", content="2")],
    )
    assert "USER" in t and "ASSISTANT" in t


def test_complete_maps_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)
    um = SimpleNamespace(
        prompt_token_count=4,
        candidates_token_count=5,
        cached_content_token_count=2,
    )
    resp = SimpleNamespace(
        text="out",
        usage_metadata=um,
        model_version="gemini-2.5-pro",
    )
    monkeypatch.setattr(g._client.models, "generate_content", lambda **kw: resp)
    out = g.complete(
        messages=[Message(role="user", content="p")],
        model="gemini-2.5-pro",
        max_tokens=20,
        temperature=0.1,
        stop=None,
        extras=None,
    )
    assert out.text == "out"
    assert out.usage.input == 4
    assert out.usage.output == 5
    assert out.usage.cache_read == 2
    assert "gemini" in out.model_id_resolved


def test_complete_records_thinking_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini 2.5+ thinking models report a dedicated ``thoughts_token_count``.

    The field is disjoint from ``candidates_token_count`` but billed at the same
    rate as output. Dropping it on the floor under-reports cost and bypasses
    ``BudgetGuard.max_cost_usd`` — see the gemini-2.5-pro pricing row.
    """
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)
    um = SimpleNamespace(
        prompt_token_count=800,
        candidates_token_count=200,
        cached_content_token_count=0,
        thoughts_token_count=4000,
    )
    resp = SimpleNamespace(
        text="out",
        usage_metadata=um,
        model_version="gemini-2.5-pro",
    )
    monkeypatch.setattr(g._client.models, "generate_content", lambda **kw: resp)

    out = g.complete(
        messages=[Message(role="user", content="p")],
        model="gemini-2.5-pro",
        max_tokens=2048,
        temperature=0.0,
        stop=None,
        extras=None,
    )

    assert out.usage.input == 800
    assert out.usage.output == 200
    assert out.usage.reasoning == 4000
    assert out.usage.cache_read == 0
    # ``total_all`` feeds BudgetGuard.max_total_tokens; must include thinking.
    assert out.usage.total_all() == 800 + 200 + 4000

    cost = estimate_cost(
        out.usage,
        provider="gemini",
        model_id="gemini-2.5-pro",
        catalogue=cat,
    )
    # gemini-2.5-pro pricing.yaml rates: input 1.25, output 10.00, reasoning 10.00.
    expected = (
        Decimal(800) / Decimal(1_000_000) * Decimal("1.25")
        + Decimal(200) / Decimal(1_000_000) * Decimal("10.00")
        + Decimal(4000) / Decimal(1_000_000) * Decimal("10.00")
    )
    assert cost.total_usd == expected


def test_budget_guard_counts_thinking_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-fix the per-call cost was visible-only, so a $0.01 cap let through
    more than 10x its budget on a thinking-heavy gemini-2.5-pro response.
    Post-fix the very first call already trips the cap because reasoning
    contributes ~$0.04 on its own.
    """
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)
    um = SimpleNamespace(
        prompt_token_count=800,
        candidates_token_count=200,
        cached_content_token_count=0,
        thoughts_token_count=4000,
    )
    resp = SimpleNamespace(
        text="out",
        usage_metadata=um,
        model_version="gemini-2.5-pro",
    )
    monkeypatch.setattr(g._client.models, "generate_content", lambda **kw: resp)

    guard = BudgetGuard(Budget(max_cost_usd=Decimal("0.01")))
    set_budget_guard(guard)
    try:
        with pytest.raises(BudgetExceededError):
            g.complete(
                messages=[Message(role="user", content="p")],
                model="gemini-2.5-pro",
                max_tokens=2048,
                temperature=0.0,
                stop=None,
                extras=None,
            )
    finally:
        set_budget_guard(None)
    # Real spend recorded on the guard is $0.043, well above the $0.01 cap.
    assert guard.state.spent_usd >= Decimal("0.04")


def test_error_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)

    def boom(**kw: object) -> None:
        e = Exception("quota")
        e.status_code = 429  # type: ignore[attr-defined]
        raise e

    monkeypatch.setattr(g._client.models, "generate_content", boom)
    with pytest.raises(LLMError) as ei:
        g.complete(
            messages=[Message(role="user", content="p")],
            model="m",
            max_tokens=1,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "rate_limit"


def test_count_tokens_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)

    def _fail(**kw: object) -> None:
        msg = "offline"
        raise RuntimeError(msg)

    monkeypatch.setattr(g._client.models, "count_tokens", _fail)
    n = g.count_tokens([Message(role="user", content="a" * 40)], "m")
    assert n == (len(_flatten_messages([Message(role="user", content="a" * 40)])) // 4)
