"""OpenAI adapter tests (mocked SDK)."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from types import SimpleNamespace

import httpx
import openai
import pytest

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import BudgetExceededError, LLMError, Message
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
from kalinov.llm.providers.openai_client import OpenAIClient


def _usage(
    prompt: int = 5,
    completion: int = 7,
    cached: int = 2,
    reasoning: int = 3,
) -> SimpleNamespace:
    pd = SimpleNamespace(cached_tokens=cached)
    cd = SimpleNamespace(reasoning_tokens=reasoning)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=pd,
        completion_tokens_details=cd,
    )


def test_usage_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)
    msg = SimpleNamespace(content="hi")
    ch = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(
        model="gpt-4o-2024-08-06",
        choices=[ch],
        usage=_usage(),
    )
    monkeypatch.setattr(c._client.chat.completions, "create", lambda **kw: resp)
    out = c.complete(
        messages=[Message(role="user", content="x")],
        model="gpt-4o",
        max_tokens=20,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    assert "4o" in out.model_id_resolved
    # `prompt_tokens` (5) already contains `cached_tokens` (2). Storing both as
    # `input` AND `cache_read` would double-count the cached portion in cost
    # estimation and budget enforcement, so input must be `prompt - cached`.
    assert out.usage.input == 3  # 5 prompt - 2 cached
    assert out.usage.cache_read == 2
    assert out.usage.reasoning == 3
    assert out.usage.output == 4  # 7 - 3 reasoning


def test_reasoning_extra_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)
    seen: dict[str, object] = {}

    def cap(**kw: object) -> SimpleNamespace:
        seen.update(kw)
        return SimpleNamespace(
            model="o1",
            choices=[SimpleNamespace(message=SimpleNamespace(content="y"), finish_reason="stop")],
            usage=_usage(reasoning=0, completion=1),
        )

    monkeypatch.setattr(c._client.chat.completions, "create", cap)
    c.complete(
        messages=[Message(role="user", content="q")],
        model="o1",
        max_tokens=50,
        temperature=None,
        stop=None,
        extras={"reasoning_effort": "high"},
    )
    assert seen.get("reasoning_effort") == "high"


def test_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)

    def boom(**kw: object) -> None:
        raise openai.RateLimitError("rl", response=resp, body={})

    monkeypatch.setattr(c._client.chat.completions, "create", boom)
    with pytest.raises(LLMError) as ei:
        c.complete(
            messages=[Message(role="user", content="a")],
            model="gpt-4o",
            max_tokens=1,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "rate_limit"


def test_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)
    req = httpx.Request("POST", "https://api.openai.com")
    resp = httpx.Response(401, request=req)

    def boom(**kw: object) -> None:
        raise openai.AuthenticationError("no", response=resp, body={})

    monkeypatch.setattr(c._client.chat.completions, "create", boom)
    with pytest.raises(LLMError) as ei:
        c.complete(
            messages=[Message(role="user", content="a")],
            model="gpt-4o",
            max_tokens=1,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "auth"


def test_cached_tokens_not_double_counted_in_cost_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``prompt_tokens`` includes ``cached_tokens``.

    Pre-fix the adapter stored both as ``input`` (= ``prompt_tokens``) and
    ``cache_read`` (= ``cached_tokens``), so:

    * cost estimation billed the cached portion twice (once at the full
      ``input_per_mtok`` rate, again at ``cache_read_per_mtok``);
    * ``usage.total_all()`` was inflated, which made ``BudgetGuard``'s
      ``max_total_tokens`` cap trip earlier than the user's real token spend.
    """
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)
    # 4000-token prompt, 3000 of which were cached, plus 500 visible output.
    resp = SimpleNamespace(
        model="gpt-4o-2024-08-06",
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
        usage=_usage(prompt=4000, cached=3000, completion=500, reasoning=0),
    )
    monkeypatch.setattr(c._client.chat.completions, "create", lambda **kw: resp)
    out = c.complete(
        messages=[Message(role="user", content="x")],
        model="gpt-4o",
        max_tokens=1024,
        temperature=0.0,
        stop=None,
        extras=None,
    )

    # Non-double-counted token bookkeeping.
    assert out.usage.input == 1000  # 4000 - 3000 cached
    assert out.usage.cache_read == 3000
    assert out.usage.output == 500
    # Real tokens the user paid for == prompt + visible_out == 4500.
    assert out.usage.total_all() == 4500, (
        f"total_all() must equal prompt + output (4500), not double-count "
        f"cached tokens (would give 7500 pre-fix); got {out.usage.total_all()}"
    )

    # Cost estimate against bundled pricing (cache_read_per_mtok defaults to 0
    # so cached portion is priced free, but the *uncached* input portion must
    # not still be billed against the full 4000 prompt count).
    cost = estimate_cost(
        out.usage,
        provider="openai",
        model_id="gpt-4o",
        catalogue=cat,
    )
    # 1000 * $2.50/M + 500 * $10.00/M = $0.0025 + $0.005 = $0.0075.
    # Pre-fix this was $0.015 (4000 * $2.50/M + 500 * $10/M).
    assert cost.total_usd == Decimal("0.007500"), cost.total_usd


def test_cached_tokens_do_not_trip_budget_prematurely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BudgetGuard`` token cap must reflect real token consumption."""
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)
    resp = SimpleNamespace(
        model="gpt-4o-2024-08-06",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=_usage(prompt=4000, cached=3000, completion=500, reasoning=0),
    )
    monkeypatch.setattr(c._client.chat.completions, "create", lambda **kw: resp)

    # Cap == real token consumption (4500). Pre-fix `total_all()` was 7500 so
    # this single call would have raised; post-fix it must succeed.
    guard = BudgetGuard(Budget(max_total_tokens=4500))
    set_budget_guard(guard)
    try:
        c.complete(
            messages=[Message(role="user", content="x")],
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.0,
            stop=None,
            extras=None,
        )
    finally:
        set_budget_guard(None)
    assert guard.state.total_tokens == 4500

    # And the next equivalent call MUST trip the cap.
    guard2 = BudgetGuard(Budget(max_total_tokens=4500))
    set_budget_guard(guard2)
    try:
        c.complete(
            messages=[Message(role="user", content="x")],
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.0,
            stop=None,
            extras=None,
        )
        with pytest.raises(BudgetExceededError):
            c.complete(
                messages=[Message(role="user", content="x")],
                model="gpt-4o",
                max_tokens=1024,
                temperature=0.0,
                stop=None,
                extras=None,
            )
    finally:
        set_budget_guard(None)


def test_stream_yields_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    c = OpenAIClient(api_key="k", catalogue=cat)

    def fake_stream(**kw: object) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="a"), finish_reason=None)],
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="b"), finish_reason="stop")],
        )

    monkeypatch.setattr(c._client.chat.completions, "create", fake_stream)
    out = "".join(
        c.stream(
            messages=[Message(role="user", content="x")],
            model="gpt-4o",
            max_tokens=5,
            temperature=None,
            stop=None,
            extras=None,
        ),
    )
    assert out == "ab"
