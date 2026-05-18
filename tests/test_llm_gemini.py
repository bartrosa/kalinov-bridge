"""Gemini adapter tests (mocked client)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import LLMError, Message
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
    # `prompt_token_count` (4) is the *total* prompt and includes
    # `cached_content_token_count` (2). `input` must subtract the cached
    # portion to avoid double-counting in cost / budget bookkeeping.
    assert out.usage.input == 2  # 4 prompt - 2 cached
    assert out.usage.output == 5
    assert out.usage.cache_read == 2
    assert "gemini" in out.model_id_resolved


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


def test_cached_tokens_not_double_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ``prompt_token_count`` includes ``cached_content_token_count``.

    Pre-fix both fields were stored as ``input`` *and* ``cache_read``, so the
    cached portion was billed twice (against ``input_per_mtok`` plus
    ``cache_read_per_mtok``) and ``usage.total_all()`` inflated the user's
    real token consumption for budget enforcement.
    """
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)
    um = SimpleNamespace(
        prompt_token_count=4000,
        candidates_token_count=500,
        cached_content_token_count=3000,
    )
    resp = SimpleNamespace(text="out", usage_metadata=um, model_version="gemini-2.5-pro")
    monkeypatch.setattr(g._client.models, "generate_content", lambda **kw: resp)
    out = g.complete(
        messages=[Message(role="user", content="p")],
        model="gemini-2.5-pro",
        max_tokens=20,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    # Non-double-counted bookkeeping.
    assert out.usage.input == 1000  # 4000 prompt - 3000 cached
    assert out.usage.cache_read == 3000
    assert out.usage.output == 500
    # `total_all()` must reflect real consumption (prompt + visible_out).
    assert out.usage.total_all() == 4500, (
        f"total_all() must equal prompt + output (4500); pre-fix it was 7500, "
        f"which tripped BudgetGuard token caps prematurely. got {out.usage.total_all()}"
    )

    cost = estimate_cost(
        out.usage,
        provider="gemini",
        model_id="gemini-2.5-pro",
        catalogue=cat,
    )
    # 1000 * $1.25/M + 500 * $10.00/M = $0.00125 + $0.005 = $0.00625.
    # Pre-fix this was $0.01 (4000 * $1.25/M + 500 * $10/M).
    assert cost.total_usd == Decimal("0.006250"), cost.total_usd


def test_count_tokens_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)

    def _fail(**kw: object) -> None:
        msg = "offline"
        raise RuntimeError(msg)

    monkeypatch.setattr(g._client.models, "count_tokens", _fail)
    n = g.count_tokens([Message(role="user", content="a" * 40)], "m")
    assert n == (len(_flatten_messages([Message(role="user", content="a" * 40)])) // 4)
