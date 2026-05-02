"""Gemini adapter tests (mocked client)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    assert out.usage.input == 4
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


def test_count_tokens_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    g = GeminiClient(api_key="k", catalogue=cat)

    def _fail(**kw: object) -> None:
        msg = "offline"
        raise RuntimeError(msg)

    monkeypatch.setattr(g._client.models, "count_tokens", _fail)
    n = g.count_tokens([Message(role="user", content="a" * 40)], "m")
    assert n == (len(_flatten_messages([Message(role="user", content="a" * 40)])) // 4)
