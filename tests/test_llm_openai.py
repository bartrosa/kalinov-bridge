"""OpenAI adapter tests (mocked SDK)."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import openai
import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import LLMError, Message
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
    assert out.usage.input == 5
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
