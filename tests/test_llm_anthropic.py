"""Anthropic adapter tests (mocked SDK)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import LLMError, Message
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.providers.anthropic_client import AnthropicClient
from kalinov.telemetry import start_run


def _fake_usage(**kw: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "input_tokens": 10,
        "output_tokens": 6,
        "thinking_tokens": 0,
        "cache_read_input_tokens": 1,
        "cache_creation_input_tokens": 2,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_maps_usage_and_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    text_block = SimpleNamespace(type="text", text="Hello")
    resp = SimpleNamespace(
        model="claude-3-5-sonnet-20241022",
        content=[text_block],
        usage=_fake_usage(),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: resp)

    out = client.complete(
        messages=[Message(role="user", content="hi")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        temperature=0.5,
        stop=None,
        extras=None,
    )
    assert out.text == "Hello"
    assert out.model_id_resolved == "claude-3-5-sonnet-20241022"
    assert out.usage.input == 10
    assert out.usage.cache_read == 1


def test_rate_limit_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp429 = httpx.Response(429, request=req)

    def boom(**kw: object) -> None:
        raise anthropic.RateLimitError("slow down", response=resp429, body={})

    monkeypatch.setattr(client._client.messages, "create", boom)
    with pytest.raises(LLMError) as ei:
        client.complete(
            messages=[Message(role="user", content="x")],
            model="m",
            max_tokens=10,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "rate_limit"


def test_auth_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp401 = httpx.Response(401, request=req)

    def boom(**kw: object) -> None:
        raise anthropic.AuthenticationError("nope", response=resp401, body={})

    monkeypatch.setattr(client._client.messages, "create", boom)
    with pytest.raises(LLMError) as ei:
        client.complete(
            messages=[Message(role="user", content="x")],
            model="m",
            max_tokens=10,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "auth"


def test_cache_hit_short_circuits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def mk(**kw: object) -> SimpleNamespace:
        calls.append(1)
        return SimpleNamespace(
            model="claude-3-5-sonnet-20241022",
            content=[SimpleNamespace(type="text", text="cached")],
            usage=_fake_usage(input_tokens=2, output_tokens=3),
        )

    def boom(**kw: object) -> None:
        raise AssertionError("SDK must not run on cache hit")

    cat = load_default_catalogue()
    cache = LLMCache(tmp_path / "c", mode=CacheMode.READ_WRITE)
    client = AnthropicClient(api_key="k", catalogue=cat, cache=cache)
    monkeypatch.setattr(client._client.messages, "create", mk)

    first = client.complete(
        messages=[Message(role="user", content="same")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        temperature=None,
        stop=None,
        extras=None,
    )
    assert first.text == "cached"
    assert calls == [1]

    monkeypatch.setattr(client._client.messages, "create", boom)

    second = client.complete(
        messages=[Message(role="user", content="same")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        temperature=None,
        stop=None,
        extras=None,
    )
    assert second.text == "cached"
    assert calls == [1]


def test_telemetry_with_active_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    resp = SimpleNamespace(
        model="m",
        content=[SimpleNamespace(type="text", text="t")],
        usage=_fake_usage(),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: resp)
    with start_run(runs_dir=tmp_path):
        client.complete(
            messages=[Message(role="user", content="u")],
            model="m",
            max_tokens=5,
            temperature=None,
            stop=None,
            extras=None,
        )
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert run_dirs
    log = run_dirs[0] / "llm_calls.jsonl"
    assert log.is_file()
