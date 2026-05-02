"""LLM response cache tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, Message
from kalinov.llm.cache import CacheMode, LLMCache


def test_key_stable(tmp_path: Path) -> None:
    c = LLMCache(tmp_path / "cache")
    msgs = [Message(role="user", content="hello")]
    k1 = c.key_for(
        provider="anthropic",
        model="m",
        messages=msgs,
        max_tokens=100,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    k2 = c.key_for(
        provider="anthropic",
        model="m",
        messages=msgs,
        max_tokens=100,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    assert k1 == k2


def test_miss_returns_none(tmp_path: Path) -> None:
    c = LLMCache(tmp_path / "cache")
    assert c.get("deadbeef" * 8) is None


def test_roundtrip_set_get(tmp_path: Path) -> None:
    c = LLMCache(tmp_path / "cache")
    key = c.key_for(
        provider="openai",
        model="gpt-4o",
        messages=[Message(role="user", content="x")],
        max_tokens=10,
        temperature=None,
        stop=None,
        extras={"reasoning_effort": "high"},
    )
    comp = Completion(
        text="ok",
        usage=TokenUsage(input=1, output=2),
        model_id_resolved="gpt-4o-2024",
        raw_response={"x": 1},
        cache_hit=False,
    )
    c.set(key, provider="openai", completion=comp)
    hit = c.get(key)
    assert hit is not None
    assert hit.text == "ok"
    assert hit.model_id_resolved == "gpt-4o-2024"
    assert hit.usage.input == 1


def test_read_only_no_write(tmp_path: Path) -> None:
    c = LLMCache(tmp_path / "cache", mode=CacheMode.READ_ONLY)
    key = "a" * 64
    comp = Completion(
        text="x",
        usage=TokenUsage(),
        model_id_resolved="m",
        raw_response=None,
    )
    c.set(key, provider="p", completion=comp)
    assert c.get(key) is None


def test_extras_subset_affects_key(tmp_path: Path) -> None:
    c = LLMCache(tmp_path / "cache")
    kw: dict[str, Any] = {
        "provider": "anthropic",
        "model": "m",
        "messages": [Message(role="user", content="h")],
        "max_tokens": 1,
        "temperature": None,
        "stop": None,
    }
    k1 = c.key_for(extras={"extended_thinking_budget_tokens": 1000}, **kw)
    k2 = c.key_for(extras={"extended_thinking_budget_tokens": 2000}, **kw)
    assert k1 != k2


def test_irrelevant_extra_ignored_in_key(tmp_path: Path) -> None:
    c = LLMCache(tmp_path / "cache")
    kw2: dict[str, Any] = {
        "provider": "anthropic",
        "model": "m",
        "messages": [Message(role="user", content="h")],
        "max_tokens": 1,
        "temperature": None,
        "stop": None,
    }
    k1 = c.key_for(extras={"foo": "bar"}, **kw2)
    k2 = c.key_for(extras=None, **kw2)
    assert k1 == k2
