"""OpenAI-compatible (local) adapter tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import Message
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.providers.openai_compat_client import OpenAICompatClient


def test_pricing_key_zero_usd(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    c = OpenAICompatClient(
        api_key="not-needed",
        base_url="http://127.0.0.1:11434/v1",
        catalogue=cat,
    )
    ch = SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
    resp = SimpleNamespace(
        model="llama3",
        choices=[ch],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )
    monkeypatch.setattr(c._client.chat.completions, "create", lambda **kw: resp)
    out = c.complete(
        messages=[Message(role="user", content="hi")],
        model="llama3",
        max_tokens=10,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    assert out.text == "ok"
    assert out.usage.input + out.usage.output == 5


def test_count_tokens_heuristic() -> None:
    cat = load_default_catalogue()
    c = OpenAICompatClient(
        api_key="x",
        base_url="http://localhost:1/v1",
        catalogue=cat,
    )
    n = c.count_tokens([Message(role="user", content="abcd")], "m")
    assert n == 1


def test_cache_namespaced_per_base_url(tmp_path: Path) -> None:
    """A shared LLM cache must not return entries from a different backend.

    Two ``openai_compat`` providers configured against different base URLs can
    legitimately publish entirely different model checkpoints under the same
    public alias (e.g. ``llama3``). Cache lookups must be scoped by base URL
    so cached responses from one endpoint are never served to another.
    """
    cat = load_default_catalogue()
    cache = LLMCache(tmp_path / "cache", mode=CacheMode.READ_WRITE)

    local = OpenAICompatClient(
        api_key="not-needed",
        base_url="http://127.0.0.1:11434/v1",
        catalogue=cat,
        cache=cache,
    )
    remote = OpenAICompatClient(
        api_key="not-needed",
        base_url="http://remote.example.com/v1",
        catalogue=cat,
        cache=cache,
    )

    def _resp(text: str) -> SimpleNamespace:
        return SimpleNamespace(
            model="llama3",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason="stop",
                ),
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    local._client.chat.completions.create = lambda **kw: _resp("LOCAL")
    remote._client.chat.completions.create = lambda **kw: _resp("REMOTE")

    msgs = [Message(role="user", content="hello")]
    r_local = local.complete(
        messages=msgs,
        model="llama3",
        max_tokens=10,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    assert r_local.text == "LOCAL"
    assert r_local.cache_hit is False

    r_remote = remote.complete(
        messages=msgs,
        model="llama3",
        max_tokens=10,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    assert r_remote.text == "REMOTE", (
        "shared cache must not serve a LOCAL response to a REMOTE openai_compat "
        "endpoint configured with a different base_url"
    )
    assert r_remote.cache_hit is False

    r_local2 = local.complete(
        messages=msgs,
        model="llama3",
        max_tokens=10,
        temperature=0.0,
        stop=None,
        extras=None,
    )
    assert r_local2.text == "LOCAL"
    assert r_local2.cache_hit is True
