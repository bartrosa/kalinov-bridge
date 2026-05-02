"""OpenAI-compatible (local) adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import Message
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
