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


def test_default_headers_forwarded_to_openai_sdk() -> None:
    """Custom ``default_headers`` (e.g. corporate gateway auth) must reach the SDK.

    Reproducer for the bug: corporate users configure ``extra_headers`` in
    ``kalinov.config.yaml`` for an ``openai_compat`` endpoint that requires
    additional auth/tracing headers. Previously those headers were silently
    dropped at the ``OpenAICompatClient`` boundary, so every request to the
    gateway hit it WITHOUT the configured headers (resulting in 401/403).
    """
    cat = load_default_catalogue()
    hdrs = {"X-Auth-Token": "shhh", "X-Tenant-Id": "team-a"}
    c = OpenAICompatClient(
        api_key="ignored",
        base_url="https://internal-llm.example/v1",
        catalogue=cat,
        default_headers=hdrs,
    )
    # The openai SDK stores user-provided headers under ``_custom_headers``.
    custom = getattr(c._client, "_custom_headers", None)
    assert custom is not None, "openai SDK no longer exposes _custom_headers"
    for k, v in hdrs.items():
        assert custom.get(k) == v
