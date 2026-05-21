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

    Also pins the regression in which a bad merge dropped
    ``self._base_url = base_url`` from ``__init__`` while keeping the
    ``cache_namespace`` property that reads it, so every ``complete()`` call
    raised ``AttributeError: 'OpenAICompatClient' object has no attribute
    '_base_url'`` and the openai_compat backend became wholly unusable.
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
    kwargs: dict[str, object] = {
        "messages": msgs,
        "model": "llama3",
        "max_tokens": 10,
        "temperature": 0.0,
        "stop": None,
        "extras": None,
    }

    r_local = local.complete(**kwargs)  # type: ignore[arg-type]
    assert r_local.text == "LOCAL"

    r_remote = remote.complete(**kwargs)  # type: ignore[arg-type]
    assert r_remote.text == "REMOTE", (
        "remote endpoint must not receive a cache hit from the local endpoint"
    )

    r_local_again = local.complete(**kwargs)  # type: ignore[arg-type]
    assert r_local_again.text == "LOCAL"
    assert r_local_again.cache_hit is True

    r_remote_again = remote.complete(**kwargs)  # type: ignore[arg-type]
    assert r_remote_again.text == "REMOTE"
    assert r_remote_again.cache_hit is True


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
