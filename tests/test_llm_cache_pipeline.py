"""Cache integration with provider pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import LLMError, Message
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.providers.anthropic_client import AnthropicClient


def test_read_only_miss_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    cache = LLMCache(tmp_path / "c", mode=CacheMode.READ_ONLY)
    client = AnthropicClient(api_key="k", catalogue=cat, cache=cache)

    def no_net(**kw: object) -> None:
        raise AssertionError("no network")

    monkeypatch.setattr(client._client.messages, "create", no_net)

    with pytest.raises(LLMError, match="read_only"):
        client.complete(
            messages=[Message(role="user", content="nope")],
            model="m",
            max_tokens=1,
            temperature=None,
            stop=None,
            extras=None,
        )
