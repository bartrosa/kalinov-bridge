"""Factory wiring tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.llm.base import LLMClient
from kalinov.llm.config import ConfigError, KalinovConfig, LLMProviderType, ProviderConfigEntry
from kalinov.llm.factory import make_client


def test_make_anthropic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SK", "secret-key")
    cfg = KalinovConfig(
        providers={
            "mine": ProviderConfigEntry(
                name="mine",
                type=LLMProviderType.ANTHROPIC,
                api_key_env="SK",
                base_url=None,
                default_model="claude-3-5-sonnet-20241022",
            ),
        },
    )
    client = make_client("mine", config=cfg, pricing=load_default_catalogue())
    assert isinstance(client, LLMClient)


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MISSING", raising=False)
    cfg = KalinovConfig(
        providers={
            "x": ProviderConfigEntry(
                name="x",
                type=LLMProviderType.OPENAI,
                api_key_env="MISSING",
                base_url=None,
                default_model="gpt-4o",
            ),
        },
    )
    with pytest.raises(ConfigError, match="MISSING"):
        make_client("x", config=cfg)


def test_openai_compat_optional_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = KalinovConfig(
        providers={
            "loc": ProviderConfigEntry(
                name="loc",
                type=LLMProviderType.OPENAI_COMPAT,
                api_key_env=None,
                base_url="http://localhost:11434/v1",
                default_model="llama",
            ),
        },
    )
    c = make_client("loc", config=cfg)
    from kalinov.llm.providers.openai_compat_client import OpenAICompatClient

    assert isinstance(c, OpenAICompatClient)


def test_unknown_provider_name() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        make_client("nope", config=KalinovConfig(providers={}))


def test_openai_compat_extra_headers_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``extra_headers`` from ``kalinov.config.yaml`` must reach the SDK.

    Corporate openai-compatible gateways frequently require additional auth
    headers (e.g. ``X-Auth-Token``, ``X-Tenant-Id``). The previous wiring
    silently dropped them for the ``openai_compat`` provider, so requests went
    out without the configured headers and were rejected by the gateway.
    """
    cfg = KalinovConfig(
        providers={
            "corp": ProviderConfigEntry(
                name="corp",
                type=LLMProviderType.OPENAI_COMPAT,
                api_key_env=None,
                base_url="https://corp-llm.internal/v1",
                default_model="internal-mistral",
                extra_headers={"X-Auth-Token": "abc", "X-Tenant-Id": "team-a"},
            ),
        },
    )
    c = make_client("corp", config=cfg)
    custom = getattr(c._client, "_custom_headers", None)
    assert custom is not None, "openai SDK no longer exposes _custom_headers"
    assert custom.get("X-Auth-Token") == "abc"
    assert custom.get("X-Tenant-Id") == "team-a"
