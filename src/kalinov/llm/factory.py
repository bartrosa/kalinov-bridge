"""Construct a concrete :class:`LLMClient` from :class:`KalinovConfig`."""

from __future__ import annotations

import os
from collections.abc import Mapping

from kalinov.cost.catalogue import PricingCatalogue, load_default_catalogue
from kalinov.llm.base import LLMClient
from kalinov.llm.cache import LLMCache
from kalinov.llm.config import (
    ConfigError,
    KalinovConfig,
    LLMProviderType,
)
from kalinov.llm.config import (
    load_config as load_llm_config,
)
from kalinov.llm.providers.anthropic_client import AnthropicClient
from kalinov.llm.providers.gemini_client import GeminiClient
from kalinov.llm.providers.openai_client import OpenAIClient
from kalinov.llm.providers.openai_compat_client import OpenAICompatClient


def _require_env(var_name: str) -> str:
    v = os.environ.get(var_name)
    if not v:
        msg = f"environment variable {var_name!r} is not set (required for this provider)"
        raise ConfigError(msg)
    return v


def make_client(
    provider_name: str,
    *,
    config: KalinovConfig | None = None,
    cache: LLMCache | None = None,
    pricing: PricingCatalogue | None = None,
) -> LLMClient:
    """Return a live client; resolve API keys from the environment now."""
    cfg = config or load_llm_config()
    cat = pricing or load_default_catalogue()
    if provider_name not in cfg.providers:
        raise ConfigError(f"unknown provider name {provider_name!r}")
    entry = cfg.providers[provider_name]

    extras_headers: Mapping[str, str] = entry.extra_headers

    if entry.type is LLMProviderType.ANTHROPIC:
        if not entry.api_key_env:
            raise ConfigError("anthropic provider requires api_key_env")
        key = _require_env(entry.api_key_env)
        return AnthropicClient(
            api_key=key,
            catalogue=cat,
            cache=cache,
            default_headers=extras_headers,
        )
    if entry.type is LLMProviderType.OPENAI:
        if not entry.api_key_env:
            raise ConfigError("openai provider requires api_key_env")
        key = _require_env(entry.api_key_env)
        return OpenAIClient(
            api_key=key,
            catalogue=cat,
            cache=cache,
            default_headers=extras_headers,
        )
    if entry.type is LLMProviderType.GEMINI:
        if not entry.api_key_env:
            raise ConfigError("gemini provider requires api_key_env")
        key = _require_env(entry.api_key_env)
        return GeminiClient(api_key=key, catalogue=cat, cache=cache)
    if entry.type is LLMProviderType.OPENAI_COMPAT:
        assert entry.base_url is not None
        key = _require_env(entry.api_key_env) if entry.api_key_env else "not-needed"
        return OpenAICompatClient(
            api_key=key,
            base_url=entry.base_url,
            catalogue=cat,
            cache=cache,
        )

    raise ConfigError(f"unsupported provider type {entry.type!r}")


__all__ = ["make_client"]
