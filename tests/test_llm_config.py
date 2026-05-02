"""Configuration loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalinov.llm.config import (
    ConfigError,
    LLMProviderType,
    load_config,
)


def test_load_sample_yaml(tmp_path: Path) -> None:
    p = tmp_path / "kalinov.config.yaml"
    p.write_text(
        """
providers:
  claude:
    type: anthropic
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-opus-4-7
  ollama:
    type: openai_compat
    base_url: http://localhost:11434/v1
    default_model: llama3.1:70b
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert "claude" in cfg.providers
    assert cfg.providers["claude"].type is LLMProviderType.ANTHROPIC
    assert cfg.providers["ollama"].base_url == "http://localhost:11434/v1"


def test_unknown_provider_type(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("providers:\n  x:\n    type: bogus\n    default_model: m\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown type"):
        load_config(p)


def test_openai_compat_requires_base_url(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "providers:\n  x:\n    type: openai_compat\n    default_model: m\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="base_url"):
        load_config(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_root_returns_empty_providers(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text("{}", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.providers == {}
