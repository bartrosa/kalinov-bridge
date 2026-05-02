from __future__ import annotations

# YAML schema (``kalinov.config.yaml``): root ``providers:`` map; each entry has
# ``type`` (anthropic|openai|gemini|openai_compat), ``default_model``,
# optional ``api_key_env``, ``base_url`` (required for openai_compat),
# optional ``extra_headers``.
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class LLMProviderType(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True, slots=True)
class ProviderConfigEntry:
    name: str
    type: LLMProviderType
    api_key_env: str | None
    base_url: str | None
    default_model: str
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KalinovConfig:
    providers: Mapping[str, ProviderConfigEntry]


class ConfigError(Exception):
    """Invalid ``kalinov.config.yaml`` or provider definition."""


def _xdg_config_candidates() -> list[Path]:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    home = Path.home()
    if xdg:
        return [Path(xdg) / "kalinov" / "config.yaml"]
    return [home / ".config" / "kalinov" / "config.yaml"]


def _empty_config() -> KalinovConfig:
    return KalinovConfig(providers={})


def _parse_entry(name: str, blob: Mapping[str, Any]) -> ProviderConfigEntry:
    try:
        t_raw = blob["type"]
        default_model = str(blob["default_model"])
    except KeyError as exc:
        raise ConfigError(f"provider {name!r} missing {exc.args[0]}") from exc

    try:
        ptype = LLMProviderType(str(t_raw))
    except ValueError as exc:
        raise ConfigError(f"provider {name!r}: unknown type {t_raw!r}") from exc

    api_key_env = blob.get("api_key_env")
    if api_key_env is not None:
        api_key_env = str(api_key_env)
    base_url = blob.get("base_url")
    if base_url is not None:
        base_url = str(base_url)

    if ptype is LLMProviderType.OPENAI_COMPAT and not base_url:
        raise ConfigError(f"provider {name!r}: openai_compat requires base_url")

    extra = blob.get("extra_headers") or {}
    if extra is not None and not isinstance(extra, Mapping):
        raise ConfigError(f"provider {name!r}: extra_headers must be a mapping")

    return ProviderConfigEntry(
        name=name,
        type=ptype,
        api_key_env=api_key_env,
        base_url=base_url,
        default_model=default_model,
        extra_headers=dict(extra) if isinstance(extra, Mapping) else {},
    )


def load_config(path: str | Path | None = None) -> KalinovConfig:
    """Load YAML configuration (see module docstring)."""
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        raw_text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
        return _config_from_mapping(data, source=str(p))

    local = Path("kalinov.config.yaml")
    if local.is_file():
        data = yaml.safe_load(local.read_text(encoding="utf-8"))
        return _config_from_mapping(data, source=str(local.resolve()))

    for cand in _xdg_config_candidates():
        if cand.is_file():
            data = yaml.safe_load(cand.read_text(encoding="utf-8"))
            return _config_from_mapping(data, source=str(cand))

    return _empty_config()


def _config_from_mapping(data: Any, *, source: str) -> KalinovConfig:
    if data is None:
        return _empty_config()
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: root must be a mapping")
    prov = data.get("providers")
    if prov is None:
        return _empty_config()
    if not isinstance(prov, Mapping):
        raise ConfigError(f"{source}: providers must be a mapping")
    out: dict[str, ProviderConfigEntry] = {}
    for name, blob in prov.items():
        if not isinstance(blob, Mapping):
            raise ConfigError(f"{source}: providers.{name} must be a mapping")
        entry = _parse_entry(str(name), blob)
        out[entry.name] = entry
    return KalinovConfig(providers=out)


__all__ = [
    "ConfigError",
    "KalinovConfig",
    "LLMProviderType",
    "ProviderConfigEntry",
    "load_config",
]
