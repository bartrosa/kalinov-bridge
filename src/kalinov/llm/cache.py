"""Disk-backed LLM completion cache with canonical keys."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, Message

_EXTRA_KEYS_FOR_KEY = frozenset(
    {
        "extended_thinking_budget_tokens",
        "reasoning_effort",
        "cache_control",
    },
)


class CacheMode(StrEnum):
    OFF = "off"
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"


def _extras_subset(extras: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not extras:
        return None
    d = {k: extras[k] for k in sorted(extras) if k in _EXTRA_KEYS_FOR_KEY}
    return d or None


def _messages_wire(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _canonical_key_blob(
    *,
    provider: str,
    model: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None,
    stop: list[str] | None,
    extras: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "messages": _messages_wire(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": list(stop) if stop else [],
        "extras": _extras_subset(dict(extras)) if extras else None,
    }


class LLMCache:
    """SHA256-keyed JSON cache under ``cache_dir``."""

    def __init__(
        self,
        cache_dir: str | Path,
        mode: CacheMode = CacheMode.READ_WRITE,
    ) -> None:
        self._root = Path(cache_dir).resolve()
        self._mode = mode

    @property
    def mode(self) -> CacheMode:
        return self._mode

    def key_for(
        self,
        *,
        provider: str,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        extras: Mapping[str, Any] | None,
    ) -> str:
        blob = _canonical_key_blob(
            provider=provider,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extras=extras,
        )
        raw = json.dumps(blob, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        prefix = key[:2]
        return self._root / prefix / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        path = self._path_for(key)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        u = data["usage"]
        usage = TokenUsage(
            input=int(u.get("input", 0)),
            output=int(u.get("output", 0)),
            reasoning=int(u.get("reasoning", 0)),
            cache_read=int(u.get("cache_read", 0)),
            cache_write=int(u.get("cache_write", 0)),
        )
        raw_resp = data.get("raw_response")
        return Completion(
            text=str(data["response_text"]),
            usage=usage,
            model_id_resolved=str(data["model_id_resolved"]),
            raw_response=raw_resp,
            cache_hit=True,
        )

    def set(self, key: str, *, provider: str, completion: Completion) -> None:
        if self._mode is not CacheMode.READ_WRITE:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        usage_dict = asdict(completion.usage)
        payload = {
            "key": key,
            "provider": provider,
            "request_summary": "",
            "response_text": completion.text,
            "raw_response": completion.raw_response,
            "usage": usage_dict,
            "timestamp": "",
            "model_id_resolved": completion.model_id_resolved,
        }
        path.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


__all__ = ["CacheMode", "LLMCache"]
