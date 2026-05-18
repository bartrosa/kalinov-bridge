"""Google Gemini via ``google-genai`` (generate_content).

Streaming is not wired to the SDK stream RPC in this PR; :meth:`stream`
delegates to the base class one-shot completion (see docstring).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from google import genai
from google.genai import types

from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, LLMClient, LLMError, Message
from kalinov.llm.cache import LLMCache
from kalinov.llm.pipeline import run_completion
from kalinov.llm.retrying import with_retries


class GeminiClient(LLMClient):
    """Gemini ``generate_content`` transport (single-turn text concatenation)."""

    provider_key = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        catalogue: PricingCatalogue,
        cache: LLMCache | None = None,
    ) -> None:
        self._catalogue = catalogue
        self._cache = cache
        self._client = genai.Client(api_key=api_key)

    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        extras: Mapping[str, Any] | None,
    ) -> Completion:
        def call_once() -> Completion:
            return self._complete_raw(model, messages, max_tokens, temperature, stop, extras)

        return run_completion(
            provider_catalog_key=self.provider_key,
            provider_label=self.provider_key,
            model_alias=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extras=extras,
            cache=self._cache,
            catalogue=self._catalogue,
            uncached=lambda: with_retries(call_once),
        )

    def _complete_raw(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        extras: Mapping[str, Any] | None,
    ) -> Completion:
        prompt = _flatten_messages(messages)
        cfg_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if stop:
            cfg_kwargs["stop_sequences"] = stop
        cfg = types.GenerateContentConfig(**cfg_kwargs)

        try:
            resp = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
        except Exception as exc:
            code, retriable = _map_exception(exc)
            raise LLMError(
                provider=self.provider_key,
                code=code,
                message=str(exc),
                retriable=retriable,
            ) from exc

        text = ""
        if resp.text:
            text = resp.text
        um = getattr(resp, "usage_metadata", None)
        prompt_tok = int(getattr(um, "prompt_token_count", 0) or 0) if um else 0
        out = int(getattr(um, "candidates_token_count", 0) or 0) if um else 0
        cached = int(getattr(um, "cached_content_token_count", 0) or 0) if um else 0
        # `prompt_token_count` is the total prompt and includes
        # `cached_content_token_count`. Subtract so we don't double-count the
        # cached portion in cost estimation or `BudgetGuard`'s token cap.
        visible_in = max(prompt_tok - cached, 0)
        usage = TokenUsage(
            input=visible_in,
            output=out,
            reasoning=0,
            cache_read=cached,
            cache_write=0,
        )

        resolved = model
        rm = getattr(resp, "model_version", None)
        if rm:
            resolved = str(rm)

        return Completion(
            text=text,
            usage=usage,
            model_id_resolved=resolved,
            raw_response=resp,
            cache_hit=False,
        )

    def count_tokens(self, messages: list[Message], model: str) -> int:
        prompt = _flatten_messages(messages)
        try:
            result = self._client.models.count_tokens(model=model, contents=prompt)
            return int(getattr(result, "total_tokens", 0) or 0) or max(
                1,
                len(prompt) // 4,
            )
        except Exception:
            return max(1, len(prompt) // 4)


def _flatten_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m.role.upper()}:\n{m.content}")
    return "\n\n".join(parts)


def _map_exception(exc: Exception) -> tuple[str, bool]:
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    if status == 429:
        return "rate_limit", True
    if status in (401, 403):
        return "auth", False
    if status is not None and status >= 500:
        return "server", True
    if "quota" in msg:
        return "rate_limit", True
    return "other", False


__all__ = ["GeminiClient"]
