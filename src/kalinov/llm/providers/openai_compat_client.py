"""OpenAI-compatible HTTP servers (Ollama, vLLM, LM Studio, …)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openai

from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, LLMClient, LLMError, Message
from kalinov.llm.cache import LLMCache
from kalinov.llm.pipeline import run_completion
from kalinov.llm.retrying import with_retries


class OpenAICompatClient(LLMClient):
    """Same contract as :class:`OpenAIClient` but with a custom ``base_url``."""

    provider_key = "openai_compat"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        catalogue: PricingCatalogue,
        cache: LLMCache | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._catalogue = catalogue
        self._cache = cache
        self._base_url = base_url
        kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if default_headers:
            kwargs["default_headers"] = dict(default_headers)
        self._client = openai.OpenAI(**kwargs)

    @property
    def cache_namespace(self) -> str:
        """Cache namespace that distinguishes endpoints sharing ``provider_key``.

        Two ``openai_compat`` providers pointed at different ``base_url`` values
        must not share cached completions: the same ``model`` alias commonly
        refers to entirely different model checkpoints across self-hosted
        backends. Including the base URL here scopes cache entries per backend.
        """
        return f"{self.provider_key}:{self._base_url}"

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
            cache_namespace=self.cache_namespace,
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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if stop:
            payload["stop"] = stop

        try:
            resp = self._client.chat.completions.create(**payload)
        except openai.APIStatusError as exc:
            code, retriable = _map_status(exc.status_code)
            raise LLMError(
                provider=self.provider_key,
                code=code,
                message=str(exc),
                retriable=retriable,
            ) from exc
        except Exception as exc:
            raise LLMError(
                provider=self.provider_key,
                code="other",
                message=str(exc),
                retriable=False,
            ) from exc

        choice = resp.choices[0]
        text = choice.message.content or ""
        u = resp.usage
        prompt_tok = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
        comp_tok = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
        usage = TokenUsage(input=prompt_tok, output=comp_tok)
        return Completion(
            text=text,
            usage=usage,
            model_id_resolved=str(resp.model),
            raw_response=resp,
            cache_hit=False,
        )

    def count_tokens(self, messages: list[Message], model: str) -> int:
        return max(1, sum(len(m.content) for m in messages) // 4)


def _map_status(status: int | None) -> tuple[str, bool]:
    if status in (401, 403):
        return "auth", False
    if status == 429:
        return "rate_limit", True
    if status is not None and status >= 500:
        return "server", True
    return "other", False


__all__ = ["OpenAICompatClient"]
