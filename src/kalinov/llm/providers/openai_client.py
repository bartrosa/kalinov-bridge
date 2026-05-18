"""OpenAI Chat Completions adapter (GPT + reasoning models)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import openai

from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, LLMClient, LLMError, Message
from kalinov.llm.cache import LLMCache
from kalinov.llm.pipeline import run_completion
from kalinov.llm.retrying import with_retries


class OpenAIClient(LLMClient):
    """OpenAI ``chat.completions`` via the ``openai`` SDK."""

    provider_key = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        catalogue: PricingCatalogue,
        cache: LLMCache | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._catalogue = catalogue
        self._cache = cache
        kwargs: dict[str, Any] = {"api_key": api_key}
        if default_headers:
            kwargs["default_headers"] = dict(default_headers)
        self._client = openai.OpenAI(**kwargs)

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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if stop:
            payload["stop"] = stop
        if extras and "reasoning_effort" in extras:
            payload["reasoning_effort"] = extras["reasoning_effort"]

        try:
            resp = self._client.chat.completions.create(**payload)
        except openai.RateLimitError as exc:
            raise LLMError(
                provider=self.provider_key,
                code="rate_limit",
                message=str(exc),
                retriable=True,
            ) from exc
        except openai.AuthenticationError as exc:
            raise LLMError(
                provider=self.provider_key,
                code="auth",
                message=str(exc),
                retriable=False,
            ) from exc
        except openai.APIStatusError as exc:
            code, retriable = _map_openai_status(exc.status_code)
            raise LLMError(
                provider=self.provider_key,
                code=code,
                message=str(exc),
                retriable=retriable,
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                provider=self.provider_key,
                code="timeout",
                message=str(exc),
                retriable=True,
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
        assert u is not None
        cached = 0
        reasoning = 0
        ptd = getattr(u, "prompt_tokens_details", None)
        if ptd is not None:
            cached = int(getattr(ptd, "cached_tokens", 0) or 0)
        ctd = getattr(u, "completion_tokens_details", None)
        if ctd is not None:
            reasoning = int(getattr(ctd, "reasoning_tokens", 0) or 0)
        comp = int(u.completion_tokens or 0)
        visible_out = max(comp - reasoning, 0)
        # OpenAI's `prompt_tokens` is the *total* prompt count and already
        # includes `prompt_tokens_details.cached_tokens`. Storing both would
        # double-count the cached portion in cost estimation and in
        # `usage.total_all()` (which feeds `BudgetGuard`'s token cap).
        prompt = int(u.prompt_tokens or 0)
        visible_in = max(prompt - cached, 0)
        usage = TokenUsage(
            input=visible_in,
            output=visible_out,
            reasoning=reasoning,
            cache_read=cached,
            cache_write=0,
        )
        return Completion(
            text=text,
            usage=usage,
            model_id_resolved=str(resp.model),
            raw_response=resp,
            cache_hit=False,
        )

    def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        extras: Mapping[str, Any] | None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if stop:
            payload["stop"] = stop
        if extras and "reasoning_effort" in extras:
            payload["reasoning_effort"] = extras["reasoning_effort"]
        try:
            stream = self._client.chat.completions.create(**payload)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except openai.APIStatusError as exc:
            code, retriable = _map_openai_status(exc.status_code)
            raise LLMError(
                provider=self.provider_key,
                code=code,
                message=str(exc),
                retriable=retriable,
            ) from exc

    def count_tokens(self, messages: list[Message], model: str) -> int:
        return max(1, sum(len(m.content) for m in messages) // 4)


def _map_openai_status(status: int | None) -> tuple[str, bool]:
    if status in (401, 403):
        return "auth", False
    if status == 429:
        return "rate_limit", True
    if status == 400:
        return "context_length", False
    if status is not None and status >= 500:
        return "server", True
    return "other", False


__all__ = ["OpenAIClient"]
