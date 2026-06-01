"""Anthropic Claude adapter (messages API + optional extended thinking)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anthropic

from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, LLMClient, LLMError, Message
from kalinov.llm.cache import LLMCache
from kalinov.llm.pipeline import run_completion
from kalinov.llm.retrying import with_retries


class AnthropicClient(LLMClient):
    """Claude via the ``anthropic`` SDK (non-streaming + base ``stream`` fallback)."""

    provider_key = "anthropic"

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
        self._client = anthropic.Anthropic(**kwargs)

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
            return self._complete_raw(
                model,
                messages,
                max_tokens,
                temperature,
                stop,
                extras,
            )

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
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        conv = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conv,
        }
        if system_text:
            body["system"] = system_text
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop_sequences"] = stop
        if extras and "extended_thinking_budget_tokens" in extras:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(extras["extended_thinking_budget_tokens"]),
            }

        try:
            resp = self._client.messages.create(**body)
        except anthropic.APIStatusError as exc:
            code, retriable = _map_api_status(exc.status_code)
            raise LLMError(
                provider=self.provider_key,
                code=code,
                message=str(exc.message),
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

        text_parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        text = "".join(text_parts)

        u = resp.usage
        # Anthropic's `Usage` exposes extended-thinking tokens via the nested
        # ``output_tokens_details.thinking_tokens`` field — there is no
        # top-level ``thinking_tokens`` attribute. Reading it directly off ``u``
        # always returns 0, which silently lumps every thinking token into the
        # ``output`` bucket and leaves ``reasoning`` empty. With the bundled
        # ``pricing.yaml`` (where ``output_per_mtok == reasoning_per_mtok`` for
        # Claude) total cost still adds up, but per-bucket telemetry and any
        # custom pricing that prices thinking tokens differently from visible
        # output (e.g. a tiered enterprise rate) come out wrong. Mirror the
        # OpenAI / Gemini adapters and read the nested details object.
        otd = getattr(u, "output_tokens_details", None)
        thinking = int(getattr(otd, "thinking_tokens", 0) or 0) if otd is not None else 0
        # ``output_tokens`` is the authoritative billed total and *includes*
        # thinking tokens, so subtract to get the visible output.
        out_reported = int(getattr(u, "output_tokens", 0) or 0)
        visible_out = max(out_reported - thinking, 0)
        usage = TokenUsage(
            input=int(getattr(u, "input_tokens", 0) or 0),
            output=visible_out,
            reasoning=thinking,
            cache_read=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_write=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        )

        return Completion(
            text=text,
            usage=usage,
            model_id_resolved=str(resp.model),
            raw_response=resp,
            cache_hit=False,
        )

    def count_tokens(self, messages: list[Message], model: str) -> int:
        return max(1, sum(len(m.content) for m in messages) // 4)


def _map_api_status(status: int | None) -> tuple[str, bool]:
    if status in (401, 403):
        return "auth", False
    if status == 429:
        return "rate_limit", True
    if status == 413:
        return "context_length", False
    if status is not None and status >= 500:
        return "server", True
    return "other", False


__all__ = ["AnthropicClient"]
