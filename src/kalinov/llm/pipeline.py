"""Shared cache, telemetry, and budget wiring for provider adapters."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import CostBreakdown, TokenUsage
from kalinov.llm.base import Completion, LLMError, Message
from kalinov.llm.budget_context import active_budget_guard
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.telemetry import extras_summary_from, log_llm_call


def _zero_cost() -> CostBreakdown:
    z = Decimal("0")
    return CostBreakdown(
        total_usd=z,
        input_usd=z,
        output_usd=z,
        reasoning_usd=z,
        cache_read_usd=z,
        cache_write_usd=z,
        pricing_source="cache",
    )


def run_completion(
    *,
    provider_catalog_key: str,
    provider_label: str,
    model_alias: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float | None,
    stop: list[str] | None,
    extras: Mapping[str, Any] | None,
    cache: LLMCache | None,
    catalogue: PricingCatalogue,
    uncached: Callable[[], Completion],
    cache_namespace: str | None = None,
) -> Completion:
    """Handle cache lookup/miss, telemetry, and budget recording.

    ``cache_namespace`` separates the cache key from the pricing/telemetry key.
    Providers whose ``provider_key`` is shared across multiple distinct
    endpoints (notably ``openai_compat``, which is the same class string for
    every base_url) must pass a namespace that uniquely identifies the
    backend so cached responses from one endpoint are not served to another.
    Defaults to ``provider_catalog_key`` for backwards compatibility.
    """
    t0 = time.perf_counter_ns()
    summary = extras_summary_from(extras)
    cache_key_provider = cache_namespace or provider_catalog_key

    if cache is not None and cache.mode is not CacheMode.OFF:
        key = cache.key_for(
            provider=cache_key_provider,
            model=model_alias,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extras=extras,
        )
        hit = cache.get(key)
        if hit is not None:
            lat = int((time.perf_counter_ns() - t0) / 1_000_000)
            log_llm_call(
                provider=provider_label,
                model_id_resolved=hit.model_id_resolved,
                usage=hit.usage,
                cost=_zero_cost(),
                latency_ms=lat,
                cache_hit=True,
                error_code=None,
                extras_summary=summary,
            )
            return hit

        if cache.mode is CacheMode.READ_ONLY:
            lat = int((time.perf_counter_ns() - t0) / 1_000_000)
            log_llm_call(
                provider=provider_label,
                model_id_resolved=model_alias,
                usage=TokenUsage(),
                cost=_zero_cost(),
                latency_ms=lat,
                cache_hit=False,
                error_code="cache_miss_read_only",
                extras_summary=summary,
            )
            raise LLMError(
                provider=provider_label,
                code="other",
                message="cache miss in read_only mode",
                retriable=False,
            )

    try:
        result = uncached()
    except LLMError as exc:
        lat = int((time.perf_counter_ns() - t0) / 1_000_000)
        log_llm_call(
            provider=provider_label,
            model_id_resolved=model_alias,
            usage=TokenUsage(),
            cost=_zero_cost(),
            latency_ms=lat,
            cache_hit=False,
            error_code=exc.code,
            extras_summary=summary,
        )
        raise

    cost = estimate_cost(
        result.usage,
        provider=provider_catalog_key,
        model_id=result.model_id_resolved,
        catalogue=catalogue,
    )
    # The provider often returns a date-versioned model id (e.g. "gpt-4o-2024-08-06")
    # that doesn't match the alias users put in pricing.yaml ("gpt-4o"). Without this
    # fallback every priced provider call collapses to pricing_source="unknown" /
    # total_usd=0, and the run-wide max_cost_usd guard is silently bypassed.
    if cost.pricing_source == "unknown" and model_alias != result.model_id_resolved:
        fallback = estimate_cost(
            result.usage,
            provider=provider_catalog_key,
            model_id=model_alias,
            catalogue=catalogue,
        )
        if fallback.pricing_source != "unknown":
            cost = fallback

    # Persist telemetry and cache the response BEFORE recording against the
    # budget. ``BudgetGuard.record`` may raise :class:`BudgetExceededError` for
    # a successful (already-billed) provider call; if we logged/cached after
    # ``guard.record``, that work would be silently dropped, which would:
    #   * mask the actual provider spend in ``llm_calls.jsonl`` /
    #     ``kalinov cost report`` (the user is told the run cost $0 when it
    #     really cost real money),
    #   * leave the cache un-populated, so a retry with a larger budget would
    #     re-bill the same prompt against the provider.
    lat = int((time.perf_counter_ns() - t0) / 1_000_000)
    log_llm_call(
        provider=provider_label,
        model_id_resolved=result.model_id_resolved,
        usage=result.usage,
        cost=cost,
        latency_ms=lat,
        cache_hit=False,
        error_code=None,
        extras_summary=summary,
    )

    if cache is not None and cache.mode is CacheMode.READ_WRITE:
        key = cache.key_for(
            provider=cache_key_provider,
            model=model_alias,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extras=extras,
        )
        cache.set(key, provider=cache_key_provider, completion=result)

    guard = active_budget_guard()
    if guard is not None:
        guard.record(cost=cost, usage=result.usage, provider=provider_label)

    return result


__all__ = ["run_completion"]
