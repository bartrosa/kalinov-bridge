"""Deterministic :class:`~kalinov.llm.base.LLMClient` for tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import load_default_catalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import Completion, LLMClient, LLMError, Message
from kalinov.llm.budget_context import active_budget_guard
from kalinov.llm.telemetry import extras_summary_from, log_llm_call


def _default_usage() -> TokenUsage:
    return TokenUsage(input=10, output=20)


@dataclass
class FakeLLMClient(LLMClient):
    """Returns scripted completions and records the last request."""

    provider_key: ClassVar[str] = "openai"
    _queue: list[str | Exception] = field(default_factory=list)
    _last_messages: list[Message] | None = None
    _model_resolved: str = "gpt-4o"
    _usage_factory: Callable[[], TokenUsage] = field(default_factory=lambda: _default_usage)

    def set_queue(self, items: list[str | Exception]) -> None:
        self._queue = list(items)

    @property
    def last_messages(self) -> list[Message] | None:
        return self._last_messages

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
        self._last_messages = list(messages)
        if not self._queue:
            raise LLMError(
                provider=self.provider_key,
                code="other",
                message="fake queue exhausted",
                retriable=False,
            )
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        usage = self._usage_factory()
        result = Completion(
            text=item,
            usage=usage,
            model_id_resolved=self._model_resolved,
            raw_response=None,
            cache_hit=False,
        )
        catalogue = load_default_catalogue()
        cost = estimate_cost(
            result.usage,
            provider=self.provider_key,
            model_id=result.model_id_resolved,
            catalogue=catalogue,
        )
        # Match real provider pipeline ordering: log + cache before budget
        # enforcement so a budget-tripping successful call is still observable.
        log_llm_call(
            provider=self.provider_key,
            model_id_resolved=result.model_id_resolved,
            usage=result.usage,
            cost=cost,
            latency_ms=0,
            cache_hit=False,
            error_code=None,
            extras_summary=extras_summary_from(None),
        )
        guard = active_budget_guard()
        if guard is not None:
            guard.record(cost=cost, usage=result.usage, provider=self.provider_key)
        return result

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
        c = self.complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extras=extras,
        )
        if c.text:
            yield c.text

    def count_tokens(self, messages: list[Message], model: str) -> int:
        return sum(len(m.content) for m in messages)


__all__ = ["FakeLLMClient"]
