"""Abstract LLM client contract (frozen for downstream orchestration)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from kalinov.cost.models import TokenUsage


@dataclass(frozen=True, slots=True)
class Message:
    """Chat message (single block text per PR 4a)."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    """Normalized completion plus unstructured provider payload."""

    text: str
    usage: TokenUsage
    model_id_resolved: str
    raw_response: Any
    cache_hit: bool = False


class LLMError(Exception):
    """Normalized failure from a provider adapter (no raw SDK types)."""

    __slots__ = ("provider", "code", "message", "retriable")

    def __init__(
        self,
        *,
        provider: str,
        code: str,
        message: str,
        retriable: bool,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.message = message
        self.retriable = retriable


class BudgetExceededError(LLMError):
    """Cumulative run budget was exceeded. Not retriable."""

    def __init__(self, *, provider: str, message: str) -> None:
        super().__init__(
            provider=provider,
            code="budget_exceeded",
            message=message,
            retriable=False,
        )


class LLMClient(ABC):
    """Provider adapter: transport only (no oracle logic)."""

    provider_key: ClassVar[str]

    @abstractmethod
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
        """Non-streaming completion."""

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
        """Yield plain-text chunks; default falls back to one-shot completion."""
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

    @abstractmethod
    def count_tokens(self, messages: list[Message], model: str) -> int:
        """Best-effort tokenizer count for budget preflight."""

    def capabilities(self, model: str) -> Mapping[str, Any]:
        """Optional capability hints for UI / harness."""
        return {"model": model}


__all__ = [
    "BudgetExceededError",
    "Completion",
    "LLMClient",
    "LLMError",
    "Message",
]
