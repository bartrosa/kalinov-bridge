"""Tenacity retry wrapper for transient :class:`LLMError` codes."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from kalinov.llm.base import LLMError

T = TypeVar("T")


def max_retry_attempts() -> int:
    raw = os.environ.get("LLM_MAX_RETRIES", "3")
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _retriable(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and exc.retriable and exc.code in ("rate_limit", "server")


def with_retries(fn: Callable[[], T]) -> T:
    """Run *fn* with exponential backoff on rate limits / 5xx-class LLM errors."""
    r = Retrying(
        stop=stop_after_attempt(max_retry_attempts()),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception(_retriable),
        reraise=True,
    )
    return r(fn)


__all__ = ["max_retry_attempts", "with_retries"]
