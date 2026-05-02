"""Pluggable content sources (pull-style, async iteration)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class SourceItem:
    """One unit of source material — typically a single document."""

    source_id: str
    url: str
    retrieved_at: datetime
    license: str | None
    title: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Source(ABC):
    """A pull-style source. Implementations expose async iteration."""

    name: ClassVar[str]
    requires_network: ClassVar[bool] = True

    @abstractmethod
    def fetch(self, query: str, *, limit: int) -> AsyncIterator[SourceItem]:
        """Yield source items for *query*, at most *limit* total (async iterator)."""
        ...


__all__ = ["Source", "SourceItem"]
