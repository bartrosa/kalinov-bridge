"""Claim extraction from source text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from kalinov.mining.sources.base import SourceItem


@dataclass(frozen=True, slots=True)
class CandidateClaim:
    """Informal statement candidate before Math-Gherkin emission."""

    text: str
    source_item: SourceItem
    span: tuple[int, int]
    kind: str
    confidence: float


class Extractor(ABC):
    name: ClassVar[str]

    @abstractmethod
    def extract(self, item: SourceItem) -> tuple[CandidateClaim, ...]: ...


__all__ = ["CandidateClaim", "Extractor"]
