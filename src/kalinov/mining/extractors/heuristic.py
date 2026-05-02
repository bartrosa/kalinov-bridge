"""Lightweight regex/keyword claim recognition for math/physics abstracts.

Sentence boundaries: ``split`` on whitespace following ``.?!`` (naive).
Known limitation: abbreviations like ``e.g.`` can split incorrectly on
abstract-sized inputs — acceptable for this skeleton; refine later if needed.
"""

from __future__ import annotations

import re
from typing import ClassVar

from kalinov.mining.extractors.base import CandidateClaim, Extractor
from kalinov.mining.sources.base import SourceItem

# Confidence table (audit): order matters — first match wins.
# (pattern tested against lowercased sentence, kind, confidence)
_PATTERN_TABLE: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"theorem\s+(\d+|\d+\.\d+)[\s\.]"), "theorem", 0.55),
    (re.compile(r"lemma\s+(\d+|\d+\.\d+)[\s\.]"), "lemma", 0.52),
    (re.compile(r"^lemma\.(?:\s|$)", re.I), "lemma", 0.5),
    (re.compile(r"we prove that\b"), "theorem", 0.45),
    (re.compile(r"we show that\b"), "theorem", 0.45),
    (re.compile(r"it is well known that\b"), "claim", 0.35),
    (re.compile(r"we establish that\b"), "theorem", 0.44),
    (re.compile(r"proposition\s+\d+[\s\.]"), "proposition", 0.5),
)


def iter_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, sentence)`` using naive ``.?!`` + whitespace splits."""
    text = text.strip()
    if not text:
        return []
    raw_parts = re.split(r"(?<=[.!?])\s+", text)
    spans: list[tuple[int, int, str]] = []
    offset = 0
    for part in raw_parts:
        chunk = part.strip()
        if not chunk:
            continue
        start = text.find(chunk, offset)
        if start < 0:
            start = offset
        end = start + len(chunk)
        spans.append((start, end, chunk))
        offset = end
    return spans


class HeuristicExtractor(Extractor):
    """Regex/keyword-driven extractor for theorem-like sentences."""

    name: ClassVar[str] = "heuristic"

    def extract(self, item: SourceItem) -> tuple[CandidateClaim, ...]:
        out: list[CandidateClaim] = []
        for start, end, sent in iter_sentence_spans(item.text):
            if len(sent) < 5:
                continue
            low = sent.lower()
            for pat, kind, conf in _PATTERN_TABLE:
                if pat.search(low):
                    out.append(
                        CandidateClaim(
                            text=sent,
                            source_item=item,
                            span=(start, end),
                            kind=kind,
                            confidence=conf,
                        ),
                    )
                    break
        return tuple(out)


__all__ = ["HeuristicExtractor", "iter_sentence_spans"]
