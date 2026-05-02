"""Heuristic claim extractor."""

from __future__ import annotations

from datetime import UTC, datetime

from kalinov.mining.extractors.heuristic import _PATTERN_TABLE, HeuristicExtractor
from kalinov.mining.sources.base import SourceItem

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _item(text: str) -> SourceItem:
    return SourceItem(
        source_id="x",
        url="https://example.com/a",
        retrieved_at=_TS,
        license="MIT",
        title="T",
        text=text,
        metadata={"source": "arxiv"},
    )


def test_heuristic_we_prove_span_and_confidence() -> None:
    raw = "First noise. We prove that primes exist. More noise."
    item = _item(raw)
    claims = HeuristicExtractor().extract(item)
    assert len(claims) == 1
    c = claims[0]
    assert c.kind == "theorem"
    assert c.confidence == 0.45
    assert "We prove that" in c.text
    start, end = c.span
    assert raw[start:end] == c.text


def test_heuristic_theorem_number() -> None:
    item = _item("Theorem 1. This is the main statement.")
    claims = HeuristicExtractor().extract(item)
    assert len(claims) == 1
    assert claims[0].kind == "theorem"
    assert claims[0].confidence == 0.55


def test_heuristic_lemma_number_and_case_insensitive_lemma_keyword() -> None:
    item = _item("Lemma 3. Auxiliary bound.")
    claims = HeuristicExtractor().extract(item)
    assert len(claims) == 1
    assert claims[0].kind == "lemma"
    assert claims[0].confidence == 0.52

    item2 = _item("LEMMA. Standalone opener.")
    claims2 = HeuristicExtractor().extract(item2)
    assert len(claims2) == 1
    assert claims2[0].kind == "lemma"
    assert claims2[0].confidence == 0.5


def test_heuristic_we_show_uppercase() -> None:
    item = _item("WE SHOW THAT gravity bends light.")
    out = HeuristicExtractor().extract(item)
    assert len(out) == 1
    assert out[0].confidence == 0.45


def test_heuristic_ignores_unmarked_sentences() -> None:
    item = _item("We discuss related work and cite prior art.")
    assert HeuristicExtractor().extract(item) == ()


def test_pattern_table_is_auditable() -> None:
    assert len(_PATTERN_TABLE) >= 5
    for _pat, _kind, conf in _PATTERN_TABLE:
        assert 0.0 <= conf <= 1.0
