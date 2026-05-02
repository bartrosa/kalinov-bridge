"""Feature emitter guards and Gherkin round-trip."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalinov.gherkin import parse_feature_file
from kalinov.mining.emit import emit_feature_file
from kalinov.mining.errors import MiningError
from kalinov.mining.extractors.base import CandidateClaim
from kalinov.mining.sources.base import SourceItem

_TS = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)


def _claim(
    text: str,
    *,
    lic: str | None = "arxiv-noEx-distribute",
    url: str = "https://arxiv.org/abs/1",
) -> CandidateClaim:
    item = SourceItem(
        source_id="2501.00001",
        url=url,
        retrieved_at=_TS,
        license=lic,
        title="Sample Paper Title Goes Here",
        text="ignored",
        metadata={"source": "arxiv"},
    )
    return CandidateClaim(
        text=text,
        source_item=item,
        span=(0, len(text)),
        kind="theorem",
        confidence=0.9,
    )


def test_emit_refuses_missing_url(tmp_path: Path) -> None:
    bad_item = SourceItem(
        source_id="x",
        url="",
        retrieved_at=_TS,
        license="L",
        title="t",
        text="t",
        metadata={"source": "arxiv"},
    )
    c = CandidateClaim(
        text="We prove that X.",
        source_item=bad_item,
        span=(0, 5),
        kind="theorem",
        confidence=0.5,
    )
    with pytest.raises(MiningError, match="url"):
        emit_feature_file([c], feature_name="F", out_dir=tmp_path)


def test_emit_refuses_missing_license(tmp_path: Path) -> None:
    item = SourceItem(
        source_id="x",
        url="https://a.test/doc",
        retrieved_at=_TS,
        license=None,
        title="t",
        text="t",
        metadata={"source": "arxiv"},
    )
    c = CandidateClaim(
        text="We prove that X.",
        source_item=item,
        span=(0, 3),
        kind="theorem",
        confidence=0.5,
    )
    with pytest.raises(MiningError, match="license"):
        emit_feature_file([c], feature_name="F", out_dir=tmp_path)


def test_emit_filename_contains_sha8(tmp_path: Path) -> None:
    path = emit_feature_file(
        [_claim("We prove that foo."), _claim("Lemma-style claim via We prove that bar.")],
        feature_name="My Feature",
        out_dir=tmp_path,
    )
    m = re.search(r"__([0-9a-f]{8})\.feature$", path.name)
    assert m is not None


def test_emit_round_trips_parser(tmp_path: Path) -> None:
    path = emit_feature_file(
        [_claim("We prove that convergence holds.")],
        feature_name="Emit test",
        out_dir=tmp_path,
    )
    ff = parse_feature_file(path)
    assert ff.feature.name == "Emit test"
    assert any("@mined" in t for t in ff.feature.tags)
    assert len(ff.feature.scenarios) == 1
    step = ff.feature.scenarios[0].steps[0]
    assert "convergence" in step.text


def test_emit_refuses_empty_candidates(tmp_path: Path) -> None:
    with pytest.raises(MiningError, match="no candidates"):
        emit_feature_file([], feature_name="X", out_dir=tmp_path)
