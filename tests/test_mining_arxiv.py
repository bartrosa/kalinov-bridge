"""arXiv Atom source — offline fixture parsing and policy."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from kalinov.mining.errors import MiningError
from kalinov.mining.pipeline import MiningConfig, mine
from kalinov.mining.sources.arxiv import ArxivSource, parse_atom_feed

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "arxiv_sample.atom"


def test_parse_atom_feed_items_and_metadata() -> None:
    body = FIXTURE.read_text(encoding="utf-8")
    when = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    items = parse_atom_feed(body, retrieved_at=when)
    assert len(items) == 2
    a, b = items
    assert a.source_id == "2501.12345"
    assert a.url == "https://arxiv.org/abs/2501.12345"
    assert a.license == "arxiv-noEx-distribute"
    assert a.title == "A Nice Paper About Numbers"
    assert "We prove that" in a.text
    assert a.metadata["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert b.source_id == "2502.99999"
    assert b.metadata["authors"] == ["Solo Author"]


def test_arxiv_rate_limit_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    body = FIXTURE.read_text(encoding="utf-8")

    class DummyClient:
        async def __aenter__(self) -> DummyClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> object:
            xml = body

            class R:
                status_code = 200
                text = xml

                def raise_for_status(self) -> None:
                    return None

            return R()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: DummyClient())
    src = ArxivSource(rate_limit_seconds=0.15, timeout_seconds=5.0)

    async def run_two_fetches() -> int:
        n = 0
        async for _ in src.fetch("test", limit=1):
            n += 1
        async for _ in src.fetch("test", limit=1):
            n += 1
        return n

    t0 = time.monotonic()
    n = asyncio.run(run_two_fetches())
    elapsed = time.monotonic() - t0
    assert n == 2
    assert elapsed >= 0.14


def test_mine_network_disabled_raises_for_arxiv() -> None:
    cfg = MiningConfig(
        source_name="arxiv",
        query="all",
        limit=2,
        network_enabled=False,
    )

    async def run() -> None:
        await mine(cfg)

    with pytest.raises(MiningError, match="network"):
        asyncio.run(run())
