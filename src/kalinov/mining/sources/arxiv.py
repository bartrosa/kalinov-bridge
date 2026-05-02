"""arXiv Atom API source (abstracts only)."""

from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import ClassVar
from urllib.parse import quote_plus

import httpx

from kalinov.mining.sources.base import Source, SourceItem

# Coarse tag for arXiv-hosted metadata; per-paper CC licenses are not in the Atom feed.
ARXIV_LICENSE_TAG = "arxiv-noEx-distribute"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_API = "http://export.arxiv.org/api/query"


class ArxivSource(Source):
    """Pulls abstracts from arXiv via its public Atom API.

    https://export.arxiv.org/api/query?search_query=...

    Honors arXiv's rate limits (1 request every 3 seconds between calls).
    """

    name: ClassVar[str] = "arxiv"
    requires_network: ClassVar[bool] = True

    def __init__(
        self,
        *,
        rate_limit_seconds: float = 3.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._rate_limit_seconds = rate_limit_seconds
        self._timeout_seconds = timeout_seconds
        self._last_request_end: float | None = None

    async def _throttle(self) -> None:
        if self._last_request_end is None:
            return
        elapsed = time.monotonic() - self._last_request_end
        wait = self._rate_limit_seconds - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

    async def fetch(self, query: str, *, limit: int = 10) -> AsyncGenerator[SourceItem, None]:
        if limit <= 0:
            return
        q = quote_plus(query)
        # Single request for modest limits (API max_results cap is large enough for our use).
        url = f"{_ARXIV_API}?search_query=all:{q}&start=0&max_results={limit}"
        await self._throttle()
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            resp = await client.get(url)
            self._last_request_end = time.monotonic()
        resp.raise_for_status()
        retrieved_at = datetime.now(tz=UTC)
        root = ET.fromstring(resp.text)
        entries = root.findall(f"{_ATOM}entry")
        for entry in entries[:limit]:
            item = _entry_to_item(entry, retrieved_at=retrieved_at)
            if item is not None:
                yield item


def _entry_to_item(entry: ET.Element, *, retrieved_at: datetime) -> SourceItem | None:
    id_el = entry.find(f"{_ATOM}id")
    title_el = entry.find(f"{_ATOM}title")
    summary_el = entry.find(f"{_ATOM}summary")
    if id_el is None or title_el is None or summary_el is None or not id_el.text:
        return None
    id_text = (id_el.text or "").strip()
    # id is like http://arxiv.org/abs/2501.12345v1
    tail = id_text.rsplit("/", maxsplit=1)[-1]
    source_id = tail.split("v")[0] if tail else id_text
    title = " ".join((title_el.text or "").split())
    abstract = " ".join((summary_el.text or "").split())
    abs_url = f"https://arxiv.org/abs/{source_id}"
    authors: list[str] = []
    for author in entry.findall(f"{_ATOM}author"):
        name_el = author.find(f"{_ATOM}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    return SourceItem(
        source_id=source_id,
        url=abs_url,
        retrieved_at=retrieved_at,
        license=ARXIV_LICENSE_TAG,
        title=title,
        text=abstract,
        metadata={
            "source": "arxiv",
            "authors": authors,
            "arxiv_id_text": id_text,
        },
    )


def parse_atom_feed(body: str, *, retrieved_at: datetime | None = None) -> tuple[SourceItem, ...]:
    """Parse an Atom XML document (test helper / offline replay)."""
    when = retrieved_at or datetime.now(tz=UTC)
    root = ET.fromstring(body)
    out: list[SourceItem] = []
    for entry in root.findall(f"{_ATOM}entry"):
        item = _entry_to_item(entry, retrieved_at=when)
        if item is not None:
            out.append(item)
    return tuple(out)


async def fetch_atom_fixture(body: str) -> AsyncGenerator[SourceItem, None]:
    """Yield items from a fixture string without HTTP (tests only)."""
    when = datetime.now(tz=UTC)
    for item in parse_atom_feed(body, retrieved_at=when):
        yield item


__all__ = [
    "ARXIV_LICENSE_TAG",
    "ArxivSource",
    "fetch_atom_fixture",
    "parse_atom_feed",
]
