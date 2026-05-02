"""End-to-end mining pipeline (offline fetch monkeypatch)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from kalinov.mining.pipeline import MiningConfig, mine
from kalinov.mining.sources.arxiv import ArxivSource, fetch_atom_fixture
from kalinov.mining.sources.base import SourceItem
from kalinov.telemetry import start_run

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "arxiv_sample.atom"


def test_pipeline_offline_emits_and_logs_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    body = FIXTURE.read_text(encoding="utf-8")

    async def fake_fetch(
        self: ArxivSource,
        query: str,
        *,
        limit: int = 10,
    ) -> AsyncGenerator[SourceItem, None]:
        del self, query, limit
        async for item in fetch_atom_fixture(body):
            yield item

    monkeypatch.setattr(ArxivSource, "requires_network", False)
    monkeypatch.setattr(ArxivSource, "fetch", fake_fetch)

    out_dir = tmp_path / "mined"
    runs_root = tmp_path / "runs"

    async def run_mine() -> tuple[Path, ...]:
        with start_run(runs_dir=runs_root):
            cfg = MiningConfig(
                source_name="arxiv",
                query="ignored",
                limit=10,
                out_dir=out_dir,
                feature_name="Fixture run",
                network_enabled=False,
            )
            return await mine(cfg)

    paths = asyncio.run(run_mine())
    assert len(paths) == 1
    assert paths[0].is_file()
    assert "__" in paths[0].name and paths[0].suffix == ".feature"

    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    log = run_dirs[0] / "mining.jsonl"
    assert log.is_file()
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) >= 2
    assert any(x.get("source_id") == "2501.12345" for x in lines)
    assert any(x.get("emitted_path") for x in lines)
