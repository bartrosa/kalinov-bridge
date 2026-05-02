"""Single-shot mining pipeline: source → extractor → ``.feature`` emission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kalinov.mining.emit import emit_feature_file
from kalinov.mining.errors import MiningError
from kalinov.mining.extractors.base import CandidateClaim, Extractor
from kalinov.mining.extractors.heuristic import HeuristicExtractor
from kalinov.mining.sources.arxiv import ArxivSource
from kalinov.mining.sources.base import Source
from kalinov.telemetry.context import active_run
from kalinov.telemetry.jsonl import append_jsonl_record


@dataclass(frozen=True, slots=True)
class MiningConfig:
    source_name: str
    query: str
    limit: int = 10
    extractor_name: str = "heuristic"
    out_dir: Path = Path("corpus/mined")
    feature_name: str = "Mined claims"
    network_enabled: bool = False
    """If False, sources that need the network raise :exc:`MiningError`."""


def _make_extractor(name: str) -> Extractor:
    key = name.strip().lower()
    if key == "heuristic":
        return HeuristicExtractor()
    raise MiningError(f"unknown extractor {name!r}")


def _make_source(name: str) -> Source:
    key = name.strip().lower()
    if key == "arxiv":
        return ArxivSource()
    raise MiningError(f"unknown source {name!r}")


def _mining_log(record: dict[str, Any]) -> None:
    run = active_run()
    if run is None:
        return
    path = run.run_dir / "mining.jsonl"
    append_jsonl_record(path, record)


async def mine(config: MiningConfig) -> tuple[Path, ...]:
    """Fetch items, extract candidates, emit one ``.feature`` path.

    Writes JSONL lines to ``mining.jsonl`` on the active run when present.
    """
    source = _make_source(config.source_name)
    if not config.network_enabled and source.requires_network:
        raise MiningError(
            "network disabled in MiningConfig but source "
            f"{config.source_name!r} requires network access "
            "(set network_enabled=True after opting in)",
        )

    extractor = _make_extractor(config.extractor_name)

    all_claims: list[CandidateClaim] = []

    async for item in source.fetch(config.query, limit=config.limit):
        claims = list(extractor.extract(item))
        all_claims.extend(claims)
        _mining_log(
            {
                "source": source.name,
                "source_id": item.source_id,
                "url": item.url,
                "license_known": item.license is not None and bool(str(item.license).strip()),
                "extractor": extractor.name,
                "candidate_count": len(claims),
                "emitted_path": None,
            },
        )

    if not all_claims:
        return ()

    out = emit_feature_file(
        all_claims,
        feature_name=config.feature_name,
        out_dir=config.out_dir,
    )

    _mining_log(
        {
            "source": source.name,
            "source_id": "*",
            "url": "",
            "license_known": True,
            "extractor": extractor.name,
            "candidate_count": len(all_claims),
            "emitted_path": str(out),
        },
    )

    return (out,)
