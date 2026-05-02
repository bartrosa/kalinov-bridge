"""Aggregate ``llm_calls.jsonl`` for the ``kalinov cost report`` command."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

GroupBy = Literal["none", "provider", "model", "day"]


@dataclass
class _Agg:
    total_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0


def _iter_llm_call_rows(run_dirs: list[Path]) -> Iterator[dict[str, Any]]:
    for rd in run_dirs:
        path = rd / "llm_calls.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def discover_run_dirs(*, runs_dir: Path, run_id: str | None) -> list[Path]:
    if run_id:
        p = runs_dir / run_id
        return [p] if p.is_dir() else []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def aggregate_run_directories(run_dirs: list[Path], *, group_by: GroupBy) -> dict[str, Any]:
    rows = list(_iter_llm_call_rows(run_dirs))
    if group_by == "none":
        agg = _Agg()
        for row in rows:
            _accumulate(agg, row)
        return {"totals": _agg_to_dict(agg), "groups": []}

    groups: dict[str, _Agg] = defaultdict(_Agg)
    for row in rows:
        key = _group_key(row, group_by)
        _accumulate(groups[key], row)

    out_groups = [
        {"key": k, **_agg_to_dict(v)} for k, v in sorted(groups.items(), key=lambda x: x[0])
    ]
    total = _Agg()
    for a in groups.values():
        _merge_total(total, a)
    return {"totals": _agg_to_dict(total), "groups": out_groups}


def _group_key(row: dict[str, Any], group_by: GroupBy) -> str:
    if group_by == "provider":
        return str(row.get("provider", ""))
    if group_by == "model":
        return str(row.get("model_id_resolved", ""))
    if group_by == "day":
        ts = int(row.get("ts_ms", 0))
        dt = datetime.fromtimestamp(ts / 1000.0, tz=UTC)
        return dt.strftime("%Y-%m-%d")
    return ""


def _accumulate(target: _Agg, row: dict[str, Any]) -> None:
    target.calls += 1
    target.total_usd += Decimal(str(row.get("cost_usd", "0")))
    u = row.get("usage") or {}
    target.input_tokens += int(u.get("input", 0))
    target.output_tokens += int(u.get("output", 0))
    target.reasoning_tokens += int(u.get("reasoning", 0))
    target.cache_read_tokens += int(u.get("cache_read", 0))
    target.cache_write_tokens += int(u.get("cache_write", 0))


def _merge_total(total: _Agg, part: _Agg) -> None:
    total.total_usd += part.total_usd
    total.input_tokens += part.input_tokens
    total.output_tokens += part.output_tokens
    total.reasoning_tokens += part.reasoning_tokens
    total.cache_read_tokens += part.cache_read_tokens
    total.cache_write_tokens += part.cache_write_tokens
    total.calls += part.calls


def _agg_to_dict(a: _Agg) -> dict[str, Any]:
    return {
        "total_usd": str(a.total_usd),
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
        "reasoning_tokens": a.reasoning_tokens,
        "cache_read_tokens": a.cache_read_tokens,
        "cache_write_tokens": a.cache_write_tokens,
        "calls": a.calls,
    }


def format_text_report(payload: dict[str, Any], *, group_by: GroupBy) -> str:
    lines: list[str] = []
    t = payload["totals"]
    lines.append("totals:")
    lines.append(
        f"  usd={t['total_usd']} calls={t['calls']} "
        f"tokens in/out/rsn/cr/cw={t['input_tokens']}/"
        f"{t['output_tokens']}/{t['reasoning_tokens']}/"
        f"{t['cache_read_tokens']}/{t['cache_write_tokens']}",
    )
    if group_by != "none" and payload.get("groups"):
        lines.append(f"by {group_by}:")
        for g in payload["groups"]:
            key = g["key"]
            lines.append(
                f"  {key}: usd={g['total_usd']} calls={g['calls']} "
                f"tokens {g['input_tokens']}/{g['output_tokens']}/"
                f"{g['reasoning_tokens']}",
            )
    return "\n".join(lines) + "\n"


def run_cost_report(
    *,
    runs_dir: Path,
    run_id: str | None,
    fmt: Literal["text", "json"],
    group_by: GroupBy,
) -> tuple[int, str]:
    """Return ``(exit_code, stdout_payload)``."""
    dirs = discover_run_dirs(runs_dir=runs_dir, run_id=run_id)
    if not dirs:
        return 1, ""
    # Optional manifest hints (not required for aggregation)
    gb: GroupBy = group_by
    if gb == "none":
        real_group: GroupBy = "none"
    elif gb == "provider":
        real_group = "provider"
    elif gb == "model":
        real_group = "model"
    else:
        real_group = "day"

    payload = aggregate_run_directories(dirs, group_by=real_group)
    if fmt == "json":
        return 0, json.dumps(payload, indent=2) + "\n"
    return 0, format_text_report(payload, group_by=real_group)


__all__ = [
    "aggregate_run_directories",
    "discover_run_dirs",
    "run_cost_report",
]
