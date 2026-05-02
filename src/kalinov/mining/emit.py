"""Emit Math-Gherkin ``.feature`` files from extracted claim candidates."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from kalinov.mining.errors import MiningError
from kalinov.mining.extractors.base import CandidateClaim


def _sha8_join(candidates: Sequence[CandidateClaim]) -> str:
    joined = "\n".join(c.text for c in candidates)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]


def _iso_z(dt: datetime) -> str:
    s = dt.isoformat()
    if s.endswith("+00:00"):
        return s[:-6] + "Z"
    return s


def _safe_feature_stub(name: str) -> str:
    x = re.sub(r"[^\w\-]+", "_", name.strip())
    return x.strip("_") or "mined"


def _truncate_title(title: str, max_len: int = 48) -> str:
    t = " ".join(title.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def emit_feature_file(
    candidates: Sequence[CandidateClaim],
    *,
    feature_name: str,
    out_dir: Path,
) -> Path:
    """Write one ``.feature`` file (one Scenario per candidate).

    File name: ``<feature_stub>__<sha8>.feature`` where *sha8* hashes joined
    candidate texts. Refuses to write when attribution fields are incomplete.
    """
    if not candidates:
        raise MiningError("refusing to emit: no candidates (cannot attach provenance)")

    out_dir.mkdir(parents=True, exist_ok=True)
    stub = _safe_feature_stub(feature_name)
    sha8 = _sha8_join(candidates)
    path = out_dir / f"{stub}__{sha8}.feature"

    for c in candidates:
        it = c.source_item
        if not (it.url and it.url.strip()):
            msg = f"incomplete attribution: missing url for source_id={it.source_id!r}"
            raise MiningError(msg)
        if not it.license or not str(it.license).strip():
            msg = f"incomplete attribution: missing license for source_id={it.source_id!r}"
            raise MiningError(msg)

    first = candidates[0].source_item
    src_tag = first.metadata.get("source", first.metadata.get("SOURCE", "unknown"))
    tag = src_tag.lower().replace(" ", "_") if isinstance(src_tag, str) else "unknown"

    when = _iso_z(first.retrieved_at)
    desc = (
        f"  Mined from {tag} on {when}.\n"
        f"  See @attribution comments below for per-claim provenance."
    )

    lines: list[str] = [
        "# language: en",
        f"@mined @{tag}",
        f"Feature: {feature_name}",
        desc,
        "",
    ]

    for c in candidates:
        it = c.source_item
        src = it.metadata.get("source", tag)
        src_s = str(src) if src else tag
        lic = it.license or ""
        scen = f"{c.kind} from {_truncate_title(it.title)}"
        body = _one_line(c.text)
        lines.extend(
            [
                "  # @attribution",
                f"  #   source: {src_s}",
                f"  #   url: {it.url.strip()}",
                f"  #   license: {lic}",
                f"  #   retrieved_at: {_iso_z(it.retrieved_at)}",
                f"  Scenario: {scen}",
                f"    Then {body}",
                "",
            ],
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


__all__ = ["emit_feature_file"]
