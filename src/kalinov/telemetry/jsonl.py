"""Append-only JSONL backed by a single global lock (cross-thread safe)."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a single line to *path* (creates parent dirs)."""
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
