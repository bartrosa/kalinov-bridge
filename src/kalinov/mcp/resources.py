"""Read-only MCP resources over ``runs/`` and redacted config."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from kalinov.llm.config import ConfigError, load_config
from kalinov.mcp.runtime import MCPServerConfig

_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.match(run_id):
        msg = "invalid run_id (expected 12 lowercase hex chars)"
        raise ValueError(msg)


def resource_list_runs(cfg: MCPServerConfig) -> str:
    root = cfg.runs_dir.resolve()
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return json.dumps([], indent=2)
    for p in root.iterdir():
        if not p.is_dir():
            continue
        rid = p.name
        if not _RUN_ID_RE.match(rid):
            continue
        manifest = p / "manifest.json"
        started = ""
        cost = ""
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                cost = str(data.get("total_cost_usd", ""))
                s = data.get("started_at")
                if s is not None:
                    started = str(s)
            except (json.JSONDecodeError, OSError):
                pass
        if not started:
            started = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()
        rows.append(
            {
                "run_id": rid,
                "started_at": started,
                "total_cost_usd": cost,
                "path": str(p),
            },
        )
    rows.sort(key=lambda r: r["started_at"], reverse=True)
    return json.dumps(rows, indent=2)


def resource_run_manifest(cfg: MCPServerConfig, run_id: str) -> str:
    validate_run_id(run_id)
    path = cfg.runs_dir.resolve() / run_id / "manifest.json"
    if not path.is_file():
        return "{}"
    return path.read_text(encoding="utf-8")


def resource_run_llm_calls(cfg: MCPServerConfig, run_id: str, limit: int) -> str:
    validate_run_id(run_id)
    path = cfg.runs_dir.resolve() / run_id / "llm_calls.jsonl"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[-limit:] if limit > 0 else lines
    return "\n".join(chunk) + ("\n" if chunk else "")


def resource_run_oracle_loop(cfg: MCPServerConfig, run_id: str, limit: int) -> str:
    validate_run_id(run_id)
    path = cfg.runs_dir.resolve() / run_id / "oracle_loop.jsonl"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[-limit:] if limit > 0 else lines
    return "\n".join(chunk) + ("\n" if chunk else "")


def resource_run_transcript(cfg: MCPServerConfig, run_id: str, name: str) -> str:
    validate_run_id(run_id)
    if not _SAFE_NAME.match(name):
        raise ValueError("invalid transcript name")
    path = cfg.runs_dir.resolve() / run_id / "transcripts" / name
    if path.is_dir() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def resource_config_summary(cfg: MCPServerConfig) -> str:
    try:
        kc = load_config(cfg.kalinov_config_path)
    except ConfigError as exc:
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)
    prov: dict[str, Any] = {}
    for name, e in kc.providers.items():
        prov[name] = {
            "type": str(e.type),
            "default_model": e.default_model,
            "api_key_env": ("<redacted>" if e.api_key_env else None),
            "base_url": e.base_url,
        }
    return json.dumps({"ok": True, "providers": prov}, indent=2)


__all__ = [
    "resource_config_summary",
    "resource_list_runs",
    "resource_run_llm_calls",
    "resource_run_manifest",
    "resource_run_oracle_loop",
    "resource_run_transcript",
    "validate_run_id",
]
