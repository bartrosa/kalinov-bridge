"""MCP resource handler tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalinov.mcp.resources import (
    resource_config_summary,
    resource_list_runs,
    resource_run_llm_calls,
    resource_run_manifest,
    resource_run_transcript,
    validate_run_id,
)
from kalinov.mcp.runtime import MCPServerConfig


def test_list_runs_returns_known_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    d1 = runs / "aaaaaaaaaaaa"
    d2 = runs / "bbbbbbbbbbbb"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    (d1 / "manifest.json").write_text(
        json.dumps(
            {
                "total_cost_usd": "1.0",
                "started_at": "2024-01-01T00:00:00+00:00",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    (d2 / "manifest.json").write_text(
        json.dumps(
            {
                "total_cost_usd": "2.0",
                "started_at": "2024-06-01T00:00:00+00:00",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = MCPServerConfig(transport="stdio", runs_dir=runs)
    out = json.loads(resource_list_runs(cfg))
    assert [r["run_id"] for r in out] == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]


def test_run_manifest_path_traversal_rejected() -> None:
    with pytest.raises(ValueError, match="run_id"):
        validate_run_id("../etc")


def test_jsonl_resource_respects_limit(tmp_path: Path) -> None:
    rid = "cccccccccccc"
    p = tmp_path / "runs" / rid
    p.mkdir(parents=True)
    lines = [json.dumps({"i": i}) for i in range(500)]
    (p / "llm_calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg = MCPServerConfig(transport="stdio", runs_dir=tmp_path / "runs")
    text100 = resource_run_llm_calls(cfg, rid, 100)
    assert len(text100.splitlines()) == 100
    text500 = resource_run_llm_calls(cfg, rid, 500)
    assert len(text500.splitlines()) == 500


def test_config_summary_redacts_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sekrit-token-unique-xyz"
    cfg_yml = tmp_path / "kalinov.config.yaml"
    cfg_yml.write_text(
        "providers:\n"
        "  p1:\n"
        "    type: anthropic\n"
        "    default_model: claude\n"
        "    api_key_env: MY_KEY_VAR\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_KEY_VAR", secret)
    cfg = MCPServerConfig(
        transport="stdio",
        runs_dir=tmp_path / "runs",
        kalinov_config_path=cfg_yml,
    )
    out = resource_config_summary(cfg)
    assert secret not in out
    assert "<redacted>" in out


def test_transcript_resource_safe_filename(tmp_path: Path) -> None:
    cfg = MCPServerConfig(transport="stdio", runs_dir=tmp_path / "runs")
    rid = "dddddddddddd"
    (tmp_path / "runs" / rid / "transcripts").mkdir(parents=True)
    with pytest.raises(ValueError, match="transcript"):
        resource_run_transcript(cfg, rid, "../x")


def test_run_manifest_reads_bytes(tmp_path: Path) -> None:
    rid = "eeeeeeeeeeee"
    d = tmp_path / "runs" / rid
    d.mkdir(parents=True)
    man = {"run_id": rid, "total_cost_usd": "0"}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    cfg = MCPServerConfig(transport="stdio", runs_dir=tmp_path / "runs")
    raw = resource_run_manifest(cfg, rid)
    assert json.loads(raw)["run_id"] == rid
