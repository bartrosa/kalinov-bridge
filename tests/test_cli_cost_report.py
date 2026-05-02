"""CLI tests for ``kalinov cost report``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalinov.cli import main


def test_empty_runs_dir(tmp_path: Path) -> None:
    rc = main(["cost", "report", "--runs-dir", str(tmp_path)])
    assert rc == 1


def test_totals_match_lines(tmp_path: Path) -> None:
    run = tmp_path / "abc123"
    run.mkdir()
    line1 = (
        '{"ts_ms":0,"provider":"openai","model_id_resolved":"gpt-4o",'
        '"usage":{"input":2,"output":2,"reasoning":0,"cache_read":0,"cache_write":0},'
        '"cost_usd":"0.10","cost_detail":{"input_usd":"0.05","output_usd":"0.05",'
        '"reasoning_usd":"0","cache_read_usd":"0","cache_write_usd":"0",'
        '"pricing_source":"catalogue"},"latency_ms":1,"cache_hit":false,'
        '"error_code":null,"extras_summary":{}}'
    )
    line2 = (
        '{"ts_ms":0,"provider":"openai","model_id_resolved":"gpt-4o",'
        '"usage":{"input":0,"output":0,"reasoning":0,"cache_read":0,"cache_write":0},'
        '"cost_usd":"0.00","cost_detail":{"input_usd":"0","output_usd":"0",'
        '"reasoning_usd":"0","cache_read_usd":"0","cache_write_usd":"0",'
        '"pricing_source":"cache"},"latency_ms":0,"cache_hit":true,'
        '"error_code":null,"extras_summary":{}}'
    )
    (run / "llm_calls.jsonl").write_text(line1 + "\n" + line2 + "\n", encoding="utf-8")

    rc = main(["cost", "report", "--runs-dir", str(tmp_path), "--format", "json"])
    assert rc == 0


def test_json_structure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "r1"
    run.mkdir()
    (run / "llm_calls.jsonl").write_text(
        '{"ts_ms":1000,"provider":"a","model_id_resolved":"m",'
        '"usage":{"input":1,"output":1,"reasoning":0,"cache_read":0,"cache_write":0},'
        '"cost_usd":"0.5","cost_detail":{"input_usd":"0.5","output_usd":"0",'
        '"reasoning_usd":"0","cache_read_usd":"0","cache_write_usd":"0",'
        '"pricing_source":"x"},"latency_ms":1,"cache_hit":false,'
        '"error_code":null,"extras_summary":{}}\n',
        encoding="utf-8",
    )
    main(["cost", "report", "--runs-dir", str(tmp_path), "--format", "json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["totals"]["calls"] == 1
    assert payload["totals"]["total_usd"] == "0.5"


def test_by_provider_grouping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("r1", "r2"):
        d = tmp_path / name
        d.mkdir()
        rows = [
            '{"ts_ms":0,"provider":"anthropic","model_id_resolved":"x",'
            '"usage":{"input":1,"output":0,"reasoning":0,"cache_read":0,"cache_write":0},'
            '"cost_usd":"1.0","cost_detail":{"input_usd":"1","output_usd":"0",'
            '"reasoning_usd":"0","cache_read_usd":"0","cache_write_usd":"0",'
            '"pricing_source":"c"},"latency_ms":0,"cache_hit":false,'
            '"error_code":null,"extras_summary":{}}',
        ]
        (d / "llm_calls.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    main(["cost", "report", "--runs-dir", str(tmp_path), "--format", "json", "--by-provider"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["groups"]) == 1
    assert payload["groups"][0]["key"] == "anthropic"
