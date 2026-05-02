"""CLI tests for ``kalinov eval``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalinov.cli import main
from tests.fixtures.fake_llm_client import FakeLLMClient


def test_eval_cli_smoke_writes_reports(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "kalinov.config.yaml"
    cfg.write_text(
        "providers:\n"
        "  fakep:\n"
        "    type: openai_compat\n"
        "    base_url: http://127.0.0.1:9/v1\n"
        "    default_model: gpt-4o\n",
        encoding="utf-8",
    )
    client = FakeLLMClient()
    client.set_queue(["theorem ok := rfl"] * 50)
    monkeypatch.setattr("kalinov.eval.runner.make_client", lambda *_a, **_k: client)

    out = tmp_path / "rep"
    suite = repo_root / "evals" / "suites" / "smoke.yaml"
    code = main(
        [
            "eval",
            "--suite",
            str(suite),
            "--prover",
            "null",
            "--provider",
            "fakep",
            "--llm-config",
            str(cfg),
            "--out",
            str(out),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert code == 0
    assert (out / "report.json").is_file()
    assert (out / "report.md").is_file()
    body = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert body["aggregate_metrics"]
