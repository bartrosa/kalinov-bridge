"""End-to-end tests for ``kalinov solve``."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from kalinov.cli import main
from tests.fixtures.fake_llm_client import FakeLLMClient


@pytest.fixture
def gauss_feature() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "gauss_sum.feature"


def test_solve_with_null_prover_always_ok(
    tmp_path: Path,
    gauss_feature: Path,
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
    client.set_queue(["theorem ok := rfl"] * 10)
    monkeypatch.setattr("kalinov.cli.make_client", lambda *_a, **_k: client)

    runs = tmp_path / "runs"
    code = main(
        [
            "solve",
            "--prover",
            "null",
            "--provider",
            "fakep",
            "--llm-config",
            str(cfg),
            "--runs-dir",
            str(runs),
            str(gauss_feature),
        ],
    )
    assert code == 0
    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert Decimal(manifest["total_cost_usd"]) > 0


def test_solve_with_budget_exceeded_exits_1(
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
    one_claim = tmp_path / "one.feature"
    one_claim.write_text(
        "# language: en\nFeature: One\n  Scenario: S\n    Then we expect $1 = 1$\n",
        encoding="utf-8",
    )
    client = FakeLLMClient()
    client.set_queue(["theorem ok := rfl"])
    monkeypatch.setattr("kalinov.cli.make_client", lambda *_a, **_k: client)

    runs = tmp_path / "runs"
    code = main(
        [
            "solve",
            "--prover",
            "null",
            "--provider",
            "fakep",
            "--llm-config",
            str(cfg),
            "--runs-dir",
            str(runs),
            "--max-cost-usd",
            "0",
            str(one_claim),
        ],
    )
    assert code == 1
    run_dir = next(runs.iterdir())
    oracle_lines = (run_dir / "oracle_loop.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert oracle_lines
    last = json.loads(oracle_lines[-1])
    assert last["outcome_so_far"] == "error"
