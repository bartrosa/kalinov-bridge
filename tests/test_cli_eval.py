"""CLI tests for ``kalinov eval``."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from kalinov.cli import main
from kalinov.eval.cli_impl import _eval_async
from kalinov.eval.matrix import ConfigMatrix
from kalinov.llm.budget import Budget
from kalinov.llm.config import KalinovConfig, LLMProviderType, ProviderConfigEntry
from kalinov.oracle.strategy import OracleConfig, OracleOutcomeKind
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


def test_budget_is_shared_across_matrix_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single ``--max-cost-usd`` cap must apply across every matrix config.

    Regression for bug where ``_eval_async`` built a fresh ``EvalRunner`` per
    expanded ``EvalConfig`` (each combination of prover × provider × seed ×
    oracle_config). The runner's ``BudgetGuard`` reset between configs, so the
    effective spend cap grew as ``cap × configs``: an experiment with
    ``seeds: [1, 2, 3]`` and ``budget.max_cost_usd: "10.00"`` could be billed
    up to $30 by the provider. See ``experiments/lean_basic_local.yaml``.

    Each fake call costs $0.000225 at bundled openai/gpt-4o pricing
    (10 input × $2.50/Mtok + 20 output × $10.00/Mtok). With a $0.0003 cap and
    three seed-configs, only the first config's call should be billed; the
    second and third configs must observe a budget-exceeded outcome.
    """
    feat = tmp_path / "f.feature"
    feat.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "suite_id: s\ndescription: ''\ntasks:\n"
        f"  - id: t\n    file: {feat.name}\n    expected: either\n",
        encoding="utf-8",
    )

    matrix = ConfigMatrix(
        provers=("null",),
        providers=(("fake", "gpt-4o"),),
        seeds=(1, 2, 3),
        oracle_configs=(OracleConfig(max_repair_attempts=0),),
    )
    llm_cfg = KalinovConfig(
        providers={
            "fake": ProviderConfigEntry(
                name="fake",
                type=LLMProviderType.OPENAI,
                api_key_env=None,
                base_url=None,
                default_model="gpt-4o",
            ),
        },
    )

    fake = FakeLLMClient()
    fake.set_queue(["theorem ok := rfl"] * 30)
    monkeypatch.setattr("kalinov.eval.runner.make_client", lambda *_a, **_k: fake)
    monkeypatch.setattr(
        "kalinov.eval.cli_impl.load_llm_config",
        lambda _p=None: llm_cfg,
    )

    budget = Budget(max_cost_usd=Decimal("0.00030"))

    _hf, results, _written, _md = asyncio.run(
        _eval_async(
            suite_yaml,
            matrix,
            llm_config_path=None,
            cache=None,
            budget=budget,
            runs_dir=tmp_path / "runs",
            out_dir=tmp_path / "out",
            formats=("json",),
            silent=True,
        ),
    )

    assert len(results) == 3, "all three seed configs should produce a RunResult"
    total_spent = sum(
        (tr.total_cost_usd for rr in results for tr in rr.task_results),
        Decimal("0"),
    )
    # Without ``BudgetGuard.ensure_not_exceeded`` (a separate pending fix), each
    # over-budget config still ships ONE billable provider call before the
    # guard refuses it post-hoc. Pre-fix, that overrun was silently dropped
    # from ``OracleOutcome.total_cost_usd`` so the summary lied about real
    # spend. Post-fix the overrun is folded into the per-task total via
    # ``BudgetExceededError.attempted_cost_usd``, so the aggregate equals the
    # provider's actual billing (3 configs × 1 obligation × $0.000225).
    unit = Decimal("0.000225")
    assert total_spent == unit * 3, (
        f"shared BudgetGuard must attribute every billed call to "
        f"total_cost_usd (was hiding overrun cost); got {total_spent}, "
        f"expected {unit * 3}"
    )

    # The first config got the single permitted call; the remaining two must
    # trip the cap and report a budget-exceeded outcome.
    outcomes_by_config = [tuple(o.kind for o in rr.task_results[0].outcomes) for rr in results]
    assert outcomes_by_config[0] == (OracleOutcomeKind.SOLVED,)
    assert all(
        OracleOutcomeKind.BUDGET_EXCEEDED in kinds for kinds in outcomes_by_config[1:]
    ), f"expected later configs to be budget-exceeded, got {outcomes_by_config}"
