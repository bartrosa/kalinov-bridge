"""CLI tests for ``kalinov eval``."""

from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from kalinov.cli import main
from kalinov.eval.cli_impl import _eval_async, _merge_experiment_cli
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
    cap = Decimal("0.00030")
    assert total_spent <= cap, (
        f"total cumulative spend {total_spent} exceeds budget cap {cap}; "
        f"BudgetGuard is being reset between matrix configs"
    )

    # The first config got the single permitted call; the remaining two must
    # trip the cap and report a budget-exceeded outcome.
    outcomes_by_config = [tuple(o.kind for o in rr.task_results[0].outcomes) for rr in results]
    assert outcomes_by_config[0] == (OracleOutcomeKind.SOLVED,)
    assert all(
        OracleOutcomeKind.BUDGET_EXCEEDED in kinds for kinds in outcomes_by_config[1:]
    ), f"expected later configs to be budget-exceeded, got {outcomes_by_config}"


def _exp_args_with_max_cost(config_file: Path, *, max_cost_usd: str) -> argparse.Namespace:
    """Argparse Namespace matching ``kalinov eval --config-file ... --max-cost-usd ...``."""
    return argparse.Namespace(
        config_file=str(config_file),
        prover=None,
        provider=None,
        model=None,
        seed=None,
        max_repair_attempts=None,
        max_tokens=None,
        temperature=None,
        max_cost_usd=max_cost_usd,
        out=None,
    )


def test_cli_max_cost_usd_preserves_other_budget_caps(
    repo_root: Path, tmp_path: Path
) -> None:
    """``--max-cost-usd`` must override only the cost cap, not wipe the budget block.

    Regression for a bug where ``_merge_experiment_cli`` rebuilt ``Budget``
    from scratch when the user passed ``--max-cost-usd`` on the command line,
    silently dropping ``max_total_tokens`` and ``max_calls`` that came from
    the experiment YAML's ``budget:`` block.

    Concrete trigger: a team pins ``max_calls: 100`` in
    ``experiments/foo.yaml`` to bound runaway repair loops on a cheap model,
    then runs ``kalinov eval --config-file experiments/foo.yaml
    --max-cost-usd 5.00`` to additionally cap real-money spend. With the bug,
    the ``max_calls`` cap was silently lost and the run could ship thousands
    of provider requests inside the $5 envelope before the cost cap tripped.
    """
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        f"suite: {repo_root / 'evals' / 'suites' / 'smoke.yaml'}\n"
        "matrix:\n"
        "  provers: ['null']\n"
        "  providers:\n"
        "    - { name: a }\n"
        "  seeds: [0]\n"
        "  oracle_configs:\n"
        "    - {}\n"
        "out: ./out\n"
        "budget:\n"
        "  max_total_tokens: 12345\n"
        "  max_calls: 100\n",
        encoding="utf-8",
    )
    args = _exp_args_with_max_cost(exp_yaml, max_cost_usd="5.00")
    _suite_path, _matrix, budget, _out_dir = _merge_experiment_cli(exp_yaml, args)

    assert budget is not None
    assert budget.max_cost_usd == Decimal("5.00"), (
        "--max-cost-usd must apply the requested cost cap"
    )
    assert budget.max_total_tokens == 12345, (
        "max_total_tokens from experiment YAML must survive --max-cost-usd; "
        "dropping it silently removes a user-configured safety cap"
    )
    assert budget.max_calls == 100, (
        "max_calls from experiment YAML must survive --max-cost-usd; "
        "dropping it silently removes a user-configured safety cap that "
        "bounds runaway repair loops"
    )


def test_cli_max_cost_usd_with_no_yaml_budget(
    repo_root: Path, tmp_path: Path
) -> None:
    """``--max-cost-usd`` works when the experiment file has no ``budget:`` block."""
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        f"suite: {repo_root / 'evals' / 'suites' / 'smoke.yaml'}\n"
        "matrix:\n"
        "  provers: ['null']\n"
        "  providers:\n"
        "    - { name: a }\n"
        "  seeds: [0]\n"
        "  oracle_configs:\n"
        "    - {}\n"
        "out: ./out\n",
        encoding="utf-8",
    )
    args = _exp_args_with_max_cost(exp_yaml, max_cost_usd="2.50")
    _suite_path, _matrix, budget, _out_dir = _merge_experiment_cli(exp_yaml, args)

    assert budget is not None
    assert budget.max_cost_usd == Decimal("2.50")
    assert budget.max_total_tokens is None
    assert budget.max_calls is None
