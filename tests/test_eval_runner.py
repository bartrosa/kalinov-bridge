"""EvalRunner integration tests."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.eval.matrix import EvalConfig
from kalinov.eval.runner import EvalRunner, RunResult
from kalinov.eval.suite import Suite, Task, TaskExpected
from kalinov.llm.budget import Budget
from kalinov.llm.config import KalinovConfig, LLMProviderType, ProviderConfigEntry
from kalinov.oracle.strategy import OracleConfig, OracleOutcomeKind
from tests.fixtures.fake_llm_client import FakeLLMClient


@pytest.fixture
def fake_kalinov_config() -> KalinovConfig:
    return KalinovConfig(
        providers={
            "fake": ProviderConfigEntry(
                name="fake",
                type=LLMProviderType.OPENAI_COMPAT,
                api_key_env=None,
                base_url="http://127.0.0.1:9/v1",
                default_model="gpt-4o",
            ),
        },
    )


@pytest.fixture
def tiny_suite(tmp_path: Path) -> Suite:
    feat = tmp_path / "one.feature"
    feat.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then we expect $1=1$\n",
        encoding="utf-8",
    )
    return Suite(
        suite_id="t",
        description="",
        tasks=(
            Task(
                id="one",
                file=feat.resolve(),
                expected=TaskExpected.EITHER,
                tags=(),
            ),
        ),
    )


async def _run_with_fake(
    suite: Suite,
    cfg: EvalConfig,
    *,
    fake_kalinov_config: KalinovConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> RunResult:
    fake = FakeLLMClient()
    fake.set_queue(["theorem ok := rfl"] * 20)
    monkeypatch.setattr(
        "kalinov.eval.runner.make_client",
        lambda *_a, **_k: fake,
    )
    runner = EvalRunner(
        kalinov_config=fake_kalinov_config,
        pricing=load_default_catalogue(),
        runs_dir=tmp_path,
    )
    return await runner.run(suite, cfg)


def test_runner_produces_outcomes_and_manifests(
    tiny_suite: Suite,
    fake_kalinov_config: KalinovConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model="gpt-4o",
        seed=0,
        oracle=OracleConfig(max_repair_attempts=1),
        label="test",
    )
    rr = asyncio.run(
        _run_with_fake(
            tiny_suite,
            cfg,
            fake_kalinov_config=fake_kalinov_config,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        ),
    )
    assert len(rr.task_results) == 1
    tr = rr.task_results[0]
    assert tr.obligations_total >= 1
    assert all(len(o.attempts) >= 1 for o in tr.outcomes if o.kind is OracleOutcomeKind.SOLVED)
    manifest = json.loads(
        (tmp_path / tr.telemetry_run_id / "manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest["eval_task_id"] == "one"
    assert (tmp_path / tr.telemetry_run_id / "llm_calls.jsonl").is_file()


def test_per_task_run_id_unique(
    tiny_suite: Suite,
    fake_kalinov_config: KalinovConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model=None,
        seed=1,
        oracle=OracleConfig(),
        label="l",
    )
    rr = asyncio.run(
        _run_with_fake(
            tiny_suite,
            cfg,
            fake_kalinov_config=fake_kalinov_config,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        ),
    )
    assert rr.task_results[0].telemetry_run_id


def test_matched_expected_solved(
    tmp_path: Path,
    fake_kalinov_config: KalinovConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feat = tmp_path / "t.feature"
    feat.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    suite = Suite(
        suite_id="s",
        description="",
        tasks=(
            Task(
                id="t",
                file=feat,
                expected=TaskExpected.SOLVED,
                tags=(),
            ),
        ),
    )
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model="gpt-4o",
        seed=0,
        oracle=OracleConfig(),
        label="x",
    )
    rr = asyncio.run(
        _run_with_fake(
            suite,
            cfg,
            fake_kalinov_config=fake_kalinov_config,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        ),
    )
    assert rr.task_results[0].matched_expected


def test_budget_is_shared_across_tasks(
    tmp_path: Path,
    fake_kalinov_config: KalinovConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single ``--max-cost-usd`` cap must apply across the whole eval run.

    Regression for bug where ``EvalRunner`` allocated a fresh ``BudgetGuard``
    per task, allowing total spend to grow as ``cap × tasks`` and silently
    over-charging users (e.g. the bundled ``lean_basic_local.yaml`` experiment
    has 10 tasks × 3 seeds, so a "$10" cap would actually permit up to $300).
    """
    # Each fake call costs $0.000225 (10 input × $2.50/Mtok + 20 output ×
    # $10.00/Mtok at the bundled openai/gpt-4o pricing). With a $0.0003 cap
    # and three tasks, only the first task's call should fit; the second and
    # third tasks must observe a budget-exceeded outcome.
    feat = tmp_path / "f.feature"
    feat.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    tasks: tuple[Task, ...] = tuple(
        Task(
            id=f"t{i}",
            file=feat.resolve(),
            expected=TaskExpected.EITHER,
            tags=(),
        )
        for i in range(3)
    )
    suite = Suite(suite_id="s_budget", description="", tasks=tasks)
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model="gpt-4o",
        seed=0,
        oracle=OracleConfig(max_repair_attempts=0),
        label="budget_share",
    )

    fake = FakeLLMClient()
    fake.set_queue(["theorem ok := rfl"] * 10)
    monkeypatch.setattr(
        "kalinov.eval.runner.make_client",
        lambda *_a, **_k: fake,
    )

    runner = EvalRunner(
        kalinov_config=fake_kalinov_config,
        pricing=load_default_catalogue(),
        budget=Budget(max_cost_usd=Decimal("0.0003")),
        runs_dir=tmp_path,
    )
    rr = asyncio.run(runner.run(suite, cfg))

    assert len(rr.task_results) == 3, "expected one TaskResult per suite task"
    kinds_by_task = [
        tuple(o.kind for o in tr.outcomes) for tr in rr.task_results
    ]
    # First task's only obligation should have spent under the cap and solved.
    assert OracleOutcomeKind.SOLVED in kinds_by_task[0], (
        "first task should fit under shared budget; got "
        f"{kinds_by_task[0]!r}"
    )
    # Subsequent tasks must observe BUDGET_EXCEEDED — meaning the guard kept
    # accumulating across task boundaries instead of being reset.
    later_kinds = kinds_by_task[1] + kinds_by_task[2]
    assert OracleOutcomeKind.BUDGET_EXCEEDED in later_kinds, (
        "shared budget guard should have tripped on later tasks; got "
        f"{kinds_by_task!r} (per-task BudgetGuard would let every task spend "
        "the full cap independently)."
    )
    # Sanity: cumulative recorded spend across all task results stays at or
    # under one extra increment beyond the cap (the call that triggers the
    # guard still counts toward the recorded total).
    total_spend = sum(
        (tr.total_cost_usd for tr in rr.task_results), Decimal("0")
    )
    assert total_spend <= Decimal("0.0005"), (
        f"shared budget should bound total spend; got {total_spend}"
    )
