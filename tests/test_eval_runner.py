"""EvalRunner integration tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kalinov.cost.catalogue import load_default_catalogue
from kalinov.eval.matrix import EvalConfig
from kalinov.eval.runner import EvalRunner, RunResult
from kalinov.eval.suite import Suite, Task, TaskExpected
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
