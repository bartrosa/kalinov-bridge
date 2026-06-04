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
    # Cumulative recorded spend across all task results matches the actual
    # provider-billed total. Each obligation that initiates a call against an
    # over-budget guard still incurs the unit cost on the provider (the guard
    # raises only after ``record`` mutates state), and that cost is now
    # propagated into ``OracleOutcome.total_cost_usd`` via
    # ``BudgetExceededError.attempted_cost_usd`` so the user-visible summary
    # no longer silently drops the overrun spend. With 3 tasks × 1 obligation
    # × $0.000225 per call this is $0.000675.
    unit = Decimal("0.000225")
    total_spend = sum(
        (tr.total_cost_usd for tr in rr.task_results), Decimal("0")
    )
    assert total_spend == unit * 3, (
        "shared budget should attribute every billed call (including the "
        f"overrun calls) to total_cost_usd; got {total_spend}, expected "
        f"{unit * 3}"
    )


def test_unparseable_task_does_not_abort_suite(
    tmp_path: Path,
    fake_kalinov_config: KalinovConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single malformed .feature must not destroy prior tasks' results.

    Regression for data-loss bug where :class:`~gherkin.errors.GherkinParseError`
    raised in :meth:`EvalRunner.run` was re-raised as :class:`RuntimeError`,
    escaped the for-task loop, and exited ``_eval_async`` without writing any
    report. A paid suite with N completed tasks ahead of the bad one would
    drop every previously-collected ``TaskResult`` (the LLM had already been
    billed but the eval report — the actual user-visible artifact — never
    materialized, forcing a full re-run).

    Expected new behaviour: the malformed task is recorded as a
    ``PROVER_ERROR`` outcome, the runner keeps processing the rest of the
    suite, and the prior tasks' results survive in the returned
    ``RunResult``.
    """
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.feature"
    # Missing 'Feature:' makes the gherkin parser raise immediately on the
    # 'Scenario:' line — a realistic failure mode when a user edits a task
    # file in the middle of an eval run.
    bad.write_text(
        "this is not a valid gherkin file\nScenario: nope\n  Then never\n",
        encoding="utf-8",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: G\n  Scenario: S2\n    Then $2=2$\n",
        encoding="utf-8",
    )
    suite = Suite(
        suite_id="s_parse",
        description="",
        tasks=(
            Task(id="good", file=good.resolve(), expected=TaskExpected.EITHER, tags=()),
            Task(id="bad", file=bad.resolve(), expected=TaskExpected.EITHER, tags=()),
            Task(id="third", file=third.resolve(), expected=TaskExpected.EITHER, tags=()),
        ),
    )
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model="gpt-4o",
        seed=0,
        oracle=OracleConfig(max_repair_attempts=0),
        label="parse_err_resilience",
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
        runs_dir=tmp_path,
    )
    rr = asyncio.run(runner.run(suite, cfg))

    assert len(rr.task_results) == 3, (
        "all three tasks must produce a TaskResult; a single parse error "
        "must not abort the suite or discard previously-completed work"
    )
    ids = [tr.task.id for tr in rr.task_results]
    assert ids == ["good", "bad", "third"], (
        f"task ordering must be preserved across the parse error; got {ids}"
    )

    bad_tr = rr.task_results[1]
    assert any(o.kind is OracleOutcomeKind.PROVER_ERROR for o in bad_tr.outcomes), (
        "the malformed task must surface as a PROVER_ERROR outcome so the "
        f"eval report reflects the failure; got {[o.kind for o in bad_tr.outcomes]!r}"
    )
    diag = bad_tr.outcomes[0].diagnostic or ""
    assert "parse error" in diag.lower() or "bad.feature" in diag, (
        f"diagnostic must identify the failing file; got {diag!r}"
    )

    # The good tasks' outcomes must be preserved with non-error kinds.
    for tr in (rr.task_results[0], rr.task_results[2]):
        assert tr.outcomes, f"task {tr.task.id} must have at least one outcome"
        assert all(
            o.kind is not OracleOutcomeKind.PROVER_ERROR for o in tr.outcomes
        ), (
            f"good task {tr.task.id} should not be marked errored; "
            f"got {[o.kind for o in tr.outcomes]!r}"
        )


def _run_three_task_suite(
    *,
    bad_factory,
    tmp_path: Path,
    fake_kalinov_config: KalinovConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> RunResult:
    """Helper: build a 3-task suite where the middle task is broken by ``bad_factory``."""
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: F\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: G\n  Scenario: S2\n    Then $2=2$\n",
        encoding="utf-8",
    )
    bad_path = bad_factory(tmp_path)
    suite = Suite(
        suite_id="s_io",
        description="",
        tasks=(
            Task(id="good", file=good.resolve(), expected=TaskExpected.EITHER, tags=()),
            Task(id="bad", file=bad_path, expected=TaskExpected.EITHER, tags=()),
            Task(id="third", file=third.resolve(), expected=TaskExpected.EITHER, tags=()),
        ),
    )
    cfg = EvalConfig(
        prover_name="null",
        provider_name="fake",
        model="gpt-4o",
        seed=0,
        oracle=OracleConfig(max_repair_attempts=0),
        label="io_err_resilience",
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
        runs_dir=tmp_path,
    )
    return asyncio.run(runner.run(suite, cfg))


def test_non_utf8_task_file_does_not_abort_suite(
    tmp_path: Path,
    fake_kalinov_config: KalinovConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-UTF-8 ``.feature`` mid-suite must not destroy prior tasks' results.

    Regression for the I/O-layer counterpart of the
    :func:`test_unparseable_task_does_not_abort_suite` data-loss bug.
    ``parse_feature_file`` reads with ``encoding="utf-8"``, which raises
    :class:`UnicodeDecodeError` (a ``ValueError`` subclass — *not* a
    :class:`GherkinParseError`) on the first non-UTF-8 byte. Without the new
    handler the exception escapes the for-task loop, exits ``_eval_async``
    without writing the report, and silently drops every previously-completed
    ``TaskResult`` — the same blast radius as the parse-error bug (LLM
    already billed for the good tasks, but the user-visible artifact those
    costs paid for never materializes).

    Concrete trigger: an editor reopens a feature file in latin-1 and saves
    accented characters, or a merge conflict marker injects ``\\xff`` bytes,
    or the file was originally UTF-16. Suite validation in
    :func:`load_suite` succeeds because it only checks ``is_file()``; the
    decoding error surfaces only at per-task parse time.
    """

    def _write_latin1(tp: Path) -> Path:
        bad = tp / "bad.feature"
        # 0xff is never a valid UTF-8 start byte (a typical signature of a
        # file mistakenly saved as latin-1 / UTF-16-LE BOM).
        bad.write_bytes(
            b"# language: en\nFeature: bad\n  Scenario: \xff oops\n"
            b"    Then nope\n",
        )
        return bad.resolve()

    rr = _run_three_task_suite(
        bad_factory=_write_latin1,
        tmp_path=tmp_path,
        fake_kalinov_config=fake_kalinov_config,
        monkeypatch=monkeypatch,
    )

    assert len(rr.task_results) == 3, (
        "all three tasks must produce a TaskResult; a UnicodeDecodeError on "
        "the middle task must not abort the suite or discard previously-"
        f"completed work. got {len(rr.task_results)} task result(s)"
    )
    ids = [tr.task.id for tr in rr.task_results]
    assert ids == ["good", "bad", "third"], (
        f"task ordering must be preserved across the decode error; got {ids}"
    )

    bad_tr = rr.task_results[1]
    assert any(o.kind is OracleOutcomeKind.PROVER_ERROR for o in bad_tr.outcomes), (
        "the un-decodable task must surface as a PROVER_ERROR outcome so the "
        f"eval report reflects the failure; got {[o.kind for o in bad_tr.outcomes]!r}"
    )
    diag = (bad_tr.outcomes[0].diagnostic or "").lower()
    assert "utf-8" in diag or "decode" in diag or "bad.feature" in diag, (
        "diagnostic must identify the decoding failure / failing file; "
        f"got {diag!r}"
    )

    for tr in (rr.task_results[0], rr.task_results[2]):
        assert tr.outcomes, f"task {tr.task.id} must have at least one outcome"
        assert all(
            o.kind is not OracleOutcomeKind.PROVER_ERROR for o in tr.outcomes
        ), (
            f"good task {tr.task.id} should not be marked errored; "
            f"got {[o.kind for o in tr.outcomes]!r}"
        )


def test_missing_task_file_does_not_abort_suite(
    tmp_path: Path,
    fake_kalinov_config: KalinovConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vanished feature file mid-suite must not destroy prior tasks' results.

    Concrete trigger: ``load_suite`` validates each task's file at load time,
    but a long-running paid suite is wide open to mid-run file movement — a
    background editor save, a CI cleanup job, a ``git checkout`` on the
    feature corpus, or a watcher script can delete or rename the file
    between ``is_file()`` validation and ``parse_feature_file``'s
    ``read_text`` call. Pre-fix the resulting :class:`FileNotFoundError`
    (an :class:`OSError` subclass — *not* a :class:`GherkinParseError`)
    escapes the for-task loop and discards every prior ``TaskResult``.
    """

    def _stage_then_delete(tp: Path) -> Path:
        bad = tp / "bad.feature"
        bad.write_text(
            "# language: en\nFeature: B\n  Scenario: S\n    Then $1=1$\n",
            encoding="utf-8",
        )
        resolved = bad.resolve()
        # Simulate the file vanishing between suite-load validation and the
        # per-task parse. ``Task`` is already constructed with the resolved
        # path; deleting now reproduces the race.
        bad.unlink()
        return resolved

    rr = _run_three_task_suite(
        bad_factory=_stage_then_delete,
        tmp_path=tmp_path,
        fake_kalinov_config=fake_kalinov_config,
        monkeypatch=monkeypatch,
    )

    assert len(rr.task_results) == 3, (
        "all three tasks must produce a TaskResult; a missing file on the "
        "middle task must not abort the suite or discard previously-"
        f"completed work. got {len(rr.task_results)} task result(s)"
    )
    ids = [tr.task.id for tr in rr.task_results]
    assert ids == ["good", "bad", "third"], (
        f"task ordering must be preserved across the I/O error; got {ids}"
    )

    bad_tr = rr.task_results[1]
    assert any(o.kind is OracleOutcomeKind.PROVER_ERROR for o in bad_tr.outcomes), (
        "the missing-file task must surface as a PROVER_ERROR outcome; "
        f"got {[o.kind for o in bad_tr.outcomes]!r}"
    )
    diag = (bad_tr.outcomes[0].diagnostic or "").lower()
    assert "read" in diag or "filenotfound" in diag or "bad.feature" in diag, (
        f"diagnostic must identify the I/O failure / failing file; got {diag!r}"
    )

    for tr in (rr.task_results[0], rr.task_results[2]):
        assert tr.outcomes, f"task {tr.task.id} must have at least one outcome"
        assert all(
            o.kind is not OracleOutcomeKind.PROVER_ERROR for o in tr.outcomes
        ), (
            f"good task {tr.task.id} should not be marked errored; "
            f"got {[o.kind for o in tr.outcomes]!r}"
        )
