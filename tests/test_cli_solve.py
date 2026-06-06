"""End-to-end tests for ``kalinov solve``."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from kalinov.cli import main
from kalinov.cli_core import SolveProgrammaticResult, run_solve_programmatic
from kalinov.llm.config import KalinovConfig, LLMProviderType, ProviderConfigEntry
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
    monkeypatch.setattr("kalinov.cli_core.make_client", lambda *_a, **_k: client)

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
    monkeypatch.setattr("kalinov.cli_core.make_client", lambda *_a, **_k: client)

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


def _solve_kalinov_config() -> KalinovConfig:
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


def _solve_paths(
    *,
    tmp_path: Path,
    paths: list[Path],
    runs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SolveProgrammaticResult:
    """Drive ``run_solve_programmatic`` directly with a fake LLM client.

    Returns the ``SolveProgrammaticResult`` so individual tests can assert on
    the structured outcome (which is what the MCP tool / CLI consumer
    actually receives — the bug under test is that this object never
    materialises when an I/O error escapes the per-file loop).
    """
    fake = FakeLLMClient()
    fake.set_queue(["theorem ok := rfl"] * 50)
    monkeypatch.setattr("kalinov.cli_core.make_client", lambda *_a, **_k: fake)
    monkeypatch.setattr(
        "kalinov.cli_core.load_llm_config",
        lambda _p=None: _solve_kalinov_config(),
    )

    return asyncio.run(
        run_solve_programmatic(
            paths=paths,
            runs_dir=runs,
            prover_name="null",
            provider="fake",
            model=None,
            llm_config_path=None,
            cache=None,
            max_repair_attempts=0,
            max_tokens=128,
            temperature=0.0,
            save_transcripts=False,
            max_cost_usd=None,
            echo=False,
        ),
    )


def test_solve_continues_past_non_utf8_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-UTF-8 file mid-list must not abort ``kalinov solve`` and lose
    the LLM-billed work done for prior feature files.

    Regression for the cli_core counterpart of the eval-runner data-loss
    bug fixed in PR #25. ``parse_feature_file`` reads with
    ``encoding="utf-8"``, which raises :class:`UnicodeDecodeError` (a
    ``ValueError`` subclass — *not* a :class:`GherkinParseError`) the
    moment it hits a stray ``\\xff`` byte from a botched editor save / merge
    or a UTF-16 BOM. Pre-fix that exception escaped the for-path loop,
    exited the ``with start_run(...)`` block without writing
    ``manifest.json``, and the caller never received a
    ``SolveProgrammaticResult`` describing the obligations that were
    actually billed.
    """
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: G\n  Scenario: GoodScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.feature"
    bad.write_bytes(
        b"# language: en\nFeature: bad\n  Scenario: \xff oops\n    Then nope\n",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: T\n  Scenario: ThirdScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    res = _solve_paths(
        tmp_path=tmp_path,
        paths=[good.resolve(), bad.resolve(), third.resolve()],
        runs=runs,
        monkeypatch=monkeypatch,
    )

    assert res.parse_failed is True, (
        "non-UTF-8 file must surface in the parse_failed flag (drives exit 2)"
    )
    obl_names = [o.obligation_name for o in res.outcomes]
    assert any("GoodScenario" in n for n in obl_names), (
        "good.feature must have produced obligation outcomes; got "
        f"{obl_names!r} (pre-fix the loop aborted before any outcomes "
        "were collected)"
    )
    assert any("ThirdScenario" in n for n in obl_names), (
        "third.feature (after the bad file) must still be processed; got "
        f"{obl_names!r} (pre-fix the UnicodeDecodeError aborted the loop)"
    )
    run_dir = runs / res.run_id
    assert (run_dir / "manifest.json").is_file(), (
        "manifest.json must be written even when a mid-list file fails to "
        "decode (pre-fix the with-start_run block exited via the unhandled "
        "exception and the manifest write was skipped)"
    )


def test_solve_continues_past_missing_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vanished feature file mid-list must not abort the solve loop.

    Trigger: user runs ``kalinov solve corpus/*.feature`` while a CI job /
    editor / version-control operation deletes one of the files between
    the CLI's ``is_file()`` check and ``parse_feature_file``'s
    ``read_text``. Pre-fix the resulting :class:`FileNotFoundError` (an
    :class:`OSError` subclass) escaped the for-path loop and discarded
    every previously-completed (already LLM-billed) ``SolveOutcomeEntry``.
    """
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: G\n  Scenario: GoodScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: T\n  Scenario: ThirdScenario\n    Then $1=1$\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.feature"
    # Path is computed but the file is never created (or imagine it was
    # deleted between the CLI's pre-flight ``is_file()`` and the for-loop
    # iteration that reaches it).

    runs = tmp_path / "runs"
    res = _solve_paths(
        tmp_path=tmp_path,
        paths=[good.resolve(), missing, third.resolve()],
        runs=runs,
        monkeypatch=monkeypatch,
    )

    assert res.parse_failed is True
    obl_names = [o.obligation_name for o in res.outcomes]
    assert any("GoodScenario" in n for n in obl_names), (
        f"good.feature must be processed before the missing one; got {obl_names!r}"
    )
    assert any("ThirdScenario" in n for n in obl_names), (
        "third.feature must still be processed after the missing one; "
        f"got {obl_names!r} (pre-fix FileNotFoundError aborted the loop)"
    )
    run_dir = runs / res.run_id
    assert (run_dir / "manifest.json").is_file()
