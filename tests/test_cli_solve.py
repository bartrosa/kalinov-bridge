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


def test_solve_non_utf8_feature_does_not_abort_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a non-UTF-8 ``.feature`` mid-run must not crash ``kalinov solve``.

    Pre-fix, :func:`run_solve_programmatic` only caught
    :class:`GherkinParseError` around ``parse_feature_file``. A file saved in
    a non-UTF-8 encoding raises :class:`UnicodeDecodeError` from
    ``read_text(encoding="utf-8")``, which escaped the per-file loop, skipped
    the per-run ``manifest.json`` write, and replaced the documented exit
    code 2 with an uncaught Python traceback (exit 1). For multi-file
    invocations, every later file was silently dropped after already-billed
    LLM work on earlier files completed; the manifest summary that records
    that paid spend never landed on disk.

    The same exception escape also crashed the MCP ``solve`` tool handler,
    which only catches ``GherkinParseError``/``LLMError``/``ProverError`` and
    therefore severed the long-running server's connection for every other
    client.
    """
    cfg = tmp_path / "kalinov.config.yaml"
    cfg.write_text(
        "providers:\n"
        "  fakep:\n"
        "    type: openai_compat\n"
        "    base_url: http://127.0.0.1:9/v1\n"
        "    default_model: gpt-4o\n",
        encoding="utf-8",
    )
    good = tmp_path / "good.feature"
    good.write_text(
        "# language: en\nFeature: G\n  Scenario: S\n    Then we expect $1 = 1$\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad.feature"
    # 0xff is never a valid UTF-8 start byte (typical of a file mistakenly
    # saved as latin-1 or with a UTF-16 BOM / merge-conflict garbage).
    bad.write_bytes(
        b"# language: en\nFeature: B\n  Scenario: bad \xff oops\n    Then nope\n",
    )
    third = tmp_path / "third.feature"
    third.write_text(
        "# language: en\nFeature: T\n  Scenario: S3\n    Then we expect $3 = 3$\n",
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
            str(good),
            str(bad),
            str(third),
        ],
    )
    assert code == 2, (
        "non-UTF-8 file must be reported as a parse failure (exit 2), not "
        "crash the CLI with a traceback (exit 1). pre-fix this returned "
        "with an uncaught UnicodeDecodeError."
    )

    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1, (
        "the run must complete and persist its manifest even when one of "
        "the input files cannot be decoded; pre-fix the manifest write "
        f"was skipped. got run_dirs={run_dirs!r}"
    )
    manifest_path = run_dirs[0] / "manifest.json"
    assert manifest_path.is_file(), (
        "manifest.json must be written even when a malformed feature is "
        "encountered mid-run; pre-fix it was lost on UnicodeDecodeError."
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Both good.feature and third.feature contribute one obligation each.
    # Pre-fix, third.feature was silently skipped because the for-path
    # loop aborted on the bad file before reaching it.
    assert manifest["obligations_total"] == 2, (
        "files after the malformed one must still be processed; pre-fix "
        "they were silently dropped. got: "
        f"obligations_total={manifest['obligations_total']}"
    )
    assert manifest["obligations_solved"] == 2

    oracle_lines = (
        (run_dirs[0] / "oracle_loop.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(oracle_lines) >= 2, (
        "oracle should record one outcome per surviving obligation; pre-fix "
        f"third.feature was never reached. got lines={len(oracle_lines)}"
    )
