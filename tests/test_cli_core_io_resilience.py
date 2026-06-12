"""I/O-layer resilience for ``run_check_programmatic`` / ``run_solve_programmatic``.

Regression coverage for the cli_core counterpart of the
:class:`~kalinov.eval.runner.EvalRunner` data-loss bug fixed in commit
``59122b9``.

``parse_feature_file`` calls ``Path.read_text(encoding="utf-8")``, which can
raise:

  * :class:`UnicodeDecodeError` — file saved in a non-UTF-8 encoding (UTF-16
    BOM, latin-1, stray bytes from a botched editor save / merge marker).
  * :class:`OSError` subclasses — ``FileNotFoundError`` /
    ``PermissionError`` / ``IsADirectoryError`` / generic I/O when the file
    races against an editor save, CI cleanup, ``git checkout``, or watcher
    between the CLI's ``is_file()`` pre-check and the per-file parse.

The CLI's ``except GherkinParseError`` branch is the bound on per-file
damage. Without an additional ``except UnicodeDecodeError`` / ``except
OSError`` arm, the exception escapes the per-file loop. For
``kalinov solve`` that means previously-billed LLM work has no
``manifest.json`` summary, no roll-up print, and the user sees an unhandled
traceback instead of the per-file outcomes they paid for.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kalinov.cli_core import (
    check_exit_code,
    run_check_programmatic,
    run_solve_programmatic,
    solve_exit_code,
)
from kalinov.provers import NullProver, NullProverConfig, NullProverMode
from tests.fixtures.fake_llm_client import FakeLLMClient


def _good_feature(p: Path, name: str) -> Path:
    p.write_text(
        f"# language: en\nFeature: {name}\n  Scenario: S\n    Then we expect $1=1$\n",
        encoding="utf-8",
    )
    return p


def _latin1_feature(p: Path) -> Path:
    # 0xff is never a valid UTF-8 start byte; this mirrors a file mistakenly
    # saved as latin-1 / UTF-16-LE BOM / corrupted by a merge marker.
    p.write_bytes(
        b"# language: en\nFeature: bad\n  Scenario: \xff oops\n    Then nope\n",
    )
    return p


def _staged_then_deleted_feature(p: Path) -> Path:
    # The CLI pre-validates ``is_file()`` but the file can be moved between
    # that check and ``parse_feature_file``'s ``read_text``. Construct the
    # path object referring to a now-vanished file.
    p.write_text(
        "# language: en\nFeature: gone\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )
    p.unlink()
    return p


# ---------------------------------------------------------------------------
# run_check_programmatic
# ---------------------------------------------------------------------------


def test_check_handles_non_utf8_file_without_crash(tmp_path: Path) -> None:
    good = _good_feature(tmp_path / "good.feature", "Good")
    bad = _latin1_feature(tmp_path / "bad.feature")
    third = _good_feature(tmp_path / "third.feature", "Third")

    prover = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))
    res = run_check_programmatic(
        prover,
        [good, bad, third],
        "null",
        tmp_path / "runs",
        echo=False,
    )
    assert res.parse_failed is True, (
        "non-UTF-8 file must surface as parse_failed=True so the CLI exits 2; "
        "this is the same exit contract the GherkinParseError path uses."
    )
    assert res.total_obligations == 2, (
        "the two good files' obligations must still be checked; the bad file "
        "must not abort the loop. got total_obligations="
        f"{res.total_obligations}"
    )
    assert res.total_ok == 2
    assert check_exit_code(res) == 2


def test_check_handles_missing_file_race_without_crash(tmp_path: Path) -> None:
    good = _good_feature(tmp_path / "good.feature", "Good")
    missing = _staged_then_deleted_feature(tmp_path / "missing.feature")
    third = _good_feature(tmp_path / "third.feature", "Third")

    prover = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))
    res = run_check_programmatic(
        prover,
        [good, missing, third],
        "null",
        tmp_path / "runs",
        echo=False,
    )
    assert res.parse_failed is True
    assert res.total_obligations == 2
    assert res.total_ok == 2


# ---------------------------------------------------------------------------
# run_solve_programmatic
# ---------------------------------------------------------------------------


def _write_llm_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "kalinov.config.yaml"
    cfg.write_text(
        "providers:\n"
        "  fakep:\n"
        "    type: openai_compat\n"
        "    base_url: http://127.0.0.1:9/v1\n"
        "    default_model: gpt-4o\n",
        encoding="utf-8",
    )
    return cfg


def _patch_fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    fake = FakeLLMClient()
    fake.set_queue(["theorem ok := rfl"] * 10)
    monkeypatch.setattr("kalinov.cli_core.make_client", lambda *_a, **_k: fake)
    return fake


def test_solve_non_utf8_file_does_not_drop_prior_billed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete trigger: ``kalinov solve a.feature b.feature c.feature`` where
    ``b.feature`` has a stray ``0xff`` byte.

    Pre-fix the ``UnicodeDecodeError`` raised by ``parse_feature_file(b)``
    escapes the per-file loop, the manifest write and summary print are
    skipped, and the user sees a traceback instead of the outcomes their
    LLM spend produced for ``a.feature``. The fix mirrors the eval runner
    handler: print a diagnostic, mark ``parse_failed=True``, and keep
    processing remaining files.
    """
    _patch_fake_client(monkeypatch)
    cfg = _write_llm_config(tmp_path)

    good = _good_feature(tmp_path / "a.feature", "A")
    bad = _latin1_feature(tmp_path / "b.feature")
    third = _good_feature(tmp_path / "c.feature", "C")
    runs = tmp_path / "runs"

    res = asyncio.run(
        run_solve_programmatic(
            paths=[good, bad, third],
            runs_dir=runs,
            prover_name="null",
            provider="fakep",
            model=None,
            llm_config_path=cfg,
            cache=None,
            max_repair_attempts=0,
            max_tokens=128,
            temperature=0.0,
            save_transcripts=False,
            max_cost_usd=None,
            echo=False,
        ),
    )

    assert res.parse_failed is True, (
        "non-UTF-8 file must surface as parse_failed=True (CLI exit 2)."
    )
    assert res.obligations_total == 2, (
        "the surrounding good files' obligations must still drive the "
        f"oracle loop; got obligations_total={res.obligations_total}"
    )
    assert len(res.outcomes) == 2
    # The manifest must be written so the run is recoverable by
    # ``kalinov cost report`` / downstream tooling.
    run_dir = next(runs.iterdir())
    manifest = run_dir / "manifest.json"
    assert manifest.is_file(), (
        "the manifest.json roll-up must be written even when one file in "
        "the batch fails to decode; pre-fix the UnicodeDecodeError escaped "
        "the for-loop and skipped this write, silently dropping the "
        "user-visible record of LLM-billed work."
    )
    assert solve_exit_code(res) == 2


def test_solve_missing_file_race_does_not_drop_prior_billed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file deleted between the CLI's ``is_file()`` pre-check and the
    per-file ``read_text`` must not crash the solve loop. Concrete trigger:
    long-running solve over a feature corpus shared with a CI job that
    moves files mid-run."""
    _patch_fake_client(monkeypatch)
    cfg = _write_llm_config(tmp_path)

    good = _good_feature(tmp_path / "a.feature", "A")
    missing = _staged_then_deleted_feature(tmp_path / "b.feature")
    third = _good_feature(tmp_path / "c.feature", "C")
    runs = tmp_path / "runs"

    res = asyncio.run(
        run_solve_programmatic(
            paths=[good, missing, third],
            runs_dir=runs,
            prover_name="null",
            provider="fakep",
            model=None,
            llm_config_path=cfg,
            cache=None,
            max_repair_attempts=0,
            max_tokens=128,
            temperature=0.0,
            save_transcripts=False,
            max_cost_usd=None,
            echo=False,
        ),
    )

    assert res.parse_failed is True
    assert res.obligations_total == 2
    assert len(res.outcomes) == 2
    run_dir = next(runs.iterdir())
    assert (run_dir / "manifest.json").is_file()
