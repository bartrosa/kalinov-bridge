"""Suite YAML loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalinov.eval.suite import SuiteError, TaskExpected, load_suite


def test_load_smoke_suite(repo_root: Path) -> None:
    path = repo_root / "evals" / "suites" / "smoke.yaml"
    suite = load_suite(path)
    assert suite.suite_id == "smoke"
    assert len(suite.tasks) >= 3
    assert suite.tasks[0].file.is_file()
    assert suite.tasks[0].expected is TaskExpected.EITHER


def test_missing_task_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "suite_id: x\ndescription: ''\ntasks:\n"
        "  - id: a\n    file: tasks/does_not_exist.feature\n"
        "    expected: either\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError):
        load_suite(bad)


def test_unknown_expected_raises(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "x.feature").write_text("# language: en\nFeature: X\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "suite_id: x\ndescription: ''\ntasks:\n"
        "  - id: a\n    file: tasks/x.feature\n    expected: maybe\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError):
        load_suite(bad)


def test_paths_resolved_relative_to_suite_file(repo_root: Path) -> None:
    suite = load_suite(repo_root / "evals" / "suites" / "smoke.yaml")
    assert suite.tasks[0].file.is_absolute()
    assert suite.tasks[0].file.name.endswith(".feature")
