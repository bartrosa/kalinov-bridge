"""YAML benchmark suite loader."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class TaskExpected(StrEnum):
    SOLVED = "solved"
    GAVE_UP = "gave_up"
    EITHER = "either"


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    file: Path
    expected: TaskExpected
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Suite:
    suite_id: str
    description: str
    tasks: tuple[Task, ...]


class SuiteError(Exception):
    """Invalid suite YAML or missing task artifacts."""


def load_suite(path: str | Path) -> Suite:
    """Resolve relative task paths against the suite YAML directory."""
    p = Path(path).resolve()
    if not p.is_file():
        raise SuiteError(f"suite file not found: {p}")
    base_dir = p.parent
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteError(f"invalid YAML in {p}") from exc
    if not isinstance(raw, dict):
        raise SuiteError("suite root must be a mapping")

    suite_id = raw.get("suite_id")
    if not suite_id or not isinstance(suite_id, str):
        raise SuiteError("missing string suite_id")

    desc = raw.get("description")
    description = str(desc).strip() if desc is not None else ""

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise SuiteError("tasks must be a non-empty list")

    out_tasks: list[Task] = []
    for i, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise SuiteError(f"tasks[{i}] must be a mapping")
        tid = item.get("id")
        if not tid or not isinstance(tid, str):
            raise SuiteError(f"tasks[{i}].id must be a non-empty string")

        rel = item.get("file")
        if not rel or not isinstance(rel, str):
            raise SuiteError(f"tasks[{tid}].file must be a path string")

        task_path = (base_dir / rel).resolve()
        if not task_path.is_file():
            raise SuiteError(f"task file does not exist for {tid}: {task_path}")

        exp_raw = item.get("expected")
        if exp_raw is None:
            raise SuiteError(f"tasks[{tid}] missing expected")
        try:
            expected = TaskExpected(str(exp_raw))
        except ValueError as exc:
            raise SuiteError(f"tasks[{tid}]: unknown expected {exp_raw!r}") from exc

        tags_raw = item.get("tags")
        if tags_raw is None:
            tags_raw = []
        elif not isinstance(tags_raw, list):
            raise SuiteError(f"tasks[{tid}].tags must be a list")
        tags = tuple(str(x) for x in tags_raw)

        out_tasks.append(Task(id=tid, file=task_path, expected=expected, tags=tags))

    return Suite(suite_id=suite_id, description=description, tasks=tuple(out_tasks))


__all__ = [
    "Suite",
    "SuiteError",
    "Task",
    "TaskExpected",
    "load_suite",
]
