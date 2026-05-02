"""Markdown diff between two eval runs."""

from __future__ import annotations

from collections.abc import Sequence

from kalinov.eval.runner import RunResult, TaskResult


def _task_signature(tr: TaskResult) -> str:
    parts = [o.kind.value for o in tr.outcomes]
    return ",".join(parts) if parts else "(empty)"


def compare_runs(a: RunResult, b: RunResult, *, by: str = "task") -> str:
    """Render a Markdown diff; highlights tasks whose obligation outcomes differ."""
    if by not in ("task", "obligation"):
        raise ValueError(f"unsupported by={by!r}")

    lines: list[str] = []
    lines.append("# Run comparison\n\n")
    lines.append(f"- **A** `{a.config.label}` (suite `{a.suite.suite_id}`)\n")
    lines.append(f"- **B** `{b.config.label}` (suite `{b.suite.suite_id}`)\n\n")

    if by == "obligation":
        lines.append(
            "_Obligation-level diff matches task mode when there is one obligation per task._\n\n",
        )

    map_a = {tr.task.id: tr for tr in a.task_results}
    map_b = {tr.task.id: tr for tr in b.task_results}
    ids: Sequence[str] = sorted(set(map_a) | set(map_b))

    lines.append("## Changed tasks\n\n")
    changes = False
    for tid in ids:
        ta = map_a.get(tid)
        tb = map_b.get(tid)
        if ta is None or tb is None:
            changes = True
            lines.append(f"- **{tid}**: present only in one run\n")
            continue
        sa = _task_signature(ta)
        sb = _task_signature(tb)
        if sa != sb:
            changes = True
            lines.append(f"- **{tid}**: `{sa}` vs `{sb}`\n")

    if not changes:
        lines.append("_No outcome differences at task granularity._\n")

    return "".join(lines)


__all__ = ["compare_runs"]
