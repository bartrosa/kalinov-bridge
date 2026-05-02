"""Active run context for telemetry (one run_id under a runs root directory)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

_current_run: ContextVar[RunContext | None] = ContextVar("kalinov_current_run", default=None)


@dataclass(frozen=True, slots=True)
class RunContext:
    """An active benchmark/check run."""

    run_id: str
    runs_root: Path

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id


def active_run() -> RunContext | None:
    """Return the innermost active run, if any."""
    return _current_run.get()


@contextmanager
def start_run(*, runs_dir: str | Path) -> Iterator[RunContext]:
    """Create a new run directory under *runs_dir* and activate it for telemetry.

    ``ContextVar`` state is not inherited by new threads; use
    ``contextvars.copy_context()`` in worker threads if they need the active
    :class:`RunContext`.
    """
    root = Path(runs_dir).resolve()
    run_id = uuid.uuid4().hex[:12]
    ctx = RunContext(run_id=run_id, runs_root=root)
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    token = _current_run.set(ctx)
    try:
        yield ctx
    finally:
        _current_run.reset(token)
