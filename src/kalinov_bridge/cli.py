"""Command-line entry for local demos and future benchmark runs."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from kalinov_bridge.lean_build import run_lake_build
from kalinov_bridge.mock_llm import fill_proof
from kalinov_bridge.pipeline import run_demo_cycle
from kalinov_bridge.repo import find_repo_root

_LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="kalinov-bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "run-demo",
        help="Mock LLM patches Scratch.lean, runs lake build, restores file.",
    )
    demo.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from cwd).",
    )
    demo.add_argument(
        "--scratch-file",
        type=Path,
        default=None,
        help="Lean file to patch (default: lean/KalinovBridge/Scratch.lean under repo root).",
    )
    demo.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for logs (default: artifacts/run-<timestamp>-<uuid> under repo root).",
    )

    args = parser.parse_args(argv)
    if args.command != "run-demo":
        return 1

    repo_root = (args.repo_root or find_repo_root()).resolve()
    scratch = (
        args.scratch_file
        if args.scratch_file is not None
        else repo_root / "lean" / "KalinovBridge" / "Scratch.lean"
    )
    lean_dir = repo_root / "lean"
    if args.artifacts_dir is not None:
        artifacts = args.artifacts_dir.resolve()
    else:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        artifacts = repo_root / "artifacts" / f"run-{stamp}-{uuid.uuid4().hex[:8]}"

    result = run_demo_cycle(
        scratch_file=scratch,
        lean_dir=lean_dir,
        artifacts_dir=artifacts,
        fill_proof=fill_proof,
        lake_build=run_lake_build,
    )
    _LOG.info(
        "task=%s success=%s returncode=%s duration_ms=%.1f artifacts=%s patched=%s",
        result.task,
        result.success,
        result.returncode,
        result.duration_ms,
        result.artifacts_dir,
        result.patched_lean_name,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
