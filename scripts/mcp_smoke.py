"""Local smoke test for the MCP server.

Spawns ``kalinov mcp`` (or ``python -m kalinov.cli mcp``) over stdio, calls a
small set of tools, and prints short JSON summaries to stderr. Exits 0 when
the server responds to each call without a transport-level failure.

Not part of automated tests — use when verifying a local install with the
``[mcp]`` extra.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _tool_payload(result: object) -> str:
    return json.dumps(
        {
            "isError": getattr(result, "isError", None),
            "content": [c.model_dump() for c in getattr(result, "content", [])],
        },
        indent=2,
    )


async def _amain() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / ".mcp_smoke_runs"
    runs_dir.mkdir(exist_ok=True)
    tiny = repo_root / ".mcp_smoke_check.feature"
    tiny.write_text(
        "# language: en\nFeature: Smoke\n  Scenario: S\n    Then $1=1$\n",
        encoding="utf-8",
    )

    src = str(repo_root / "src")
    env = {**os.environ, "PYTHONPATH": src}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kalinov.cli", "mcp", "--runs-dir", str(runs_dir)],
        env=env,
        cwd=str(repo_root),
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        steps: list[tuple[str, dict[str, Any]]] = [
            (
                "cost_report",
                {
                    "req": {
                        "group_by": "none",
                        "runs_dir": str(runs_dir),
                        "run_id": None,
                    },
                },
            ),
            (
                "check",
                {
                    "req": {
                        "feature_path": str(tiny),
                        "prover": "null",
                        "no_forthel": False,
                        "null_mode": "always_ok",
                        "null_fail_after": 0,
                    },
                },
            ),
            (
                "mine",
                {
                    "req": {
                        "source": "arxiv",
                        "query": "smoke",
                        "limit": 1,
                        "out_dir": str(runs_dir / "mined"),
                        "network": False,
                    },
                },
            ),
        ]
        for name, args in steps:
            res = await session.call_tool(name, args)
            _err(f"=== {name} ===\n{_tool_payload(res)}")

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
