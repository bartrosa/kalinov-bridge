"""FastMCP wiring tests."""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from pathlib import Path

from kalinov.mcp.runtime import MCPServerConfig, setup_logging_for_stdio
from kalinov.mcp.server import build_server


def test_build_server_registers_all_tools() -> None:
    async def inner() -> None:
        app = build_server(MCPServerConfig(transport="stdio", runs_dir=Path("runs")))
        tools = await app.list_tools()
        names = sorted(t.name for t in tools)
        assert names == ["check", "cost_report", "eval", "mine", "solve"]

    asyncio.run(inner())


def test_build_server_registers_all_resources() -> None:
    async def inner() -> None:
        app = build_server(MCPServerConfig(transport="stdio", runs_dir=Path("runs")))
        static = await app.list_resources()
        uri_static = {str(r.uri) for r in static}
        assert "kalinov://runs" in uri_static
        assert "kalinov://config" in uri_static
        tpl = await app.list_resource_templates()
        uri_tpl = {t.uriTemplate for t in tpl}
        assert "kalinov://runs/{run_id}/manifest" in uri_tpl
        assert "kalinov://runs/{run_id}/llm_calls/{limit}" in uri_tpl
        assert "kalinov://runs/{run_id}/oracle_loop/{limit}" in uri_tpl
        assert "kalinov://runs/{run_id}/transcripts/{name}" in uri_tpl

    asyncio.run(inner())


def test_stdio_logging_does_not_pollute_stdout() -> None:
    spy = io.StringIO()
    old = sys.stdout
    sys.stdout = spy
    try:
        setup_logging_for_stdio()
        logging.getLogger("kalinov.mcp.test_stdio").warning("hello from logging")
    finally:
        sys.stdout = old
    assert spy.getvalue() == ""
