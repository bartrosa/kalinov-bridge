"""Model Context Protocol server package for kalinov.

``build_server`` / ``run_server`` are loaded lazily so ``import kalinov.mcp.runtime``
(direct or via :mod:`kalinov.mcp`) does not pull the MCP server stack before
:func:`setup_logging_for_stdio` runs in the stdio entry path.
"""

from __future__ import annotations

import importlib
from typing import Any

from kalinov.mcp.runtime import MCPServerConfig, load_server_config, setup_logging_for_stdio

__all__ = [
    "MCPServerConfig",
    "build_server",
    "load_server_config",
    "run_server",
    "setup_logging_for_stdio",
]


def __getattr__(name: str) -> Any:
    if name == "build_server":
        mod = importlib.import_module("kalinov.mcp.server")
        return mod.build_server
    if name == "run_server":
        mod = importlib.import_module("kalinov.mcp.server")
        return mod.run_server
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
