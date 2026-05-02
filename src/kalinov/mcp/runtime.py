"""MCP server process configuration and stdio logging hygiene.

Stdout hygiene rules:

- The MCP SDK reserves stdout for JSON-RPC framing on stdio transport.
  Any ``print()`` to stdout corrupts the stream and breaks the client.
- All Python logging in the server process MUST go to stderr (or a file).
  :func:`setup_logging_for_stdio` configures the root logger accordingly.
"""

from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    transport: str
    bind_host: str = "127.0.0.1"
    bind_port: int = 8765
    runs_dir: Path = Path("runs")
    kalinov_config_path: Path | None = None
    cache_dir: Path | None = None
    cache_mode: str = "read_write"
    default_max_cost_usd: str | None = None
    """Default budget for tools that invoke the LLM. Per-call override via tool args."""


def setup_logging_for_stdio() -> None:
    """Reconfigure the root logger to write to stderr only. Idempotent enough for tests."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
            force=True,
        )
    else:
        for h in list(root.handlers):
            root.removeHandler(h)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
            stream=sys.stderr,
            force=True,
        )


def load_server_config(args: Namespace) -> MCPServerConfig:
    """Build :class:`MCPServerConfig` from argparse Namespace and environment."""
    transport = getattr(args, "transport", "stdio")
    host = os.environ.get("KALINOV_MCP_HOST", getattr(args, "host", "127.0.0.1"))
    port = int(os.environ.get("KALINOV_MCP_PORT", str(getattr(args, "port", 8765))))
    runs = Path(os.environ.get("KALINOV_RUNS_DIR", getattr(args, "runs_dir", Path("runs"))))
    cfg_path = getattr(args, "kalinov_config", None)
    cache_dir = getattr(args, "cache_dir", None)
    cache_mode = getattr(args, "cache_mode", "read_write")
    max_cost = getattr(args, "max_cost_usd", None)
    return MCPServerConfig(
        transport=transport,
        bind_host=host,
        bind_port=port,
        runs_dir=runs,
        kalinov_config_path=cfg_path,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
        default_max_cost_usd=max_cost,
    )


__all__ = ["MCPServerConfig", "load_server_config", "setup_logging_for_stdio"]
