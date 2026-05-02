"""Lazy entry for ``kalinov mcp`` (avoids importing MCP stack when unused)."""

from __future__ import annotations

import sys
from argparse import Namespace

_MCP_INSTALL = (
    "MCP server requires the [mcp] extra. Install with: pip install 'kalinov-bridge[mcp]'"
)


def run_mcp_main(args: Namespace) -> int:
    try:
        import mcp  # noqa: F401
    except ImportError:
        print(_MCP_INSTALL, file=sys.stderr)
        return 4
    # Stdio transport reserves stdout for JSON-RPC; configure logging before pulling
    # the full server (which may import subsystems that log on import).
    if getattr(args, "transport", "stdio") == "stdio":
        from kalinov.mcp.runtime import setup_logging_for_stdio

        setup_logging_for_stdio()
    from kalinov.mcp.runtime import load_server_config
    from kalinov.mcp.server import run_server

    cfg = load_server_config(args)
    if (
        cfg.transport == "streamable-http"
        and cfg.bind_host == "0.0.0.0"
        and not bool(getattr(args, "allow_public", False))
    ):
        print(
            "refusing to bind 0.0.0.0 without --allow-public (irreversible network exposure).",
            file=sys.stderr,
        )
        return 5
    return run_server(cfg)


__all__ = ["run_mcp_main"]
