"""FastMCP server factory and transport runner."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from kalinov.mcp import tools as mcp_tools
from kalinov.mcp.resources import (
    resource_config_summary,
    resource_list_runs,
    resource_run_llm_calls,
    resource_run_manifest,
    resource_run_oracle_loop,
    resource_run_transcript,
)
from kalinov.mcp.runtime import MCPServerConfig
from kalinov.mcp.schemas import (
    CheckRequest,
    CheckResponse,
    CostReportRequest,
    CostReportResponse,
    EvalRequest,
    EvalResponse,
    MineRequest,
    MineResponse,
    SolveRequest,
    SolveResponse,
)

logger = logging.getLogger(__name__)


def build_server(config: MCPServerConfig) -> FastMCP:
    """Construct FastMCP with tools and resources."""
    instructions = (
        "Strong oracle bridging informal mathematics to formal proofs. "
        "Tools to check, solve, evaluate, and mine mathematical statements; "
        "resources to inspect run telemetry."
    )
    app = FastMCP(
        name="kalinov-bridge",
        instructions=instructions,
        host=config.bind_host,
        port=config.bind_port,
    )

    @app.tool()
    async def solve(req: SolveRequest) -> SolveResponse:
        return await mcp_tools.tool_solve(req, config)

    @app.tool()
    async def check(req: CheckRequest) -> CheckResponse:
        return await mcp_tools.tool_check(req, config)

    @app.tool()
    async def eval(req: EvalRequest) -> EvalResponse:
        return await mcp_tools.tool_eval(req, config)

    @app.tool()
    async def mine(req: MineRequest) -> MineResponse:
        return await mcp_tools.tool_mine(req, config)

    @app.tool()
    async def cost_report(req: CostReportRequest) -> CostReportResponse:
        return await mcp_tools.tool_cost_report(req, config)

    @app.resource("kalinov://runs")
    async def list_runs_resource() -> str:
        return resource_list_runs(config)

    @app.resource("kalinov://runs/{run_id}/manifest")
    async def manifest_resource(run_id: str) -> str:
        return resource_run_manifest(config, run_id)

    @app.resource("kalinov://runs/{run_id}/llm_calls/{limit}")
    async def llm_calls_resource(run_id: str, limit: str) -> str:
        return resource_run_llm_calls(config, run_id, int(limit))

    @app.resource("kalinov://runs/{run_id}/oracle_loop/{limit}")
    async def oracle_loop_resource(run_id: str, limit: str) -> str:
        return resource_run_oracle_loop(config, run_id, int(limit))

    @app.resource("kalinov://runs/{run_id}/transcripts/{name}")
    async def transcript_resource(run_id: str, name: str) -> str:
        return resource_run_transcript(config, run_id, name)

    @app.resource("kalinov://config")
    async def config_resource() -> str:
        return resource_config_summary(config)

    return app


def run_server(config: MCPServerConfig) -> int:
    """Run MCP transport; blocks until shutdown."""
    try:
        app = build_server(config)
        if config.transport == "stdio":
            app.run(transport="stdio")
        elif config.transport == "streamable-http":
            app.run(transport="streamable-http")
        else:
            logger.error("unknown transport %s", config.transport)
            return 1
    except Exception:
        logger.exception("mcp server failed")
        return 1
    return 0


__all__ = ["build_server", "run_server"]
