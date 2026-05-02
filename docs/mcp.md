# Model Context Protocol (MCP) integration

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open
standard for connecting LLM clients to tools and data. Hosts (Cursor, Claude
Desktop, VS Code, and others) spawn a local or remote process and exchange
JSON-RPC messages so the model can call registered **tools** and read
**resources** in a structured way.

This project exposes `solve`, `check`, `eval`, `mine`, and `cost_report` as MCP
tools, backed by the same programmatic cores as the `kalinov` CLI. Telemetry
under `runs/<run_id>/` matches CLI runs.

## Install

The MCP Python SDK is optional:

```bash
uv sync --group dev --extra mcp
# or
pip install 'kalinov-bridge[mcp]'
```

## Run the server

Default transport is **stdio** (for local IDE integration):

```bash
kalinov mcp
```

Streamable HTTP (bind defaults to `127.0.0.1`; binding `0.0.0.0` requires
`--allow-public`):

```bash
kalinov mcp --transport streamable-http --host 127.0.0.1 --port 8765
```

## Cursor (`~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "kalinov": {
      "command": "kalinov",
      "args": ["mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "${env:ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

Adjust `command` / `args` if `kalinov` is not on `PATH` (e.g. use `uv run kalinov mcp` via a small wrapper script).

## Claude Desktop

Same JSON shape under:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

## Tool arguments (`req`)

Tools are registered with a single Pydantic parameter named `req`. Low-level
clients must pass arguments under that key, for example:

```json
{
  "req": {
    "feature_path": "examples/gauss_sum.feature",
    "prover": "null",
    "provider": "my_provider"
  }
}
```

IDE hosts usually map the tool schema so you do not type `req` yourself.

## Money fields

All USD amounts in tool JSON are **strings** (decimal text), never floats.

## MCP tools vs CLI

| MCP tool      | CLI equivalent        |
| ------------- | --------------------- |
| `solve`       | `kalinov solve`       |
| `check`       | `kalinov check`       |
| `eval`        | `kalinov eval`        |
| `mine`        | `kalinov mine`        |
| `cost_report` | `kalinov cost report` |

## Troubleshooting

- **Corrupted stdio / host disconnects**: ensure nothing writes to **stdout**
  except MCP JSON-RPC (use stderr for logs). The server sets stderr logging in
  stdio mode.
- **`MCP server requires the [mcp] extra`**: install `kalinov-bridge[mcp]`.
- **Lean / `lean4` prover**: requires `elan`, built `provers/lean/runtime`, and
  expected PATH tools — same as the CLI.

## Smoke script

From the repo root (with `[mcp]` installed):

```bash
uv run python scripts/mcp_smoke.py
```

This spawns `python -m kalinov.cli mcp` and calls a few tools; summaries are
printed to stderr.
