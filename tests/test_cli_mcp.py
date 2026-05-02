"""CLI tests for ``kalinov mcp``."""

from __future__ import annotations

import builtins

import pytest

from kalinov.cli import main


def test_mcp_subcommand_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        main(["mcp", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "--transport" in out and "--host" in out and "--port" in out


def test_mcp_missing_extra_exits_4(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_arg: dict[str, object] | None = None,
        locals_arg: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("simulated missing mcp extra")
        return real_import(name, globals_arg, locals_arg, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(["mcp"])
    assert code == 4


def test_mcp_invalid_bind_exits_5() -> None:
    code = main(
        [
            "mcp",
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
        ],
    )
    assert code == 5
