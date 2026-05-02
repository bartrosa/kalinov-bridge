"""``kalinov mine`` exit codes and wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalinov.cli import main


def test_mine_without_network_exits_4(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["mine", "--source", "arxiv", "--query", "test"])
    err = capsys.readouterr().err
    assert code == 4
    assert "network" in err.lower()


def test_mine_with_network_uses_patched_mine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "Mined_claims__deadbeef.feature"
    out.write_text(
        "\n".join(
            [
                "# language: en",
                "@mined @arxiv",
                "Feature: Mined claims",
                "  ok",
                "",
                "  Scenario: s",
                "    Then x",
                "",
            ],
        ),
        encoding="utf-8",
    )

    async def fake_mine(_cfg: object) -> tuple[Path, ...]:
        return (out,)

    monkeypatch.setattr("kalinov.cli.mine", fake_mine)
    code = main(
        [
            "mine",
            "--source",
            "arxiv",
            "--query",
            "q",
            "--network",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert code == 0
