"""Unit tests for oracle prompt helpers."""

from __future__ import annotations

from kalinov.oracle.prompts import format_diagnostics, format_obligation
from kalinov.provers.base import ProofObligation
from kalinov.provers.errors import StructuredError


def test_format_obligation_includes_hypotheses() -> None:
    obl = ProofObligation(
        name="g",
        statement="n > 0",
        hypotheses=("n is a natural", "n is prime"),
    )
    text = format_obligation(obl)
    assert "n is a natural" in text
    assert "n is prime" in text


def test_format_obligation_empty_hypotheses() -> None:
    obl = ProofObligation(name="g", statement="true", hypotheses=())
    text = format_obligation(obl)
    assert "(none)" in text


def test_format_diagnostics_groups_severities() -> None:
    diags = (
        StructuredError("warning", "w1", None, None, None, None),
        StructuredError("error", "e1", None, None, None, None),
        StructuredError("error", "e2", None, None, None, None),
    )
    block = format_diagnostics(diags)
    lines = block.splitlines()
    assert lines[0].startswith("[error]")
    assert lines[1].startswith("[error]")
    assert lines[2].startswith("[warning]")
