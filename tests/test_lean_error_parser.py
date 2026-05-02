"""Unit tests for Lean diagnostic parsing (no Lean install required)."""

from __future__ import annotations

from kalinov.provers.lean.error_parser import parse_lean_output


def test_single_error() -> None:
    raw = "File.lean:5:10: error: foo\n"
    diags = parse_lean_output(raw)
    assert len(diags) == 1
    d = diags[0]
    assert d.severity == "error"
    assert d.message == "foo"
    assert d.file == "File.lean"
    assert d.line == 5
    assert d.column == 10


def test_warning_severity() -> None:
    raw = "X.lean:1:2: warning: unused variable\n"
    d = parse_lean_output(raw)[0]
    assert d.severity == "warning"


def test_continuation_lines_appended() -> None:
    raw = """\
Foo.lean:2:3: error: type mismatch
  line two of message
  line three
"""
    d = parse_lean_output(raw)[0]
    assert "line two" in d.message
    assert "line three" in d.message


def test_multiple_diagnostics() -> None:
    raw = """\
A.lean:1:1: error: first
B.lean:3:4: error: second
"""
    diags = parse_lean_output(raw)
    assert len(diags) == 2
    assert diags[0].message == "first"
    assert diags[1].file == "B.lean"


def test_empty_input() -> None:
    assert parse_lean_output("") == ()
    assert parse_lean_output("\n\n") == ()


def test_irrelevant_lines_ignored() -> None:
    raw = """\
Building Mathlib.Data.Nat.Basic
Some.lean:4:0: error: real problem
"""
    diags = parse_lean_output(raw)
    assert len(diags) == 1
    assert "Building" not in diags[0].message
    assert diags[0].message == "real problem"
