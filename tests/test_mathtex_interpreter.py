"""Tests for MathTexInterpreter."""

from __future__ import annotations

import pytest

from kalinov.gherkin.ast import DocString, Location, Step
from kalinov.interpreters.mathtex import MathTexInterpreter

_LOC = Location(line=1, column=1)


def _step(
    keyword: str,
    text: str,
    *,
    doc_string: DocString | None = None,
) -> Step:
    return Step(
        keyword=keyword,
        text=text,
        doc_string=doc_string,
        data_table=None,
        location=_LOC,
    )


@pytest.fixture
def mathtex() -> MathTexInterpreter:
    return MathTexInterpreter()


def test_no_math_returns_none(mathtex: MathTexInterpreter) -> None:
    step = _step("Then ", "plain prose with no delimiters")
    assert mathtex.interpret(step, {}) is None


def test_inline_dollar_extracted(mathtex: MathTexInterpreter) -> None:
    step = _step("Then ", "Then $a^2 + b^2 = c^2$ holds")
    out = mathtex.interpret(step, {})
    assert out is not None
    frags = out.payload["fragments"]
    assert len(frags) == 1
    assert frags[0].inner == "a^2 + b^2 = c^2"
    assert frags[0].delimiter == "$"


def test_display_double_dollar(mathtex: MathTexInterpreter) -> None:
    step = _step("When ", "Display $$a+b$$ here")
    out = mathtex.interpret(step, {})
    assert out is not None
    assert out.payload["fragments"][0].delimiter == "$$"
    assert out.payload["fragments"][0].inner == "a+b"


def test_latex_paren_and_bracket(mathtex: MathTexInterpreter) -> None:
    step = _step("Given ", r"Inline \(\alpha\) and display \[x+y\] done")
    out = mathtex.interpret(step, {})
    assert out is not None
    frags = out.payload["fragments"]
    assert len(frags) == 2
    assert frags[0].delimiter == r"\("
    assert frags[0].inner == r"\alpha"
    assert frags[1].delimiter == r"\["
    assert frags[1].inner == "x+y"


def test_escaped_dollar_not_matched(mathtex: MathTexInterpreter) -> None:
    step = _step("Then ", r"Then the price is \$5 today")
    assert mathtex.interpret(step, {}) is None


def test_multiple_fragments(mathtex: MathTexInterpreter) -> None:
    step = _step("Then ", r"First $1$ gap second $2$ end")
    out = mathtex.interpret(step, {})
    assert out is not None
    frags = out.payload["fragments"]
    assert len(frags) == 2
    assert frags[0].inner == "1"
    assert frags[0].start < frags[1].start
    assert frags[0].end <= frags[1].start
    assert frags[1].inner == "2"


def test_doc_string_math_content_type(mathtex: MathTexInterpreter) -> None:
    doc = DocString(
        content="n + 0 = n\n",
        content_type="math",
        location=_LOC,
    )
    step = _step("Then ", "see block below", doc_string=doc)
    out = mathtex.interpret(step, {})
    assert out is not None
    frags = out.payload["fragments"]
    assert len(frags) == 1
    assert frags[0].delimiter == "math-block"
    assert "n + 0 = n" in frags[0].inner


def test_kind_classification(mathtex: MathTexInterpreter) -> None:
    g = _step("Given ", "$x$")
    w = _step("When ", "$y$")
    t = _step("Then ", "$z$")
    og = mathtex.interpret(g, {})
    ow = mathtex.interpret(w, {})
    ot = mathtex.interpret(t, {})
    assert og is not None and og.kind == "context"
    assert ow is not None and ow.kind == "action"
    assert ot is not None and ot.kind == "claim"


def test_payload_stripped_text(mathtex: MathTexInterpreter) -> None:
    step = _step("Then ", "Some   prose with $a+b$ here\n\tand more")
    out = mathtex.interpret(step, {})
    assert out is not None
    assert out.payload["stripped_text"] == "Some prose with here and more"
