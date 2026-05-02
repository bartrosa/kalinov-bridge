"""LaTeX-style math fragment extraction from Gherkin steps (structurer, not validator)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from kalinov.gherkin.ast import DocString, Step
from kalinov.interpreters.base import InterpretedStep, StepInterpreter


@dataclass(frozen=True, slots=True)
class MathFragment:
    """A single recognized math fragment within step text."""

    raw: str
    inner: str
    delimiter: str
    start: int
    end: int


_MATH_DOC_TYPES = frozenset({"math", "tex"})


def _math_kind_for_keyword(keyword: str) -> str:
    """Map Gherkin step keyword to InterpretedStep.kind for math-bearing steps."""
    stripped = keyword.strip()
    if not stripped:
        return "claim"
    head = stripped.split(maxsplit=1)[0].lower()
    if head in frozenset({"given"}):
        return "context"
    if head in frozenset({"when"}):
        return "action"
    if head in frozenset({"then", "and", "but", "*"}):
        return "claim"
    return "claim"


def _find_close_single_dollar(text: str, open_idx: int) -> int:
    """Find closing unescaped `$` for inline `$...`, starting after ``open_idx``."""
    j = open_idx + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "$":
            return j
        j += 1
    return -1


def _try_fragment_at(text: str, i: int) -> MathFragment | None:
    if text.startswith("$$", i):
        close = text.find("$$", i + 2)
        if close == -1:
            return None
        inner = text[i + 2 : close]
        end = close + 2
        return MathFragment(
            raw=text[i:end],
            inner=inner,
            delimiter="$$",
            start=i,
            end=end,
        )
    if i < len(text) and text[i] == "$":
        close = _find_close_single_dollar(text, i)
        if close == -1:
            return None
        inner = text[i + 1 : close]
        end = close + 1
        return MathFragment(
            raw=text[i:end],
            inner=inner,
            delimiter="$",
            start=i,
            end=end,
        )
    if text.startswith(r"\(", i):
        close = text.find(r"\)", i + 2)
        if close == -1:
            return None
        inner = text[i + 2 : close]
        end = close + 2
        return MathFragment(
            raw=text[i:end],
            inner=inner,
            delimiter=r"\(",
            start=i,
            end=end,
        )
    if text.startswith(r"\[", i):
        close = text.find(r"\]", i + 2)
        if close == -1:
            return None
        inner = text[i + 2 : close]
        end = close + 2
        return MathFragment(
            raw=text[i:end],
            inner=inner,
            delimiter=r"\[",
            start=i,
            end=end,
        )
    return None


def _scan_text_fragments(text: str) -> list[MathFragment]:
    """Scan *text* for non-overlapping math regions; `\\$` is literal, not a delimiter."""
    out: list[MathFragment] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text.startswith(r"\$", pos):
            pos += 2
            continue
        frag = _try_fragment_at(text, pos)
        if frag is not None:
            out.append(frag)
            pos = frag.end
        else:
            pos += 1
    return out


def _doc_string_fragment(doc: DocString, text_len: int) -> MathFragment:
    body = doc.content
    at = text_len
    return MathFragment(
        raw=body,
        inner=body,
        delimiter="math-block",
        start=at,
        end=at,
    )


def _collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _stripped_text(step_text: str, fragments: tuple[MathFragment, ...]) -> str:
    """Remove fragments that lie inside *step_text* (delimiter != math-block)."""
    inline_frags = [f for f in fragments if f.delimiter != "math-block"]
    if not inline_frags:
        return _collapse_whitespace(step_text)
    parts: list[str] = []
    cursor = 0
    for f in sorted(inline_frags, key=lambda x: x.start):
        if f.start > cursor:
            parts.append(step_text[cursor : f.start])
        cursor = max(cursor, f.end)
    if cursor < len(step_text):
        parts.append(step_text[cursor:])
    return _collapse_whitespace("".join(parts))


class MathTexInterpreter(StepInterpreter):
    """Extracts inline and display math from a step."""

    name: ClassVar[str] = "mathtex"

    def interpret(self, step: Step, context: Mapping[str, Any]) -> InterpretedStep | None:
        _ = context
        text_frags = _scan_text_fragments(step.text)
        doc_fragments: list[MathFragment] = []
        if step.doc_string is not None:
            ct = step.doc_string.content_type
            if ct is not None and ct.lower() in _MATH_DOC_TYPES:
                doc_fragments.append(_doc_string_fragment(step.doc_string, len(step.text)))

        combined = tuple(text_frags + doc_fragments)
        if not combined:
            return None

        kind = _math_kind_for_keyword(step.keyword)
        payload: dict[str, Any] = {
            "fragments": combined,
            "stripped_text": _stripped_text(step.text, combined),
        }
        return InterpretedStep(
            original=step,
            kind=kind,
            payload=payload,
            interpreter_name=self.name,
        )
