"""Deterministic stand-in for an LLM: patch Lean sources before verification."""

from __future__ import annotations


def fill_proof(lean_source: str) -> str:
    """Replace the first ``by sorry`` with ``by trivial`` (demo contract for ``Scratch.lean``)."""
    needle = "by sorry"
    replacement = "by trivial"
    if needle not in lean_source:
        msg = f"Expected {needle!r} in Lean source for mock fill"
        raise ValueError(msg)
    return lean_source.replace(needle, replacement, 1)
