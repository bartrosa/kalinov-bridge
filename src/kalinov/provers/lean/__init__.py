"""Lean 4 prover adapter (Lake / Mathlib runtime)."""

from __future__ import annotations

from kalinov.provers.lean.adapter import LeanProver, LeanProverConfig
from kalinov.provers.lean.error_parser import parse_lean_output
from kalinov.provers.lean.toolchain import (
    ToolchainInfo,
    ToolchainNotFoundError,
    detect_toolchain,
)

__all__ = [
    "LeanProver",
    "LeanProverConfig",
    "ToolchainInfo",
    "ToolchainNotFoundError",
    "detect_toolchain",
    "parse_lean_output",
]
