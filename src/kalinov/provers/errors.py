"""Structured diagnostics for prover backends."""

from __future__ import annotations

from dataclasses import dataclass


class ProverError(Exception):
    """Base class for prover-related runtime errors."""


@dataclass(frozen=True, slots=True)
class StructuredError:
    """A single diagnostic from a prover: typed, locatable error or warning."""

    severity: str
    message: str
    file: str | None
    line: int | None
    column: int | None
    code: str | None
