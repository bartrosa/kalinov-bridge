"""Prover backends and shared contracts."""

from __future__ import annotations

from kalinov.provers.base import (
    CheckResult,
    CompileResult,
    ProofArtifact,
    ProofObligation,
    Prover,
    SpecDocument,
)
from kalinov.provers.errors import ProverError, StructuredError
from kalinov.provers.null import NullProver, NullProverConfig, NullProverMode

__all__ = [
    "CheckResult",
    "CompileResult",
    "NullProver",
    "NullProverConfig",
    "NullProverMode",
    "ProofArtifact",
    "ProofObligation",
    "Prover",
    "ProverError",
    "SpecDocument",
    "StructuredError",
]
