"""Abstract prover interface and shared datatypes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from kalinov.gherkin.ast import FeatureFile
from kalinov.interpreters.base import InterpretedStep
from kalinov.provers.errors import StructuredError


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """A single goal the prover must discharge."""

    name: str
    statement: str
    hypotheses: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProofArtifact:
    """A candidate proof or proof-fragment to compile/check."""

    obligation: ProofObligation
    body: str
    language: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompileResult:
    ok: bool
    duration_ms: int
    diagnostics: tuple[StructuredError, ...]
    raw_output: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Whether the artifact actually proves the obligation."""

    ok: bool
    duration_ms: int
    diagnostics: tuple[StructuredError, ...]
    obligation: ProofObligation
    raw_output: str


@dataclass(frozen=True, slots=True)
class SpecDocument:
    """Feature file plus interpreted steps (caller-defined pairing)."""

    feature_file: FeatureFile
    interpreted_steps: tuple[InterpretedStep, ...]


class Prover(ABC):
    """Abstract prover backend."""

    backend_name: ClassVar[str]
    language: ClassVar[str]

    @abstractmethod
    def compile(self, artifact: ProofArtifact) -> CompileResult:
        """Type-check / parse-check the artifact without proving."""

    @abstractmethod
    def check(self, artifact: ProofArtifact) -> CheckResult:
        """Verify that the artifact discharges its obligation."""

    @abstractmethod
    def extract_obligations(self, spec: SpecDocument) -> tuple[ProofObligation, ...]:
        """Slice obligations from an interpreted spec."""

    @abstractmethod
    def parse_error(self, raw: str) -> tuple[StructuredError, ...]:
        """Normalize backend-native error text into structured diagnostics."""


__all__ = [
    "CheckResult",
    "CompileResult",
    "ProofArtifact",
    "ProofObligation",
    "Prover",
    "SpecDocument",
]
