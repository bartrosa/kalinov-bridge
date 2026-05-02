"""Oracle configuration and outcome records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from kalinov.cost.models import CostBreakdown
from kalinov.provers.base import CheckResult, ProofArtifact, ProofObligation

ProviderExtras = Mapping[str, Any]


class OracleStrategyKind(StrEnum):
    PROPOSE_THEN_REPAIR = "propose_then_repair"


@dataclass(frozen=True, slots=True)
class OracleConfig:
    strategy: OracleStrategyKind = OracleStrategyKind.PROPOSE_THEN_REPAIR
    max_repair_attempts: int = 3
    max_tokens_per_call: int = 2048
    temperature: float = 0.0
    """Default 0 for reproducibility; bump higher in experiments."""
    extras: ProviderExtras | None = None
    """Optional per-call provider tuning (e.g. extended thinking)."""
    save_transcripts: bool = False


class OracleOutcomeKind(StrEnum):
    SOLVED = "solved"
    GAVE_UP = "gave_up"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVER_ERROR = "prover_error"
    LLM_ERROR = "llm_error"


@dataclass(frozen=True, slots=True)
class OracleAttempt:
    iteration: int  # 0 = initial proposal
    artifact: ProofArtifact
    check_result: CheckResult | None  # None if LLM/prover errored before check
    cost: CostBreakdown | None  # None when call cached or errored
    duration_ms: int


@dataclass(frozen=True, slots=True)
class OracleOutcome:
    obligation: ProofObligation
    kind: OracleOutcomeKind
    attempts: tuple[OracleAttempt, ...]
    final_artifact: ProofArtifact | None
    total_cost_usd: Decimal
    diagnostic: str | None  # reason for non-SOLVED outcomes


__all__ = [
    "OracleAttempt",
    "OracleConfig",
    "OracleOutcome",
    "OracleOutcomeKind",
    "OracleStrategyKind",
    "ProviderExtras",
]
