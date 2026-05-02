"""Oracle loop: propose → verify → repair."""

from __future__ import annotations

from kalinov.oracle.loop import OracleLoop
from kalinov.oracle.strategy import (
    OracleAttempt,
    OracleConfig,
    OracleOutcome,
    OracleOutcomeKind,
    OracleStrategyKind,
)
from kalinov.oracle.transcript import Transcript, TranscriptMessage

__all__ = [
    "OracleAttempt",
    "OracleConfig",
    "OracleLoop",
    "OracleOutcome",
    "OracleOutcomeKind",
    "OracleStrategyKind",
    "Transcript",
    "TranscriptMessage",
]
