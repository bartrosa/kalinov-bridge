"""Conversation transcripts for oracle debugging / figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class Transcript:
    """Full conversation that produced an :class:`~kalinov.oracle.strategy.OracleOutcome`.

    Stored as JSON under ``runs/<run_id>/transcripts/<obligation_name>.json`` when enabled.
    """

    messages: tuple[TranscriptMessage, ...]


__all__ = ["MessageRole", "Transcript", "TranscriptMessage"]
