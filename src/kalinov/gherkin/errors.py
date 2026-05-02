"""Errors raised by the Gherkin frontend."""

from __future__ import annotations

from pathlib import Path


class GherkinError(Exception):
    """Base class for Gherkin frontend errors."""


class GherkinParseError(GherkinError):
    """Raised when the underlying parser rejects a ``.feature`` file."""

    def __init__(
        self,
        message: str,
        *,
        source_path: Path | None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.source_path = source_path
        self.line = line
        self.column = column

    def __str__(self) -> str:
        parts: list[str] = []
        if self.source_path is not None:
            parts.append(str(self.source_path))
        if self.line is not None:
            loc = str(self.line)
            if self.column is not None:
                loc = f"{self.line}:{self.column}"
            parts.append(loc)
        prefix = ":".join(parts) + ": " if parts else ""
        return f"{prefix}{self.args[0]}"
