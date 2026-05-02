"""Typed AST for parsed Gherkin feature files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Location:
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class DocString:
    content: str
    content_type: str | None
    location: Location


@dataclass(frozen=True, slots=True)
class DataTable:
    rows: tuple[tuple[str, ...], ...]
    location: Location


@dataclass(frozen=True, slots=True)
class Step:
    keyword: str
    text: str
    doc_string: DocString | None
    data_table: DataTable | None
    location: Location


@dataclass(frozen=True, slots=True)
class Background:
    name: str
    description: str
    steps: tuple[Step, ...]
    location: Location


@dataclass(frozen=True, slots=True)
class Examples:
    tags: tuple[str, ...]
    name: str
    description: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    location: Location


@dataclass(frozen=True, slots=True)
class Scenario:
    tags: tuple[str, ...]
    name: str
    description: str
    steps: tuple[Step, ...]
    examples: tuple[Examples, ...]
    location: Location

    @property
    def is_outline(self) -> bool:
        return len(self.examples) > 0


@dataclass(frozen=True, slots=True)
class Feature:
    tags: tuple[str, ...]
    language: str
    name: str
    description: str
    background: Background | None
    scenarios: tuple[Scenario, ...]
    location: Location


@dataclass(frozen=True, slots=True)
class FeatureFile:
    source_path: Path | None
    feature: Feature
