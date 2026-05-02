"""Parse ``.feature`` files using ``gherkin-official`` into our typed AST."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gherkin.errors import CompositeParserException, ParserError, ParserException
from gherkin.parser import Parser as GherkinParser

from kalinov.gherkin.ast import (
    Background,
    DataTable,
    DocString,
    Examples,
    Feature,
    FeatureFile,
    Location,
    Scenario,
    Step,
)
from kalinov.gherkin.errors import GherkinParseError


def parse_feature_file(path: str | Path) -> FeatureFile:
    """Parse a UTF-8 encoded ``.feature`` file from disk."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return parse_feature_text(text, source_path=p.resolve())


def parse_feature_text(text: str, *, source_path: Path | None = None) -> FeatureFile:
    """Parse Gherkin source *text* into a :class:`FeatureFile`."""
    parser = GherkinParser()
    try:
        doc = parser.parse(text)
    except CompositeParserException as e:
        if e.errors:
            _raise_parse_error(e.errors[0], source_path)
        raise GherkinParseError(str(e), source_path=source_path) from e
    except ParserException as e:
        _raise_parse_error(e, source_path)
    except ParserError as e:
        raise GherkinParseError(str(e), source_path=source_path) from e

    raw_feature = doc["feature"]
    feature = _convert_feature(raw_feature)
    return FeatureFile(source_path=source_path, feature=feature)


def _raise_parse_error(
    err: ParserException,
    source_path: Path | None,
) -> None:
    loc_raw = err.location
    line = int(loc_raw["line"])
    col_raw = loc_raw.get("column")
    column = int(col_raw) if col_raw is not None else None
    raise GherkinParseError(
        str(err),
        source_path=source_path,
        line=line,
        column=column,
    ) from err


def _convert_location(raw: Mapping[str, Any]) -> Location:
    line = int(raw["line"])
    col = raw.get("column")
    return Location(line=line, column=int(col) if col is not None else 1)


def _tags(raw_tags: list[Any]) -> tuple[str, ...]:
    return tuple(str(t["name"]) for t in raw_tags)


def _convert_doc_string(raw: Mapping[str, Any]) -> DocString:
    mt = raw.get("mediaType")
    return DocString(
        content=str(raw["content"]),
        content_type=str(mt) if mt is not None else None,
        location=_convert_location(raw["location"]),
    )


def _convert_data_table(raw: Mapping[str, Any]) -> DataTable:
    rows_out: list[tuple[str, ...]] = []
    for row in raw["rows"]:
        rows_out.append(tuple(str(c["value"]) for c in row["cells"]))
    return DataTable(
        rows=tuple(rows_out),
        location=_convert_location(raw["location"]),
    )


def _convert_step(raw: Mapping[str, Any]) -> Step:
    ds = raw.get("docString")
    dt = raw.get("dataTable")
    return Step(
        keyword=str(raw["keyword"]),
        text=str(raw["text"]),
        doc_string=_convert_doc_string(ds) if ds else None,
        data_table=_convert_data_table(dt) if dt else None,
        location=_convert_location(raw["location"]),
    )


def _convert_background(raw: Mapping[str, Any]) -> Background:
    return Background(
        name=str(raw["name"]),
        description=str(raw["description"]),
        steps=tuple(_convert_step(s) for s in raw["steps"]),
        location=_convert_location(raw["location"]),
    )


def _convert_examples(raw: Mapping[str, Any]) -> Examples:
    header_cells = raw["tableHeader"]["cells"]
    headers = tuple(str(c["value"]) for c in header_cells)
    body_rows: list[tuple[str, ...]] = []
    for row in raw["tableBody"]:
        body_rows.append(tuple(str(c["value"]) for c in row["cells"]))
    return Examples(
        tags=_tags(raw["tags"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        headers=headers,
        rows=tuple(body_rows),
        location=_convert_location(raw["location"]),
    )


def _convert_scenario(raw: Mapping[str, Any]) -> Scenario:
    return Scenario(
        tags=_tags(raw["tags"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        steps=tuple(_convert_step(s) for s in raw["steps"]),
        examples=tuple(_convert_examples(e) for e in raw["examples"]),
        location=_convert_location(raw["location"]),
    )


def _convert_feature(raw: Mapping[str, Any]) -> Feature:
    background: Background | None = None
    scenarios: list[Scenario] = []
    for child in raw["children"]:
        if "background" in child:
            background = _convert_background(child["background"])
        elif "scenario" in child:
            scenarios.append(_convert_scenario(child["scenario"]))
        elif "rule" in child:
            continue

    return Feature(
        tags=_tags(raw["tags"]),
        language=str(raw["language"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        background=background,
        scenarios=tuple(scenarios),
        location=_convert_location(raw["location"]),
    )
