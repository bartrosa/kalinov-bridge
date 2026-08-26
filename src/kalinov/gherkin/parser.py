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

    # gherkin-official's parser accepts ``.feature`` files with no ``Feature:``
    # at all (e.g. a totally empty file, whitespace-only, or comment-only) and
    # returns a doc whose ``feature`` key is either missing or set to ``None``.
    # Letting ``doc["feature"]`` propagate as ``KeyError`` (or feeding ``None``
    # into ``_convert_feature``) would escape every caller's
    # ``except GherkinParseError`` branch, which is what bounds the per-file
    # damage in ``kalinov check`` / ``kalinov solve`` and what
    # ``EvalRunner._spec_error_outcome`` relies on to keep a paid eval suite
    # running past a single bad task. Without this branch, dropping an empty
    # file into a 100-task suite aborts the whole run mid-flight (after
    # already-billed tasks have completed) and discards every prior
    # TaskResult — exactly the data-loss class fixed for malformed-syntax
    # files in PR #23.
    raw_feature = doc.get("feature")
    if raw_feature is None:
        raise GherkinParseError(
            "no Feature defined in file (empty, whitespace-only, "
            "or comment-only .feature)",
            source_path=source_path,
        )
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
            # Gherkin v6+ allows grouping scenarios under a ``Rule:`` block.
            # ``gherkin-official`` returns rules as ``children`` entries whose
            # own ``children`` list holds the nested scenarios (and an
            # optional rule-level background). Previously we ``continue``'d
            # past every rule, silently dropping every nested scenario. The
            # blast radius was severe:
            #
            #   * ``kalinov check`` returned exit code 0 on files whose
            #     proofs were never actually run (every scenario lived
            #     under a Rule), masking real failures from CI.
            #   * ``kalinov solve`` exited "success" with zero LLM calls
            #     when every obligation was nested under a Rule — the user
            #     thought their proofs were attempted (and the prover was
            #     blamed for any later breakage) but the LLM was never
            #     invoked.
            #   * ``kalinov eval`` reported ``obligations_total=0`` /
            #     ``matched_expected=True`` for any task file using Rule
            #     syntax (see ``_outcomes_match_expected``: empty kinds is
            #     treated as "matched"), silently corrupting benchmark
            #     numbers.
            #
            # Flatten the rule's scenarios into the feature-level list and
            # inherit the rule's tags onto each nested scenario (matches
            # Gherkin's tag-inheritance semantics, so ``@lean`` /
            # ``@math`` on a Rule still flows through to its scenarios).
            _flatten_rule_scenarios(child["rule"], scenarios)

    return Feature(
        tags=_tags(raw["tags"]),
        language=str(raw["language"]),
        name=str(raw["name"]),
        description=str(raw["description"]),
        background=background,
        scenarios=tuple(scenarios),
        location=_convert_location(raw["location"]),
    )


def _flatten_rule_scenarios(
    rule_raw: Mapping[str, Any],
    scenarios: list[Scenario],
) -> None:
    """Append every ``Scenario`` nested under a ``Rule:`` to *scenarios*.

    Per Gherkin's tag-inheritance contract, tags declared on the ``Rule:``
    apply to every scenario it contains. We prepend the rule's tags to each
    scenario's own tags (de-duplicating while preserving order) so
    downstream tag-based dispatch (notably ``@lean``) keeps working when
    users write ``@lean`` on a ``Rule:`` instead of on every nested
    ``Scenario:``.

    Rule-level ``Background:`` blocks (if any) are intentionally not merged
    here: the rest of the codebase does not consume background steps for
    proving, so ignoring them is behaviour-preserving relative to the
    feature-level background that was already supported.
    """
    rule_tags = _tags(rule_raw.get("tags") or [])
    for sub in rule_raw.get("children") or ():
        if "scenario" not in sub:
            continue
        sc = _convert_scenario(sub["scenario"])
        if rule_tags:
            merged: list[str] = []
            seen: set[str] = set()
            for t in (*rule_tags, *sc.tags):
                if t in seen:
                    continue
                seen.add(t)
                merged.append(t)
            sc = Scenario(
                tags=tuple(merged),
                name=sc.name,
                description=sc.description,
                steps=sc.steps,
                examples=sc.examples,
                location=sc.location,
            )
        scenarios.append(sc)
