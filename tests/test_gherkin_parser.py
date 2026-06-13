"""Tests for the Gherkin parser and AST mapping."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kalinov.gherkin import (
    parse_feature_file,
    parse_feature_text,
)
from kalinov.gherkin.errors import GherkinParseError

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_parse_minimal_feature() -> None:
    text = """
Feature: Minimal
  Scenario: One
    Given a condition
    When an action
    Then an outcome
""".strip()
    ff = parse_feature_text(text)
    assert ff.source_path is None
    assert ff.feature.name == "Minimal"
    assert len(ff.feature.scenarios) == 1
    sc = ff.feature.scenarios[0]
    assert sc.name == "One"
    assert not sc.is_outline
    assert len(sc.steps) == 3
    assert [s.text for s in sc.steps] == ["a condition", "an action", "an outcome"]


def test_parse_with_background(tmp_path: Path) -> None:
    path = tmp_path / "bg.feature"
    path.write_text(
        """
Feature: With background
  Background:
    Given the world is set up
  Scenario: Act
    When we run
    Then it works
""".strip(),
        encoding="utf-8",
    )
    ff = parse_feature_file(path)
    assert ff.feature.background is not None
    assert ff.feature.background.name == ""
    assert len(ff.feature.background.steps) == 1
    assert ff.feature.background.steps[0].text == "the world is set up"
    assert len(ff.feature.scenarios) == 1
    assert ff.feature.scenarios[0].steps[0].text == "we run"


def test_parse_scenario_outline_with_examples(tmp_path: Path) -> None:
    path = tmp_path / "outline.feature"
    path.write_text(
        """
Feature: Outline
  Scenario Outline: row expansion
    Given the value is <a>
    Then it matches <b>

    Examples:
      | a  | b  |
      | 2  | 2  |
      | 3  | 3  |
""".strip(),
        encoding="utf-8",
    )
    ff = parse_feature_file(path)
    sc = ff.feature.scenarios[0]
    assert sc.is_outline
    ex = sc.examples[0]
    assert ex.headers == ("a", "b")
    assert ex.rows[0] == ("2", "2")
    assert ex.rows[1] == ("3", "3")


def test_parse_tags_propagate() -> None:
    ff = parse_feature_file(EXAMPLES_DIR / "gauss_sum.feature")
    assert ff.feature.tags == ("@math", "@arithmetic")
    sc = ff.feature.scenarios[0]
    assert sc.tags == ("@lean", "@step_series")
    assert sc.examples[0].tags == ("@first_rows",)


def test_parse_doc_string_in_step() -> None:
    ff = parse_feature_file(EXAMPLES_DIR / "pythagoras.feature")
    step = ff.feature.scenarios[0].steps[-1]
    assert step.doc_string is not None
    assert step.doc_string.content.strip() == "a^2 + b^2 = c^2"
    assert step.doc_string.content_type == "tex"


def test_parse_data_table_in_step() -> None:
    ff = parse_feature_file(EXAMPLES_DIR / "triangle_inequality.feature")
    step = ff.feature.scenarios[0].steps[-1]
    assert step.data_table is not None
    assert step.data_table.rows[0][0] == "form of left member"
    assert "abs(x + y)" in step.data_table.rows[1][0]


def test_parse_error_has_location(tmp_path: Path) -> None:
    src = tmp_path / "bad.feature"
    src.write_text("bad", encoding="utf-8")
    with pytest.raises(GherkinParseError) as excinfo:
        parse_feature_text(src.read_text(encoding="utf-8"), source_path=src)
    err = excinfo.value
    msg = str(err)
    assert "bad.feature" in msg
    assert ":1:" in msg  # line (and column as 1:1 style)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("empty", ""),
        ("whitespace_only", "   \n   \n"),
        ("single_comment", "# only a comment\n"),
        ("multiple_comments", "# c1\n# c2\n# c3\n"),
    ],
)
def test_parse_featureless_file_raises_gherkin_parse_error(
    tmp_path: Path,
    label: str,
    text: str,
) -> None:
    """A ``.feature`` with no ``Feature:`` block must surface as a
    :class:`GherkinParseError`, not a raw ``KeyError``.

    The underlying ``gherkin-official`` parser accepts these inputs and
    returns a doc whose ``feature`` key is missing or ``None``. Letting
    that escape as ``KeyError`` bypasses every caller's
    ``except GherkinParseError`` branch — which is exactly what bounds
    the per-file damage in :func:`kalinov.cli_core.run_check_programmatic`,
    :func:`kalinov.cli_core.run_solve_programmatic`, and
    :class:`kalinov.eval.runner.EvalRunner`. Without this guard, dropping
    an empty / comment-only file into a paid eval suite mid-run aborts
    the whole run and discards every previously-completed task result
    (the same data-loss class fixed for malformed-syntax files in PR #23).
    """
    src = tmp_path / f"{label}.feature"
    src.write_text(text, encoding="utf-8")
    with pytest.raises(GherkinParseError) as excinfo:
        parse_feature_file(src)
    err = excinfo.value
    assert err.source_path == src.resolve()
    assert "no Feature" in str(err) or "feature" in str(err).lower()


@pytest.mark.parametrize(
    "path",
    sorted(EXAMPLES_DIR.glob("*.feature")),
    ids=lambda p: p.name,
)
def test_parse_all_example_files(path: Path) -> None:
    ff = parse_feature_file(path)
    assert ff.feature.name.strip() != ""


def test_locations_are_one_indexed(tmp_path: Path) -> None:
    path = tmp_path / "loc.feature"
    path.write_text(
        """
Feature: Loc
  Scenario: S
    Given x
""".strip(),
        encoding="utf-8",
    )
    ff = parse_feature_file(path)
    loc = ff.feature.location
    step_loc = ff.feature.scenarios[0].steps[0].location
    assert loc.line >= 1 and loc.column >= 1
    assert step_loc.line >= 1 and step_loc.column >= 1


def test_rule_scenarios_are_flattened(tmp_path: Path) -> None:
    """Scenarios nested under ``Rule:`` must surface as feature-level scenarios.

    Pre-fix, :func:`_convert_feature` ``continue``d past every ``"rule"``
    child returned by ``gherkin-official``, silently dropping every
    scenario nested under a ``Rule:`` block. The blast radius covered all
    three top-level commands:

    - ``kalinov check`` returned exit code 0 (no failures, no parse
      errors) on files whose proofs were never actually run, masking real
      regressions in CI.
    - ``kalinov solve`` exited "success" with zero LLM calls when every
      obligation lived under a Rule, even though the user thought their
      proofs were attempted.
    - ``kalinov eval`` reported ``obligations_total=0`` and
      ``matched_expected=True`` (empty kinds is treated as "matched") for
      any task file using Rule syntax, silently corrupting benchmark
      numbers.
    """
    path = tmp_path / "rule.feature"
    path.write_text(
        """
Feature: With a rule
  Scenario: outside
    Given x

  Rule: business rule
    Scenario: nested_a
      Given a
    Scenario: nested_b
      Given b
""".strip(),
        encoding="utf-8",
    )
    ff = parse_feature_file(path)
    names = [s.name for s in ff.feature.scenarios]
    assert names == ["outside", "nested_a", "nested_b"]


def test_rule_tags_inherit_to_nested_scenarios(tmp_path: Path) -> None:
    """Tags on a ``Rule:`` propagate to every scenario it contains.

    Gherkin's tag-inheritance contract states that tags declared on a
    parent (Feature, Rule) apply to every nested scenario. Kalinov's
    tag-based dispatch (notably ``@lean``) depends on this: silently
    dropping the inherited tags would change which scenarios drive Lean
    obligation extraction, leading to a different — and silently smaller
    — set of obligations than the user expects.
    """
    path = tmp_path / "rule_tags.feature"
    path.write_text(
        """
Feature: F
  @lean @math
  Rule: r
    Scenario: a
      Given x
    @arith
    Scenario: b
      Given y
""".strip(),
        encoding="utf-8",
    )
    ff = parse_feature_file(path)
    assert [s.name for s in ff.feature.scenarios] == ["a", "b"]
    # Rule tags inherit; scenario-level tags are preserved and de-duplicated.
    assert ff.feature.scenarios[0].tags == ("@lean", "@math")
    assert ff.feature.scenarios[1].tags == ("@lean", "@math", "@arith")


def test_immutability(tmp_path: Path) -> None:
    path = tmp_path / "frozen.feature"
    path.write_text(
        """
Feature: F
  Scenario: S
    Given x
""".strip(),
        encoding="utf-8",
    )
    ff = parse_feature_file(path)
    with pytest.raises(FrozenInstanceError):
        ff.feature.name = "y"  # type: ignore[misc]
