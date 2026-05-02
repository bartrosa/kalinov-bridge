"""Tests for NullProver and prover dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kalinov.gherkin import parse_feature_text
from kalinov.interpreters import InterpreterChain, MathTexInterpreter, RawInterpreter
from kalinov.interpreters.base import InterpretedStep
from kalinov.provers import (
    NullProver,
    NullProverConfig,
    NullProverMode,
    ProofArtifact,
    ProofObligation,
    SpecDocument,
)

_CHAIN = InterpreterChain(
    [MathTexInterpreter(), RawInterpreter()],
)


def _spec_from_text(text: str) -> SpecDocument:
    ff = parse_feature_text(text)
    steps: list[InterpretedStep] = []
    for sc in ff.feature.scenarios:
        for st in sc.steps:
            steps.append(_CHAIN.interpret(st))
    return SpecDocument(feature_file=ff, interpreted_steps=tuple(steps))


def test_always_ok_mode() -> None:
    p = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))
    obl = ProofObligation(name="o", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    assert p.compile(art).ok is True
    assert p.check(art).ok is True


def test_always_fail_mode() -> None:
    p = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_FAIL))
    obl = ProofObligation(name="o", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    cr = p.compile(art)
    assert cr.ok is False
    assert len(cr.diagnostics) >= 1
    chk = p.check(art)
    assert chk.ok is False
    assert len(chk.diagnostics) >= 1


def test_fail_after_n() -> None:
    p = NullProver(
        NullProverConfig(mode=NullProverMode.FAIL_AFTER_N, fail_after=2),
    )
    obl = ProofObligation(name="o", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    assert p.compile(art).ok is True
    assert p.check(art).ok is True
    assert p.compile(art).ok is False


def test_call_count_increments() -> None:
    p = NullProver()
    obl = ProofObligation(name="o", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    assert p.call_count == 0
    p.compile(art)
    assert p.call_count == 1
    p.check(art)
    assert p.call_count == 2


def test_extract_obligations_from_claims() -> None:
    text_claim = """
Feature: F
  Scenario: One
    Then $x$
""".strip()
    spec = _spec_from_text(text_claim)
    p = NullProver()
    obs = p.extract_obligations(spec)
    assert len(obs) == 1
    assert obs[0].statement == "$x$"
    assert obs[0].name == "One#0"

    text_raw = """
Feature: G
  Scenario: Two
    Then no math here
""".strip()
    spec2 = _spec_from_text(text_raw)
    assert p.extract_obligations(spec2) == ()


def test_parse_error_returns_structured() -> None:
    p = NullProver()
    out = p.parse_error("raw stderr line")
    assert len(out) == 1
    assert out[0].message == "raw stderr line"


def test_proof_artifact_immutable() -> None:
    obl = ProofObligation(name="o", statement="s")
    art = ProofArtifact(obligation=obl, body="", language="null", metadata={})
    with pytest.raises(FrozenInstanceError):
        art.body = "x"  # type: ignore[misc]
