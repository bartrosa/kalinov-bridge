"""Lean integration tests (require elan, lake, and a built vendored runtime)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalinov.gherkin import parse_feature_text
from kalinov.gherkin.ast import FeatureFile
from kalinov.interpreters import InterpreterChain, MathTexInterpreter, RawInterpreter
from kalinov.interpreters.base import InterpretedStep
from kalinov.provers.base import ProofArtifact, SpecDocument
from kalinov.provers.lean import (
    LeanProver,
    LeanProverConfig,
    ToolchainInfo,
    detect_toolchain,
)
from kalinov.telemetry import start_run

pytestmark = pytest.mark.lean

_CHAIN = InterpreterChain([MathTexInterpreter(), RawInterpreter()])


def _interpret(ff: FeatureFile) -> SpecDocument:
    steps: list[InterpretedStep] = []
    for sc in ff.feature.scenarios:
        for st in sc.steps:
            steps.append(_CHAIN.interpret(st))
    return SpecDocument(feature_file=ff, interpreted_steps=tuple(steps))


@pytest.fixture
def lean_toolchain() -> ToolchainInfo:
    return detect_toolchain()


@pytest.fixture
def lean_prover(lean_toolchain: ToolchainInfo) -> LeanProver:
    return LeanProver(toolchain=lean_toolchain)


def test_toolchain_detection(lean_toolchain: ToolchainInfo) -> None:
    assert lean_toolchain.lake_path.name == "lake"
    assert "Lake" in lean_toolchain.lake_version or "lake" in lean_toolchain.lake_version.lower()


def test_compile_trivial_lemma(lean_prover: LeanProver) -> None:
    from kalinov.provers import ProofObligation

    body = "import Mathlib.Tactic\n\ntheorem t : 1 = 1 := rfl\n"
    o = ProofObligation(name="t", statement="s", metadata={})
    art = ProofArtifact(obligation=o, body=body, language="lean4", metadata={})
    res = lean_prover.compile(art)
    assert res.ok is True


def test_compile_syntax_error(lean_prover: LeanProver) -> None:
    from kalinov.provers import ProofObligation

    art = ProofArtifact(
        obligation=ProofObligation(name="bad", statement="s"),
        body="import Mathlib.Tactic\n\ntheorem t : := rfl\n",
        language="lean4",
        metadata={},
    )
    res = lean_prover.compile(art)
    assert res.ok is False
    assert any(d.severity == "error" for d in res.diagnostics)


def test_check_trivial_lemma(lean_prover: LeanProver) -> None:
    from kalinov.provers import ProofObligation

    body = "import Mathlib.Tactic\n\ntheorem t_check : 1 = 1 := rfl\n"
    art = ProofArtifact(
        obligation=ProofObligation(name="ck", statement="s"),
        body=body,
        language="lean4",
        metadata={},
    )
    res = lean_prover.check(art)
    assert res.ok is True


def test_check_false_lemma(lean_prover: LeanProver) -> None:
    from kalinov.provers import ProofObligation

    body = "import Mathlib.Tactic\n\ntheorem t_bad : 1 = 2 := rfl\n"
    art = ProofArtifact(
        obligation=ProofObligation(name="no", statement="s"),
        body=body,
        language="lean4",
        metadata={},
    )
    res = lean_prover.check(art)
    assert res.ok is False


def test_compile_timeout(lean_toolchain: ToolchainInfo) -> None:
    cfg = LeanProverConfig(compile_timeout_seconds=0.01)
    p = LeanProver(config=cfg, toolchain=lean_toolchain)
    from kalinov.provers import ProofObligation

    body = "import Mathlib\n\ntheorem slow : True := trivial\n"
    art = ProofArtifact(
        obligation=ProofObligation(name="to", statement="s"),
        body=body,
        language="lean4",
        metadata={},
    )
    res = p.compile(art)
    assert res.ok is False
    assert any(d.code == "timeout" for d in res.diagnostics)


def test_extract_obligations_filters_lean_tag(lean_prover: LeanProver) -> None:
    text = """
Feature: F
  @lean
  Scenario: With tag
    Then $1$

  Scenario: No tag
    Then $2$
""".strip()
    ff = parse_feature_text(text)
    spec = _interpret(ff)
    obs = lean_prover.extract_obligations(spec)
    assert len(obs) == 1
    assert obs[0].name.startswith("With tag")


def test_telemetry_written(lean_prover: LeanProver, tmp_path: Path) -> None:
    from kalinov.provers import ProofObligation

    body = "import Mathlib.Tactic\n\ntheorem tel : 1 = 1 := rfl\n"
    art = ProofArtifact(
        obligation=ProofObligation(name="tel", statement="s"),
        body=body,
        language="lean4",
        metadata={},
    )
    with start_run(runs_dir=tmp_path) as run:
        lean_prover.check(art)
        logf = run.run_dir / "prover_calls.jsonl"
        lines = logf.read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(lines[-1])
    assert row["backend"] == "lean4"
