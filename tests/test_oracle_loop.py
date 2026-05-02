"""Oracle loop behaviour with FakeLLMClient and NullProver / ScriptedProver."""

from __future__ import annotations

import asyncio

import pytest

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import load_default_catalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import BudgetExceededError, LLMError
from kalinov.oracle import OracleConfig, OracleLoop, OracleOutcome, OracleOutcomeKind
from kalinov.provers import NullProver, NullProverConfig, NullProverMode, ProofObligation
from tests.fixtures.fake_llm_client import FakeLLMClient
from tests.fixtures.scripted_prover import ScriptedProver


@pytest.fixture
def obl() -> ProofObligation:
    return ProofObligation(name="goal", statement="1 + 1 = 2", hypotheses=())


async def _run(loop: OracleLoop, o: ProofObligation) -> OracleOutcome:
    return await loop.run(o)


def test_solved_on_first_attempt(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["theorem ok := rfl"])
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK)),
        llm=llm,
        model="gpt-4o",
    )
    out = asyncio.run(_run(loop, obl))
    assert out.kind is OracleOutcomeKind.SOLVED
    assert len(out.attempts) == 1


def test_solved_after_repair(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["first bad", "theorem ok := rfl"])
    prover = ScriptedProver(rounds=[(True, False), (True, True)])
    loop = OracleLoop(prover=prover, llm=llm, model="gpt-4o")
    out = asyncio.run(_run(loop, obl))
    assert out.kind is OracleOutcomeKind.SOLVED
    assert len(out.attempts) == 2


def test_gave_up_after_max_attempts(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["a", "b", "c"])
    cfg = OracleConfig(max_repair_attempts=2)
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_FAIL)),
        llm=llm,
        model="gpt-4o",
        config=cfg,
    )
    out = asyncio.run(_run(loop, obl))
    assert out.kind is OracleOutcomeKind.GAVE_UP
    assert len(out.attempts) == 3


def test_compile_failure_uses_compile_diagnostics_in_repair_prompt(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["bad body", "theorem ok := rfl"])
    prover = ScriptedProver(
        rounds=[(False, True), (True, True)],
        compile_fail_message="COMPILE_DIAG_XYZ",
    )
    loop = OracleLoop(prover=prover, llm=llm, model="gpt-4o")
    asyncio.run(_run(loop, obl))
    msgs = llm.last_messages
    assert msgs is not None
    user = [m for m in msgs if m.role == "user"][-1].content
    assert "COMPILE_DIAG_XYZ" in user


def test_budget_exceeded_terminates_cleanly(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(
        [
            "theorem a := rfl",
            BudgetExceededError(provider="openai", message="cut"),
        ],
    )
    prover = ScriptedProver(rounds=[(True, False), (True, True)])
    loop = OracleLoop(prover=prover, llm=llm, model="gpt-4o")
    out = asyncio.run(_run(loop, obl))
    assert out.kind is OracleOutcomeKind.BUDGET_EXCEEDED


def test_llm_error_terminates_cleanly(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(
        [
            LLMError(
                provider="openai",
                code="other",
                message="boom",
                retriable=False,
            ),
        ],
    )
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK)),
        llm=llm,
        model="gpt-4o",
    )
    out = asyncio.run(_run(loop, obl))
    assert out.kind is OracleOutcomeKind.LLM_ERROR
    assert out.diagnostic == "boom"


def test_total_cost_aggregates(obl: ProofObligation) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["theorem a := rfl", "theorem b := rfl"])
    prover = ScriptedProver(rounds=[(True, False), (True, True)])
    loop = OracleLoop(prover=prover, llm=llm, model="gpt-4o")
    out = asyncio.run(_run(loop, obl))
    cat = load_default_catalogue()
    usage = TokenUsage(input=10, output=20)
    unit = estimate_cost(usage, provider="openai", model_id="gpt-4o", catalogue=cat).total_usd
    assert out.total_cost_usd == unit * 2
