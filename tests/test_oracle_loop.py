"""Oracle loop behaviour with FakeLLMClient and NullProver / ScriptedProver."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import load_default_catalogue
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import BudgetExceededError, LLMError
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
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


def test_budget_overrun_call_cost_is_included_in_total_cost_usd(
    obl: ProofObligation,
) -> None:
    """Regression: when a billable provider call trips ``max_cost_usd``,
    that call's cost MUST be folded into ``OracleOutcome.total_cost_usd``.

    Before the fix, ``BudgetGuard.record`` raised ``BudgetExceededError``
    after the call had already been billed (and logged to
    ``llm_calls.jsonl``), but the oracle loop's ``except`` branch never
    added the cost to ``total_cost``. The result: the OracleOutcome — and
    every downstream surface that aggregates it (``kalinov solve``'s
    ``summary: total_usd=...``, eval ``report.json`` per-task
    ``total_cost_usd``) — reported ``$0`` for the overrun obligation even
    though the provider charged real money for the call that crossed the
    cap. The summary user-visibly contradicted ``kalinov cost report``
    (which reads ``llm_calls.jsonl`` and stayed correct), and a user
    inspecting only the summary would not realise their cap was breached.
    """
    llm = FakeLLMClient()
    llm.set_queue(["theorem a := rfl"])
    cat = load_default_catalogue()
    unit = estimate_cost(
        TokenUsage(input=10, output=20),
        provider="openai",
        model_id="gpt-4o",
        catalogue=cat,
    ).total_usd
    assert unit > Decimal("0"), "fixture pricing should yield non-zero unit cost"

    # Cap below the cost of a single call: ``record`` will raise after
    # the call has been logged + billed.
    guard = BudgetGuard(Budget(max_cost_usd=unit / 2))
    set_budget_guard(guard)
    try:
        loop = OracleLoop(
            prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK)),
            llm=llm,
            model="gpt-4o",
        )
        out = asyncio.run(_run(loop, obl))
    finally:
        set_budget_guard(None)

    assert out.kind is OracleOutcomeKind.BUDGET_EXCEEDED
    assert out.total_cost_usd == unit, (
        "OracleOutcome.total_cost_usd must include the cost of the call "
        "that tripped the budget cap; otherwise the solve/eval summary "
        f"silently drops the overrun spend (expected {unit}, got {out.total_cost_usd})"
    )
    # Sanity: the guard state agrees with the attributed cost.
    assert guard.state.spent_usd == unit
