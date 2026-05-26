"""Propose → verify → repair orchestration for a single obligation."""

from __future__ import annotations

import time
from dataclasses import asdict
from decimal import Decimal

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import PricingCatalogue, load_default_catalogue
from kalinov.cost.models import CostBreakdown
from kalinov.llm.base import BudgetExceededError, Completion, LLMClient, LLMError, Message
from kalinov.llm.telemetry import take_last_llm_call_id
from kalinov.oracle.prompts import (
    REPAIR_TEMPLATE,
    SYSTEM_PROMPT_LEAN,
    body_from_llm_text,
    format_diagnostics,
    format_obligation,
)
from kalinov.oracle.strategy import OracleAttempt, OracleConfig, OracleOutcome, OracleOutcomeKind
from kalinov.oracle.telemetry import log_iteration, save_transcript_json
from kalinov.oracle.transcript import Transcript, TranscriptMessage
from kalinov.provers.base import ProofArtifact, ProofObligation, Prover
from kalinov.provers.telemetry import (
    take_last_prover_check_call_id,
    take_last_prover_compile_call_id,
)


def _elapsed_ms(t0_ns: int) -> int:
    return int((time.perf_counter_ns() - t0_ns) / 1_000_000)


def _oracle_line(
    *,
    obligation_name: str,
    iteration: int,
    outcome_so_far: str,
    t0_ns: int,
    llm_call_id: str | None,
    prover_call_id: str | None,
    cost_usd: str | None,
) -> None:
    log_iteration(
        obligation_name=obligation_name,
        iteration=iteration,
        outcome_so_far=outcome_so_far,
        duration_ms=_elapsed_ms(t0_ns),
        llm_call_id=llm_call_id,
        prover_call_id=prover_call_id,
        cost_usd=cost_usd,
    )


class OracleLoop:
    """Drives the propose → verify → repair cycle for one obligation
    against one prover and one LLM client.

    The loop is single-obligation; batch processing over a SpecDocument
    is the caller's job (see ``kalinov solve`` CLI). Single-obligation scope
    keeps state machines small and telemetry rows clean.
    """

    def __init__(
        self,
        *,
        prover: Prover,
        llm: LLMClient,
        model: str,
        config: OracleConfig | None = None,
        catalogue: PricingCatalogue | None = None,
    ) -> None:
        self._prover = prover
        self._llm = llm
        self._model = model
        self._config = config or OracleConfig()
        self._catalogue = catalogue or load_default_catalogue()

    async def run(self, obligation: ProofObligation) -> OracleOutcome:
        """Execute the loop. Writes one ``oracle_loop.jsonl`` line per iteration."""
        return self._run_sync(obligation)

    def _run_sync(self, obligation: ProofObligation) -> OracleOutcome:
        attempts: list[OracleAttempt] = []
        transcript_buf: list[TranscriptMessage] = []
        total_cost = Decimal("0")
        previous_artifact: ProofArtifact | None = None
        last_diagnostic_text = ""

        max_rounds = self._config.max_repair_attempts + 1

        for iter_idx in range(max_rounds):
            t0 = time.perf_counter_ns()
            system_content = SYSTEM_PROMPT_LEAN
            if iter_idx == 0:
                user_content = format_obligation(obligation)
            else:
                assert previous_artifact is not None
                user_content = REPAIR_TEMPLATE.format(
                    previous_body=previous_artifact.body,
                    diagnostics_block=last_diagnostic_text,
                )

            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=user_content),
            ]
            transcript_buf.extend(
                [
                    TranscriptMessage(role="system", content=system_content),
                    TranscriptMessage(role="user", content=user_content),
                ]
            )

            completion: Completion | None = None
            artifact_for_attempt: ProofArtifact | None = None
            cost_br: CostBreakdown | None = None
            llm_id: str | None = None
            prover_join_id: str | None = None

            try:
                completion = self._llm.complete(
                    messages=messages,
                    model=self._model,
                    max_tokens=self._config.max_tokens_per_call,
                    temperature=self._config.temperature,
                    stop=None,
                    extras=self._config.extras,
                )
            except (BudgetExceededError, LLMError) as exc:
                llm_id = take_last_llm_call_id()
                # When ``BudgetGuard.record`` raises after a successful
                # (already-billed) provider call, ``attempted_cost_usd``
                # carries the cost of that call. Without folding it into
                # ``total_cost`` the OracleOutcome would report ``$0`` for
                # the obligation that tripped the cap even though the
                # provider charged real money for the overrun, which
                # silently understates the per-task / per-run spend the
                # user sees in eval reports and the solve summary.
                attempted_cost: str | None = None
                if isinstance(exc, BudgetExceededError) and exc.attempted_cost_usd is not None:
                    total_cost += exc.attempted_cost_usd
                    attempted_cost = str(exc.attempted_cost_usd)
                _oracle_line(
                    obligation_name=obligation.name,
                    iteration=iter_idx,
                    outcome_so_far="error",
                    t0_ns=t0,
                    llm_call_id=llm_id,
                    prover_call_id=None,
                    cost_usd=attempted_cost,
                )
                self._maybe_save_transcript(obligation.name, transcript_buf)
                kind = (
                    OracleOutcomeKind.BUDGET_EXCEEDED
                    if isinstance(exc, BudgetExceededError)
                    else OracleOutcomeKind.LLM_ERROR
                )
                return OracleOutcome(
                    obligation=obligation,
                    kind=kind,
                    attempts=tuple(attempts),
                    final_artifact=previous_artifact,
                    total_cost_usd=total_cost,
                    diagnostic=exc.message,
                )

            assert completion is not None
            llm_id = take_last_llm_call_id()
            body = body_from_llm_text(completion.text)
            artifact_for_attempt = ProofArtifact(
                obligation=obligation,
                body=body,
                language=self._prover.language,
                metadata={},
            )
            previous_artifact = artifact_for_attempt
            transcript_buf.append(TranscriptMessage(role="assistant", content=completion.text))

            if completion.cache_hit:
                cost_br = None
            else:
                cost_br = estimate_cost(
                    completion.usage,
                    provider=self._llm.provider_key,
                    model_id=completion.model_id_resolved,
                    catalogue=self._catalogue,
                )
                # Mirror the pipeline alias fallback so total_cost_usd stays
                # accurate when the provider returns a date-versioned model id
                # that doesn't match pricing.yaml directly.
                if (
                    cost_br.pricing_source == "unknown"
                    and self._model != completion.model_id_resolved
                ):
                    fallback = estimate_cost(
                        completion.usage,
                        provider=self._llm.provider_key,
                        model_id=self._model,
                        catalogue=self._catalogue,
                    )
                    if fallback.pricing_source != "unknown":
                        cost_br = fallback
                total_cost += cost_br.total_usd

            cost_usd_str = str(cost_br.total_usd) if cost_br is not None else None

            try:
                compile_res = self._prover.compile(artifact_for_attempt)
            except Exception as exc:
                compile_join = take_last_prover_compile_call_id()
                _oracle_line(
                    obligation_name=obligation.name,
                    iteration=iter_idx,
                    outcome_so_far="error",
                    t0_ns=t0,
                    llm_call_id=llm_id,
                    prover_call_id=compile_join,
                    cost_usd=cost_usd_str,
                )
                attempts.append(
                    OracleAttempt(
                        iteration=iter_idx,
                        artifact=artifact_for_attempt,
                        check_result=None,
                        cost=cost_br,
                        duration_ms=_elapsed_ms(t0),
                    ),
                )
                self._maybe_save_transcript(obligation.name, transcript_buf)
                return OracleOutcome(
                    obligation=obligation,
                    kind=OracleOutcomeKind.PROVER_ERROR,
                    attempts=tuple(attempts),
                    final_artifact=artifact_for_attempt,
                    total_cost_usd=total_cost,
                    diagnostic=str(exc),
                )

            compile_join = take_last_prover_compile_call_id()
            prover_join_id = compile_join

            if not compile_res.ok:
                last_diagnostic_text = format_diagnostics(compile_res.diagnostics)
                _oracle_line(
                    obligation_name=obligation.name,
                    iteration=iter_idx,
                    outcome_so_far="compile_fail",
                    t0_ns=t0,
                    llm_call_id=llm_id,
                    prover_call_id=prover_join_id,
                    cost_usd=cost_usd_str,
                )
                attempts.append(
                    OracleAttempt(
                        iteration=iter_idx,
                        artifact=artifact_for_attempt,
                        check_result=None,
                        cost=cost_br,
                        duration_ms=_elapsed_ms(t0),
                    ),
                )
                if iter_idx >= self._config.max_repair_attempts:
                    self._maybe_save_transcript(obligation.name, transcript_buf)
                    return OracleOutcome(
                        obligation=obligation,
                        kind=OracleOutcomeKind.GAVE_UP,
                        attempts=tuple(attempts),
                        final_artifact=artifact_for_attempt,
                        total_cost_usd=total_cost,
                        diagnostic=last_diagnostic_text,
                    )
                continue

            try:
                check_res = self._prover.check(artifact_for_attempt)
            except Exception as exc:
                check_join = take_last_prover_check_call_id()
                _oracle_line(
                    obligation_name=obligation.name,
                    iteration=iter_idx,
                    outcome_so_far="error",
                    t0_ns=t0,
                    llm_call_id=llm_id,
                    prover_call_id=check_join or prover_join_id,
                    cost_usd=cost_usd_str,
                )
                attempts.append(
                    OracleAttempt(
                        iteration=iter_idx,
                        artifact=artifact_for_attempt,
                        check_result=None,
                        cost=cost_br,
                        duration_ms=_elapsed_ms(t0),
                    ),
                )
                self._maybe_save_transcript(obligation.name, transcript_buf)
                return OracleOutcome(
                    obligation=obligation,
                    kind=OracleOutcomeKind.PROVER_ERROR,
                    attempts=tuple(attempts),
                    final_artifact=artifact_for_attempt,
                    total_cost_usd=total_cost,
                    diagnostic=str(exc),
                )

            check_join = take_last_prover_check_call_id()
            prover_join_id = check_join or compile_join

            elapsed = _elapsed_ms(t0)
            attempts.append(
                OracleAttempt(
                    iteration=iter_idx,
                    artifact=artifact_for_attempt,
                    check_result=check_res,
                    cost=cost_br,
                    duration_ms=elapsed,
                ),
            )

            if check_res.ok:
                _oracle_line(
                    obligation_name=obligation.name,
                    iteration=iter_idx,
                    outcome_so_far="ok",
                    t0_ns=t0,
                    llm_call_id=llm_id,
                    prover_call_id=prover_join_id,
                    cost_usd=cost_usd_str,
                )
                self._maybe_save_transcript(obligation.name, transcript_buf)
                return OracleOutcome(
                    obligation=obligation,
                    kind=OracleOutcomeKind.SOLVED,
                    attempts=tuple(attempts),
                    final_artifact=artifact_for_attempt,
                    total_cost_usd=total_cost,
                    diagnostic=None,
                )

            last_diagnostic_text = format_diagnostics(check_res.diagnostics)
            _oracle_line(
                obligation_name=obligation.name,
                iteration=iter_idx,
                outcome_so_far="check_fail",
                t0_ns=t0,
                llm_call_id=llm_id,
                prover_call_id=prover_join_id,
                cost_usd=cost_usd_str,
            )

            if iter_idx >= self._config.max_repair_attempts:
                self._maybe_save_transcript(obligation.name, transcript_buf)
                return OracleOutcome(
                    obligation=obligation,
                    kind=OracleOutcomeKind.GAVE_UP,
                    attempts=tuple(attempts),
                    final_artifact=artifact_for_attempt,
                    total_cost_usd=total_cost,
                    diagnostic=last_diagnostic_text,
                )

        self._maybe_save_transcript(obligation.name, transcript_buf)
        return OracleOutcome(
            obligation=obligation,
            kind=OracleOutcomeKind.GAVE_UP,
            attempts=tuple(attempts),
            final_artifact=previous_artifact,
            total_cost_usd=total_cost,
            diagnostic=last_diagnostic_text or "exhausted repair attempts",
        )

    def _maybe_save_transcript(self, obligation_name: str, buf: list[TranscriptMessage]) -> None:
        if not self._config.save_transcripts:
            return
        tr = Transcript(messages=tuple(buf))
        payload = {"messages": [asdict(m) for m in tr.messages]}
        save_transcript_json(obligation_name=obligation_name, payload=payload)


__all__ = ["OracleLoop"]
