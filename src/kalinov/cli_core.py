"""Programmatic cores for CLI commands (shared by ``kalinov`` CLI and MCP tools).

Observable behaviour matches the former inlined argparse handlers.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from kalinov.bridges.forthel_lean import TranslationOutcomeKind, translate_step
from kalinov.gherkin import parse_feature_file
from kalinov.gherkin.errors import GherkinParseError
from kalinov.interpreters import (
    ForTheLInterpreter,
    InterpreterChain,
    MathTexInterpreter,
    RawInterpreter,
)
from kalinov.interpreters.base import InterpretedStep
from kalinov.llm.budget import Budget, BudgetGuard
from kalinov.llm.budget_context import set_budget_guard
from kalinov.llm.cache import LLMCache
from kalinov.llm.config import ConfigError
from kalinov.llm.config import load_config as load_llm_config
from kalinov.llm.factory import make_client
from kalinov.oracle import OracleConfig, OracleLoop, OracleOutcome, OracleOutcomeKind
from kalinov.provers import (
    NullProver,
    NullProverConfig,
    NullProverMode,
    ProofArtifact,
    ProofObligation,
)
from kalinov.provers.base import Prover, SpecDocument
from kalinov.provers.errors import ProverError
from kalinov.provers.lean import LeanProver, detect_toolchain
from kalinov.telemetry import start_run


class ClientConfigError(Exception):
    """Raised when ``make_client`` fails configuration (CLI historically exits 3)."""


def _scenario_has_lean_tag(tags: tuple[str, ...]) -> bool:
    return any(t.strip().lower() == "@lean" for t in tags)


def _interpret_chain() -> InterpreterChain:
    return InterpreterChain(
        [
            MathTexInterpreter(),
            ForTheLInterpreter(),
            RawInterpreter(),
        ],
    )


@dataclass(frozen=True, slots=True)
class CheckObligationResult:
    obligation_name: str
    ok: bool
    diagnostics: tuple[str, ...]
    duration_ms: int


@dataclass(frozen=True, slots=True)
class CheckProgrammaticResult:
    run_id: str
    runs_dir: Path
    results: tuple[CheckObligationResult, ...]
    parse_failed: bool
    total_ok: int
    total_fail: int
    total_obligations: int


def run_check_programmatic(
    prover: Prover,
    paths: list[Path],
    artifact_language: str,
    runs_dir: Path,
    *,
    forthel_bridge: bool = False,
    echo: bool = True,
) -> CheckProgrammaticResult:
    """Run prover checks like ``kalinov check``. When *echo* is True, mirror CLI prints."""
    chain = _interpret_chain()
    total_ok = 0
    total_fail = 0
    total_obligations = 0
    parse_failed = False
    rows: list[CheckObligationResult] = []
    captured_run_id = ""
    captured_runs_root = runs_dir.resolve()

    def emit_ok(name: str, duration_ms: int) -> None:
        nonlocal total_ok
        total_ok += 1
        rows.append(
            CheckObligationResult(name, True, (), duration_ms),
        )
        if echo:
            print(f"OK {name}")

    def emit_fail(name: str, msg: str, duration_ms: int) -> None:
        nonlocal total_fail
        total_fail += 1
        rows.append(
            CheckObligationResult(name, False, (msg,), duration_ms),
        )
        if echo:
            print(f"FAIL {name}: {msg}")

    with start_run(runs_dir=runs_dir) as run:
        captured_run_id = run.run_id
        captured_runs_root = run.runs_root
        for path in paths:
            try:
                ff = parse_feature_file(path)
            except GherkinParseError as exc:
                if echo:
                    print(f"parse error in {path}: {exc}", file=sys.stderr)
                parse_failed = True
                continue

            interpreted: list[InterpretedStep] = []
            for scenario in ff.feature.scenarios:
                for step in scenario.steps:
                    interpreted.append(chain.interpret(step))

            spec = SpecDocument(
                feature_file=ff,
                interpreted_steps=tuple(interpreted),
            )

            skip_obligation_names: set[str] = set()

            if forthel_bridge and isinstance(prover, LeanProver):
                it = iter(spec.interpreted_steps)
                broken = False
                for scenario in ff.feature.scenarios:
                    if broken:
                        break
                    for step_idx, _step in enumerate(scenario.steps):
                        try:
                            interp = next(it)
                        except StopIteration:
                            if echo:
                                print(
                                    f"parse error in {path}: "
                                    "interpreted_steps shorter than scenario step walk",
                                    file=sys.stderr,
                                )
                            parse_failed = True
                            broken = True
                            break

                        name = f"{scenario.name}#{step_idx}"
                        lean_here = _scenario_has_lean_tag(scenario.tags)

                        if interp.interpreter_name == "forthel" and interp.kind == "claim":
                            tout = translate_step(interp)
                            if tout.kind == TranslationOutcomeKind.SKIPPED:
                                if echo:
                                    print(f"SKIP FORTHEL {name}: {tout.diagnostic}")
                            elif tout.kind == TranslationOutcomeKind.FAILED:
                                total_fail += 1
                                msg = tout.diagnostic or "forthel translation failed"
                                rows.append(CheckObligationResult(name, False, (msg,), 0))
                                if echo:
                                    print(f"FAIL {name}: {msg}")
                                if lean_here:
                                    skip_obligation_names.add(name)
                            else:
                                obl = ProofObligation(
                                    name=name,
                                    statement=interp.original.text,
                                    hypotheses=(),
                                    metadata={},
                                )
                                artifact = ProofArtifact(
                                    obligation=obl,
                                    body=tout.lean_source or "",
                                    language=artifact_language,
                                    metadata={},
                                )
                                result = prover.check(artifact)
                                total_obligations += 1
                                if result.ok:
                                    emit_ok(obl.name, result.duration_ms)
                                else:
                                    msg = (
                                        result.diagnostics[0].message
                                        if result.diagnostics
                                        else "check failed"
                                    )
                                    emit_fail(obl.name, msg, result.duration_ms)
                                if lean_here:
                                    skip_obligation_names.add(name)
                    if broken:
                        break

                if not broken:
                    try:
                        next(it)
                    except StopIteration:
                        pass
                    else:
                        if echo:
                            print(
                                f"parse error in {path}: "
                                "interpreted_steps longer than scenario step walk",
                                file=sys.stderr,
                            )
                        parse_failed = True
                        broken = True

                if broken:
                    continue

            try:
                obligations = prover.extract_obligations(spec)
            except ProverError as exc:
                if echo:
                    print(f"error in {path}: {exc}", file=sys.stderr)
                parse_failed = True
                continue

            for obl in obligations:
                if obl.name in skip_obligation_names:
                    continue
                total_obligations += 1
                artifact = ProofArtifact(
                    obligation=obl,
                    body="",
                    language=artifact_language,
                    metadata={},
                )
                result = prover.check(artifact)
                if result.ok:
                    emit_ok(obl.name, result.duration_ms)
                else:
                    msg = result.diagnostics[0].message if result.diagnostics else "check failed"
                    emit_fail(obl.name, msg, result.duration_ms)

        if echo:
            print()
            print(
                "summary:",
                f"obligations={total_obligations}",
                f"pass={total_ok}",
                f"fail={total_fail}",
                f"run_id={run.run_id}",
                f"telemetry_dir={run.run_dir}",
                sep=" ",
            )

    return CheckProgrammaticResult(
        run_id=captured_run_id,
        runs_dir=captured_runs_root,
        results=tuple(rows),
        parse_failed=parse_failed,
        total_ok=total_ok,
        total_fail=total_fail,
        total_obligations=total_obligations,
    )


def check_exit_code(res: CheckProgrammaticResult) -> int:
    if res.parse_failed:
        return 2
    if res.total_fail > 0:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class SolveOutcomeEntry:
    obligation_name: str
    kind: str
    iterations: int
    total_cost_usd: str
    final_artifact: str | None
    diagnostic: str | None


@dataclass(frozen=True, slots=True)
class SolveProgrammaticResult:
    run_id: str
    runs_dir: Path
    outcomes: tuple[SolveOutcomeEntry, ...]
    total_cost_usd: str
    duration_ms: int
    parse_failed: bool
    obligations_total: int
    obligations_solved: int


def _outcome_kind_str(kind: OracleOutcomeKind) -> str:
    return kind.value


def _oracle_line_text(o: OracleOutcome) -> str:
    k = len(o.attempts)
    cost_s = f"${o.total_cost_usd}"
    name = o.obligation.name
    if o.kind is OracleOutcomeKind.SOLVED:
        return f"{name}: SOLVED in {k} iterations, {cost_s}"
    reason = o.diagnostic or ""
    return f"{name}: {o.kind.value.upper()} after {k} iterations: {reason}"


async def run_solve_programmatic(
    *,
    paths: list[Path],
    runs_dir: Path,
    prover_name: str,
    provider: str,
    model: str | None,
    llm_config_path: Path | None,
    cache: LLMCache | None,
    max_repair_attempts: int,
    max_tokens: int,
    temperature: float,
    save_transcripts: bool,
    max_cost_usd: str | None,
    echo: bool = True,
) -> SolveProgrammaticResult:
    """Async core for ``kalinov solve``."""
    t0 = time.perf_counter()

    try:
        llm_cfg = load_llm_config(llm_config_path)
    except ConfigError:
        raise

    if provider not in llm_cfg.providers:
        raise ConfigError(f"unknown provider {provider!r}")

    prov_entry = llm_cfg.providers[provider]
    resolved_model = model or prov_entry.default_model

    if prover_name == "lean4":
        tc = detect_toolchain()
        prover: Prover = LeanProver(toolchain=tc)
    else:
        prover = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))

    oc = OracleConfig(
        max_repair_attempts=max_repair_attempts,
        max_tokens_per_call=max_tokens,
        temperature=temperature,
        save_transcripts=save_transcripts,
    )

    max_cost: Decimal | None = None
    if max_cost_usd is not None:
        max_cost = Decimal(str(max_cost_usd))

    try:
        client = make_client(provider, config=llm_cfg, cache=cache)
    except ConfigError as exc:
        raise ClientConfigError(str(exc)) from exc

    chain = _interpret_chain()
    parse_failed = False
    obligations_total = 0
    solved = 0
    sum_usd = Decimal("0")
    outcome_rows: list[SolveOutcomeEntry] = []
    captured_run_id = ""
    captured_runs_root = runs_dir.resolve()

    with start_run(runs_dir=runs_dir) as run:
        captured_run_id = run.run_id
        captured_runs_root = run.runs_root
        guard: BudgetGuard | None = None
        if max_cost is not None:
            guard = BudgetGuard(Budget(max_cost_usd=max_cost))
        set_budget_guard(guard)
        try:
            loop = OracleLoop(prover=prover, llm=client, model=resolved_model, config=oc)
            for path in paths:
                try:
                    ff = parse_feature_file(path)
                except GherkinParseError as exc:
                    if echo:
                        print(f"parse error in {path}: {exc}", file=sys.stderr)
                    parse_failed = True
                    continue

                interpreted: list[InterpretedStep] = []
                for scenario in ff.feature.scenarios:
                    for step in scenario.steps:
                        interpreted.append(chain.interpret(step))

                spec = SpecDocument(
                    feature_file=ff,
                    interpreted_steps=tuple(interpreted),
                )
                try:
                    obligations = prover.extract_obligations(spec)
                except ProverError as exc:
                    if echo:
                        print(f"error in {path}: {exc}", file=sys.stderr)
                    parse_failed = True
                    continue

                for obl in obligations:
                    obligations_total += 1
                    out = await loop.run(obl)
                    if echo:
                        print(_oracle_line_text(out))
                    sum_usd += out.total_cost_usd
                    if out.kind is OracleOutcomeKind.SOLVED:
                        solved += 1
                    body = None
                    if out.final_artifact is not None:
                        body = out.final_artifact.body
                    outcome_rows.append(
                        SolveOutcomeEntry(
                            obligation_name=out.obligation.name,
                            kind=_outcome_kind_str(out.kind),
                            iterations=len(out.attempts),
                            total_cost_usd=str(out.total_cost_usd),
                            final_artifact=body,
                            diagnostic=out.diagnostic,
                        ),
                    )
        finally:
            set_budget_guard(None)

        manifest = {
            "run_id": run.run_id,
            "total_cost_usd": str(sum_usd),
            "obligations_total": obligations_total,
            "obligations_solved": solved,
        }
        (run.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if echo:
            print()
            print(
                "summary:",
                f"obligations={obligations_total}",
                f"solved={solved}",
                f"total_usd={sum_usd}",
                f"run_id={run.run_id}",
                f"telemetry_dir={run.run_dir}",
                sep=" ",
            )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return SolveProgrammaticResult(
        run_id=captured_run_id,
        runs_dir=captured_runs_root,
        outcomes=tuple(outcome_rows),
        total_cost_usd=str(sum_usd),
        duration_ms=elapsed_ms,
        parse_failed=parse_failed,
        obligations_total=obligations_total,
        obligations_solved=solved,
    )


def solve_exit_code(res: SolveProgrammaticResult) -> int:
    if res.parse_failed:
        return 2
    if res.obligations_solved < res.obligations_total:
        return 1
    return 0


__all__ = [
    "CheckObligationResult",
    "CheckProgrammaticResult",
    "ClientConfigError",
    "SolveOutcomeEntry",
    "SolveProgrammaticResult",
    "check_exit_code",
    "run_check_programmatic",
    "run_solve_programmatic",
    "solve_exit_code",
]
