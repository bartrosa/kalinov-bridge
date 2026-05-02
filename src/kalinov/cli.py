"""``kalinov`` CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

import httpx

from kalinov.bridges.forthel_lean import TranslationOutcomeKind, translate_step
from kalinov.eval.cli_impl import run_eval
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
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.config import ConfigError
from kalinov.llm.config import load_config as load_llm_config
from kalinov.llm.factory import make_client
from kalinov.llm.run_report import run_cost_report
from kalinov.mining import MiningConfig, MiningError, mine
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
from kalinov.provers.lean import LeanProver, ToolchainNotFoundError, detect_toolchain
from kalinov.telemetry import start_run


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


def _run_checks(
    prover: Prover,
    paths: list[Path],
    artifact_language: str,
    runs_dir: Path,
    *,
    forthel_bridge: bool = False,
) -> int:
    chain = _interpret_chain()
    total_ok = 0
    total_fail = 0
    total_obligations = 0
    parse_failed = False

    with start_run(runs_dir=runs_dir) as run:
        for path in paths:
            try:
                ff = parse_feature_file(path)
            except GherkinParseError as exc:
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
                                print(f"SKIP FORTHEL {name}: {tout.diagnostic}")
                            elif tout.kind == TranslationOutcomeKind.FAILED:
                                total_fail += 1
                                msg = tout.diagnostic or "forthel translation failed"
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
                                    total_ok += 1
                                    print(f"OK {obl.name}")
                                else:
                                    total_fail += 1
                                    msg = (
                                        result.diagnostics[0].message
                                        if result.diagnostics
                                        else "check failed"
                                    )
                                    print(f"FAIL {obl.name}: {msg}")
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
                    total_ok += 1
                    print(f"OK {obl.name}")
                else:
                    total_fail += 1
                    msg = result.diagnostics[0].message if result.diagnostics else "check failed"
                    print(f"FAIL {obl.name}: {msg}")

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

    if parse_failed:
        return 2
    if total_fail > 0:
        return 1
    return 0


def _parse_files(args: argparse.Namespace) -> int:
    paths = [Path(s) for s in args.files]
    for path in paths:
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2

    runs_dir = Path(args.runs_dir)

    if args.prover == "null":
        mode_map = {
            "always_ok": NullProverMode.ALWAYS_OK,
            "always_fail": NullProverMode.ALWAYS_FAIL,
            "fail_after_n": NullProverMode.FAIL_AFTER_N,
        }
        cfg = NullProverConfig(
            mode=mode_map[args.mode],
            fail_after=args.fail_after,
        )
        prover: Prover = NullProver(cfg)
        return _run_checks(prover, paths, "null", runs_dir, forthel_bridge=False)

    if args.prover == "lean4":
        try:
            tc = detect_toolchain()
        except ToolchainNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        prover = LeanProver(toolchain=tc)
        use_bridge = not args.no_forthel
        return _run_checks(prover, paths, "lean4", runs_dir, forthel_bridge=use_bridge)

    print(f"error: unknown prover {args.prover!r}", file=sys.stderr)
    return 2


def _oracle_line(o: OracleOutcome) -> str:
    k = len(o.attempts)
    cost_s = f"${o.total_cost_usd}"
    name = o.obligation.name
    if o.kind is OracleOutcomeKind.SOLVED:
        return f"{name}: SOLVED in {k} iterations, {cost_s}"
    reason = o.diagnostic or ""
    return f"{name}: {o.kind.value.upper()} after {k} iterations: {reason}"


async def _solve_async(args: argparse.Namespace) -> int:
    paths = [Path(s) for s in args.files]
    for path in paths:
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2

    runs_dir = Path(args.runs_dir).resolve()

    if args.cache_mode != "off" and args.cache_dir is None:
        print("error: --cache-dir is required when --cache-mode is not off", file=sys.stderr)
        return 2

    cache: LLMCache | None = None
    if args.cache_mode != "off" and args.cache_dir is not None:
        cache = LLMCache(Path(args.cache_dir), mode=CacheMode(args.cache_mode))

    try:
        llm_cfg = load_llm_config(args.llm_config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.provider not in llm_cfg.providers:
        print(f"error: unknown provider {args.provider!r}", file=sys.stderr)
        return 2
    prov_entry = llm_cfg.providers[args.provider]
    model = args.model or prov_entry.default_model

    if args.prover == "lean4":
        try:
            tc = detect_toolchain()
        except ToolchainNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        prover: Prover = LeanProver(toolchain=tc)
    else:
        prover = NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK))

    oc = OracleConfig(
        max_repair_attempts=args.max_repair_attempts,
        max_tokens_per_call=args.max_tokens,
        temperature=args.temperature,
        save_transcripts=args.save_transcripts,
    )

    max_cost: Decimal | None = None
    if args.max_cost_usd is not None:
        max_cost = Decimal(str(args.max_cost_usd))

    try:
        client = make_client(args.provider, config=llm_cfg, cache=cache)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    chain = _interpret_chain()
    parse_failed = False
    obligations_total = 0
    solved = 0
    sum_usd = Decimal("0")

    with start_run(runs_dir=runs_dir) as run:
        guard: BudgetGuard | None = None
        if max_cost is not None:
            guard = BudgetGuard(Budget(max_cost_usd=max_cost))
        set_budget_guard(guard)
        try:
            loop = OracleLoop(prover=prover, llm=client, model=model, config=oc)
            for path in paths:
                try:
                    ff = parse_feature_file(path)
                except GherkinParseError as exc:
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
                    print(f"error in {path}: {exc}", file=sys.stderr)
                    parse_failed = True
                    continue

                for obl in obligations:
                    obligations_total += 1
                    out = await loop.run(obl)
                    print(_oracle_line(out))
                    sum_usd += out.total_cost_usd
                    if out.kind is OracleOutcomeKind.SOLVED:
                        solved += 1
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

    if parse_failed:
        return 2
    if solved < obligations_total:
        return 1
    return 0


def _run_solve(args: argparse.Namespace) -> int:
    return asyncio.run(_solve_async(args))


def _run_mine(args: argparse.Namespace) -> int:
    if not args.network:
        print(
            "Mining is gated behind --network; run with --network to proceed.",
            file=sys.stderr,
        )
        return 4
    cfg = MiningConfig(
        source_name=args.source,
        query=args.query,
        limit=args.limit,
        extractor_name=args.extractor,
        out_dir=Path(args.out),
        feature_name=args.feature_name,
        network_enabled=True,
    )
    runs_dir = Path(args.runs_dir)

    async def run_inner() -> tuple[Path, ...]:
        with start_run(runs_dir=runs_dir):
            return await mine(cfg)

    try:
        paths = asyncio.run(run_inner())
    except MiningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except httpx.HTTPError as exc:
        print(f"error: HTTP fetch failed: {exc}", file=sys.stderr)
        return 1

    for path in paths:
        text = path.read_text(encoding="utf-8")
        n = sum(1 for line in text.splitlines() if line.lstrip().startswith("Scenario:"))
        print(f"{path} candidates={n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalinov")
    sub = parser.add_subparsers(dest="command", required=True)

    cost = sub.add_parser("cost", help="Inspect recorded LLM spend.")
    cost_sub = cost.add_subparsers(dest="cost_command", required=True)
    cost_rep = cost_sub.add_parser("report", help="Aggregate llm_calls.jsonl under runs.")
    cost_rep.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Runs root directory.",
    )
    cost_rep.add_argument("--run-id", type=str, default=None, help="Single run id folder.")
    cost_rep.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="report_format",
    )
    grp = cost_rep.add_mutually_exclusive_group()
    grp.add_argument(
        "--by-provider",
        action="store_const",
        const="provider",
        dest="group_by",
        help="Group totals by provider name.",
    )
    grp.add_argument(
        "--by-model",
        action="store_const",
        const="model",
        dest="group_by",
        help="Group totals by resolved model id.",
    )
    grp.add_argument(
        "--by-day",
        action="store_const",
        const="day",
        dest="group_by",
        help="Group totals by UTC calendar day (from ts_ms).",
    )
    cost_rep.set_defaults(group_by="none")

    check = sub.add_parser("check", help="Parse features and run prover checks.")
    check.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more .feature files.",
    )
    check.add_argument("--prover", choices=["null", "lean4"], default="null")
    check.add_argument(
        "--mode",
        choices=["always_ok", "always_fail", "fail_after_n"],
        default="always_ok",
        help="Only applies to --prover null.",
    )
    check.add_argument(
        "--fail-after",
        type=int,
        default=0,
        metavar="N",
        help="With fail_after_n: succeed on the first N compile/check calls (shared counter).",
    )
    check.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory under which per-run folders are created.",
    )
    check.add_argument(
        "--no-forthel",
        action="store_true",
        help="Disable ForTheL→Lean translation for --prover lean4 (obligation path only).",
    )

    solve = sub.add_parser("solve", help="LLM oracle: propose → verify → repair per obligation.")
    solve.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more .feature files.",
    )
    solve.add_argument("--prover", choices=["null", "lean4"], required=True)
    solve.add_argument(
        "--provider",
        type=str,
        required=True,
        help="Provider name from kalinov.config.yaml.",
    )
    solve.add_argument("--model", type=str, default=None, help="Override provider default_model.")
    solve.add_argument("--max-repair-attempts", type=int, default=3, metavar="N")
    solve.add_argument(
        "--max-cost-usd",
        type=str,
        default=None,
        metavar="D",
        help="Abort LLM calls when cumulative USD exceeds this budget.",
    )
    solve.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        metavar="N",
        help="Per-completion max_tokens passed to the LLM.",
    )
    solve.add_argument("--temperature", type=float, default=0.0, metavar="T")
    solve.add_argument(
        "--save-transcripts",
        action="store_true",
        help="Write transcripts/<obligation>.json under the run directory.",
    )
    solve.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for the LLM response cache.",
    )
    solve.add_argument(
        "--cache-mode",
        choices=["read_write", "read_only", "off"],
        default="off",
        help="Cache behavior (requires --cache-dir unless off).",
    )
    solve.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory under which per-run folders are created.",
    )
    solve.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="Optional path to kalinov.config.yaml (defaults to search path).",
    )

    ev = sub.add_parser("eval", help="Run a benchmark suite across one or more LLM configurations.")
    ev.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="Path to suite YAML (required without --config-file).",
    )
    ev.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Experiment YAML (suite + matrix + out); CLI flags override when set.",
    )
    ev.add_argument("--prover", choices=["null", "lean4"], default=None)
    ev.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="PROVIDER_NAME",
        help="LLM provider name from kalinov.config.yaml (repeat for several).",
    )
    ev.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for all listed providers.",
    )
    ev.add_argument("--seed", action="append", type=int, default=[], dest="seed")
    ev.add_argument("--max-repair-attempts", type=int, default=None, metavar="N")
    ev.add_argument(
        "--max-cost-usd",
        type=str,
        default=None,
        metavar="D",
        help="Run budget in USD (overrides experiment file).",
    )
    ev.add_argument("--max-tokens", type=int, default=None, metavar="N")
    ev.add_argument("--temperature", type=float, default=None, metavar="T")
    ev.add_argument("--cache-dir", type=Path, default=None)
    ev.add_argument(
        "--cache-mode",
        choices=["read_write", "read_only", "off"],
        default="off",
    )
    ev.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report output directory (required for flag-based invocations).",
    )
    ev.add_argument(
        "--format",
        type=str,
        default="json,md",
        help="Comma-separated report formats: json, md.",
    )
    ev.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="Path to kalinov.config.yaml.",
    )
    ev.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root for per-task eval telemetry runs.",
    )

    mine_p = sub.add_parser(
        "mine",
        help="Mine candidate .feature files from external sources (network opt-in).",
    )
    mine_p.add_argument("--source", type=str, required=True, help="Source id (e.g. arxiv).")
    mine_p.add_argument("--query", type=str, required=True, help="Source-specific query string.")
    mine_p.add_argument("--limit", type=int, default=10, metavar="N")
    mine_p.add_argument(
        "--extractor",
        type=str,
        default="heuristic",
        help="Extractor name (default: heuristic).",
    )
    mine_p.add_argument(
        "--out",
        type=Path,
        default=Path("corpus/mined"),
        help="Output directory for .feature files.",
    )
    mine_p.add_argument(
        "--feature-name",
        type=str,
        default="Mined claims",
        help="Feature title used in emitted Gherkin.",
    )
    mine_p.add_argument(
        "--network",
        action="store_true",
        help="Allow outbound HTTP (required; mining is off by default).",
    )
    mine_p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Runs root for mining.jsonl telemetry.",
    )

    args = parser.parse_args(argv)
    if args.command == "cost" and args.cost_command == "report":
        code, out = run_cost_report(
            runs_dir=args.runs_dir,
            run_id=args.run_id,
            fmt=args.report_format,
            group_by=args.group_by,
        )
        if code != 0:
            print("no matching runs under", args.runs_dir, file=sys.stderr)
            return code
        print(out, end="")
        return 0
    if args.command == "check":
        return _parse_files(args)
    if args.command == "solve":
        return _run_solve(args)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "mine":
        return _run_mine(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
