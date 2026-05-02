"""``kalinov`` CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kalinov")
    sub = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args(argv)
    if args.command == "check":
        return _parse_files(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
