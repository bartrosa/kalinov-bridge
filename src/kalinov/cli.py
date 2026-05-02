"""``kalinov`` CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kalinov.gherkin import parse_feature_file
from kalinov.gherkin.errors import GherkinParseError
from kalinov.interpreters import (
    ForTheLInterpreter,
    InterpreterChain,
    MathTexInterpreter,
    RawInterpreter,
)
from kalinov.interpreters.base import InterpretedStep
from kalinov.provers import NullProver, NullProverConfig, NullProverMode, ProofArtifact
from kalinov.provers.base import Prover, SpecDocument
from kalinov.provers.lean import LeanProver, ToolchainNotFoundError, detect_toolchain
from kalinov.telemetry import start_run


def _interpret_chain() -> InterpreterChain:
    return InterpreterChain(
        [
            MathTexInterpreter(),
            ForTheLInterpreter(),
            RawInterpreter(),
        ],
    )


def _run_checks(prover: Prover, paths: list[Path], artifact_language: str, runs_dir: Path) -> int:
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
            obligations = prover.extract_obligations(spec)
            for obl in obligations:
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
        return _run_checks(prover, paths, "null", runs_dir)

    if args.prover == "lean4":
        try:
            tc = detect_toolchain()
        except ToolchainNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        prover = LeanProver(toolchain=tc)
        return _run_checks(prover, paths, "lean4", runs_dir)

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

    args = parser.parse_args(argv)
    if args.command == "check":
        return _parse_files(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
