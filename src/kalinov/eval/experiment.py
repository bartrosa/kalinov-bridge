"""Load experiment YAML (suite path + matrix + output + optional budget)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from kalinov.eval.matrix import ConfigMatrix
from kalinov.llm.budget import Budget
from kalinov.oracle.strategy import OracleConfig, OracleStrategyKind


class ExperimentError(Exception):
    """Invalid experiment YAML."""


def _oracle_from_blob(blob: dict[str, Any] | None) -> OracleConfig:
    if not blob:
        return OracleConfig()
    strat_raw = blob.get("strategy", OracleStrategyKind.PROPOSE_THEN_REPAIR.value)
    strategy = OracleStrategyKind(str(strat_raw))
    extras = blob.get("extras")
    return OracleConfig(
        strategy=strategy,
        max_repair_attempts=int(blob.get("max_repair_attempts", 3)),
        max_tokens_per_call=int(blob.get("max_tokens_per_call", 2048)),
        temperature=float(blob.get("temperature", 0.0)),
        extras=extras if isinstance(extras, dict) else None,
        save_transcripts=bool(blob.get("save_transcripts", False)),
    )


def matrix_from_experiment_mapping(matrix_blob: dict[str, Any]) -> ConfigMatrix:
    prov_raw = matrix_blob.get("providers")
    if not isinstance(prov_raw, list) or not prov_raw:
        raise ExperimentError("matrix.providers must be a non-empty list")
    providers: list[tuple[str, str | None]] = []
    for i, item in enumerate(prov_raw):
        if not isinstance(item, dict):
            raise ExperimentError(f"providers[{i}] must be a mapping")
        name = item.get("name")
        if not name:
            raise ExperimentError(f"providers[{i}].name required")
        model = item.get("model")
        if model is not None:
            model = str(model)
        providers.append((str(name), model))

    provers_raw = matrix_blob.get("provers")
    if not isinstance(provers_raw, list) or not provers_raw:
        raise ExperimentError("matrix.provers must be a non-empty list")
    provers = tuple(str(x) for x in provers_raw)

    seeds_raw = matrix_blob.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ExperimentError("matrix.seeds must be a non-empty list")
    seeds = tuple(int(x) for x in seeds_raw)

    oc_raw = matrix_blob.get("oracle_configs")
    oracle_configs: tuple[OracleConfig, ...]
    if not isinstance(oc_raw, list) or not oc_raw:
        oracle_configs = (OracleConfig(),)
    else:
        ocs: list[OracleConfig] = []
        for x in oc_raw:
            ocs.append(_oracle_from_blob(x) if isinstance(x, dict) else OracleConfig())
        oracle_configs = tuple(ocs)

    return ConfigMatrix(
        provers=provers,
        providers=tuple(providers),
        seeds=seeds,
        oracle_configs=oracle_configs,
    )


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    suite_path: Path
    matrix: ConfigMatrix
    out_dir: Path
    budget: Budget | None


def load_experiment(path: str | Path) -> ExperimentSpec:
    p = Path(path).resolve()
    if not p.is_file():
        raise ExperimentError(f"experiment file not found: {p}")
    base = p.parent
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentError(f"invalid YAML in {p}") from exc
    if not isinstance(raw, dict):
        raise ExperimentError("experiment root must be a mapping")

    suite_rel = raw.get("suite")
    if not suite_rel:
        raise ExperimentError("missing suite path")
    suite_path = (base / str(suite_rel)).resolve()
    if not suite_path.is_file():
        raise ExperimentError(f"suite file not found: {suite_path}")

    matrix_blob = raw.get("matrix")
    if not isinstance(matrix_blob, dict):
        raise ExperimentError("missing matrix mapping")
    matrix = matrix_from_experiment_mapping(matrix_blob)

    out_rel = raw.get("out")
    if out_rel is None:
        raise ExperimentError("missing out directory")
    out_dir = (base / str(out_rel)).resolve()

    budget: Budget | None = None
    budget_blob = raw.get("budget")
    if isinstance(budget_blob, dict):
        max_cost = budget_blob.get("max_cost_usd")
        max_tok = budget_blob.get("max_total_tokens")
        max_calls = budget_blob.get("max_calls")
        budget = Budget(
            max_cost_usd=Decimal(str(max_cost)) if max_cost is not None else None,
            max_total_tokens=int(max_tok) if max_tok is not None else None,
            max_calls=int(max_calls) if max_calls is not None else None,
        )
        if (
            budget.max_cost_usd is None
            and budget.max_total_tokens is None
            and budget.max_calls is None
        ):
            budget = None

    return ExperimentSpec(suite_path=suite_path, matrix=matrix, out_dir=out_dir, budget=budget)


__all__ = ["ExperimentError", "ExperimentSpec", "load_experiment", "matrix_from_experiment_mapping"]
