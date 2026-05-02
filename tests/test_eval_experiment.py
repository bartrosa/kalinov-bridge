"""Experiment YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalinov.eval.experiment import ExperimentError, load_experiment, matrix_from_experiment_mapping
from kalinov.oracle.strategy import OracleConfig


def test_load_experiment_file(repo_root: Path, tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text(
        f"suite: {repo_root / 'evals' / 'suites' / 'smoke.yaml'}\n"
        "matrix:\n"
        "  provers: ['null']\n"
        "  providers:\n"
        "    - { name: a }\n"
        "  seeds: [0]\n"
        "  oracle_configs:\n"
        "    - max_repair_attempts: 2\n"
        "out: ./out\n",
        encoding="utf-8",
    )
    spec = load_experiment(p)
    assert spec.out_dir == (tmp_path / "out").resolve()
    assert spec.matrix.provers == ("null",)
    (oc,) = spec.matrix.oracle_configs
    assert isinstance(oc, OracleConfig)
    assert oc.max_repair_attempts == 2


def test_experiment_missing_suite_errors(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text(
        "suite: ./nope.yaml\n"
        "matrix:\n  provers: [null]\n  providers: [{name: a}]\n  seeds: [0]\n"
        "  oracle_configs: [{}]\n"
        "out: ./o\n",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentError):
        load_experiment(p)


def test_matrix_from_experiment_mapping_oracle_list() -> None:
    m = matrix_from_experiment_mapping(
        {
            "provers": ["null"],
            "providers": [{"name": "p", "model": "m"}],
            "seeds": [1],
            "oracle_configs": [{"max_repair_attempts": 4}],
        },
    )
    (oc,) = m.oracle_configs
    assert oc.max_repair_attempts == 4
