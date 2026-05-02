"""Config matrix expansion."""

from __future__ import annotations

from kalinov.eval.matrix import ConfigMatrix
from kalinov.oracle.strategy import OracleConfig


def test_expand_cartesian_product() -> None:
    m = ConfigMatrix(
        provers=("null",),
        providers=(("a", None), ("b", "gpt-4o")),
        seeds=(1, 2),
        oracle_configs=(
            OracleConfig(max_repair_attempts=1),
            OracleConfig(max_repair_attempts=2),
        ),
    )
    cfgs = m.expand()
    assert len(cfgs) == 1 * 2 * 2 * 2  # 8
    labels = [c.label for c in cfgs]
    assert len(set(labels)) == len(labels)


def test_expand_single_config() -> None:
    m = ConfigMatrix(
        provers=("null",),
        providers=(("p", None),),
        seeds=(42,),
        oracle_configs=(OracleConfig(),),
    )
    cfgs = m.expand()
    assert len(cfgs) == 1
    assert cfgs[0].provider_name == "p"
    assert cfgs[0].seed == 42
