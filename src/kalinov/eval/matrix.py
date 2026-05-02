"""Cartesian configuration grid for multi-model / multi-seed evals."""

from __future__ import annotations

from dataclasses import dataclass

from kalinov.oracle.strategy import OracleConfig


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """A single point in the configuration space."""

    prover_name: str  # "null" | "lean4"
    provider_name: str
    model: str | None  # None = use provider default
    seed: int  # reserved for reproducibility hooks on adapters
    oracle: OracleConfig
    label: str


@dataclass(frozen=True, slots=True)
class ConfigMatrix:
    """Cartesian product over selected dimensions."""

    provers: tuple[str, ...]
    providers: tuple[tuple[str, str | None], ...]
    seeds: tuple[int, ...]
    oracle_configs: tuple[OracleConfig, ...]

    def expand(self) -> tuple[EvalConfig, ...]:
        """Return one :class:`EvalConfig` per combination with an auto-generated label."""
        out: list[EvalConfig] = []
        seen: set[str] = set()
        for prover in self.provers:
            for provider_name, model in self.providers:
                for seed in self.seeds:
                    for oracle in self.oracle_configs:
                        label = _make_label(prover, provider_name, model, seed, oracle)
                        if label in seen:
                            label = f"{label}__{len(seen)}"
                        seen.add(label)
                        out.append(
                            EvalConfig(
                                prover_name=prover,
                                provider_name=provider_name,
                                model=model,
                                seed=seed,
                                oracle=oracle,
                                label=label,
                            ),
                        )
        return tuple(out)


def _make_label(
    prover: str,
    provider: str,
    model: str | None,
    seed: int,
    oracle: OracleConfig,
) -> str:
    m = model or "default_model"
    safe_m = m.replace("/", "_").replace(" ", "_")
    return (
        f"{prover}__{provider}__{safe_m}__s{seed}"
        f"__r{oracle.max_repair_attempts}_t{oracle.temperature}"
    )


__all__ = ["ConfigMatrix", "EvalConfig"]
