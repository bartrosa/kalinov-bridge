"""Load YAML pricing tables into a lookup structure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import yaml


class PricingSchemaError(Exception):
    """Invalid pricing configuration."""


@dataclass(frozen=True, slots=True)
class ModelPricingRow:
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    reasoning_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    cache_write_per_mtok: Decimal


@dataclass(frozen=True, slots=True)
class PricingCatalogue:
    """Resolved pricing keyed by ``provider`` → ``model_id`` → rates."""

    raw: Mapping[str, Any]
    models: Mapping[str, Mapping[str, ModelPricingRow]]
    wildcard_self_hosted: ClassVar[str] = "self_hosted"

    def row_for(self, provider: str, model_id: str) -> ModelPricingRow | None:
        prov = self.models.get(provider)
        if prov is None:
            return None
        if model_id in prov:
            return prov[model_id]
        if provider == "openai_compat" and "*" in prov:
            return prov["*"]
        return None


def _dec(val: Any) -> Decimal:
    if isinstance(val, Decimal):
        return val
    if isinstance(val, (int, float)):
        return Decimal(str(val))
    if isinstance(val, str):
        return Decimal(val)
    raise PricingSchemaError(f"not a decimal-compatible value: {val!r}")


def _parse_providers_blob(data: Mapping[str, Any]) -> dict[str, dict[str, ModelPricingRow]]:
    provs = data.get("providers")
    if not isinstance(provs, Mapping):
        raise PricingSchemaError("pricing.yaml: missing 'providers' mapping")
    out: dict[str, dict[str, ModelPricingRow]] = {}
    for p_name, models_blob in provs.items():
        if not isinstance(models_blob, Mapping):
            raise PricingSchemaError(f"providers.{p_name} must be a mapping")
        out[str(p_name)] = {}
        for m_name, row in models_blob.items():
            if not isinstance(row, Mapping):
                raise PricingSchemaError(f"providers.{p_name}.{m_name} must be a mapping")
            if row.get("pricing_source") == PricingCatalogue.wildcard_self_hosted or m_name == "*":
                out[str(p_name)][str(m_name)] = ModelPricingRow(
                    input_per_mtok=Decimal("0"),
                    output_per_mtok=Decimal("0"),
                    reasoning_per_mtok=Decimal("0"),
                    cache_read_per_mtok=Decimal("0"),
                    cache_write_per_mtok=Decimal("0"),
                )
                continue
            try:
                inp = _dec(row["input_per_mtok"])
                outp = _dec(row["output_per_mtok"])
                rsn = _dec(row.get("reasoning_per_mtok", outp))
                cr = _dec(row.get("cache_read_per_mtok", "0"))
                cw = _dec(row.get("cache_write_per_mtok", "0"))
            except KeyError as exc:
                raise PricingSchemaError(f"pricing row missing field: {exc}") from exc
            out[str(p_name)][str(m_name)] = ModelPricingRow(
                input_per_mtok=inp,
                output_per_mtok=outp,
                reasoning_per_mtok=rsn,
                cache_read_per_mtok=cr,
                cache_write_per_mtok=cw,
            )
    return out


def load_catalogue_from_path(path: Path) -> PricingCatalogue:
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, Mapping):
        raise PricingSchemaError("pricing root must be a mapping")
    models = _parse_providers_blob(data)
    return PricingCatalogue(raw=dict(data), models=models)


def load_default_catalogue() -> PricingCatalogue:
    """Load bundled ``pricing.yaml`` next to this package."""
    here = Path(__file__).resolve().parent
    return load_catalogue_from_path(here / "pricing.yaml")
