"""Turn token usage + catalogue row into USD :class:`CostBreakdown`."""

from __future__ import annotations

from decimal import Decimal

from kalinov.cost.catalogue import PricingCatalogue
from kalinov.cost.models import CostBreakdown, TokenUsage

_MTOK = Decimal("1000000")


def estimate_cost(
    usage: TokenUsage,
    *,
    provider: str,
    model_id: str,
    catalogue: PricingCatalogue,
) -> CostBreakdown:
    """Compute USD cost using catalogue rates (zeros for self-hosted wildcard)."""
    row = catalogue.row_for(provider, model_id)
    if row is None:
        return CostBreakdown(
            total_usd=Decimal("0"),
            input_usd=Decimal("0"),
            output_usd=Decimal("0"),
            reasoning_usd=Decimal("0"),
            cache_read_usd=Decimal("0"),
            cache_write_usd=Decimal("0"),
            pricing_source="unknown",
        )

    src = "self_hosted" if provider == "openai_compat" else "catalogue"

    inp_u = Decimal(usage.input) / _MTOK * row.input_per_mtok
    out_u = Decimal(usage.output) / _MTOK * row.output_per_mtok
    rsn_u = Decimal(usage.reasoning) / _MTOK * row.reasoning_per_mtok
    cr_u = Decimal(usage.cache_read) / _MTOK * row.cache_read_per_mtok
    cw_u = Decimal(usage.cache_write) / _MTOK * row.cache_write_per_mtok
    total = inp_u + out_u + rsn_u + cr_u + cw_u

    return CostBreakdown(
        total_usd=total,
        input_usd=inp_u,
        output_usd=out_u,
        reasoning_usd=rsn_u,
        cache_read_usd=cr_u,
        cache_write_usd=cw_u,
        pricing_source=src,
    )
