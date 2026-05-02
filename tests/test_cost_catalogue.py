"""Pricing catalogue and calculator tests."""

from __future__ import annotations

from decimal import Decimal

from kalinov.cost import TokenUsage, estimate_cost, load_default_catalogue


def test_load_bundled_yaml() -> None:
    c = load_default_catalogue()
    row = c.row_for("openai", "gpt-4o")
    assert row is not None
    assert row.input_per_mtok > 0


def test_openai_compat_wildcard_zero() -> None:
    c = load_default_catalogue()
    u = TokenUsage(input=1_000_000, output=1_000_000)
    out = estimate_cost(
        u,
        provider="openai_compat",
        model_id="any",
        catalogue=c,
    )
    assert out.total_usd == 0
    assert out.pricing_source == "self_hosted"


def test_estimate_nonzero() -> None:
    c = load_default_catalogue()
    u = TokenUsage(input=1_000_000, output=0)
    out = estimate_cost(
        u,
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        catalogue=c,
    )
    assert out.total_usd == Decimal("3.00")
