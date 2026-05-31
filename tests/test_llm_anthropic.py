"""Anthropic adapter tests (mocked SDK)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from kalinov.cost.calculator import estimate_cost
from kalinov.cost.catalogue import (
    ModelPricingRow,
    PricingCatalogue,
    load_default_catalogue,
)
from kalinov.cost.models import TokenUsage
from kalinov.llm.base import LLMError, Message
from kalinov.llm.cache import CacheMode, LLMCache
from kalinov.llm.providers.anthropic_client import AnthropicClient
from kalinov.telemetry import start_run


def _fake_usage(**kw: Any) -> SimpleNamespace:
    """Build a Usage stub matching the real Anthropic SDK shape.

    Real ``anthropic.types.Usage`` exposes thinking tokens via the nested
    ``output_tokens_details.thinking_tokens`` field — there is no top-level
    ``thinking_tokens`` attribute. Tests must mirror that shape so we don't
    accidentally pin behaviour that only works against the unrealistic stub.
    """
    base: dict[str, Any] = {
        "input_tokens": 10,
        "output_tokens": 6,
        "cache_read_input_tokens": 1,
        "cache_creation_input_tokens": 2,
    }
    thinking = int(kw.pop("thinking_tokens", 0))
    base.update(kw)
    base["output_tokens_details"] = SimpleNamespace(thinking_tokens=thinking)
    return SimpleNamespace(**base)


def test_maps_usage_and_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    text_block = SimpleNamespace(type="text", text="Hello")
    resp = SimpleNamespace(
        model="claude-3-5-sonnet-20241022",
        content=[text_block],
        usage=_fake_usage(),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: resp)

    out = client.complete(
        messages=[Message(role="user", content="hi")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        temperature=0.5,
        stop=None,
        extras=None,
    )
    assert out.text == "Hello"
    assert out.model_id_resolved == "claude-3-5-sonnet-20241022"
    assert out.usage.input == 10
    assert out.usage.cache_read == 1


def test_rate_limit_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp429 = httpx.Response(429, request=req)

    def boom(**kw: object) -> None:
        raise anthropic.RateLimitError("slow down", response=resp429, body={})

    monkeypatch.setattr(client._client.messages, "create", boom)
    with pytest.raises(LLMError) as ei:
        client.complete(
            messages=[Message(role="user", content="x")],
            model="m",
            max_tokens=10,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "rate_limit"


def test_auth_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp401 = httpx.Response(401, request=req)

    def boom(**kw: object) -> None:
        raise anthropic.AuthenticationError("nope", response=resp401, body={})

    monkeypatch.setattr(client._client.messages, "create", boom)
    with pytest.raises(LLMError) as ei:
        client.complete(
            messages=[Message(role="user", content="x")],
            model="m",
            max_tokens=10,
            temperature=None,
            stop=None,
            extras=None,
        )
    assert ei.value.code == "auth"


def test_cache_hit_short_circuits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def mk(**kw: object) -> SimpleNamespace:
        calls.append(1)
        return SimpleNamespace(
            model="claude-3-5-sonnet-20241022",
            content=[SimpleNamespace(type="text", text="cached")],
            usage=_fake_usage(input_tokens=2, output_tokens=3),
        )

    def boom(**kw: object) -> None:
        raise AssertionError("SDK must not run on cache hit")

    cat = load_default_catalogue()
    cache = LLMCache(tmp_path / "c", mode=CacheMode.READ_WRITE)
    client = AnthropicClient(api_key="k", catalogue=cat, cache=cache)
    monkeypatch.setattr(client._client.messages, "create", mk)

    first = client.complete(
        messages=[Message(role="user", content="same")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        temperature=None,
        stop=None,
        extras=None,
    )
    assert first.text == "cached"
    assert calls == [1]

    monkeypatch.setattr(client._client.messages, "create", boom)

    second = client.complete(
        messages=[Message(role="user", content="same")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        temperature=None,
        stop=None,
        extras=None,
    )
    assert second.text == "cached"
    assert calls == [1]


def test_telemetry_with_active_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    resp = SimpleNamespace(
        model="m",
        content=[SimpleNamespace(type="text", text="t")],
        usage=_fake_usage(),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: resp)
    with start_run(runs_dir=tmp_path):
        client.complete(
            messages=[Message(role="user", content="u")],
            model="m",
            max_tokens=5,
            temperature=None,
            stop=None,
            extras=None,
        )
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert run_dirs
    log = run_dirs[0] / "llm_calls.jsonl"
    assert log.is_file()


def test_extended_thinking_tokens_recorded_under_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extended-thinking tokens must land in ``TokenUsage.reasoning``.

    Anthropic's ``Usage`` returns thinking tokens nested under
    ``output_tokens_details.thinking_tokens`` and counts them inside
    ``output_tokens`` (the billed total). The previous adapter probed for a
    non-existent top-level ``thinking_tokens`` field, so it always read 0 and
    silently lumped every thinking token into the ``output`` bucket.

    Concrete trigger: a Claude run with extended thinking enabled (e.g.
    ``extras={"extended_thinking_budget_tokens": 5000}``) where the response
    reports ``output_tokens=1000`` of which 700 are thinking. Without the
    fix:

      * per-bucket telemetry (``llm_calls.jsonl`` / ``kalinov cost report``)
        misclassifies thinking tokens as visible output, breaking any
        downstream analysis that relies on the breakdown,
      * any custom ``pricing.yaml`` that prices ``reasoning_per_mtok``
        differently from ``output_per_mtok`` (e.g. an enterprise tier that
        bills thinking at a different rate) computes the wrong USD total.
    """
    cat = load_default_catalogue()
    client = AnthropicClient(api_key="k", catalogue=cat)
    resp = SimpleNamespace(
        model="claude-3-5-sonnet-20241022",
        content=[SimpleNamespace(type="text", text="answer")],
        usage=_fake_usage(
            input_tokens=20,
            output_tokens=1000,
            thinking_tokens=700,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: resp)

    out = client.complete(
        messages=[Message(role="user", content="solve")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        temperature=None,
        stop=None,
        extras={"extended_thinking_budget_tokens": 5000},
    )

    assert out.usage.reasoning == 700, (
        "thinking tokens must be classified as `reasoning`, not silently "
        "dropped into `output` (matches Gemini / OpenAI adapter semantics)."
    )
    assert out.usage.output == 300, (
        "visible output is the billed `output_tokens` minus the thinking "
        "portion; storing the full output_tokens here would also corrupt "
        "the per-bucket cost breakdown."
    )
    assert out.usage.input == 20
    # Total tokens are unchanged either way (output_tokens already includes
    # thinking on Anthropic), but the per-bucket split now mirrors reality.
    assert out.usage.total_all() == 20 + 300 + 700


def test_extended_thinking_priced_at_reasoning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom pricing that splits reasoning vs output must apply to thinking.

    Builds a synthetic catalogue where ``reasoning_per_mtok`` differs from
    ``output_per_mtok`` and verifies that a Claude response with thinking
    tokens is billed at the reasoning rate for those tokens. Pre-fix the
    thinking tokens were attributed to ``output`` and therefore priced at the
    output rate, silently mis-billing the run.
    """
    pricing = PricingCatalogue(
        raw={},
        models={
            "anthropic": {
                "claude-3-5-sonnet-20241022": ModelPricingRow(
                    input_per_mtok=Decimal("3.00"),
                    output_per_mtok=Decimal("15.00"),
                    reasoning_per_mtok=Decimal("60.00"),
                    cache_read_per_mtok=Decimal("0"),
                    cache_write_per_mtok=Decimal("0"),
                ),
            },
        },
    )
    client = AnthropicClient(api_key="k", catalogue=pricing)
    resp = SimpleNamespace(
        model="claude-3-5-sonnet-20241022",
        content=[SimpleNamespace(type="text", text="t")],
        usage=_fake_usage(
            input_tokens=0,
            output_tokens=1000,
            thinking_tokens=900,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    monkeypatch.setattr(client._client.messages, "create", lambda **kw: resp)

    out = client.complete(
        messages=[Message(role="user", content="q")],
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        temperature=None,
        stop=None,
        extras={"extended_thinking_budget_tokens": 5000},
    )

    cost = estimate_cost(
        out.usage,
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        catalogue=pricing,
    )
    # 100 visible output @ $15/M = $0.0015; 900 thinking @ $60/M = $0.054.
    expected = Decimal("0.0015") + Decimal("0.054")
    assert cost.total_usd == expected, (
        f"custom reasoning rate must apply to thinking tokens "
        f"(got {cost.total_usd}, want {expected})"
    )
    assert cost.reasoning_usd == Decimal("0.054")
    assert cost.output_usd == Decimal("0.0015")
    # Sanity: budget guard's token cap still sees every billed token.
    assert (
        TokenUsage(
            input=out.usage.input,
            output=out.usage.output,
            reasoning=out.usage.reasoning,
        ).total_non_cache()
        == 1000
    )
