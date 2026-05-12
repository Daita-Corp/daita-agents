from decimal import Decimal

import pytest

from daita.llm.pricing import (
    PricingRegistry,
    TokenUsage,
    estimate_llm_cost,
    estimate_llm_cost_from_counts,
)


def test_openai_gpt_4o_mini_uses_input_and_output_rates():
    estimate = estimate_llm_cost_from_counts(
        "openai",
        "gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    assert estimate.usd == Decimal("0.75")
    assert estimate.pricing_confidence == "medium"
    assert estimate.pricing_model == "gpt-4o-mini*"


def test_cached_input_tokens_use_cached_rate_when_available():
    estimate = estimate_llm_cost_from_counts(
        "openai",
        "gpt-4o-mini",
        input_tokens=1_000_000,
        cached_input_tokens=500_000,
    )

    assert estimate.usd == Decimal("0.1125")


def test_unknown_model_uses_explicit_low_confidence_fallback():
    estimate = estimate_llm_cost(
        "provider-x",
        "new-model",
        TokenUsage.from_counts(input_tokens=1_000, output_tokens=1_000),
    )

    assert estimate.as_float() == pytest.approx(0.004)
    assert estimate.pricing_confidence == "low"
    assert estimate.warning


def test_registry_can_be_extended_with_catalog_data():
    registry = PricingRegistry(
        catalog={
            "test": {
                "tiny-*": {
                    "input_per_1m": "0.01",
                    "output_per_1m": "0.02",
                    "source": "unit-test",
                }
            }
        }
    )

    estimate = registry.estimate(
        "test",
        "tiny-v1",
        TokenUsage.from_counts(input_tokens=1_000_000, output_tokens=1_000_000),
    )

    assert estimate.usd == Decimal("0.03")
    assert estimate.pricing_source == "unit-test"

