from __future__ import annotations

from decimal import Decimal

import pytest

from daita.llm import ModelProfile


def test_model_profile_records_exact_limits_capabilities_and_costs() -> None:
    profile = ModelProfile(
        id="openai:gpt-test",
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        supports_tools=True,
        supports_parallel_tools=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=True,
        supports_prompt_caching=True,
        supports_native_continuation=True,
        input_cost_per_million_usd=Decimal("0.40"),
        output_cost_per_million_usd=Decimal("1.60"),
        data_routing_classification="approved_remote",
    )

    assert profile.maximum_input_tokens == 123_904
    assert profile.supports_parallel_tools
    assert profile.input_cost_per_million_usd == Decimal("0.40")
    assert profile.data_routing_classification == "approved_remote"


@pytest.mark.parametrize(
    "identity",
    ("missing-separator", ":missing-provider", "UPPER:model", "openai: "),
)
def test_model_profile_requires_canonical_provider_model_identity(
    identity: str,
) -> None:
    with pytest.raises(ValueError, match="provider:model"):
        ModelProfile(
            id=identity,
            context_window_tokens=100,
            max_output_tokens=10,
        )


def test_model_profile_fails_invalid_limits_flags_and_costs_closed() -> None:
    with pytest.raises(ValueError, match="positive model input"):
        ModelProfile(
            id="mock:small",
            context_window_tokens=100,
            max_output_tokens=100,
        )
    with pytest.raises(ValueError, match="parallel tool"):
        ModelProfile(
            id="mock:small",
            context_window_tokens=100,
            max_output_tokens=10,
            supports_parallel_tools=True,
        )
    with pytest.raises(TypeError, match="Decimal"):
        ModelProfile(
            id="mock:small",
            context_window_tokens=100,
            max_output_tokens=10,
            input_cost_per_million_usd=0.1,  # type: ignore[arg-type]
        )
