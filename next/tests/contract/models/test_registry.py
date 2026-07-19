from __future__ import annotations

import pytest

import daita
from daita.llm import (
    AnthropicProvider,
    GeminiProvider,
    GrokProvider,
    ModelProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    create_llm_provider,
)


@pytest.mark.parametrize(
    ("model_id", "expected_type"),
    (
        ("openai:gpt-test", OpenAIProvider),
        ("anthropic:claude-test", AnthropicProvider),
        ("gemini:gemini-test", GeminiProvider),
        ("grok:grok-test", GrokProvider),
        ("ollama:llama-test", OllamaProvider),
    ),
)
def test_factory_constructs_retained_adapters_without_sdk_io(
    model_id: str,
    expected_type: type[object],
) -> None:
    provider: ModelProvider = create_llm_provider(model_id)

    assert isinstance(provider, expected_type)
    assert provider.provider_id == model_id


def test_factory_requires_explicit_secure_compatible_endpoint() -> None:
    with pytest.raises(ValueError, match="explicit base_url"):
        create_llm_provider("acme:model")
    with pytest.raises(ValueError, match="HTTP.*loopback"):
        create_llm_provider("acme:model", base_url="http://models.example/v1")

    provider = create_llm_provider(
        "acme:model",
        base_url="https://models.example/v1",
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.provider_id == "acme:model"


def test_factory_preserves_fixed_endpoint_and_loopback_boundaries() -> None:
    with pytest.raises(ValueError, match="fixed provider endpoint"):
        create_llm_provider(
            "openai:gpt-test",
            base_url="https://proxy.example/v1",
        )
    with pytest.raises(ValueError, match="loopback"):
        create_llm_provider(
            "ollama:llama-test",
            base_url="http://models.example/v1",
        )


@pytest.mark.parametrize(
    "model_id",
    ("", "openai", "OpenAI:model", "openai:bad model", " openai:model"),
)
def test_factory_rejects_noncanonical_model_ids(model_id: str) -> None:
    with pytest.raises(ValueError, match="canonical|provider:model"):
        create_llm_provider(model_id)


def test_factory_is_the_same_public_root_and_llm_surface() -> None:
    assert daita.create_llm_provider is create_llm_provider
