from unittest.mock import patch

import pytest

import daita.hosting.embedded as embedded
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
)
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.anthropic import AnthropicMessagesProvider
from daita.llm.providers.gemini import GeminiProvider
from daita.llm.providers.mock import MockModelProvider
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.llm.providers.openai_compatible import OpenAICompatibleProvider
from daita.llm.routing import ModelRoute, ModelRouteCandidate, RetryPolicy
from daita.security import EmptySecretProvider

_REVIEWED_REASONING_MODELS = (
    "openai:gpt-5.6-sol",
    "openai:gpt-5.6-terra",
    "openai:gpt-5.6-luna",
    "gemini:gemini-3.6-flash",
    "gemini:gemini-3.5-flash",
    "gemini:gemini-3.5-flash-lite",
)


@pytest.mark.parametrize("provider_id", _REVIEWED_REASONING_MODELS)
def test_reviewed_reasoning_models_are_marked_as_reasoning(provider_id):
    profile = reviewed_model_profile(provider_id)

    assert profile is not None
    assert profile.supports_reasoning is True


@pytest.mark.parametrize(
    ("supports_reasoning", "expected_output_tokens"),
    ((False, 16), (True, 25_000)),
)
async def test_model_validation_reserves_output_for_reasoning(
    supports_reasoning,
    expected_output_tokens,
):
    profile = ModelProfile(
        id="openai:test-model",
        context_window_tokens=100_000,
        max_output_tokens=50_000,
        supports_tools=True,
        supports_reasoning=supports_reasoning,
    )
    route = ModelRoute(
        (ModelRouteCandidate(provider_id=profile.id, profile=profile),),
        RetryPolicy(attempts=1),
    )
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="validation-call",
                        name="daita_validate_tool_support",
                        arguments={},
                    ),
                ),
                provider_id=profile.id,
            ),
        ),
        provider_id=profile.id,
    )
    captured_routes = []

    def create_provider(validation_route, *, secret_provider):
        del secret_provider
        captured_routes.append(validation_route)
        return provider

    with patch.object(
        embedded,
        "create_model_route_provider",
        side_effect=create_provider,
    ):
        await embedded._validate_model_route(
            route,
            secret_provider=EmptySecretProvider(),
            injected_provider=None,
        )

    assert len(captured_routes) == 1
    assert (
        captured_routes[0].candidates[0].profile.max_output_tokens
        == expected_output_tokens
    )


def test_openai_reasoning_only_incomplete_response_is_an_output_limit():
    provider = OpenAIResponsesProvider("test-model")
    response = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": ({"type": "reasoning"},),
        "output_text": "",
    }

    with pytest.raises(ModelProviderError) as caught:
        provider._decode_response(response)

    assert caught.value.code is ProviderErrorCode.OUTPUT_LIMIT


def test_anthropic_reasoning_only_incomplete_response_is_an_output_limit():
    provider = AnthropicMessagesProvider("test-model")
    response = {
        "type": "message",
        "role": "assistant",
        "id": "response-id",
        "stop_reason": "max_tokens",
        "content": (),
    }

    with pytest.raises(ModelProviderError) as caught:
        provider._decode_response(response)

    assert caught.value.code is ProviderErrorCode.OUTPUT_LIMIT


def test_gemini_reasoning_only_incomplete_response_is_an_output_limit():
    provider = GeminiProvider("test-model")
    response = {
        "prompt_feedback": None,
        "candidates": (
            {
                "finish_reason": "MAX_TOKENS",
                "content": {
                    "parts": (
                        {
                            "thought": True,
                            "text": "internal reasoning",
                        },
                    ),
                },
            },
        ),
    }

    with pytest.raises(ModelProviderError) as caught:
        provider._decode_response(response)

    assert caught.value.code is ProviderErrorCode.OUTPUT_LIMIT


def test_compatible_reasoning_only_incomplete_response_is_an_output_limit():
    provider = OpenAICompatibleProvider(
        "test-model",
        provider="custom",
        base_url="https://example.test/v1",
    )
    response = {
        "choices": (
            {
                "message": {
                    "content": None,
                    "tool_calls": (),
                },
                "finish_reason": "length",
            },
        ),
    }

    with pytest.raises(ModelProviderError) as caught:
        provider._decode_response(response)

    assert caught.value.code is ProviderErrorCode.OUTPUT_LIMIT
