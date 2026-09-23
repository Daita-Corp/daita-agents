from daita.llm import profiles
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.subscription_cli.grok import (
    GrokBuildSubscriptionProvider,
)
from daita.tui.models import MODEL_SUGGESTIONS

_PICKER_SUGGESTIONS = {
    "openai": (
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ),
    "anthropic": (
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
    ),
    "codex": (
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ),
    "claude-code": (
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
    ),
    "grok-build": ("grok-4.7", "grok-4.5"),
    "gemini": (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ),
    "grok": ("grok-4.7", "grok-4.5"),
    "ollama": ("qwen3.8", "qwen3", "llama3.1", "mistral-small3.2"),
}


def test_model_picker_suggestions_match_the_reviewed_catalog():
    assert {
        provider_id: tuple(suggestion.model_id for suggestion in suggestions)
        for provider_id, suggestions in MODEL_SUGGESTIONS.items()
    } == _PICKER_SUGGESTIONS


def test_current_suggestions_are_unique_and_bound_to_their_provider():
    for provider_id, suggestions in MODEL_SUGGESTIONS.items():
        model_ids = tuple(suggestion.model_id for suggestion in suggestions)

        assert len(model_ids) == len(set(model_ids))
        assert all(suggestion.provider_id == provider_id for suggestion in suggestions)


def test_every_reviewed_model_is_available_in_the_picker():
    picker_ids = {
        f"{provider_id}:{suggestion.model_id}"
        for provider_id, suggestions in MODEL_SUGGESTIONS.items()
        for suggestion in suggestions
    }

    assert set(profiles._REVIEWED_MODEL_PROFILES) <= picker_ids


def test_published_hard_limits_are_reviewed_for_current_hosted_suggestions():
    reviewed_providers = {
        "openai",
        "anthropic",
        "codex",
        "claude-code",
        "gemini",
    }

    for provider_id in reviewed_providers:
        for suggestion in MODEL_SUGGESTIONS[provider_id]:
            profile = reviewed_model_profile(f"{provider_id}:{suggestion.model_id}")

            assert profile is not None
            assert profile.supports_tools is True
            assert profile.supports_reasoning is True


def test_new_model_limits_match_the_published_provider_contracts():
    expected_limits = {
        "openai:gpt-6-astra": (1_050_000, 128_000),
        "anthropic:claude-opus-5-5": (1_000_000, 128_000),
        "anthropic:claude-opus-4-8": (1_000_000, 128_000),
        "anthropic:claude-haiku-4-5-20251001": (200_000, 64_000),
        "gemini:gemini-3.8-flash": (1_048_576, 65_536),
    }

    for provider_id, expected in expected_limits.items():
        profile = reviewed_model_profile(provider_id)

        assert profile is not None
        assert (profile.context_window_tokens, profile.max_output_tokens) == expected


def test_grok_build_accepts_the_current_model_and_keeps_the_previous_release():
    current = GrokBuildSubscriptionProvider("grok-4.7")
    previous = GrokBuildSubscriptionProvider("grok-4.5")

    assert current.provider_id == "grok-build:grok-4.7"
    assert previous.provider_id == "grok-build:grok-4.5"
