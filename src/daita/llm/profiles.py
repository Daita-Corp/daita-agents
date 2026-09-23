"""Return reviewed capabilities and limits for exact provider and model identities."""

from __future__ import annotations

from .models import ModelProfile

# Reviewed 2026-09-22 against the exact stable-model pages under
# https://developers.openai.com/api/docs/models/ and
# https://ai.google.dev/gemini-api/docs/models/, plus Anthropic's exact model
# overview at https://platform.claude.com/docs/en/models/overview. Deliberately
# omit terminal suggestions whose provider does not publish both hard token
# limits and function-calling support for that exact identity.
_REVIEWED_MODEL_PROFILES = {
    profile.id: profile
    for profile in (
        ModelProfile(
            id="openai:gpt-6-astra",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="openai:gpt-6-sol",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="openai:gpt-6-luna",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="openai:gpt-5.6-sol",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="openai:gpt-5.6-terra",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="openai:gpt-5.6-luna",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="codex:gpt-6-astra",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="codex:gpt-6-sol",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="codex:gpt-6-luna",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="codex:gpt-5.6-sol",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="codex:gpt-5.6-terra",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="codex:gpt-5.6-luna",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="anthropic:claude-opus-5-5",
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="anthropic:claude-fable-5-1",
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="anthropic:claude-sonnet-5",
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="anthropic:claude-haiku-4-5-20251001",
            context_window_tokens=200_000,
            max_output_tokens=64_000,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="claude-code:claude-opus-5-5",
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="claude-code:claude-fable-5-1",
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="claude-code:claude-sonnet-5",
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="claude-code:claude-haiku-4-5-20251001",
            context_window_tokens=200_000,
            max_output_tokens=64_000,
            supports_tools=True,
            supports_parallel_tools=True,
            supports_structured_output=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.8-flash",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.7-flash",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.6-flash",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.5-flash",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.5-flash-lite",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
            supports_streaming=True,
            supports_reasoning=True,
        ),
    )
}


def reviewed_model_profile(provider_id: str) -> ModelProfile | None:
    """Return exact release-reviewed facts, never presentation-derived defaults."""

    if not isinstance(provider_id, str):
        raise TypeError("provider_id must be text")
    return _REVIEWED_MODEL_PROFILES.get(provider_id)


__all__ = ["reviewed_model_profile"]
