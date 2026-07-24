"""Release-reviewed model profile facts keyed by exact provider identity."""

from __future__ import annotations

from .models import ModelProfile

# Reviewed 2026-07-23 against the exact stable-model pages under
# https://developers.openai.com/api/docs/models/ and
# https://ai.google.dev/gemini-api/docs/models/. Deliberately omit every
# terminal suggestion whose provider does not publish both hard token limits
# and function-calling support for that exact identity.
_REVIEWED_MODEL_PROFILES = {
    profile.id: profile
    for profile in (
        ModelProfile(
            id="openai:gpt-5.6-sol",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
        ),
        ModelProfile(
            id="openai:gpt-5.6-terra",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
        ),
        ModelProfile(
            id="openai:gpt-5.6-luna",
            context_window_tokens=1_050_000,
            max_output_tokens=128_000,
            supports_tools=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.6-flash",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.5-flash",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
        ),
        ModelProfile(
            id="gemini:gemini-3.5-flash-lite",
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_tools=True,
        ),
    )
}


def reviewed_model_profile(provider_id: str) -> ModelProfile | None:
    """Return exact release-reviewed facts, never presentation-derived defaults."""

    if not isinstance(provider_id, str):
        raise TypeError("provider_id must be text")
    return _REVIEWED_MODEL_PROFILES.get(provider_id)


__all__ = ["reviewed_model_profile"]
