"""Shared helpers extracted from ``test_routing.py``."""

from __future__ import annotations

from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelSensitivity,
    TextBlock,
)
from daita.llm.routing import (
    ModelProviderRegistration,
)


def request():
    return ModelRequest(
        messages=(
            CanonicalMessage(role=MessageRole.USER, content=(TextBlock("hello"),)),
        )
    )


def registration(provider, *, streaming=False, allowed_sensitivities=None):
    return ModelProviderRegistration(
        provider=provider,
        profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=10_000,
            max_output_tokens=1_000,
            supports_streaming=streaming,
        ),
        allowed_sensitivities=(
            frozenset({ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL})
            if allowed_sensitivities is None
            else allowed_sensitivities
        ),
    )
