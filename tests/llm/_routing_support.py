"""Shared helpers extracted from ``test_routing.py``."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from daita.llm.errors import (
    ModelProviderError,
    ProviderErrorCode,
    ProviderFailureDiagnostic,
    ProviderFailurePhase,
)
from daita.llm.factory import create_model_route_provider
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
    TextBlock,
)
from daita.llm.pricing import (
    CostBasis,
    CostEstimate,
    CostEstimateStatus,
)
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
    RetryPolicy,
    autonomous_request_is_admissible,
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
