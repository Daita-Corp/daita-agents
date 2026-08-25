"""Define model-provider protocols and request-policy capability checks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .models import ModelRequest, ModelResponse, ModelStreamEvent


class ModelProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def supports_request_policy(self, request: ModelRequest) -> bool: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


@runtime_checkable
class StreamingModelProvider(ModelProvider, Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...


def provider_has_complete_pricing(
    provider: object,
    request: ModelRequest,
) -> bool:
    """Fail closed for providers without an admitted pricing preflight."""

    probe = getattr(provider, "has_complete_pricing", None)
    if not callable(probe):
        return False
    try:
        return probe(request) is True
    except Exception:
        return False


def provider_supports_request_policy(
    provider: object,
    request: ModelRequest,
) -> bool:
    """Fail closed when route eligibility cannot be established exactly."""

    probe = getattr(provider, "supports_request_policy", None)
    if not callable(probe):
        return False
    try:
        return probe(request) is True
    except Exception:
        return False


__all__ = [
    "ModelProvider",
    "StreamingModelProvider",
    "provider_has_complete_pricing",
    "provider_supports_request_policy",
]
