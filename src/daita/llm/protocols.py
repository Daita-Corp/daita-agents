"""Model provider boundaries used by the agent loop."""

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
    return callable(probe) and probe(request) is True


__all__ = [
    "ModelProvider",
    "StreamingModelProvider",
    "provider_has_complete_pricing",
]
