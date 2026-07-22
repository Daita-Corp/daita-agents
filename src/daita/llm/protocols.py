"""Model provider boundaries used by the agent loop."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .models import ModelRequest, ModelResponse, ModelStreamEvent


class ModelProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def supports_request_policy(self, request: ModelRequest) -> bool: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


class StreamingModelProvider(ModelProvider, Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...


__all__ = ["ModelProvider", "StreamingModelProvider"]
