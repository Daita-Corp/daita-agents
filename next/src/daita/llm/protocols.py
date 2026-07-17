"""Protocols implemented by model-provider adapters."""

from __future__ import annotations

from typing import Protocol

from .models import ModelRequest, ModelResponse


class ModelProvider(Protocol):
    """The provider-neutral inference boundary used by the generic loop."""

    @property
    def provider_id(self) -> str: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...
