"""Protocols implemented by model-provider adapters."""

from __future__ import annotations

from typing import Protocol

from .models import ModelProfile, ModelRequest, ModelResponse


class ModelProfileRepositoryError(RuntimeError):
    """Base failure for an agent's restart-stable model-profile binding."""


class ModelProfileConflictError(ModelProfileRepositoryError):
    """Raised when a different profile is proposed for an already-bound agent."""


class ModelProfileRepository(Protocol):
    """Persist or load the one exact built-in model profile bound to an agent."""

    async def bind_model_profile(
        self,
        agent_id: str,
        profile: ModelProfile,
    ) -> ModelProfile: ...

    async def load_model_profile(self, agent_id: str) -> ModelProfile | None: ...


class ModelProvider(Protocol):
    """The provider-neutral inference boundary used by the generic loop."""

    @property
    def provider_id(self) -> str: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


__all__ = [
    "ModelProfileConflictError",
    "ModelProfileRepository",
    "ModelProfileRepositoryError",
    "ModelProvider",
]
