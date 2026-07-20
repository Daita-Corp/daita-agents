"""Protocols implemented by model-provider adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, TYPE_CHECKING

from .models import ModelProfile, ModelRequest, ModelResponse, ModelStreamEvent

if TYPE_CHECKING:
    from .routing import ModelRoute


class ModelProfileRepositoryError(RuntimeError):
    """Base failure for an agent's restart-stable model-profile binding."""


class ModelProfileConflictError(ModelProfileRepositoryError):
    """Raised when a different profile is proposed for an already-bound agent."""


class ModelRouteRepositoryError(RuntimeError):
    """Base failure for versioned model-route persistence."""


class ModelRouteConflictError(ModelRouteRepositoryError):
    """Raised when model-route compare-and-set or active-operation checks fail."""


class ModelProfileRepository(Protocol):
    """Persist or load the one exact built-in model profile bound to an agent."""

    async def bind_model_profile(
        self,
        agent_id: str,
        profile: ModelProfile,
    ) -> ModelProfile: ...

    async def load_model_profile(self, agent_id: str) -> ModelProfile | None: ...


class ModelRouteRepository(Protocol):
    """Persist and load the active versioned provider-neutral model route."""

    async def set_model_route(
        self,
        agent_id: str,
        route: ModelRoute,
        *,
        expected_revision: int,
    ) -> ModelRoute: ...

    async def load_model_route(self, agent_id: str) -> ModelRoute | None: ...


class ModelProvider(Protocol):
    """The provider-neutral inference boundary used by the generic loop."""

    @property
    def provider_id(self) -> str: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...


class StreamingModelProvider(ModelProvider, Protocol):
    """Advertised provider contract for canonical incremental inference."""

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...


__all__ = [
    "ModelProfileConflictError",
    "ModelProfileRepository",
    "ModelProfileRepositoryError",
    "ModelRouteConflictError",
    "ModelRouteRepository",
    "ModelRouteRepositoryError",
    "ModelProvider",
    "StreamingModelProvider",
]
