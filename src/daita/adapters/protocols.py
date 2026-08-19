"""Small source lifecycle and adapter contracts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..errors import DaitaError, ErrorRetryability
from .models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)


def _text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class ResourceAdapterError(DaitaError):
    """Normalized resource-adapter failure safe for control-plane handling."""

    def __init__(self, source_id: str, code: str, message: str) -> None:
        _text(source_id, "adapter error source_id")
        _text(code, "adapter error code")
        _text(message, "adapter error message")
        self.source_id = source_id
        self.code = code
        super().__init__(
            message,
            error_code=code,
            retryability=ErrorRetryability.UNKNOWN,
        )


class SourceClosedError(ResourceAdapterError):
    def __init__(self, source_id: str) -> None:
        super().__init__(source_id, "source_closed", "resource source is closed")


class DiscoveryLimitError(ResourceAdapterError):
    def __init__(self, source_id: str, message: str) -> None:
        super().__init__(source_id, "discovery_limit_exceeded", message)


class ResourceNotFoundError(ResourceAdapterError):
    def __init__(self, source_id: str, resource_id: str) -> None:
        _text(resource_id, "missing resource_id")
        self.resource_id = resource_id
        super().__init__(
            source_id,
            "resource_not_found",
            f"resource is not present in source {source_id}: {resource_id}",
        )


class StaleResourceError(ResourceAdapterError):
    def __init__(self, source_id: str, resource_id: str) -> None:
        _text(resource_id, "stale resource_id")
        self.resource_id = resource_id
        super().__init__(
            source_id,
            "stale_resource",
            f"resource revision is stale for source {source_id}: {resource_id}",
        )


@runtime_checkable
class ResourceAdapter(Protocol):
    """Bounded control-plane access to one registered source."""

    @property
    def registration(self) -> SourceRegistration: ...

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult: ...

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot: ...

    async def health(self) -> SourceHealth: ...

    async def close(self) -> None: ...


@runtime_checkable
class ResourceSource(Protocol):
    """Configuration object that opens one bounded resource adapter."""

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> ResourceAdapter: ...


@runtime_checkable
class SourceStore(Protocol):
    """Persist source registration lifecycle without exposing storage details."""

    async def register_source(
        self,
        registration: SourceRegistration,
    ) -> SourceRegistration: ...

    async def load_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration | None: ...

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]: ...

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ) -> SourceRegistration: ...


__all__ = [
    "DiscoveryLimitError",
    "ResourceAdapter",
    "ResourceAdapterError",
    "ResourceNotFoundError",
    "ResourceSource",
    "SourceClosedError",
    "SourceStore",
    "StaleResourceError",
]
