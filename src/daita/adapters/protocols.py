"""Define source lifecycle, resource adapter, and source-store protocols."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..errors import DaitaError, ErrorRetryability

if TYPE_CHECKING:
    from ..storage.sqlite_records import RelationalWriteScope, SourceReadScope

from .models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)

# Bounds on one live catalog_describe result.
DESCRIBE_MAX_COLUMNS = 256
DESCRIBE_MAX_LISTED_TABLES = 100
SUMMARY_MAX_COLUMNS = 12


def column_summary(columns: str | None, total: int) -> str | None:
    """One line of leading ``name type`` pairs, saying how many more there are."""

    if not columns:
        return None
    hidden = total - SUMMARY_MAX_COLUMNS
    return f"{columns} (+{hidden} more)" if hidden > 0 else columns


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


@runtime_checkable
class SourceReadScopeStore(SourceStore, Protocol):
    """A source store that also answers what a source may currently read."""

    async def load_source_read_scope(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceReadScope | None: ...


class SourcePermissionStore(SourceReadScopeStore, Protocol):
    """A source store that also answers what a person allowed writing."""

    async def list_relational_write_scopes(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[RelationalWriteScope, ...]: ...


__all__ = [
    "DESCRIBE_MAX_COLUMNS",
    "DESCRIBE_MAX_LISTED_TABLES",
    "SUMMARY_MAX_COLUMNS",
    "column_summary",
    "ResourceAdapter",
    "ResourceAdapterError",
    "ResourceNotFoundError",
    "ResourceSource",
    "SourceClosedError",
    "SourceStore",
    "StaleResourceError",
]
