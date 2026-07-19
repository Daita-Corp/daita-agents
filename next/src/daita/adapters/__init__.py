"""Portable resource-adapter contracts and built-in sources."""

from .models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
    source_registration_id,
)
from .protocols import (
    DiscoveryLimitError,
    ResourceAdapter,
    ResourceAdapterError,
    ResourceNotFoundError,
    ResourceSource,
    SourceClosedError,
    SourceStore,
    StaleResourceError,
)
from .sqlite import SQLiteResourceAdapter, SQLiteSource, SQLiteSourceError
from .sqlite_query import SQLiteQueryBackend, SQLiteQueryError

__all__ = [
    "DiscoveryLimitError",
    "DiscoveryRequest",
    "DiscoveryResult",
    "ResourceAdapter",
    "ResourceAdapterError",
    "ResourceNotFoundError",
    "ResourceSource",
    "ResourceRef",
    "ResourceSnapshot",
    "SQLiteQueryBackend",
    "SQLiteQueryError",
    "SQLiteResourceAdapter",
    "SQLiteSource",
    "SQLiteSourceError",
    "SourceClosedError",
    "SourceHealth",
    "SourceRegistration",
    "SourceStore",
    "StaleResourceError",
    "source_registration_id",
]
