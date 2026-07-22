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
from .local_files import (
    LocalDirectoryReadBackend,
    LocalDirectoryResourceAdapter,
    LocalDirectorySource,
    LocalDirectorySourceError,
)
from .sqlite import SQLiteResourceAdapter, SQLiteSource, SQLiteSourceError
from .sqlite_query import SQLiteQueryBackend, SQLiteQueryError
from .postgresql import (
    PostgreSQLResourceAdapter,
    PostgreSQLSource,
    PostgreSQLSourceError,
)
from .postgresql_query import PostgreSQLQueryBackend, PostgreSQLQueryError

__all__ = [
    "DiscoveryLimitError",
    "DiscoveryRequest",
    "DiscoveryResult",
    "LocalDirectoryReadBackend",
    "LocalDirectoryResourceAdapter",
    "LocalDirectorySource",
    "LocalDirectorySourceError",
    "ResourceAdapter",
    "ResourceAdapterError",
    "ResourceNotFoundError",
    "ResourceSource",
    "ResourceRef",
    "ResourceSnapshot",
    "PostgreSQLResourceAdapter",
    "PostgreSQLQueryBackend",
    "PostgreSQLQueryError",
    "PostgreSQLSource",
    "PostgreSQLSourceError",
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
