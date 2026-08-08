"""Portable resource-adapter contracts and built-in sources."""

# Import order is intentional: the shared records initialize catalog models before
# source modules import the capability registry, avoiding a package-init cycle.
# isort: off
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
    PostgreSQLProbeResult,
    PostgreSQLProbeSchema,
    PostgreSQLResourceAdapter,
    PostgreSQLSource,
    PostgreSQLSourceError,
)
from .postgresql_query import PostgreSQLQueryBackend, PostgreSQLQueryError

# isort: on

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
    "PostgreSQLProbeResult",
    "PostgreSQLProbeSchema",
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
