"""
CatalogPlugin for schema discovery and metadata management.

Provides tools for discovering database schemas, API structures, and other
organizational metadata across multiple platforms.
"""

from .catalog import (
    CatalogPlugin,
    catalog as create_catalog,
    register_catalog_backend_factory,
)
from .base_discoverer import (
    BaseDiscoverer,
    DiscoveredStore,
    DiscoveryError,
    DiscoveryResult,
)
from .base_profiler import (
    BaseProfiler,
    NormalizedColumn,
    NormalizedForeignKey,
    NormalizedSchema,
    NormalizedTable,
)

# Backward compat: `from daita.plugins.catalog import catalog` still works
catalog = create_catalog

__all__ = [
    "CatalogPlugin",
    "catalog",
    "register_catalog_backend_factory",
    "BaseDiscoverer",
    "DiscoveredStore",
    "DiscoveryError",
    "DiscoveryResult",
    "BaseProfiler",
    "NormalizedColumn",
    "NormalizedForeignKey",
    "NormalizedSchema",
    "NormalizedTable",
]
