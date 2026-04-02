"""
CatalogPlugin — orchestrator for schema discovery and metadata management.

Provides pluggable infrastructure discovery, schema profiling, and a unified
catalog of data stores. Extends LifecyclePlugin for agent lifecycle hooks.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..base import LifecyclePlugin
from .base_discoverer import (
    BaseDiscoverer,
    DiscoveredStore,
    DiscoveryError,
    DiscoveryResult,
)
from .base_profiler import BaseProfiler, NormalizedSchema
from .normalizer import (
    normalize_postgresql as _normalize_postgresql,
    normalize_mysql as _normalize_mysql,
    normalize_mongodb as _normalize_mongodb,
    normalize_discovery as _normalize_discovery,
)
from .comparator import compare_schemas as _compare_schemas
from .diagram import export_diagram as _export_diagram
from .persistence import (
    persist_schema as _persist_schema,
    prune_stale_catalog,
)

if TYPE_CHECKING:
    from ...core.tools import AgentTool

logger = logging.getLogger(__name__)

# Module-level registry. Set once at startup via register_catalog_backend_factory().
# None means persist to .daita/catalog.json (local default).
_CATALOG_BACKEND_FACTORY: Optional[Callable[[], Any]] = None


def register_catalog_backend_factory(factory: Optional[Callable[[], Any]]) -> None:
    """
    Register a factory that creates catalog backends.

    Called once at application startup to inject a storage backend for schema
    documents. Pass None to reset to the default (local .daita/catalog.json),
    which is useful in tests.

    Args:
        factory: Callable that takes no arguments and returns a catalog backend
                 with a ``persist_schema(schema: dict) -> bool`` coroutine method.
                 Pass None to clear the registered factory and revert to default.
    """
    global _CATALOG_BACKEND_FACTORY
    _CATALOG_BACKEND_FACTORY = factory


class CatalogPlugin(LifecyclePlugin):
    """
    Plugin for schema discovery and metadata cataloging.

    Works standalone (returns data directly) or with optional graph storage
    for building an organizational knowledge graph.

    Supports:
    - PostgreSQL, MySQL, MongoDB schema discovery
    - GraphQL introspection
    - OpenAPI/Swagger spec parsing
    - Multi-cloud infrastructure discovery (AWS, GCP, Azure)
    - Schema comparison and validation
    - Pluggable discoverers and profilers
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        organization_id: Optional[int] = None,
        auto_persist: bool = False,
    ):
        self._graph_backend = backend
        self._catalog_backend: Optional[Any] = None
        self._organization_id = organization_id
        self._auto_persist = auto_persist
        self._agent_id: Optional[str] = None

        # Discoverer and profiler registries
        self._discoverers: List[BaseDiscoverer] = []
        self._profilers: List[BaseProfiler] = []

        # In-memory catalog state
        self._discovered_stores: Dict[str, DiscoveredStore] = {}
        self._schemas: Dict[str, NormalizedSchema] = {}
        self._last_scan: Optional[str] = None

        logger.debug(
            "CatalogPlugin initialized (backend: %s, auto_persist: %s)",
            backend is not None,
            auto_persist,
        )

    def initialize(self, agent_id: str) -> None:
        self._agent_id = agent_id
        if self._graph_backend is None:
            from daita.core.graph.backend import auto_select_backend

            self._graph_backend = auto_select_backend(graph_type="catalog")
            logger.debug(
                "CatalogPlugin: using graph backend %s",
                type(self._graph_backend).__name__,
            )

        if _CATALOG_BACKEND_FACTORY is not None and self._catalog_backend is None:
            try:
                self._catalog_backend = _CATALOG_BACKEND_FACTORY()
                logger.debug(
                    "CatalogPlugin: using catalog backend %s",
                    type(self._catalog_backend).__name__,
                )
            except Exception as exc:
                logger.warning(
                    "CatalogPlugin: catalog backend factory failed: %s. "
                    "Schema persistence will use local JSON.",
                    exc,
                )

    # ------------------------------------------------------------------
    # Discoverer / profiler registry
    # ------------------------------------------------------------------

    def add_discoverer(self, discoverer: BaseDiscoverer) -> None:
        """Register a discoverer for infrastructure enumeration."""
        self._discoverers.append(discoverer)

    def add_profiler(self, profiler: BaseProfiler) -> None:
        """Register a profiler for schema extraction."""
        self._profilers.append(profiler)

    def _find_profiler(self, store_type: str) -> Optional[BaseProfiler]:
        """Find a registered profiler that supports the given store type."""
        for p in self._profilers:
            if p.supports(store_type):
                return p
        return None

    # ------------------------------------------------------------------
    # High-level discovery operations
    # ------------------------------------------------------------------

    async def discover_all(self, concurrency: int = 5) -> DiscoveryResult:
        """
        Run all registered discoverers in parallel with bounded concurrency.

        Partial failures are captured — stores from successful discoverers
        are still returned alongside errors from failed ones.
        """
        import asyncio
        from datetime import datetime, timezone

        from .normalizer import deduplicate_stores

        result = DiscoveryResult()
        if not self._discoverers:
            return result

        sem = asyncio.Semaphore(concurrency)

        async def _run(d: BaseDiscoverer):
            async with sem:
                try:
                    await d.authenticate()
                    stores = [s async for s in d.enumerate()]
                    return d.name, stores, None
                except Exception as exc:
                    return d.name, [], exc

        raw_results = await asyncio.gather(
            *[_run(d) for d in self._discoverers], return_exceptions=True
        )

        all_stores: List[DiscoveredStore] = []
        for item in raw_results:
            if isinstance(item, Exception):
                result.errors.append(
                    DiscoveryError(
                        discoverer_name="unknown",
                        error=str(item),
                        exception_type=type(item).__name__,
                    )
                )
                continue
            name, stores, exc = item
            if exc:
                result.errors.append(
                    DiscoveryError(
                        discoverer_name=name,
                        error=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )
            all_stores.extend(stores)

        result.stores = deduplicate_stores(all_stores)

        # Update in-memory catalog
        for store in result.stores:
            self._discovered_stores[store.id] = store

        self._last_scan = datetime.now(timezone.utc).isoformat()
        return result

    async def discover_and_profile(
        self, concurrency: int = 5
    ) -> DiscoveryResult:
        """
        Discover all stores, then profile each one with a matching profiler.
        """
        result = await self.discover_all(concurrency=concurrency)

        for store in result.stores:
            profiler = self._find_profiler(store.store_type)
            if profiler:
                try:
                    schema = await profiler.profile(store)
                    self._schemas[store.id] = schema
                except Exception as exc:
                    logger.warning(
                        "Failed to profile store %s (%s): %s",
                        store.display_name,
                        store.store_type,
                        exc,
                    )

        return result

    # ------------------------------------------------------------------
    # Public accessor API
    # ------------------------------------------------------------------

    def get_stores(
        self,
        store_type: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[DiscoveredStore]:
        """Return discovered stores, optionally filtered by type and/or environment."""
        stores = list(self._discovered_stores.values())
        if store_type:
            stores = [s for s in stores if s.store_type == store_type]
        if environment:
            stores = [s for s in stores if s.environment == environment]
        return stores

    def get_store(self, store_id: str) -> Optional[DiscoveredStore]:
        """Get a single store by its fingerprint ID."""
        return self._discovered_stores.get(store_id)

    def get_schema(self, store_id: str) -> Optional[NormalizedSchema]:
        """Get the profiled schema for a store, if available."""
        return self._schemas.get(store_id)

    # ------------------------------------------------------------------
    # Lifecycle hooks (LifecyclePlugin)
    # ------------------------------------------------------------------

    async def on_before_run(self, prompt: str) -> Optional[str]:
        """Inject catalog context into the system prompt."""
        if self._discovered_stores:
            return (
                f"Catalog: {len(self._discovered_stores)} data stores known. "
                f"Last scan: {self._last_scan or 'never'}."
            )
        return None

    async def on_agent_stop(self) -> None:
        """Persist catalog state and close discoverer sessions."""
        await self.prune_stale_catalog(max_age_seconds=7 * 86400)
        for d in self._discoverers:
            await d.close()

    # ------------------------------------------------------------------
    # Discovery wrappers (delegate to discovery.py, then persist+wrap)
    # ------------------------------------------------------------------

    async def discover_postgres(
        self,
        connection_string: str,
        schema: str = "public",
        persist: bool = False,
        ssl_mode: str = "verify-full",
    ) -> Dict[str, Any]:
        """Discover PostgreSQL database schema."""
        from .discovery import discover_postgres

        result = await discover_postgres(connection_string, schema, ssl_mode)
        persisted = await self._persist_schema(result) if persist else False
        return self._persist_response(result, persisted)

    async def discover_mysql(
        self,
        connection_string: str,
        schema: Optional[str] = None,
        persist: bool = False,
    ) -> Dict[str, Any]:
        """Discover MySQL/MariaDB database schema."""
        from .discovery import discover_mysql

        result = await discover_mysql(connection_string, schema)
        persisted = await self._persist_schema(result) if persist else False
        return self._persist_response(result, persisted)

    async def discover_mongodb(
        self,
        connection_string: str,
        database: str,
        sample_size: int = 100,
        persist: bool = False,
    ) -> Dict[str, Any]:
        """Discover MongoDB schema by sampling documents."""
        from .discovery import discover_mongodb

        result = await discover_mongodb(connection_string, database, sample_size)
        persisted = await self._persist_schema(result) if persist else False
        return self._persist_response(result, persisted)

    async def discover_openapi(
        self, spec_url: str, service_name: Optional[str] = None, persist: bool = False
    ) -> Dict[str, Any]:
        """Discover API structure from OpenAPI/Swagger spec."""
        from .discovery import discover_openapi

        result = await discover_openapi(spec_url, service_name)
        persisted = await self._persist_schema(result) if persist else False
        return self._persist_response(result, persisted)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _persist_response(
        result: Dict[str, Any], persisted: bool
    ) -> Dict[str, Any]:
        """Wrap a discover_* result dict in a standard tool response."""
        response: Dict[str, Any] = {"schema": result, "persisted": persisted}
        if not persisted:
            response["persist_skipped"] = "catalog backend not configured"
        return response

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    async def compare_schemas(
        self, schema_a: Dict[str, Any], schema_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two schemas to identify differences."""
        return await _compare_schemas(schema_a, schema_b)

    async def export_diagram(
        self, schema: Dict[str, Any], format: str = "mermaid"
    ) -> Dict[str, Any]:
        """Export schema as a visual diagram."""
        return await _export_diagram(schema, format)

    async def _persist_schema(self, schema: Dict[str, Any]) -> bool:
        """Persist schema to the catalog store and graph backend."""
        return await _persist_schema(
            schema, self._catalog_backend, self._graph_backend, self._agent_id
        )

    async def prune_stale_catalog(self, max_age_seconds: int) -> dict:
        """Remove catalog entries whose last_seen is older than max_age_seconds."""
        return await prune_stale_catalog(max_age_seconds)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def get_tools(self) -> List["AgentTool"]:
        """Expose schema discovery operations as agent tools."""
        from .tools import build_catalog_tools

        return build_catalog_tools(self)

    # ------------------------------------------------------------------
    # Schema normalization — static method wrappers (backward compat)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_postgresql(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CatalogPlugin PostgreSQL discovery output."""
        return _normalize_postgresql(raw)

    @staticmethod
    def normalize_mysql(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CatalogPlugin MySQL discovery output."""
        return _normalize_mysql(raw)

    @staticmethod
    def normalize_mongodb(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CatalogPlugin MongoDB discovery output."""
        return _normalize_mongodb(raw)

    @staticmethod
    def normalize_discovery(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ``discover_*`` output to the normalized schema shape.

        Dispatches by ``raw["database_type"]``; returns *raw* unchanged for
        unrecognized types.
        """
        return _normalize_discovery(raw)


def catalog(**kwargs) -> CatalogPlugin:
    """Create CatalogPlugin with simplified interface."""
    return CatalogPlugin(**kwargs)
