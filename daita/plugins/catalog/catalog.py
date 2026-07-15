"""
CatalogPlugin — orchestrator for schema discovery and metadata management.

Provides pluggable infrastructure discovery, schema profiling, and a unified
catalog of data stores. Declares catalog capabilities, evidence schemas,
context providers, and model-visible tool views.
"""

import logging
import fnmatch
import hashlib
import json
import re
from collections import deque
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from daita.core.db_type_metadata import native_type_from_db_type

from ..base import DomainServicePlugin, PluginContext
from .base_discoverer import (
    BaseDiscoverer,
    DiscoveredStore,
    DiscoveryError,
    DiscoveryResult,
)
from .base_profiler import (
    BaseProfiler,
    NormalizedColumnValue,
    NormalizedColumnValueProfile,
    NormalizedSchema,
)
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
from .extensions import (
    CATALOG_MANIFEST,
    CatalogExecutor,
    CatalogSummaryContextProvider,
    catalog_capabilities,
    catalog_evidence_schemas,
    catalog_tool_views,
    catalog_workers,
)

logger = logging.getLogger(__name__)

MAX_CATALOG_SEARCH_LIMIT = 50
MAX_ASSET_FIELD_LIMIT = 200
MAX_RELATIONSHIP_HOPS = 6
MAX_RELATIONSHIP_PATHS = 8
MATCHED_FIELDS_LIMIT = 12
MAX_COLUMN_VALUE_HINTS = 12
MAX_VALUE_GROUNDING_PROFILE_BUDGET = 4

# Module-level registry. Set once at startup via register_catalog_backend_factory().
# None means persist to .daita/catalog.json (local default).
_CATALOG_BACKEND_FACTORY: Optional[Callable[[], Any]] = None


def _required_arg(args: Dict[str, Any], field: str) -> str:
    value = args.get(field)
    if value is None or str(value).strip() == "":
        from ...core.exceptions import ValidationError

        raise ValidationError(f"{field} is required", field=field)
    return str(value)


def _required_string_list_arg(args: Dict[str, Any], field: str) -> List[str]:
    value = args.get(field)
    if not isinstance(value, list) or not value:
        from ...core.exceptions import ValidationError

        raise ValidationError(f"{field} must be a non-empty list", field=field)
    items = [str(item) for item in value if str(item).strip()]
    if not items:
        from ...core.exceptions import ValidationError

        raise ValidationError(f"{field} must contain at least one value", field=field)
    return items


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


class CatalogPlugin(DomainServicePlugin):
    """
    Plugin for schema discovery and metadata cataloging.

    Works standalone (returns data directly) or with optional graph storage
    for building an organizational knowledge graph.

    Supports:
    - PostgreSQL, MySQL, MongoDB schema discovery
    - OpenAPI/Swagger spec parsing
    - Multi-cloud infrastructure discovery:
        * AWS — RDS, DynamoDB, S3, ElastiCache, Redshift, API Gateway,
          SQS, SNS, OpenSearch, DocumentDB, Kinesis
        * GCP — Cloud SQL, GCS, BigQuery, Firestore, Bigtable, Pub/Sub,
          Memorystore (Redis), API Gateway
        * Azure — SQL, PostgreSQL, MySQL, Cosmos DB, Blob Storage, Redis,
          Event Hubs, Service Bus, API Management
        * GitHub — connection-string scanning in config files
    - Schema comparison and validation
    - Pluggable discoverers and profilers
    """

    manifest = CATALOG_MANIFEST

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

    async def setup(self, context: PluginContext) -> None:
        self._configure_runtime_backends(context.agent_id)

    def _configure_runtime_backends(self, agent_id: Optional[str]) -> None:
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
    # Runtime extension declarations
    # ------------------------------------------------------------------

    def declare_capabilities(self):
        return catalog_capabilities()

    def get_executors(self):
        return (
            CatalogExecutor(
                id="catalog.register_source",
                capability_ids=frozenset({"catalog.source.register"}),
                evidence_kind="catalog.source_registered",
                handler=self._execute_register_source,
            ),
            CatalogExecutor(
                id="catalog.profile_source",
                capability_ids=frozenset({"catalog.source.profile"}),
                evidence_kind="catalog.profile",
                handler=self._execute_profile_source,
            ),
            CatalogExecutor(
                id="catalog.search_schema",
                capability_ids=frozenset({"catalog.schema.search"}),
                evidence_kind="schema.search_result",
                handler=self._execute_search_schema,
            ),
            CatalogExecutor(
                id="catalog.inspect_asset",
                capability_ids=frozenset({"catalog.asset.inspect"}),
                evidence_kind="schema.asset_profile",
                handler=self._execute_inspect_asset,
            ),
            CatalogExecutor(
                id="catalog.find_relationship_paths",
                capability_ids=frozenset({"catalog.relationship_paths.find"}),
                evidence_kind="schema.relationship_path",
                handler=self._execute_find_relationship_paths,
            ),
            CatalogExecutor(
                id="catalog.register_column_values",
                capability_ids=frozenset({"catalog.column_values.register"}),
                evidence_kind="schema.column_value_profile",
                handler=self._execute_register_column_values,
            ),
            CatalogExecutor(
                id="catalog.search_column_values",
                capability_ids=frozenset({"catalog.column_values.search"}),
                evidence_kind="schema.column_value_search_result",
                handler=self._execute_search_column_values,
            ),
            CatalogExecutor(
                id="catalog.resolve_column_value_hints",
                capability_ids=frozenset({"catalog.column_value_hints.resolve"}),
                evidence_kind="schema.column_value_hint",
                handler=self._execute_resolve_column_value_hints,
            ),
            CatalogExecutor(
                id="catalog.plan_value_grounding",
                capability_ids=frozenset({"catalog.value_grounding.plan"}),
                evidence_kind="catalog.value_grounding.plan",
                handler=self._execute_plan_value_grounding,
            ),
            CatalogExecutor(
                id="catalog.discover_infrastructure",
                capability_ids=frozenset({"catalog.infrastructure.discover"}),
                evidence_kind="catalog.infrastructure_inventory",
                handler=self._execute_discover_infrastructure,
            ),
            CatalogExecutor(
                id="catalog.compare_schema",
                capability_ids=frozenset({"catalog.schema.compare"}),
                evidence_kind="schema.comparison",
                handler=self._execute_compare_schema,
            ),
            CatalogExecutor(
                id="catalog.export_diagram",
                capability_ids=frozenset({"catalog.diagram.export"}),
                evidence_kind="schema.diagram",
                handler=self._execute_export_diagram,
            ),
        )

    def declare_evidence_schemas(self):
        return catalog_evidence_schemas()

    def get_context_providers(self):
        return (CatalogSummaryContextProvider(self),)

    def get_tool_views(self):
        return catalog_tool_views()

    def get_workers(self):
        return catalog_workers()

    async def _execute_register_source(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        schema = args.get("schema")
        if not isinstance(schema, dict):
            from ...core.exceptions import ValidationError

            raise ValidationError("schema is required", field="schema")
        return await self.register_schema(
            schema,
            store_type=args.get("store_type"),
            connection_string=args.get("connection_string"),
            store_id=args.get("store_id"),
            persist=bool(args.get("persist", False)),
            options=args.get("options"),
        )

    async def _execute_profile_source(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        store_id = args.get("store_id")
        if store_id:
            return self.summarize_store(
                store_id,
                profile=str(args.get("profile") or "runtime"),
                limit=int(args.get("limit") or 50),
            )
        result = await self.discover_and_profile(
            concurrency=int(args.get("concurrency") or 5)
        )
        return {
            "store_count": result.store_count,
            "error_count": result.error_count,
            "stores": [asdict(store) for store in result.stores],
            "errors": [asdict(error) for error in result.errors],
        }

    async def _execute_search_schema(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return self.catalog_search_schema(
            _required_arg(args, "store_id"),
            str(args.get("query") or ""),
            limit=int(args.get("limit") or 20),
        )

    async def _execute_inspect_asset(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return self.inspect_asset(
            _required_arg(args, "store_id"),
            _required_arg(args, "asset_ref"),
            field_filter=args.get("field_filter"),
            offset=int(args.get("offset") or 0),
            limit=int(args.get("limit") or 100),
            include_fields=bool(args.get("include_fields", True)),
            include_indexes=bool(args.get("include_indexes", True)),
            include_relationships=bool(args.get("include_relationships", True)),
            blocked_fields=args.get("blocked_fields"),
        )

    async def _execute_find_relationship_paths(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return self.find_relationship_paths(
            _required_arg(args, "store_id"),
            _required_string_list_arg(args, "from_assets"),
            _required_string_list_arg(args, "to_assets"),
            relationship_types=args.get("relationship_types"),
            max_hops=int(args.get("max_hops") or 4),
            max_paths=int(args.get("max_paths") or 5),
        )

    async def _execute_register_column_values(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        profile = args.get("profile")
        profiles = args.get("profiles")
        if profile is not None and profiles is None:
            profiles = [profile]
        return await self.register_column_value_profiles(
            str(args["store_id"]),
            list(profiles or []),
            persist=bool(args.get("persist", False)),
            source_evidence_id=args.get("source_evidence_id"),
        )

    async def _execute_search_column_values(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return self.search_column_value_profiles(
            str(args["store_id"]),
            str(args.get("query") or ""),
            tables=args.get("tables"),
            columns=args.get("columns"),
            limit=int(args.get("limit") or 20),
            include_ineligible=bool(args.get("include_ineligible", False)),
            max_age_seconds=(
                float(args["max_profile_age_seconds"])
                if args.get("max_profile_age_seconds") is not None
                else None
            ),
        )

    async def _execute_resolve_column_value_hints(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return self.resolve_column_value_hints(
            str(args["store_id"]),
            str(args.get("prompt") or args.get("query") or ""),
            tables=args.get("tables"),
            columns=args.get("columns"),
            limit=int(args.get("limit") or MAX_COLUMN_VALUE_HINTS),
        )

    async def _execute_plan_value_grounding(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return self.plan_value_grounding(
            _required_arg(args, "store_id"),
            str(args.get("prompt") or args.get("query") or ""),
            validation_facts=args.get("validation_facts"),
            warnings=args.get("warnings") or args.get("validation_warnings"),
            session_query_scopes=(
                args.get("session_query_scopes") or args.get("query_scopes")
            ),
            targets=args.get("targets"),
            profile_pairs=args.get("profile_pairs"),
            profile_budget=_value_grounding_profile_budget(
                args.get(
                    "profile_budget",
                    args.get("max_profile_budget", args.get("max_profile_targets")),
                )
            ),
            policy_frame=args.get("policy_frame") or args.get("policy"),
            blocked_tables=args.get("blocked_tables"),
            blocked_columns=args.get("blocked_columns") or args.get("blocked_fields"),
        )

    async def _execute_discover_infrastructure(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        result = await self.discover_all(concurrency=int(args.get("concurrency") or 5))
        return {
            "store_count": result.store_count,
            "error_count": result.error_count,
            "stores": [asdict(store) for store in result.stores],
            "errors": [asdict(error) for error in result.errors],
            "last_scan": self._last_scan,
        }

    async def _execute_compare_schema(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self.compare_schemas(
            dict(args.get("schema_a") or {}),
            dict(args.get("schema_b") or {}),
        )

    async def _execute_export_diagram(self, payload: Any) -> Dict[str, Any]:
        args = dict(payload or {})
        return await self.export_diagram(
            dict(args.get("schema") or {}),
            format=str(args.get("format") or "mermaid"),
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
            if isinstance(item, BaseException):
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

    async def discover_and_profile(self, concurrency: int = 5) -> DiscoveryResult:
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
                    if self._auto_persist:
                        await _persist_schema(
                            schema.to_dict(),
                            self._catalog_backend,
                            self._graph_backend,
                            self._agent_id,
                        )
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
    # Runtime teardown
    # ------------------------------------------------------------------

    async def teardown(self) -> None:
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
        return await self._register_direct_discovery(
            result,
            store_type="postgresql",
            connection_string=connection_string,
            persist=persist,
            options={"schema": schema},
        )

    async def discover_mysql(
        self,
        connection_string: str,
        schema: Optional[str] = None,
        persist: bool = False,
        ssl_mode: str = "verify-full",
    ) -> Dict[str, Any]:
        """Discover MySQL/MariaDB database schema."""
        from .discovery import discover_mysql

        result = await discover_mysql(connection_string, schema, ssl_mode)
        return await self._register_direct_discovery(
            result,
            store_type="mysql",
            connection_string=connection_string,
            persist=persist,
            options={"schema": schema},
        )

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
        return await self._register_direct_discovery(
            result,
            store_type="mongodb",
            connection_string=connection_string,
            persist=persist,
            options={"database": database},
        )

    async def discover_openapi(
        self, spec_url: str, service_name: Optional[str] = None, persist: bool = False
    ) -> Dict[str, Any]:
        """Discover API structure from OpenAPI/Swagger spec."""
        from .discovery import discover_openapi

        result = await discover_openapi(spec_url, service_name)
        return await self._register_direct_discovery(
            result,
            store_type="openapi",
            connection_string=spec_url,
            persist=persist,
            options={"service_name": service_name},
        )

    async def register_schema(
        self,
        schema: Dict[str, Any],
        *,
        store_type: Optional[str] = None,
        connection_string: Optional[str] = None,
        store_id: Optional[str] = None,
        persist: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a normalized or raw schema dict in catalog state."""
        return await self._register_direct_discovery(
            schema,
            store_type=store_type,
            connection_string=connection_string,
            store_id=store_id,
            persist=persist,
            options=options,
        )

    def search_catalog(
        self,
        store_id: str,
        query: str,
        *,
        asset_types: Optional[List[str]] = None,
        include_fields: bool = True,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search assets and fields within one profiled catalog store."""
        schema = self._schema_dict_for_store(store_id)
        if schema is None:
            return self._missing_schema_response(store_id)

        tokens = _query_tokens(query)
        limit = _clamp_int(
            limit, default=20, minimum=1, maximum=MAX_CATALOG_SEARCH_LIMIT
        )
        wanted_types = {item.lower() for item in asset_types or []}
        scored = []
        for table in schema.get("tables", []) or []:
            asset_type = str(table.get("metadata", {}).get("asset_type") or "table")
            if wanted_types and asset_type.lower() not in wanted_types:
                continue
            score, matched_fields, reasons = _score_asset(table, tokens, include_fields)
            if score <= 0 and tokens:
                continue
            scored.append(
                {
                    **_asset_summary(table, store_id=store_id, asset_type=asset_type),
                    "score": round(score, 3),
                    "matched_fields": matched_fields[:MATCHED_FIELDS_LIMIT],
                    "match_reasons": reasons[:8],
                    "relationships": self._relationships_for_asset(
                        schema, table["name"]
                    )[:8],
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["name"]))
        return {
            "store_id": store_id,
            "query": query,
            "tokens": tokens,
            "total_matches": len(scored),
            "limit": limit,
            "truncated": len(scored) > limit,
            "assets": scored[:limit],
        }

    def catalog_search_schema(
        self,
        store_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Relational view over search_catalog for tables/views/collections."""
        result = self.search_catalog(
            store_id,
            query,
            asset_types=["table", "view", "collection"],
            include_fields=True,
            limit=limit,
        )
        if "assets" in result:
            result["tables"] = result.pop("assets")
        return result

    def inspect_asset(
        self,
        store_id: str,
        asset_ref: str,
        *,
        field_filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        include_fields: bool = True,
        include_indexes: bool = True,
        include_relationships: bool = True,
        blocked_fields: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Inspect one bounded catalog asset by name/reference."""
        schema = self._schema_dict_for_store(store_id)
        if schema is None:
            return self._missing_schema_response(store_id)

        table = _find_asset(schema, asset_ref)
        if table is None:
            candidates = self.search_catalog(store_id, asset_ref, limit=10).get(
                "assets", []
            )
            return {
                "success": False,
                "store_id": store_id,
                "asset_ref": asset_ref,
                "error": f"Asset not found: {asset_ref}",
                "candidates": candidates,
            }

        fields = table.get("columns", []) or []
        filtered_fields = [
            field for field in fields if _matches_field_pattern(field, field_filter)
        ]
        limit = _clamp_int(limit, default=100, minimum=1, maximum=MAX_ASSET_FIELD_LIMIT)
        offset = _clamp_int(
            offset, default=0, minimum=0, maximum=max(len(filtered_fields), 0)
        )
        page = filtered_fields[offset : offset + limit]
        blocked = {field.lower() for field in blocked_fields or []}

        result: Dict[str, Any] = {
            "success": True,
            "store_id": store_id,
            "database_type": schema.get("database_type"),
            "database_name": schema.get("database_name") or schema.get("schema"),
            "database_dialect": schema.get("database_type"),
            "asset": _asset_summary(table, store_id=store_id),
        }
        value_profiles = _column_value_profiles(schema)
        if include_fields:
            result.update(
                {
                    "fields": [
                        _field_summary(
                            field,
                            blocked_fields=blocked,
                            store_id=store_id,
                            asset_ref=str(table.get("name") or asset_ref),
                            database_dialect=str(schema.get("database_type") or ""),
                            value_profile=_fresh_column_value_profile(
                                value_profiles.get(
                                    f"{table.get('name')}.{field.get('name')}"
                                ),
                                schema,
                            ),
                        )
                        for field in page
                    ],
                    "field_count": len(fields),
                    "matched_field_count": len(filtered_fields),
                    "field_filter": field_filter or None,
                    "offset": offset,
                    "limit": limit,
                    "truncated": offset + len(page) < len(filtered_fields),
                }
            )
        if include_indexes:
            result["indexes"] = table.get("indexes", []) or []
        if include_relationships:
            result["relationships"] = self._relationships_for_asset(
                schema, table["name"]
            )
        if table.get("metadata"):
            result["metadata"] = table["metadata"]
        return result

    def get_table_schema(
        self,
        store_id: str,
        table_name: str,
        *,
        column_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_indexes: bool = True,
        include_foreign_keys: bool = True,
        blocked_columns: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Relational inspection view over inspect_asset()."""
        result = self.inspect_asset(
            store_id,
            table_name,
            field_filter=column_pattern,
            offset=offset,
            limit=limit,
            include_fields=True,
            include_indexes=include_indexes,
            include_relationships=include_foreign_keys,
            blocked_fields=blocked_columns,
        )
        if not result.get("success"):
            return result
        table = result.pop("asset")
        result["table"] = table
        result["table_name"] = table.get("name")
        result["columns"] = result.pop("fields", [])
        result["column_count"] = result.pop("field_count", 0)
        result["matched_column_count"] = result.pop("matched_field_count", 0)
        result["column_pattern"] = result.pop("field_filter", None)
        if include_foreign_keys:
            result["foreign_keys"] = result.pop("relationships", [])
        return result

    def find_relationship_paths(
        self,
        store_id: str,
        from_assets: List[str],
        to_assets: List[str],
        *,
        relationship_types: Optional[List[str]] = None,
        max_hops: int = 4,
        max_paths: int = 5,
    ) -> Dict[str, Any]:
        """Find bounded relationship paths between catalog assets."""
        schema = self._schema_dict_for_store(store_id)
        if schema is None:
            return self._missing_schema_response(store_id)

        sources, source_errors = _resolve_asset_refs(
            self, store_id, schema, from_assets
        )
        targets, target_errors = _resolve_asset_refs(self, store_id, schema, to_assets)
        errors = source_errors + target_errors
        max_hops = _clamp_int(
            max_hops, default=4, minimum=1, maximum=MAX_RELATIONSHIP_HOPS
        )
        max_paths = _clamp_int(
            max_paths, default=5, minimum=1, maximum=MAX_RELATIONSHIP_PATHS
        )

        if errors or not sources or not targets:
            error_payload: list[object] = list(errors)
            if not error_payload:
                error_payload.append("from_assets and to_assets are required")
            return {
                "success": False,
                "store_id": store_id,
                "from_assets": from_assets,
                "to_assets": to_assets,
                "max_hops": max_hops,
                "path_count": 0,
                "reachable": False,
                "errors": error_payload,
            }

        adjacency = _relationship_adjacency(schema, relationship_types)
        target_set = set(targets)
        paths: List[Dict[str, object]] = []
        for source in sources:
            queue: deque[tuple[str, List[str], List[Dict[str, object]]]] = deque(
                [(source, [source], [])]
            )
            while queue and len(paths) < max_paths:
                current, assets, relationships = queue.popleft()
                if current in target_set and relationships:
                    paths.append(_relationship_path_result(assets, relationships))
                    continue
                if len(relationships) >= max_hops:
                    continue
                for next_asset, relationship in adjacency.get(current, []):
                    if next_asset in assets:
                        continue
                    queue.append(
                        (
                            next_asset,
                            assets + [next_asset],
                            relationships + [relationship],
                        )
                    )

        return {
            "success": True,
            "store_id": store_id,
            "from_assets": sources,
            "to_assets": targets,
            "max_hops": max_hops,
            "path_count": len(paths),
            "reachable": bool(paths),
            "paths": paths,
        }

    async def register_column_value_profiles(
        self,
        store_id: str,
        profiles: List[Dict[str, Any]],
        *,
        persist: bool = False,
        source_evidence_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register canonical column value profiles under schema metadata."""
        schema = self.get_schema(store_id)
        if schema is None:
            return self._missing_schema_response(store_id)

        stored: list[dict[str, Any]] = []
        metadata = dict(schema.metadata or {})
        canonical = dict(metadata.get("column_value_profiles") or {})
        profile_key = metadata.get("profile_key") or schema.metadata.get("profile_key")
        schema_fingerprint = _schema_structure_fingerprint(schema.to_dict())

        for raw in profiles:
            if not isinstance(raw, dict):
                continue
            profile = _normalize_column_value_profile(
                raw,
                source_evidence_id=source_evidence_id,
            )
            if not profile.table or not profile.column:
                continue
            if profile_key and "profile_key" not in profile.policy:
                profile.policy["profile_key"] = str(profile_key)
            profile.policy.setdefault("schema_fingerprint", schema_fingerprint)
            canonical[profile.ref] = profile.to_dict()
            stored.append(profile.to_dict())

        metadata["column_value_profiles"] = canonical
        schema.metadata = metadata

        persisted = await self._persist_schema(schema.to_dict()) if persist else False
        return {
            "success": True,
            "store_id": store_id,
            "profile_count": len(stored),
            "profiles": stored,
            "persisted": persisted,
            "canonical_path": "metadata.column_value_profiles",
        }

    def search_column_value_profiles(
        self,
        store_id: str,
        query: str,
        *,
        tables: Optional[Iterable[str]] = None,
        columns: Optional[Iterable[str]] = None,
        limit: int = 20,
        include_ineligible: bool = False,
        max_age_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search canonical column value profiles for prompt-relevant values."""
        schema = self._schema_dict_for_store(store_id)
        if schema is None:
            return self._missing_schema_response(store_id)
        limit = _clamp_int(limit, default=20, minimum=1, maximum=50)
        tokens = _query_tokens(query)
        table_filter = {str(item).lower() for item in tables or []}
        column_filter = {str(item).lower() for item in columns or []}

        matches: List[Dict[str, object]] = []
        for ref, profile in _column_value_profiles(schema).items():
            profile = _profile_with_freshness(
                profile,
                schema,
                max_age_seconds=max_age_seconds,
            )
            table = str(profile.get("table") or ref.rsplit(".", 1)[0])
            column = str(profile.get("column") or ref.rsplit(".", 1)[-1])
            if table_filter and table.lower() not in table_filter:
                continue
            if column_filter and column.lower() not in column_filter:
                continue
            if not include_ineligible and (
                profile.get("redacted") or profile.get("profile_status") == "skipped"
            ):
                continue
            score, reasons = _score_column_value_profile(profile, tokens)
            if tokens and score <= 0:
                continue
            matches.append(
                {
                    "store_id": store_id,
                    "profile_ref": ref,
                    "table": table,
                    "column": column,
                    "score": round(score, 3),
                    "match_reasons": reasons[:8],
                    "distinct_count": profile.get("distinct_count"),
                    "top_values": list(profile.get("top_values") or [])[:25],
                    "profile_kind": profile.get("profile_kind") or "categorical_values",
                    "profile_status": profile.get("profile_status") or "profiled",
                    "sampled": bool(profile.get("sampled", False)),
                    "redacted": bool(profile.get("redacted", False)),
                    "truncated": bool(profile.get("truncated", False)),
                    "stale": bool(profile.get("stale", False)),
                    "stale_reason": profile.get("stale_reason"),
                    "source_fingerprint": profile.get("source_fingerprint"),
                    "source_fingerprint_status": profile.get(
                        "source_fingerprint_status"
                    ),
                    "source_fingerprint_reason": profile.get(
                        "source_fingerprint_reason"
                    ),
                }
            )

        matches.sort(
            key=lambda item: (-float(str(item["score"])), str(item["profile_ref"]))
        )
        return {
            "success": True,
            "store_id": store_id,
            "query": query,
            "tokens": tokens,
            "profile_count": len(matches),
            "profiles": matches[:limit],
            "truncated": len(matches) > limit,
            "include_ineligible": include_ineligible,
            "max_profile_age_seconds": max_age_seconds,
        }

    def resolve_column_value_hints(
        self,
        store_id: str,
        prompt: str,
        *,
        tables: Optional[Iterable[str]] = None,
        columns: Optional[Iterable[str]] = None,
        limit: int = MAX_COLUMN_VALUE_HINTS,
    ) -> Dict[str, Any]:
        """Resolve prompt-scoped value hints from canonical profiles."""
        result = self.search_column_value_profiles(
            store_id,
            prompt,
            tables=tables,
            columns=columns,
            limit=limit,
        )
        if not result.get("success"):
            return result
        tokens = _query_tokens(prompt)
        hints = []
        for profile in result.get("profiles", []) or []:
            top_values = (
                [
                    item
                    for item in profile.get("top_values", []) or []
                    if isinstance(item, dict)
                ]
                if _catalog_inline_value_profile_eligible(profile)
                else []
            )
            hint: Dict[str, Any] = {
                "table": profile.get("table"),
                "column": profile.get("column"),
                "profile_ref": profile.get("profile_ref"),
                "distinct_count": profile.get("distinct_count"),
                "observed_values": top_values[:25],
                "profile_status": profile.get("profile_status") or "profiled",
                "sampled": bool(profile.get("sampled", False)),
                "truncated": bool(profile.get("truncated", False)),
                "redacted": bool(profile.get("redacted", False)),
                "stale": bool(profile.get("stale", False)),
                "stale_reason": profile.get("stale_reason"),
                "source_fingerprint": profile.get("source_fingerprint"),
                "source_fingerprint_status": profile.get("source_fingerprint_status"),
                "source_fingerprint_reason": profile.get("source_fingerprint_reason"),
            }
            mapping = _candidate_value_mapping(tokens, top_values)
            if mapping:
                hint["candidate_mapping"] = mapping
            hints.append(hint)
        return {
            "success": True,
            "store_id": store_id,
            "prompt": prompt,
            "prompt_terms": tokens,
            "hints": hints,
            "hint_count": len(hints),
            "truncated": result.get("truncated", False),
        }

    def plan_value_grounding(
        self,
        store_id: str,
        prompt: str,
        *,
        validation_facts: Any = None,
        warnings: Any = None,
        session_query_scopes: Any = None,
        targets: Any = None,
        profile_pairs: Any = None,
        profile_budget: int = MAX_VALUE_GROUNDING_PROFILE_BUDGET,
        policy_frame: Any = None,
        blocked_tables: Any = None,
        blocked_columns: Any = None,
    ) -> Dict[str, Any]:
        """Plan value-grounding targets from catalog facts and structured input."""
        schema = self._schema_dict_for_store(store_id)
        profile_budget = max(0, int(profile_budget))
        if schema is None:
            return _value_grounding_plan_response(
                store_id,
                prompt,
                targets=[],
                skipped=[
                    {"table": "", "column": "", "reason": "schema_not_registered"}
                ],
                profile_budget=profile_budget,
            )

        candidates: list[dict[str, Any]] = []
        candidates.extend(
            _value_grounding_candidates_from_validation(
                validation_facts,
                source_kind="validation_fact",
            )
        )
        candidates.extend(
            _value_grounding_candidates_from_validation(
                warnings,
                source_kind="validation_warning",
            )
        )
        candidates.extend(
            _value_grounding_candidates_from_session_scopes(session_query_scopes)
        )
        candidates.extend(
            _value_grounding_candidates_from_explicit_targets(
                targets,
                reason="explicit_target",
                source_kind="explicit_target",
                confidence=1.0,
            )
        )
        candidates.extend(
            _value_grounding_candidates_from_explicit_targets(
                profile_pairs,
                reason="explicit_profile_pair",
                source_kind="profile_pair",
                confidence=0.92,
            )
        )
        for ref, profile in sorted(_column_value_profiles(schema).items()):
            fresh_profile = _profile_with_freshness(profile, schema)
            if (
                fresh_profile.get("profile_status", "profiled") != "profiled"
                or fresh_profile.get("redacted")
                or fresh_profile.get("stale")
            ):
                continue
            candidates.append(
                {
                    "table": fresh_profile.get("table") or ref.rsplit(".", 1)[0],
                    "column": fresh_profile.get("column") or ref.rsplit(".", 1)[-1],
                    "reason": "catalog_profile",
                    "confidence": 0.9,
                    "requires_profile_read": False,
                    "source": {
                        "kind": "catalog_profile",
                        "profile_ref": ref,
                    },
                }
            )

        blocked_table_names, blocked_column_names = _blocked_value_grounding_refs(
            policy_frame,
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )
        ordered_keys: list[tuple[str, str]] = []
        target_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        skipped: list[dict[str, str]] = []
        for candidate in candidates:
            resolved, reason = _resolve_value_grounding_target(schema, candidate)
            if resolved is None:
                skipped.append(
                    {
                        "table": str(candidate.get("table") or ""),
                        "column": str(candidate.get("column") or ""),
                        "reason": reason,
                    }
                )
                continue
            table, column = resolved
            blocked_reason = _value_grounding_blocked_reason(
                table,
                column,
                blocked_tables=blocked_table_names,
                blocked_columns=blocked_column_names,
            )
            if blocked_reason:
                skipped.append(
                    {"table": table, "column": column, "reason": blocked_reason}
                )
                continue
            key = (table.lower(), column.lower())
            target = {
                "table": table,
                "column": column,
                "reason": str(candidate.get("reason") or "structured_input"),
                "confidence": _confidence(candidate.get("confidence")),
                "requires_profile_read": bool(
                    candidate.get("requires_profile_read", True)
                ),
                "source": candidate.get("source") or {"kind": "structured_input"},
            }
            existing = target_by_key.get(key)
            if existing is None:
                target_by_key[key] = target
                ordered_keys.append(key)
            else:
                _merge_value_grounding_target(existing, target)

        selected: list[dict[str, Any]] = []
        read_target_count = 0
        for key in ordered_keys:
            target = target_by_key[key]
            if target["requires_profile_read"]:
                if read_target_count >= profile_budget:
                    skipped.append(
                        {
                            "table": target["table"],
                            "column": target["column"],
                            "reason": "profile_budget_exhausted",
                        }
                    )
                    continue
                read_target_count += 1
            selected.append(target)

        return _value_grounding_plan_response(
            store_id,
            prompt,
            targets=selected,
            skipped=skipped,
            profile_budget=profile_budget,
        )

    def collect_evidence(
        self,
        store_id: str,
        prompt: str,
        intent: Dict[str, Any],
        *,
        asset_types: Optional[List[str]] = None,
        limit: int = 8,
    ) -> Dict[str, Any]:
        """Return compact catalog evidence for planning or exploration."""
        search_terms = [prompt]
        for value in intent.values():
            if isinstance(value, str):
                search_terms.append(value)
            elif isinstance(value, list):
                search_terms.extend(
                    str(item) for item in value if isinstance(item, str)
                )
        query = " ".join(term for term in search_terms if term).strip()
        result = self.search_catalog(
            store_id,
            query,
            asset_types=asset_types,
            include_fields=True,
            limit=limit,
        )
        assets = result.get("assets", [])
        relationships = []
        names = [asset["name"] for asset in assets[:4]]
        if len(names) >= 2:
            relationships = self.find_relationship_paths(
                store_id,
                names[:1],
                names[1:],
                relationship_types=["foreign_key", "references"],
                max_hops=3,
                max_paths=3,
            ).get("paths", [])
        return {
            "store_id": store_id,
            "query": query,
            "assets": assets,
            "relationships": relationships,
            "truncated": result.get("truncated", False),
        }

    def summarize_store(
        self,
        store_id: str,
        *,
        profile: str = "agent",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Return a bounded summary of one profiled store."""
        schema = self._schema_dict_for_store(store_id)
        if schema is None:
            return self._missing_schema_response(store_id)
        limit = _clamp_int(limit, default=50, minimum=1, maximum=200)
        tables = schema.get("tables", []) or []
        return {
            "store_id": store_id,
            "profile": profile,
            "database_type": schema.get("database_type"),
            "database_name": schema.get("database_name") or schema.get("schema"),
            "table_count": int(schema.get("table_count") or len(tables)),
            "asset_count": len(tables),
            "assets": [
                _asset_summary(table, store_id=store_id) for table in tables[:limit]
            ],
            "truncated": len(tables) > limit,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _persist_response(result: Dict[str, Any], persisted: bool) -> Dict[str, Any]:
        """Wrap a discover_* result dict in a standard tool response."""
        response: Dict[str, Any] = {"schema": result, "persisted": persisted}
        if not persisted:
            response["persist_skipped"] = "catalog backend not configured"
        return response

    async def _register_direct_discovery(
        self,
        result: Dict[str, Any],
        *,
        store_type: Optional[str] = None,
        connection_string: Optional[str] = None,
        store_id: Optional[str] = None,
        persist: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from datetime import datetime, timezone

        normalized = _ensure_normalized(result)
        resolved_type = str(store_type or normalized.get("database_type", "unknown"))
        resolved_store_id = store_id or _manual_store_id(
            resolved_type, connection_string, normalized, options or {}
        )
        profiled_at = datetime.now(timezone.utc).isoformat()

        normalized["store_id"] = resolved_store_id
        normalized.setdefault("profiled_at", profiled_at)
        metadata = dict(normalized.get("metadata", {}) or {})
        existing = self._schemas.get(resolved_store_id)
        existing_metadata = dict(getattr(existing, "metadata", {}) or {})
        existing_profiles = existing_metadata.get("column_value_profiles")
        if existing_profiles and "column_value_profiles" not in metadata:
            metadata["column_value_profiles"] = existing_profiles
        if connection_string:
            metadata.setdefault("connection_hint", _connection_hint(connection_string))
        metadata.setdefault("source", "manual")
        normalized["metadata"] = metadata

        schema = NormalizedSchema.from_dict(normalized)
        schema.store_id = resolved_store_id
        schema.profiled_at = normalized.get("profiled_at") or profiled_at
        self._schemas[resolved_store_id] = schema

        store = _manual_store(
            resolved_store_id,
            resolved_type,
            normalized,
            connection_string=connection_string,
            options=options or {},
        )
        self._discovered_stores[resolved_store_id] = store

        persisted = await self._persist_schema(schema.to_dict()) if persist else False
        response = self._persist_response(schema.to_dict(), persisted)
        response["store_id"] = resolved_store_id
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
        """Persist schema to the catalog store and graph backend.

        Raw ``discover_*`` output uses a flat shape (``table_name`` keys,
        columns as a sibling array). The graph persistence layer expects the
        *normalized* shape (``tables[].name`` + nested ``tables[].columns``).
        Apply ``normalize_discovery`` at the boundary so both the catalog JSON
        and the graph backend see the same normalized form.
        """
        normalized = _ensure_normalized(schema)
        if schema.get("store_id"):
            normalized["store_id"] = schema["store_id"]
        if schema.get("profiled_at"):
            normalized["profiled_at"] = schema["profiled_at"]
        if schema.get("metadata"):
            normalized["metadata"] = schema["metadata"]
        return await _persist_schema(
            normalized, self._catalog_backend, self._graph_backend, self._agent_id
        )

    def _schema_dict_for_store(self, store_id: str) -> Optional[Dict[str, Any]]:
        schema = self.get_schema(store_id)
        if schema is None:
            return None
        return schema.to_dict()

    @staticmethod
    def _missing_schema_response(store_id: str) -> Dict[str, Any]:
        return {
            "success": False,
            "store_id": store_id,
            "error": (
                f"No profiled schema for store '{store_id}'. "
                "Profile or register the schema first."
            ),
        }

    @staticmethod
    def _relationships_for_asset(
        schema: Dict[str, Any],
        asset_name: str,
    ) -> List[Dict[str, Any]]:
        wanted = asset_name.lower()
        relationships = []
        for fk in schema.get("foreign_keys", []) or []:
            source = str(fk.get("source_table", "")).lower()
            target = str(fk.get("target_table", "")).lower()
            if source == wanted or target == wanted:
                direction = "outgoing" if source == wanted else "incoming"
                relationships.append(
                    {
                        "relationship_type": "foreign_key",
                        "source_asset": fk.get("source_table"),
                        "source_field": fk.get("source_column"),
                        "target_asset": fk.get("target_table"),
                        "target_field": fk.get("target_column"),
                        "direction": direction,
                    }
                )
        return relationships

    async def prune_stale_catalog(self, max_age_seconds: int) -> dict:
        """Remove catalog entries whose last_seen is older than max_age_seconds."""
        return await prune_stale_catalog(max_age_seconds)

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


def _manual_store(
    store_id: str,
    store_type: str,
    schema: Dict[str, Any],
    *,
    connection_string: Optional[str],
    options: Dict[str, Any],
) -> DiscoveredStore:
    name = (
        schema.get("database_name")
        or schema.get("schema")
        or schema.get("service_name")
        or store_type
    )
    return DiscoveredStore(
        id=store_id,
        store_type=store_type,
        display_name=str(name),
        connection_hint=(
            _connection_hint(connection_string) if connection_string else {}
        ),
        source="manual",
        confidence=1.0,
        tags=["manual"],
        metadata={
            "options": {k: v for k, v in options.items() if v is not None},
            "profiled_at": schema.get("profiled_at"),
        },
    )


def _manual_store_id(
    store_type: str,
    connection_string: Optional[str],
    schema: Dict[str, Any],
    options: Dict[str, Any],
) -> str:
    hint = _connection_hint(connection_string) if connection_string else {}
    parts = [
        str(store_type),
        str(hint.get("host", "")),
        str(hint.get("port", "")),
        str(hint.get("database", "")),
        str(schema.get("database_name") or schema.get("schema") or ""),
        str(options.get("schema") or options.get("database") or ""),
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"manual:{store_type}:{digest}"


def _connection_hint(connection_string: str) -> Dict[str, Any]:
    from urllib.parse import urlparse

    parsed = urlparse(connection_string)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": (parsed.path or "/").lstrip("/") or None,
    }


def _score_asset(
    table: Dict[str, Any],
    tokens: List[str],
    include_fields: bool,
) -> tuple[float, list, list]:
    if not tokens:
        return 0.0, [], []
    name = str(table.get("name", "")).lower()
    metadata = table.get("metadata", {}) or {}
    searchable_metadata = " ".join(str(value).lower() for value in metadata.values())
    score = 0.0
    reasons = []
    matched_fields = []

    for token in tokens:
        if token == name:
            score += 6.0
            reasons.append(f"exact asset:{token}")
        elif token in name or token in _split_identifier(name):
            score += 3.0
            reasons.append(f"asset:{token}")
        elif searchable_metadata and token in searchable_metadata:
            score += 0.5
            reasons.append(f"metadata:{token}")

    if include_fields:
        for field in table.get("columns", []) or []:
            field_name = str(field.get("name", "")).lower()
            field_comment = str(
                field.get("column_comment") or field.get("comment") or ""
            ).lower()
            field_score = 0.0
            field_reasons = []
            for token in tokens:
                if token == field_name:
                    field_score += 4.0
                    field_reasons.append(f"exact:{token}")
                elif token in field_name or token in _split_identifier(field_name):
                    field_score += 2.0
                    field_reasons.append(token)
                elif field_comment and token in field_comment:
                    field_score += 0.5
                    field_reasons.append(f"comment:{token}")
            if field_score:
                score += min(field_score, 8.0)
                matched_fields.append(
                    {
                        "name": field.get("name"),
                        "type": field.get("type"),
                        "score": round(field_score, 3),
                        "reasons": field_reasons[:5],
                    }
                )

    matched_fields.sort(key=lambda item: (-item["score"], item["name"]))
    return score, matched_fields, reasons


def _asset_summary(
    table: Dict[str, Any],
    *,
    store_id: str,
    asset_type: str = "table",
) -> Dict[str, Any]:
    name = table.get("name")
    return {
        "store_id": store_id,
        "name": name,
        "asset_ref": name,
        "asset_type": asset_type,
        "row_count": table.get("row_count"),
        "field_count": len(table.get("columns", []) or []),
        "column_count": len(table.get("columns", []) or []),
    }


def _field_summary(
    field: Dict[str, Any],
    *,
    blocked_fields: set[str],
    store_id: str,
    asset_ref: str,
    database_dialect: str,
    value_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    physical_type = field.get("physical_type") or field.get("type")
    native_type = field.get("native_type") or native_type_from_db_type(physical_type)
    out = {
        "name": field.get("name"),
        "type": field.get("type"),
        "physical_type": physical_type,
        "native_type": native_type,
        "database_dialect": field.get("database_dialect") or database_dialect,
        "nullable": field.get("nullable"),
        "is_primary_key": bool(field.get("is_primary_key")),
    }
    for key in (
        "is_identity",
        "is_generated",
        "is_autoincrement",
        "is_monotonic",
        "default_value",
        "extra",
    ):
        if field.get(key) is not None:
            out[key] = field[key]

    field_name = str(field.get("name") or "")
    identity_proof = field.get("identity_proof")
    if isinstance(identity_proof, dict) and identity_proof:
        out["identity_proof"] = {
            **identity_proof,
            "owner": "catalog",
            "store_id": store_id,
            "asset_ref": asset_ref,
            "column": field_name,
            "database_dialect": database_dialect,
        }

    logical_type = field.get("logical_type")
    logical_type_proof = field.get("logical_type_proof")
    if not logical_type:
        if native_type == "datetime":
            logical_type = "timestamp"
        elif native_type == "date":
            logical_type = "date"
        if logical_type:
            logical_type_proof = {
                "owner": "catalog",
                "source_kind": "declared_physical_type",
                "confidence": 1.0,
                "physical_type": physical_type,
                "store_id": store_id,
                "asset_ref": asset_ref,
                "column": field_name,
                "database_dialect": database_dialect,
            }

    profiled_logical_type = _profiled_logical_type_trait(
        value_profile,
        store_id=store_id,
        asset_ref=asset_ref,
        column=field_name,
        database_dialect=database_dialect,
    )
    if profiled_logical_type is not None:
        logical_type, logical_type_proof = profiled_logical_type
    if logical_type:
        out["logical_type"] = logical_type
    if isinstance(logical_type_proof, dict) and logical_type_proof:
        out["logical_type_proof"] = {
            **logical_type_proof,
            "owner": "catalog",
            "store_id": store_id,
            "asset_ref": asset_ref,
            "column": field_name,
            "database_dialect": database_dialect,
        }
    comment = field.get("column_comment") or field.get("comment")
    if comment:
        out["comment"] = _truncate(str(comment), 160)
    if str(field.get("name", "")).lower() in blocked_fields:
        out["blocked_by_policy"] = True
    if value_profile and _catalog_inline_value_profile_eligible(value_profile):
        out["column_value_hint"] = _column_value_hint_projection(value_profile)
    return out


def _profiled_logical_type_trait(
    profile: Optional[Dict[str, Any]],
    *,
    store_id: str,
    asset_ref: str,
    column: str,
    database_dialect: str,
) -> tuple[str, Dict[str, Any]] | None:
    if not isinstance(profile, dict):
        return None
    if (
        profile.get("profile_status") != "profiled"
        or profile.get("stale")
        or profile.get("redacted")
    ):
        return None
    logical_type = str(profile.get("logical_type") or "").strip().lower()
    proof = profile.get("logical_type_proof")
    if logical_type not in {"timestamp", "date"} or not isinstance(proof, dict):
        return None
    try:
        confidence = float(proof.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return None
    if (
        proof.get("method") != "bounded_value_profile"
        or proof.get("all_values_matched") is not True
        or proof.get("lexicographically_sortable") is not True
        or not profile.get("source_evidence_id")
        or confidence < 0.9
    ):
        return None
    profile_table = str(profile.get("table") or "")
    profile_column = str(profile.get("column") or "")
    if profile_table != asset_ref or profile_column != column:
        return None
    return (
        logical_type,
        {
            **proof,
            "owner": "catalog",
            "source_kind": "schema.column_value_profile",
            "profile_ref": f"{profile_table}.{profile_column}",
            "source_evidence_id": profile.get("source_evidence_id"),
            "source_fingerprint": profile.get("source_fingerprint"),
            "store_id": store_id,
            "asset_ref": asset_ref,
            "column": column,
            "database_dialect": database_dialect,
        },
    )


def _column_value_profiles(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata = schema.get("metadata") or {}
    raw = metadata.get("column_value_profiles") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)
    }


def _profile_with_freshness(
    profile: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    max_age_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    metadata = schema.get("metadata") or {}
    schema_profile_key = (
        str(metadata.get("profile_key"))
        if isinstance(metadata, dict) and metadata.get("profile_key")
        else None
    )
    raw_policy = profile.get("policy")
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    profile_key = str(policy.get("profile_key")) if policy.get("profile_key") else None
    if schema_profile_key and profile_key and profile_key != schema_profile_key:
        return {
            **profile,
            "profile_status": "stale",
            "stale": True,
            "stale_reason": "profile_key_mismatch",
        }
    schema_fingerprint = _schema_structure_fingerprint(schema)
    profile_schema_fingerprint = (
        str(policy.get("schema_fingerprint"))
        if policy.get("schema_fingerprint")
        else None
    )
    if (
        schema_fingerprint
        and profile_schema_fingerprint
        and profile_schema_fingerprint != schema_fingerprint
    ):
        return {
            **profile,
            "profile_status": "stale",
            "stale": True,
            "stale_reason": "schema_fingerprint_mismatch",
        }
    if max_age_seconds is not None and _profile_age_exceeds(
        profile.get("profiled_at"),
        max_age_seconds=max_age_seconds,
    ):
        return {
            **profile,
            "profile_status": "stale",
            "stale": True,
            "stale_reason": "profile_ttl_expired",
        }
    return profile


def _profile_age_exceeds(value: Any, *, max_age_seconds: float) -> bool:
    if max_age_seconds < 0:
        return True
    if value is None:
        return True
    from datetime import datetime, timezone

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    return age.total_seconds() > max_age_seconds


def _schema_structure_fingerprint(schema: Dict[str, Any]) -> str:
    """Fingerprint schema shape without catalog-owned value profile metadata."""
    tables = []
    for table in schema.get("tables", []) or []:
        if not isinstance(table, dict):
            continue
        tables.append(
            {
                "name": table.get("name"),
                "columns": [
                    {
                        "name": column.get("name"),
                        "type": column.get("type") or column.get("data_type"),
                        "physical_type": column.get("physical_type"),
                        "native_type": column.get("native_type"),
                        "database_dialect": column.get("database_dialect"),
                        "nullable": column.get("nullable"),
                        "is_primary_key": column.get("is_primary_key"),
                        "is_identity": column.get("is_identity"),
                        "is_generated": column.get("is_generated"),
                        "is_autoincrement": column.get("is_autoincrement"),
                        "is_monotonic": column.get("is_monotonic"),
                        "identity_proof": column.get("identity_proof"),
                        "logical_type": column.get("logical_type"),
                        "logical_type_proof": column.get("logical_type_proof"),
                        "comment": column.get("column_comment")
                        or column.get("comment"),
                    }
                    for column in table.get("columns", []) or []
                    if isinstance(column, dict)
                ],
                "indexes": [
                    {
                        "name": index.get("name"),
                        "type": index.get("type"),
                        "columns": list(index.get("columns", []) or []),
                        "unique": index.get("unique"),
                        "metadata": dict(index.get("metadata", {}) or {}),
                    }
                    for index in table.get("indexes", []) or []
                    if isinstance(index, dict)
                ],
                "metadata": dict(table.get("metadata", {}) or {}),
            }
        )
    structural = {
        "database_type": schema.get("database_type"),
        "database_name": schema.get("database_name"),
        "tables": tables,
        "foreign_keys": [
            dict(item)
            for item in schema.get("foreign_keys", []) or []
            if isinstance(item, dict)
        ],
    }
    payload = json.dumps(structural, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _column_value_hint_projection(profile: Dict[str, Any]) -> Dict[str, Any]:
    values = []
    for item in profile.get("top_values", []) or []:
        if not isinstance(item, dict):
            continue
        values.append(item.get("value"))
    return {
        "distinct_count": profile.get("distinct_count"),
        "top_values": values[:10],
        "profile_ref": f"{profile.get('table')}.{profile.get('column')}",
    }


def _fresh_column_value_profile(
    profile: Optional[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if profile is None:
        return None
    return _profile_with_freshness(profile, schema)


def _catalog_inline_value_profile_eligible(profile: Dict[str, Any]) -> bool:
    if profile.get("profile_kind") != "categorical_values":
        return False
    if profile.get("profile_status") != "profiled":
        return False
    if profile.get("stale") or profile.get("redacted") or profile.get("sampled"):
        return False
    if profile.get("truncated"):
        return False
    top_values = profile.get("top_values")
    if not isinstance(top_values, list) or not top_values or len(top_values) > 25:
        return False
    for item in top_values:
        if not isinstance(item, dict) or item.get("value") is None:
            return False
    return True


def _value_grounding_plan_response(
    store_id: str,
    prompt: str,
    *,
    targets: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    profile_budget: int,
) -> Dict[str, Any]:
    return {
        "store_id": str(store_id),
        "prompt": str(prompt or ""),
        "targets": targets,
        "skipped": skipped,
        "diagnostics": {
            "profile_budget": int(profile_budget),
            "target_count": len(targets),
            "skipped_count": len(skipped),
        },
    }


def _value_grounding_profile_budget(value: Any) -> int:
    if value is None:
        return MAX_VALUE_GROUNDING_PROFILE_BUDGET
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = MAX_VALUE_GROUNDING_PROFILE_BUDGET
    return max(0, min(50, parsed))


def _value_grounding_candidates_from_validation(
    value: Any,
    *,
    source_kind: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in _value_grounding_items(value):
        if isinstance(item, dict):
            table, column = _table_column_from_mapping(item)
            literal = (
                item.get("literal")
                if "literal" in item
                else item.get("value", item.get("filter_literal"))
            )
            if not table or not column or literal is None:
                continue
            source = {"kind": source_kind, "literal": literal}
            if item.get("kind"):
                source["fact_kind"] = str(item["kind"])
            candidates.append(
                {
                    "table": table,
                    "column": column,
                    "reason": "validation_literal",
                    "confidence": 0.95,
                    "requires_profile_read": True,
                    "source": source,
                }
            )
            continue
        parsed = _parse_value_grounding_warning(str(item))
        if parsed is None:
            continue
        table, column, literal = parsed
        candidates.append(
            {
                "table": table,
                "column": column,
                "reason": "validation_literal",
                "confidence": 0.9,
                "requires_profile_read": True,
                "source": {
                    "kind": source_kind,
                    "literal": literal,
                },
            }
        )
    return candidates


def _value_grounding_candidates_from_session_scopes(value: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for scope in _value_grounding_items(value):
        if not isinstance(scope, dict):
            continue
        tables = _value_grounding_string_list(scope.get("tables"))
        for filter_item in _value_grounding_items(scope.get("filters")):
            if not isinstance(filter_item, dict):
                continue
            table, column = _table_column_from_mapping(filter_item)
            if not table and len(tables) == 1:
                table = tables[0]
            values = _value_grounding_string_list(
                filter_item.get("values")
                if "values" in filter_item
                else filter_item.get("value")
            )
            if not column or not values:
                continue
            source: dict[str, Any] = {
                "kind": "session_query_scope",
                "values": values[:8],
            }
            if scope.get("operation_id"):
                source["operation_id"] = str(scope["operation_id"])
            candidates.append(
                {
                    "table": table,
                    "column": column,
                    "reason": "session_query_scope_filter",
                    "confidence": 0.88,
                    "requires_profile_read": True,
                    "source": source,
                }
            )
    return candidates


def _value_grounding_candidates_from_explicit_targets(
    value: Any,
    *,
    reason: str,
    source_kind: str,
    confidence: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in _value_grounding_items(value):
        table = ""
        column = ""
        requires_profile_read = True
        candidate_reason = reason
        candidate_confidence = confidence
        source: dict[str, Any] = {"kind": source_kind}

        if isinstance(item, dict):
            table, column = _table_column_from_mapping(item)
            requires_profile_read = bool(item.get("requires_profile_read", True))
            candidate_reason = str(item.get("reason") or reason)
            candidate_confidence = _confidence(item.get("confidence", confidence))
            if item.get("source") is not None:
                source["input_source"] = item.get("source")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            table = str(item[0])
            column = str(item[1])
        else:
            table, column = _split_table_column_ref(str(item))

        if not column:
            continue
        candidates.append(
            {
                "table": table,
                "column": column,
                "reason": candidate_reason,
                "confidence": candidate_confidence,
                "requires_profile_read": requires_profile_read,
                "source": source,
            }
        )
    return candidates


def _value_grounding_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _value_grounding_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _table_column_from_mapping(value: dict[str, Any]) -> tuple[str, str]:
    table = _first_value_grounding_string(
        value,
        "table",
        "table_name",
        "target_table",
        "asset",
        "asset_ref",
    )
    column = _first_value_grounding_string(
        value,
        "column",
        "column_name",
        "target_column",
        "field",
        "field_name",
    )
    ref = _first_value_grounding_string(
        value,
        "ref",
        "column_ref",
        "target",
        "target_ref",
    )
    if ref and (not table or not column):
        ref_table, ref_column = _split_table_column_ref(ref)
        if not table:
            table = ref_table
        if not column:
            column = ref_column
    if column and "." in column:
        ref_table, ref_column = _split_table_column_ref(column)
        if not table:
            table = ref_table
        column = ref_column
    return table, column


def _first_value_grounding_string(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = value.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return ""


def _parse_value_grounding_warning(value: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"(?:filter_literal_requires_grounding|unobserved_filter_literal|"
        r"ambiguous_literal_column):"
        r"\s*([^=\s;]+)\s*=\s*([^;]+)",
        value,
    )
    if match is None:
        return None
    table, column = _split_table_column_ref(match.group(1))
    literal = _unquote_value(match.group(2).strip())
    if not table or not column or not literal:
        return None
    return table, column, literal


def _split_table_column_ref(value: str) -> tuple[str, str]:
    parts = [
        part.strip().strip('"`[]')
        for part in str(value or "").split(".")
        if part.strip().strip('"`[]')
    ]
    if len(parts) >= 2:
        return ".".join(parts[:-1]), parts[-1]
    return "", parts[0] if parts else ""


def _unquote_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and (
        (text[0] == text[-1] == "'")
        or (text[0] == text[-1] == '"')
        or (text[0] == "`" and text[-1] == "`")
    ):
        return text[1:-1]
    return text


def _blocked_value_grounding_refs(
    policy_frame: Any,
    *,
    blocked_tables: Any,
    blocked_columns: Any,
) -> tuple[set[str], set[str]]:
    tables = {item.lower() for item in _value_grounding_string_list(blocked_tables)}
    columns = {item.lower() for item in _value_grounding_string_list(blocked_columns)}
    if isinstance(policy_frame, dict):
        for key in ("blocked_tables", "deny_tables", "restricted_tables"):
            tables.update(
                item.lower()
                for item in _value_grounding_string_list(policy_frame.get(key))
            )
        for key in (
            "blocked_columns",
            "blocked_fields",
            "deny_columns",
            "restricted_columns",
            "sensitive_columns",
        ):
            columns.update(
                item.lower()
                for item in _value_grounding_string_list(policy_frame.get(key))
            )
    return tables, columns


def _resolve_value_grounding_target(
    schema: Dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[tuple[str, str] | None, str]:
    raw_table = str(candidate.get("table") or "").strip()
    raw_column = str(candidate.get("column") or "").strip()
    ref_table, ref_column = _split_table_column_ref(raw_column)
    if ref_table and ref_column:
        if not raw_table:
            raw_table = ref_table
        raw_column = ref_column
    if not raw_column:
        return None, "missing_column"

    tables = [
        table
        for table in schema.get("tables", []) or []
        if isinstance(table, dict) and str(table.get("name") or "").strip()
    ]
    if raw_table:
        table = _find_value_grounding_table(tables, raw_table)
        if table is None:
            return None, "unknown_table"
        column = _find_value_grounding_column(table, raw_column)
        if column is None:
            return None, "unknown_column"
        return (str(table["name"]), str(column["name"])), ""

    column_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for table in tables:
        column = _find_value_grounding_column(table, raw_column)
        if column is not None:
            column_matches.append((table, column))
    if not column_matches:
        return None, "unknown_column"
    if len(column_matches) > 1:
        return None, "ambiguous_column"
    table, column = column_matches[0]
    return (str(table["name"]), str(column["name"])), ""


def _find_value_grounding_table(
    tables: list[dict[str, Any]],
    raw_table: str,
) -> dict[str, Any] | None:
    wanted = raw_table.strip().lower()
    if not wanted:
        return None
    direct = [table for table in tables if str(table.get("name", "")).lower() == wanted]
    if len(direct) == 1:
        return direct[0]
    short = [
        table
        for table in tables
        if _short_name(str(table.get("name", "")).lower()) == wanted
        or str(table.get("name", "")).lower().endswith(f".{wanted}")
        or wanted.endswith(f'.{str(table.get("name", "")).lower()}')
    ]
    return short[0] if len(short) == 1 else None


def _find_value_grounding_column(
    table: dict[str, Any],
    raw_column: str,
) -> dict[str, Any] | None:
    wanted = raw_column.strip().lower()
    matches = [
        column
        for column in table.get("columns", []) or []
        if isinstance(column, dict) and str(column.get("name", "")).lower() == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def _value_grounding_blocked_reason(
    table: str,
    column: str,
    *,
    blocked_tables: set[str],
    blocked_columns: set[str],
) -> str | None:
    table_key = table.lower()
    short_table = _short_name(table_key)
    column_key = column.lower()
    if table_key in blocked_tables or short_table in blocked_tables:
        return "blocked_by_policy"
    refs = {
        column_key,
        f"{table_key}.{column_key}",
        f"{short_table}.{column_key}",
    }
    if refs & blocked_columns:
        return "blocked_by_policy"
    return None


def _merge_value_grounding_target(
    existing: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    candidate_confidence = _confidence(candidate.get("confidence"))
    if candidate_confidence > _confidence(existing.get("confidence")):
        existing["confidence"] = candidate_confidence
        existing["reason"] = candidate["reason"]
        existing["source"] = candidate["source"]
    if existing.get("requires_profile_read") and not candidate.get(
        "requires_profile_read"
    ):
        existing["requires_profile_read"] = False
        existing["reason"] = candidate["reason"]
        existing["source"] = candidate["source"]


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    return round(max(0.0, min(1.0, parsed)), 3)


def _normalize_column_value_profile(
    raw: Dict[str, Any],
    *,
    source_evidence_id: Optional[str],
) -> NormalizedColumnValueProfile:
    from datetime import datetime, timezone

    top_values = raw.get("top_values")
    if top_values is None:
        top_values = raw.get("values")
    normalized_values: list[NormalizedColumnValue] = []
    for item in top_values or []:
        if isinstance(item, dict):
            normalized_values.append(NormalizedColumnValue.from_dict(item))
        else:
            normalized_values.append(NormalizedColumnValue(value=item))

    logical_type_proof = dict(raw.get("logical_type_proof", {}) or {})
    if logical_type_proof:
        logical_type_proof.setdefault("source_evidence_id", source_evidence_id)
        logical_type_proof.setdefault(
            "source_owner", str(raw.get("source_owner") or "connector")
        )
    profile = NormalizedColumnValueProfile(
        table=str(raw.get("table") or raw.get("table_name") or ""),
        column=str(raw.get("column") or raw.get("column_name") or ""),
        profile_kind=str(raw.get("profile_kind") or "categorical_values"),
        profile_status=str(raw.get("profile_status") or "profiled"),
        distinct_count=raw.get("distinct_count"),
        null_count=raw.get("null_count"),
        row_count=raw.get("row_count"),
        top_values=normalized_values,
        max_values=int(raw.get("max_values") or len(normalized_values) or 25),
        sampled=bool(raw.get("sampled", False)),
        truncated=bool(raw.get("truncated", False)),
        redacted=bool(raw.get("redacted", False)),
        skipped_reason=raw.get("skipped_reason"),
        policy=dict(raw.get("policy", {}) or {}),
        profiled_at=raw.get("profiled_at") or datetime.now(timezone.utc).isoformat(),
        source_evidence_id=raw.get("source_evidence_id") or source_evidence_id,
        source_fingerprint=raw.get("source_fingerprint"),
        source_fingerprint_status=raw.get("source_fingerprint_status"),
        source_fingerprint_reason=raw.get("source_fingerprint_reason"),
        source_revision=raw.get("source_revision"),
        logical_type=raw.get("logical_type"),
        logical_type_proof=logical_type_proof,
    )
    if profile.profile_status == "profiled" and profile.redacted:
        profile.profile_status = "redacted"
    return profile


def _score_column_value_profile(
    profile: Dict[str, Any], tokens: List[str]
) -> tuple[float, List[str]]:
    if not tokens:
        return 1.0, ["no_query_tokens"]
    score = 0.0
    reasons: list[str] = []
    table = str(profile.get("table") or "").lower()
    column = str(profile.get("column") or "").lower()
    table_parts = set(_split_identifier(table))
    column_parts = set(_split_identifier(column))
    for token in tokens:
        if token == table or token in table_parts:
            score += 3.0
            reasons.append(f"table:{token}")
        if token == column or token in column_parts:
            score += 4.0
            reasons.append(f"column:{token}")
        for value in profile.get("top_values", []) or []:
            if not isinstance(value, dict):
                continue
            observed = str(value.get("value") or "").lower()
            if not observed:
                continue
            if token == observed:
                score += 6.0
                reasons.append(f"value:{token}")
            elif _lexical_value_match(token, observed):
                score += 2.5
                reasons.append(f"value_like:{token}->{observed}")
    return score, reasons


def _candidate_value_mapping(
    tokens: List[str], values: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    best: tuple[float, str, Any, str] | None = None
    for token in tokens:
        for item in values:
            observed = item.get("value")
            observed_text = str(observed or "").lower()
            if not observed_text:
                continue
            confidence = 0.0
            reason = ""
            if token == observed_text:
                confidence = 1.0
                reason = "exact_match"
            elif _lexical_value_match(token, observed_text):
                confidence = 0.82
                reason = "lexical_stem_match"
            if confidence and (best is None or confidence > best[0]):
                best = (confidence, token, observed, reason)
    if best is None:
        return None
    confidence, token, observed, reason = best
    return {
        "prompt_term": token,
        "closest_value": observed,
        "confidence": confidence,
        "reason": reason,
    }


def _lexical_value_match(left: str, right: str) -> bool:
    left_stem = _simple_stem(left)
    right_stem = _simple_stem(right)
    return (
        left_stem == right_stem
        or left_stem + "e" == right_stem
        or left_stem == right_stem + "e"
    )


def _simple_stem(value: str) -> str:
    text = value.lower().strip()
    for suffix in ("ing", "ed", "es", "s"):
        if len(text) > len(suffix) + 3 and text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _find_asset(schema: Dict[str, Any], asset_ref: str) -> Optional[Dict[str, Any]]:
    matches = _matching_assets(schema, asset_ref)
    return matches[0] if len(matches) == 1 else None


def _matching_assets(schema: Dict[str, Any], asset_ref: str) -> List[Dict[str, Any]]:
    wanted = str(asset_ref or "").strip().lower()
    if not wanted:
        return []
    direct = []
    short = []
    for table in schema.get("tables", []) or []:
        name = str(table.get("name", ""))
        lowered = name.lower()
        if lowered == wanted:
            direct.append(table)
        elif _short_name(lowered) == wanted:
            short.append(table)
    return direct or short


def _resolve_asset_refs(
    plugin: CatalogPlugin,
    store_id: str,
    schema: Dict[str, Any],
    asset_refs: List[str],
) -> tuple[List[str], List[Dict[str, Any]]]:
    resolved = []
    errors = []
    for raw_name in asset_refs or []:
        name = str(raw_name or "").strip()
        if not name:
            continue
        matches = _matching_assets(schema, name)
        if len(matches) == 1:
            resolved.append(str(matches[0].get("name")))
            continue
        candidates = plugin.search_catalog(store_id, name, limit=5).get("assets", [])
        errors.append(
            {
                "asset": name,
                "error": "ambiguous asset" if matches else "asset not found",
                "matches": [
                    _asset_summary(match, store_id=store_id) for match in matches[:5]
                ],
                "candidates": candidates,
            }
        )
    return _unique_preserving_order(resolved), errors


def _relationship_adjacency(
    schema: Dict[str, Any],
    relationship_types: Optional[List[str]],
) -> Dict[str, List[tuple[str, Dict[str, Any]]]]:
    wanted_types = {item.lower() for item in relationship_types or []}
    include_fk = not wanted_types or bool(wanted_types & {"foreign_key", "references"})
    adjacency: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    if not include_fk:
        return adjacency
    for fk in schema.get("foreign_keys", []) or []:
        source_table = str(fk.get("source_table") or "")
        source_column = str(fk.get("source_column") or "")
        target_table = str(fk.get("target_table") or "")
        target_column = str(fk.get("target_column") or "")
        if not all((source_table, source_column, target_table, target_column)):
            continue
        forward = {
            "relationship_type": "foreign_key",
            "left_asset": source_table,
            "left_field": source_column,
            "right_asset": target_table,
            "right_field": target_column,
            "predicate": f"{source_table}.{source_column} = {target_table}.{target_column}",
            "relationship_direction": "forward",
        }
        reverse = {
            "relationship_type": "foreign_key",
            "left_asset": target_table,
            "left_field": target_column,
            "right_asset": source_table,
            "right_field": source_column,
            "predicate": f"{target_table}.{target_column} = {source_table}.{source_column}",
            "relationship_direction": "reverse",
        }
        adjacency.setdefault(source_table, []).append((target_table, forward))
        adjacency.setdefault(target_table, []).append((source_table, reverse))
    return adjacency


def _relationship_path_result(
    assets: List[str], relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    hop_count = len(relationships)
    if hop_count <= 1:
        confidence = "high"
    elif hop_count <= 3:
        confidence = "medium"
    else:
        confidence = "low"
    joins = [
        {
            "left_table": rel["left_asset"],
            "left_column": rel["left_field"],
            "right_table": rel["right_asset"],
            "right_column": rel["right_field"],
            "predicate": rel["predicate"],
            "relationship_direction": rel["relationship_direction"],
        }
        for rel in relationships
        if rel.get("relationship_type") == "foreign_key"
    ]
    warnings = []
    bridge_assets = [asset for asset in assets[1:-1] if asset.endswith("_members")]
    if bridge_assets:
        warnings.append(
            "Path crosses membership-style bridge assets; check whether this "
            "attributes facts to all members or to a specific actor."
        )
    return {
        "assets": assets,
        "tables": assets,
        "hop_count": hop_count,
        "relationships": relationships,
        "joins": joins,
        "confidence": confidence,
        "warnings": warnings,
    }


def _matches_field_pattern(field: Dict[str, Any], pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    name = str(field.get("name", "")).lower()
    lowered_pattern = pattern.lower()
    if any(ch in lowered_pattern for ch in "*?[]"):
        return fnmatch.fnmatch(name, lowered_pattern)
    if lowered_pattern in name:
        return True
    pattern_parts = _split_identifier(lowered_pattern)
    name_parts = _split_identifier(name)
    return bool(pattern_parts) and all(
        any(part in name_part or name_part.startswith(part) for name_part in name_parts)
        for part in pattern_parts
    )


def _query_tokens(query: str) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", (query or "").lower())
    tokens: List[str] = []
    for raw in raw_tokens:
        tokens.extend(_split_identifier(raw))
    seen: set[str] = set()
    unique: List[str] = []
    for token in tokens:
        if len(token) <= 1 or token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def _split_identifier(value: str) -> List[str]:
    return [part for part in re.split(r"[^a-zA-Z0-9]+|_", value.lower()) if part]


def _short_name(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _ensure_normalized(schema: Dict[str, Any]) -> Dict[str, Any]:
    tables = schema.get("tables")
    if isinstance(tables, list) and (
        not tables or "name" in tables[0] or "columns" in tables[0]
    ):
        return dict(schema)
    return _normalize_discovery(schema)
