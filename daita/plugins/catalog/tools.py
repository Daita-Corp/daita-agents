"""
Catalog tool definitions and handlers.

Builds the list of AgentTool instances exposed by CatalogPlugin and
provides all tool handler functions.
"""

import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.exceptions import ValidationError
from .comparator import compare_schemas as _compare_schemas

if TYPE_CHECKING:
    from ...core.tools import AgentTool
    from .base_discoverer import DiscoveredStore
    from .catalog import CatalogPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def store_to_dict(store: "DiscoveredStore") -> Dict[str, Any]:
    """Serialize a DiscoveredStore to the dict shape returned by tools."""
    return {
        "id": store.id,
        "store_type": store.store_type,
        "display_name": store.display_name,
        "source": store.source,
        "region": store.region,
        "environment": store.environment,
        "confidence": store.confidence,
        "tags": store.tags,
    }


def apply_table_filter(
    schema: Dict[str, Any],
    table_filter: Optional[str],
    max_tables: int,
) -> None:
    """Apply glob filter and truncation to a schema dict's tables in place."""
    if "tables" not in schema:
        return
    tables = schema["tables"]
    if table_filter:
        tables = [
            t
            for t in tables
            if fnmatch.fnmatch(t.get("name", ""), table_filter)
        ]
    total_tables = len(tables)
    schema["tables"] = tables[:max_tables]
    schema["total_tables"] = total_tables
    schema["truncated"] = total_tables > max_tables


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_discover_postgres(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    connection_string = args.get("connection_string")
    if not connection_string:
        raise ValidationError("connection_string is required")

    result = await plugin.discover_postgres(
        connection_string=connection_string,
        schema=args.get("schema", "public"),
        persist=args.get("persist", plugin._auto_persist),
        ssl_mode=args.get("ssl_mode", "verify-full"),
    )

    apply_table_filter(
        result.get("schema", result),
        args.get("table_filter"),
        int(args.get("max_tables") or 50),
    )
    return result


async def _handle_discover_mysql(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    connection_string = args.get("connection_string")
    if not connection_string:
        raise ValidationError("connection_string is required")

    result = await plugin.discover_mysql(
        connection_string=connection_string,
        schema=args.get("schema"),
        persist=args.get("persist", plugin._auto_persist),
    )

    apply_table_filter(
        result.get("schema", result),
        args.get("table_filter"),
        int(args.get("max_tables") or 50),
    )
    return result


async def _handle_discover_mongodb(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    connection_string = args.get("connection_string")
    if not connection_string:
        raise ValidationError("connection_string is required")

    database = args.get("database")
    if not database:
        raise ValidationError("database is required")

    return await plugin.discover_mongodb(
        connection_string=connection_string,
        database=database,
        sample_size=args.get("sample_size", 100),
        persist=args.get("persist", plugin._auto_persist),
    )


async def _handle_discover_openapi(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    spec_url = args.get("spec_url")
    if not spec_url:
        raise ValidationError("spec_url is required")

    return await plugin.discover_openapi(
        spec_url=spec_url,
        service_name=args.get("service_name"),
        persist=args.get("persist", plugin._auto_persist),
    )


async def _handle_compare_schemas(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    return await plugin.compare_schemas(args.get("schema_a"), args.get("schema_b"))


async def _handle_export_diagram(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    return await plugin.export_diagram(args.get("schema"), args.get("format", "mermaid"))


async def _handle_discover_infrastructure(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    concurrency = int(args.get("concurrency") or 5)
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or 50)
    refresh = args.get("refresh", False)

    # Reuse cached results for pagination; only re-scan on first call or explicit refresh
    if refresh or not plugin._discovered_stores:
        result = await plugin.discover_all(concurrency=concurrency)
        errors = result.errors
    else:
        errors = []

    all_stores = list(plugin._discovered_stores.values())
    stores = all_stores[offset : offset + limit]

    return {
        "stores": [store_to_dict(s) for s in stores],
        "total_stores": len(all_stores),
        "offset": offset,
        "limit": limit,
        "last_scan": plugin._last_scan,
        "errors": [
            {"discoverer": e.discoverer_name, "error": e.error}
            for e in errors
        ],
    }


async def _handle_profile_store(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    store_id = args.get("store_id")
    if not store_id:
        raise ValidationError("store_id is required")

    store = plugin.get_store(store_id)
    if not store:
        return {"error": f"Store {store_id} not found in catalog"}

    profiler = plugin._find_profiler(store.store_type)
    if not profiler:
        return {
            "error": f"No profiler registered for store type '{store.store_type}'"
        }

    schema = await profiler.profile(store)
    plugin._schemas[store_id] = schema

    # Persist to catalog.json and graph backend
    await plugin._persist_schema(schema.to_dict())

    return {"store_id": store_id, "schema": schema.to_dict()}


async def _handle_compare_store_to_baseline(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    store_id = args.get("store_id")
    if not store_id:
        raise ValidationError("store_id is required")

    current = plugin._schemas.get(store_id)
    if not current:
        return {"error": f"No profiled schema for store {store_id}. Run profile_store first."}

    # Load baseline from persistence
    catalog_path = Path(".daita") / "catalog.json"
    if not catalog_path.exists():
        return {"error": "No baseline catalog found"}

    try:
        with open(catalog_path, "r") as f:
            existing = json.load(f)
    except json.JSONDecodeError:
        return {"error": "Baseline catalog is corrupt"}

    # Find the baseline schema for this store
    baseline = None
    for key, val in existing.items():
        if val.get("store_id") == store_id:
            baseline = val
            break

    if not baseline:
        return {"error": f"No baseline found for store {store_id}"}

    current_dict = current.to_dict()
    return await _compare_schemas(baseline, current_dict)


async def _handle_find_store(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    query = args.get("query", "")
    store_type = args.get("store_type")
    environment = args.get("environment")
    tag = args.get("tag")
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or 50)

    stores = plugin.get_stores(store_type=store_type, environment=environment)

    if query:
        query_lower = query.lower()
        stores = [
            s for s in stores if query_lower in s.display_name.lower()
        ]

    if tag:
        stores = [s for s in stores if tag in s.tags]

    total = len(stores)
    page = stores[offset : offset + limit]

    return {
        "stores": [store_to_dict(s) for s in page],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------


def build_catalog_tools(plugin: "CatalogPlugin") -> List["AgentTool"]:
    """Build the list of AgentTool instances for CatalogPlugin."""
    from ...core.tools import AgentTool

    return [
        AgentTool(
            name="discover_postgres",
            description="Discover PostgreSQL database schema including tables, columns, foreign keys, and indexes",
            parameters={
                "type": "object",
                "properties": {
                    "connection_string": {
                        "type": "string",
                        "description": "PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)",
                    },
                    "schema": {
                        "type": "string",
                        "description": "Schema name to introspect (default: 'public')",
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Whether to persist schema to graph storage if available (default: auto_persist setting)",
                    },
                    "ssl_mode": {
                        "type": "string",
                        "description": "SSL mode: 'verify-full' (default, validates cert) or 'require' (encrypt only, for pgbouncer poolers)",
                    },
                    "table_filter": {
                        "type": "string",
                        "description": "Glob pattern to filter tables (e.g. 'sales_*', 'orders*'). Leave empty for all tables.",
                    },
                    "max_tables": {
                        "type": "integer",
                        "description": "Maximum number of tables to include (default: 50)",
                    },
                },
                "required": ["connection_string"],
            },
            handler=lambda args, p=plugin: _handle_discover_postgres(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=120,
        ),
        AgentTool(
            name="discover_mysql",
            description="Discover MySQL/MariaDB database schema including tables, columns, and relationships",
            parameters={
                "type": "object",
                "properties": {
                    "connection_string": {
                        "type": "string",
                        "description": "MySQL connection string (e.g., mysql://user:pass@host:port/db)",
                    },
                    "schema": {
                        "type": "string",
                        "description": "Schema/database name to introspect",
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Whether to persist schema to graph storage if available",
                    },
                    "table_filter": {
                        "type": "string",
                        "description": "Glob pattern to filter tables (e.g. 'sales_*'). Leave empty for all tables.",
                    },
                    "max_tables": {
                        "type": "integer",
                        "description": "Maximum number of tables to include (default: 50)",
                    },
                },
                "required": ["connection_string"],
            },
            handler=lambda args, p=plugin: _handle_discover_mysql(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=120,
        ),
        AgentTool(
            name="discover_mongodb",
            description="Discover MongoDB schema by sampling documents to infer structure",
            parameters={
                "type": "object",
                "properties": {
                    "connection_string": {
                        "type": "string",
                        "description": "MongoDB connection string (e.g., mongodb://user:pass@host:port/db)",
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name to introspect",
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of documents to sample per collection (default: 100)",
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Whether to persist schema to graph storage if available",
                    },
                },
                "required": ["connection_string", "database"],
            },
            handler=lambda args, p=plugin: _handle_discover_mongodb(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=120,
        ),
        AgentTool(
            name="discover_openapi",
            description="Discover API structure from OpenAPI/Swagger specification",
            parameters={
                "type": "object",
                "properties": {
                    "spec_url": {
                        "type": "string",
                        "description": "URL to OpenAPI spec (JSON or YAML)",
                    },
                    "service_name": {
                        "type": "string",
                        "description": "Optional service name override",
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Whether to persist schema to graph storage if available",
                    },
                },
                "required": ["spec_url"],
            },
            handler=lambda args, p=plugin: _handle_discover_openapi(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=60,
        ),
        AgentTool(
            name="compare_schemas",
            description="Compare two schemas to identify differences for migration planning",
            parameters={
                "type": "object",
                "properties": {
                    "schema_a": {
                        "type": "object",
                        "description": "First schema (from discover_* tools)",
                    },
                    "schema_b": {
                        "type": "object",
                        "description": "Second schema to compare against",
                    },
                },
                "required": ["schema_a", "schema_b"],
            },
            handler=lambda args, p=plugin: _handle_compare_schemas(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=30,
        ),
        AgentTool(
            name="export_diagram",
            description="Export schema as a visual diagram in Mermaid, DBDiagram, or JSON Schema format",
            parameters={
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "object",
                        "description": "Schema object (from discover_* tools)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'mermaid', 'dbdiagram', or 'json_schema' (default: 'mermaid')",
                    },
                },
                "required": ["schema"],
            },
            handler=lambda args, p=plugin: _handle_export_diagram(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=30,
        ),
        # --- New discovery tools ---
        AgentTool(
            name="discover_infrastructure",
            description="Run all registered infrastructure discoverers to find data stores across cloud providers, config files, and service registries",
            parameters={
                "type": "object",
                "properties": {
                    "concurrency": {
                        "type": "integer",
                        "description": "Max concurrent discoverers (default: 5)",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N results (default: 0)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 50)",
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": "Force a fresh discovery sweep (default: false, reuses cached results)",
                    },
                },
            },
            handler=lambda args, p=plugin: _handle_discover_infrastructure(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=300,
        ),
        AgentTool(
            name="profile_store",
            description="Profile a discovered store by ID to extract its full schema (tables, columns, foreign keys)",
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {
                        "type": "string",
                        "description": "The store fingerprint ID (from discover_infrastructure)",
                    },
                },
                "required": ["store_id"],
            },
            handler=lambda args, p=plugin: _handle_profile_store(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=120,
        ),
        AgentTool(
            name="compare_store_to_baseline",
            description="Compare a store's current schema against its last persisted baseline",
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {
                        "type": "string",
                        "description": "The store fingerprint ID",
                    },
                },
                "required": ["store_id"],
            },
            handler=lambda args, p=plugin: _handle_compare_store_to_baseline(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=120,
        ),
        AgentTool(
            name="find_store",
            description="Search the catalog for stores by name, type, environment, or tags",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (matches display name)",
                    },
                    "store_type": {
                        "type": "string",
                        "description": "Filter by store type (postgresql, mysql, mongodb, s3, etc.)",
                    },
                    "environment": {
                        "type": "string",
                        "description": "Filter by environment (production, staging, development)",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N results (default: 0)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 50)",
                    },
                },
            },
            handler=lambda args, p=plugin: _handle_find_store(p, args),
            category="catalog",
            source="plugin",
            plugin_name="Catalog",
            timeout_seconds=30,
        ),
    ]
