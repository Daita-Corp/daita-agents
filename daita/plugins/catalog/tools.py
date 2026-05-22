"""
Catalog tool definitions and handlers.

Builds the list of AgentTool instances exposed by CatalogPlugin and
provides all tool handler functions.

The agent-facing surface is organised along a two-plane split:

  * Control plane — "what stores exist?"
        discover_infrastructure

  * Data plane (known endpoint) — "here is a connection string, profile it"
        discover_schema (dispatches by store_type)

  * Data plane (discovered store) — "profile a store I already found"
        profile_store
        get_table_schema (point lookup for a single table)

  * Cross-cutting
        find_store
        compare_schemas (live vs live, or live vs persisted baseline)
        export_diagram
"""

import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...core.exceptions import ValidationError
from .comparator import compare_schemas as _compare_schemas
from .diagram import export_diagram as _export_diagram

if TYPE_CHECKING:
    from ...core.tools import AgentTool
    from .base_discoverer import DiscoveredStore
    from .base_profiler import NormalizedSchema
    from .catalog import CatalogPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STORE_TYPE_ENUM = ["postgresql", "mysql", "mongodb", "openapi"]


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
        tables = [t for t in tables if fnmatch.fnmatch(t.get("name", ""), table_filter)]
    total_tables = len(tables)
    schema["tables"] = tables[:max_tables]
    schema["total_tables"] = total_tables
    schema["truncated"] = total_tables > max_tables


def _require(args: Dict[str, Any], *keys: str) -> None:
    """Raise ValidationError if any required key is missing from args."""
    for key in keys:
        if not args.get(key):
            raise ValidationError(f"{key} is required")


def _schema_dict(schema: "NormalizedSchema") -> Dict[str, Any]:
    """NormalizedSchema → dict, accepting both dataclass and dict inputs."""
    if hasattr(schema, "to_dict"):
        return schema.to_dict()
    return dict(schema)


def _load_baseline_for_store(store_id: str) -> Optional[Dict[str, Any]]:
    """Return the persisted baseline schema for ``store_id`` or None."""
    catalog_path = Path(".daita") / "catalog.json"
    if not catalog_path.exists():
        return None
    try:
        with open(catalog_path, "r") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return next(
        (val for val in existing.values() if val.get("store_id") == store_id),
        None,
    )


async def _resolve_schema_target(
    plugin: "CatalogPlugin", target: str
) -> Dict[str, Any]:
    """
    Resolve a schema reference to a concrete schema dict.

    ``target`` is either a store_id (returns the live profiled schema) or the
    literal ``"baseline:<store_id>"`` (returns the last persisted baseline for
    that store).
    """
    if target.startswith("baseline:"):
        store_id = target[len("baseline:") :]
        baseline = _load_baseline_for_store(store_id)
        if baseline is None:
            raise ValidationError(
                f"No baseline found for store '{store_id}'. "
                f"Profile the store first with profile_store."
            )
        return baseline

    # Assume plain store_id — require it to be profiled already.
    schema = plugin.get_schema(target)
    if schema is None:
        raise ValidationError(
            f"No profiled schema for store '{target}'. " f"Run profile_store first."
        )
    return _schema_dict(schema)


# ---------------------------------------------------------------------------
# Tool handlers — discovery / profiling
# ---------------------------------------------------------------------------


async def _handle_discover_infrastructure(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    concurrency = int(args.get("concurrency") or 5)
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or 50)
    refresh = args.get("refresh", False)

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
        "errors": [{"discoverer": e.discoverer_name, "error": e.error} for e in errors],
    }


async def _handle_discover_schema(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    """Dispatch a connection-string profile to the correct discoverer.

    Replaces the per-database discover_postgres / discover_mysql / etc. tools
    with a single entry point parameterised by ``store_type``.
    """
    _require(args, "store_type", "connection_string")
    store_type = args["store_type"].lower()
    if store_type not in _STORE_TYPE_ENUM:
        raise ValidationError(
            f"Unsupported store_type '{store_type}'. " f"Use one of: {_STORE_TYPE_ENUM}"
        )

    options = args.get("options") or {}
    connection_string = args["connection_string"]
    persist = args.get("persist", plugin._auto_persist)

    if store_type == "postgresql":
        result = await plugin.discover_postgres(
            connection_string=connection_string,
            schema=options.get("schema", "public"),
            persist=persist,
            ssl_mode=options.get("ssl_mode", "verify-full"),
        )
    elif store_type == "mysql":
        result = await plugin.discover_mysql(
            connection_string=connection_string,
            schema=options.get("schema"),
            persist=persist,
            ssl_mode=options.get("ssl_mode", "verify-full"),
        )
    elif store_type == "mongodb":
        database = options.get("database")
        if not database:
            raise ValidationError("mongodb requires options.database")
        result = await plugin.discover_mongodb(
            connection_string=connection_string,
            database=database,
            sample_size=int(options.get("sample_size") or 100),
            persist=persist,
        )
    elif store_type == "openapi":
        # connection_string is treated as the spec URL for OpenAPI.
        result = await plugin.discover_openapi(
            spec_url=connection_string,
            service_name=options.get("service_name"),
            persist=persist,
        )

    apply_table_filter(
        result.get("schema", result),
        options.get("table_filter"),
        int(options.get("max_tables") or 50),
    )
    return result


async def _handle_profile_store(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    _require(args, "store_id")
    store_id = args["store_id"]

    store = plugin.get_store(store_id)
    if not store:
        return {"error": f"Store {store_id} not found in catalog"}

    profiler = plugin._find_profiler(store.store_type)
    if not profiler:
        return {"error": f"No profiler registered for store type '{store.store_type}'"}

    schema = await profiler.profile(store)
    schema.store_id = store_id
    plugin._schemas[store_id] = schema
    await plugin._persist_schema(schema.to_dict())

    return {"store_id": store_id, "schema": schema.to_dict()}


async def _handle_get_table_schema(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    """Point lookup for a single table within a profiled store.

    Avoids shipping a 1000-table store schema back to the LLM when the agent
    only needs one table's columns / indexes / FKs.
    """
    _require(args, "store_id", "table_name")
    store_id = args["store_id"]
    table_name = args["table_name"]
    include_indexes = args.get("include_indexes", True)
    include_foreign_keys = args.get("include_foreign_keys", True)
    return plugin.get_table_schema(
        store_id,
        table_name,
        column_pattern=args.get("column_pattern"),
        limit=int(args.get("limit") or 100),
        offset=int(args.get("offset") or 0),
        include_indexes=include_indexes,
        include_foreign_keys=include_foreign_keys,
        blocked_columns=getattr(plugin, "_db_blocked_columns", None),
    )


async def _handle_search_catalog(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    _require(args, "store_id", "query")
    return plugin.search_catalog(
        args["store_id"],
        args["query"],
        asset_types=args.get("asset_types"),
        include_fields=args.get("include_fields", True),
        limit=int(args.get("limit") or 20),
    )


async def _handle_inspect_asset(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    _require(args, "store_id", "asset_ref")
    return plugin.inspect_asset(
        args["store_id"],
        args["asset_ref"],
        field_filter=args.get("field_filter"),
        offset=int(args.get("offset") or 0),
        limit=int(args.get("limit") or 100),
    )


async def _handle_find_relationship_paths(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    _require(args, "store_id", "from_assets", "to_assets")
    return plugin.find_relationship_paths(
        args["store_id"],
        list(args.get("from_assets") or []),
        list(args.get("to_assets") or []),
        relationship_types=args.get("relationship_types"),
        max_hops=int(args.get("max_hops") or 4),
        max_paths=int(args.get("max_paths") or 5),
    )


async def _handle_catalog_search_schema(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    _require(args, "store_id", "query")
    return plugin.catalog_search_schema(
        args["store_id"],
        args["query"],
        limit=int(args.get("limit") or 20),
    )


async def _handle_catalog_inspect_table(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    return await _handle_get_table_schema(plugin, args)


async def _handle_catalog_find_join_paths(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    _require(args, "store_id", "from_tables", "to_tables")
    return plugin.find_relationship_paths(
        args["store_id"],
        list(args.get("from_tables") or []),
        list(args.get("to_tables") or []),
        relationship_types=["foreign_key", "references"],
        max_hops=int(args.get("max_hops") or 4),
        max_paths=int(args.get("max_paths") or 5),
    )


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
        stores = [s for s in stores if query_lower in s.display_name.lower()]

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


async def _handle_compare_schemas(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two schemas. Each of ``source`` / ``target`` is either:

    * A store_id (uses the live profiled schema — call profile_store first)
    * ``"baseline:<store_id>"`` (uses the last persisted baseline)

    Covers drift detection (``target="baseline:<id>"``) and cross-store diffs
    (``source=<id_a>, target=<id_b>``) through a single tool.
    """
    _require(args, "source", "target")
    source_dict = await _resolve_schema_target(plugin, args["source"])
    target_dict = await _resolve_schema_target(plugin, args["target"])
    result = await _compare_schemas(source_dict, target_dict)
    return {
        "source": args["source"],
        "target": args["target"],
        "comparison": result,
    }


async def _handle_export_diagram(
    plugin: "CatalogPlugin", args: Dict[str, Any]
) -> Dict[str, Any]:
    """Export a visual diagram for a profiled store, by store_id."""
    _require(args, "store_id")
    store_id = args["store_id"]
    fmt = args.get("format", "mermaid")

    schema = plugin.get_schema(store_id)
    if schema is None:
        raise ValidationError(
            f"No profiled schema for store '{store_id}'. " f"Run profile_store first."
        )
    return await _export_diagram(_schema_dict(schema), fmt)


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------

# Common kwargs shared by all catalog tools
_CATALOG_TOOL_DEFAULTS = {
    "category": "catalog",
    "source": "plugin",
    "plugin_name": "Catalog",
}


def _catalog_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler_fn,
    plugin: "CatalogPlugin",
    timeout_seconds: int = 60,
    replay_safe: bool = True,
    retry_safe: bool = True,
    side_effecting: bool = False,
) -> "AgentTool":
    """Create an AgentTool with catalog defaults and bound handler."""
    from ...core.tools import AgentTool

    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        handler=lambda args, p=plugin: handler_fn(p, args),
        timeout_seconds=timeout_seconds,
        retry_safe=retry_safe,
        replay_safe=replay_safe,
        side_effecting=side_effecting,
        **_CATALOG_TOOL_DEFAULTS,
    )


def build_catalog_tools(plugin: "CatalogPlugin") -> List["AgentTool"]:
    """Build the agent-facing CatalogPlugin tools."""
    t = lambda **kw: _catalog_tool(plugin=plugin, **kw)

    return [
        t(
            name="discover_infrastructure",
            description=(
                "Control plane: find every data store reachable from registered "
                "discoverers (AWS, GCP, Azure, GitHub, config files, service registries). "
                "Use when you don't yet know what stores exist. Stores returned "
                "here can be profiled with profile_store."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "concurrency": {
                        "type": "integer",
                        "description": "Max concurrent discoverers (default 5).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N results (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 50).",
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": "Force a fresh discovery sweep (default false — reuses cached results).",
                    },
                },
            },
            handler_fn=_handle_discover_infrastructure,
            timeout_seconds=300,
        ),
        t(
            name="discover_schema",
            description=(
                "Data plane: profile a single known endpoint from a connection "
                "string. Dispatches by store_type (postgresql, mysql, mongodb, "
                "openapi). Use when the user hands you a connection string and "
                "you haven't run discover_infrastructure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_type": {
                        "type": "string",
                        "enum": _STORE_TYPE_ENUM,
                        "description": "The kind of endpoint to profile.",
                    },
                    "connection_string": {
                        "type": "string",
                        "description": (
                            "Connection string (postgresql/mysql/mongodb URL, "
                            "or OpenAPI spec URL for store_type='openapi')."
                        ),
                    },
                    "options": {
                        "type": "object",
                        "description": (
                            "Store-type-specific options. "
                            "postgresql: {schema, ssl_mode ('disable'|'require'|"
                            "'verify-full'; pass 'disable' for local/127.0.0.1 "
                            "or containerized instances that don't speak TLS), "
                            "table_filter, max_tables}. "
                            "mysql: {schema, ssl_mode ('disable'|'require'|"
                            "'verify-full'; 'disable' for local containers), "
                            "table_filter, max_tables}. "
                            "mongodb: {database (required), sample_size}. "
                            "openapi: {service_name}."
                        ),
                    },
                    "persist": {
                        "type": "boolean",
                        "description": "Persist the resulting schema to the catalog backend (default: plugin's auto_persist setting).",
                    },
                },
                "required": ["store_type", "connection_string"],
            },
            handler_fn=_handle_discover_schema,
            timeout_seconds=120,
        ),
        t(
            name="profile_store",
            description=(
                "Extract the full schema (tables, columns, indexes, foreign "
                "keys) from a store previously found by discover_infrastructure. "
                "Persists the profile so subsequent calls to get_table_schema / "
                "compare_schemas / export_diagram can reference it by store_id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {
                        "type": "string",
                        "description": "Store fingerprint ID from discover_infrastructure or find_store.",
                    },
                },
                "required": ["store_id"],
            },
            handler_fn=_handle_profile_store,
            timeout_seconds=120,
        ),
        t(
            name="get_table_schema",
            description=(
                "Return the columns, indexes, and foreign keys for a single "
                "table inside a profiled store. Use this instead of pulling a "
                "full profile_store response when you only need one table — "
                "avoids shipping a large schema payload to the LLM."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {
                        "type": "string",
                        "description": "Store ID that has already been profiled.",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to inspect.",
                    },
                    "include_indexes": {
                        "type": "boolean",
                        "description": "Include index definitions (default true).",
                    },
                    "include_foreign_keys": {
                        "type": "boolean",
                        "description": "Include foreign keys touching this table (default true).",
                    },
                    "column_pattern": {
                        "type": "string",
                        "description": "Optional substring or glob for columns.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N matched columns (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max columns to return (default 100).",
                    },
                },
                "required": ["store_id", "table_name"],
            },
            handler_fn=_handle_get_table_schema,
            timeout_seconds=30,
        ),
        t(
            name="search_catalog",
            description=(
                "Search profiled catalog assets and fields inside one store. "
                "Returns bounded, scored candidates rather than the full profile."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Profiled store ID."},
                    "query": {"type": "string", "description": "Search text."},
                    "asset_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional asset type filter.",
                    },
                    "include_fields": {
                        "type": "boolean",
                        "description": "Search and return matched fields (default true).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max assets to return (default 20).",
                    },
                },
                "required": ["store_id", "query"],
            },
            handler_fn=_handle_search_catalog,
            timeout_seconds=30,
        ),
        t(
            name="inspect_asset",
            description=(
                "Inspect one catalog asset with bounded/paginated fields and "
                "nearby relationships."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Profiled store ID."},
                    "asset_ref": {
                        "type": "string",
                        "description": "Asset name or reference to inspect.",
                    },
                    "field_filter": {
                        "type": "string",
                        "description": "Optional substring or glob for fields.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N matched fields (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max fields to return (default 100).",
                    },
                },
                "required": ["store_id", "asset_ref"],
            },
            handler_fn=_handle_inspect_asset,
            timeout_seconds=30,
        ),
        t(
            name="find_relationship_paths",
            description=(
                "Find bounded relationship paths between catalog assets without "
                "executing SQL."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Profiled store ID."},
                    "from_assets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Starting asset references.",
                    },
                    "to_assets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target asset references.",
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional relationship type filter.",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum path length (default 4).",
                    },
                    "max_paths": {
                        "type": "integer",
                        "description": "Maximum paths to return (default 5).",
                    },
                },
                "required": ["store_id", "from_assets", "to_assets"],
            },
            handler_fn=_handle_find_relationship_paths,
            timeout_seconds=30,
        ),
        t(
            name="catalog_search_schema",
            description=(
                "Relational alias over search_catalog for tables, views, and "
                "collections."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Profiled store ID."},
                    "query": {"type": "string", "description": "Search text."},
                    "limit": {
                        "type": "integer",
                        "description": "Max tables to return (default 20).",
                    },
                },
                "required": ["store_id", "query"],
            },
            handler_fn=_handle_catalog_search_schema,
            timeout_seconds=30,
        ),
        t(
            name="catalog_inspect_table",
            description="Relational alias over inspect_asset for table metadata.",
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Profiled store ID."},
                    "table_name": {
                        "type": "string",
                        "description": "Table to inspect.",
                    },
                    "column_pattern": {
                        "type": "string",
                        "description": "Optional substring or glob for columns.",
                    },
                    "offset": {"type": "integer", "description": "Column offset."},
                    "limit": {"type": "integer", "description": "Column limit."},
                },
                "required": ["store_id", "table_name"],
            },
            handler_fn=_handle_catalog_inspect_table,
            timeout_seconds=30,
        ),
        t(
            name="catalog_find_join_paths",
            description=(
                "Relational alias over find_relationship_paths for FK/reference "
                "join paths."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Profiled store ID."},
                    "from_tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Starting tables.",
                    },
                    "to_tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target tables.",
                    },
                    "max_hops": {"type": "integer", "description": "Default 4."},
                    "max_paths": {"type": "integer", "description": "Default 5."},
                },
                "required": ["store_id", "from_tables", "to_tables"],
            },
            handler_fn=_handle_catalog_find_join_paths,
            timeout_seconds=30,
        ),
        t(
            name="find_store",
            description=(
                "Search the catalog of discovered stores by display name, "
                "store type, environment, or tag. Returns matching stores with "
                "their store_ids for use with profile_store / get_table_schema."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring match against store display name.",
                    },
                    "store_type": {
                        "type": "string",
                        "description": "Filter by store type (postgresql, mysql, mongodb, s3, ...).",
                    },
                    "environment": {
                        "type": "string",
                        "description": "Filter by environment (production, staging, development).",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N results (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 50).",
                    },
                },
            },
            handler_fn=_handle_find_store,
            timeout_seconds=30,
        ),
        t(
            name="compare_schemas",
            description=(
                "Diff two schemas. Each of 'source' / 'target' is either a "
                "store_id (for the live profiled schema) or 'baseline:<store_id>' "
                "for the last persisted baseline. Covers drift detection "
                "(target='baseline:<id>') and cross-store diffs "
                "(source=<id_a>, target=<id_b>) in one tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Left side of the diff. store_id or 'baseline:<store_id>'.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Right side of the diff. store_id or 'baseline:<store_id>'.",
                    },
                },
                "required": ["source", "target"],
            },
            handler_fn=_handle_compare_schemas,
            timeout_seconds=30,
        ),
        t(
            name="export_diagram",
            description=(
                "Render a profiled store's schema as a visual diagram "
                "(Mermaid, DBDiagram, or JSON Schema). Takes a store_id that "
                "has already been profiled — no need to pass a schema dict."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "store_id": {
                        "type": "string",
                        "description": "Store ID that has already been profiled.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["mermaid", "dbdiagram", "json_schema"],
                        "description": "Output format (default 'mermaid').",
                    },
                },
                "required": ["store_id"],
            },
            handler_fn=_handle_export_diagram,
            timeout_seconds=30,
        ),
    ]
