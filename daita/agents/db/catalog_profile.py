"""Catalog-backed active database profiling helpers for ``from_db``.

The catalog plugin owns persistent profile registration, search, inspection,
and traversal. This module stays on the ``from_db`` side of the boundary: it
knows how to obtain a normalized relational profile for the active execution
plugin before that profile is registered with catalog state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ...plugins.catalog import CatalogPlugin
from .resolve import extract_sslmode

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin

SCHEMA_FALLBACK_CONCURRENCY = 10
NUMERIC_TYPES = (
    "int",
    "integer",
    "bigint",
    "smallint",
    "tinyint",
    "float",
    "double",
    "real",
    "decimal",
    "numeric",
    "number",
    "money",
    "currency",
    "smallmoney",
    "int2",
    "int4",
    "int8",
    "float4",
    "float8",
)


def normalize_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DB discovery output using catalog normalizer ownership."""
    return CatalogPlugin.normalize_discovery(raw)


def is_numeric_type(type_name: Optional[str]) -> bool:
    """Return True when a database type string looks numeric."""
    lowered = (type_name or "").lower()
    return any(token in lowered for token in NUMERIC_TYPES)


async def discover_schema(
    plugin: "BaseDatabasePlugin",
    connection_string: Optional[str],
    db_schema: Optional[str],
) -> Dict[str, Any]:
    """
    Discover the active database schema for ``from_db``.

    Direct connection-string discovery delegates to catalog discovery modules;
    generic plugin fallback uses the already-connected DB execution plugin.
    The returned profile is normalized but not persisted. ``from_db`` registers
    it with ``CatalogPlugin.register_schema()`` after applying its freshness and
    drift policy.
    """
    dialect = getattr(plugin, "sql_dialect", None)
    handler = _DISCOVERY_DISPATCH.get(dialect)
    if handler is not None and connection_string:
        result = await handler(connection_string, db_schema, plugin)
        return normalize_schema(result)

    return await discover_schema_fallback(plugin)


async def discover_schema_fallback(
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    """Fallback schema discovery using the plugin's ``tables()`` and ``describe()``."""
    table_names = await plugin.tables()
    semaphore = asyncio.Semaphore(SCHEMA_FALLBACK_CONCURRENCY)

    async def describe_one(table_name: str) -> Any:
        async with semaphore:
            return await plugin.describe(table_name)

    descriptions = await asyncio.gather(*[describe_one(t) for t in table_names])

    db_type = getattr(plugin, "sql_dialect", "unknown")
    db_name = (
        getattr(plugin, "database_name", None)
        or getattr(plugin, "path", None)
        or "unknown"
    )

    tables: List[Dict[str, Any]] = []
    for table_name, columns in zip(table_names, descriptions):
        normalized_columns = [
            {
                "name": column.get("column_name", column.get("name", "")),
                "type": column.get("data_type", column.get("type", "")),
                "nullable": column.get("is_nullable", "YES") == "YES",
                "is_primary_key": bool(column.get("is_primary_key", False)),
            }
            for column in columns
        ]
        tables.append(
            {"name": table_name, "row_count": None, "columns": normalized_columns}
        )

    return {
        "database_type": db_type,
        "database_name": str(db_name),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
    }


async def _discover_postgres(
    connection_string: str,
    db_schema: Optional[str],
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    from ...plugins.catalog.discovery import discover_postgres

    return await discover_postgres(
        connection_string,
        db_schema or "public",
        extract_sslmode(connection_string),
    )


async def _discover_mysql(
    connection_string: str,
    db_schema: Optional[str],
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    from ...plugins.catalog.discovery import discover_mysql

    return await discover_mysql(
        connection_string,
        db_schema,
        extract_sslmode(connection_string),
    )


async def _discover_mongodb(
    connection_string: str,
    db_schema: Optional[str],
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    from ...plugins.catalog.discovery import discover_mongodb

    return await discover_mongodb(
        connection_string,
        getattr(plugin, "database_name", ""),
    )


_DISCOVERY_DISPATCH: Dict[str, Callable[..., Any]] = {
    "postgresql": _discover_postgres,
    "mysql": _discover_mysql,
    "mongodb": _discover_mongodb,
}
