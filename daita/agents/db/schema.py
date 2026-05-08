"""
Schema discovery and normalization.

``discover_schema`` uses a small dialect-keyed dispatch table instead of the
full CatalogPlugin runtime. ``from_db`` only needs database schema discovery and
normalization, not catalog persistence, graph backends, or catalog tools.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ...plugins.catalog.normalizer._mongodb import normalize_mongodb
from ...plugins.catalog.normalizer._mysql import normalize_mysql
from ...plugins.catalog.normalizer._postgresql import normalize_postgresql
from .resolve import extract_sslmode

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)
SCHEMA_FALLBACK_CONCURRENCY = 10

_DB_NORMALIZERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "postgresql": normalize_postgresql,
    "mysql": normalize_mysql,
    "mongodb": normalize_mongodb,
}


def normalize_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DB discovery output into the compact ``from_db`` schema shape."""
    handler = _DB_NORMALIZERS.get(raw.get("database_type", "unknown"))
    return handler(raw) if handler else raw


# ---------------------------------------------------------------------------
# Dialect-keyed DB discovery dispatch
# ---------------------------------------------------------------------------


async def _discover_postgres(
    connection_string: str,
    db_schema: Optional[str],
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    from ...plugins.catalog.discovery._postgres import discover_postgres

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
    from ...plugins.catalog.discovery._mysql import discover_mysql

    return await discover_mysql(connection_string, db_schema)


async def _discover_mongodb(
    connection_string: str,
    db_schema: Optional[str],
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    from ...plugins.catalog.discovery._mongodb import discover_mongodb

    return await discover_mongodb(
        connection_string,
        getattr(plugin, "database_name", ""),
    )


_DISCOVERY_DISPATCH: Dict[str, Callable[..., Any]] = {
    "postgresql": _discover_postgres,
    "mysql": _discover_mysql,
    "mongodb": _discover_mongodb,
}


async def discover_schema(
    plugin: "BaseDatabasePlugin",
    connection_string: Optional[str],
    db_schema: Optional[str],
) -> Dict[str, Any]:
    """
    Discover the database schema via focused DB discovery or plugin fallback.

    Returns a normalized schema dict.
    Uses dialect-keyed dispatch — no isinstance checks on concrete plugin types.
    """
    dialect = getattr(plugin, "sql_dialect", None)

    if dialect in _DISCOVERY_DISPATCH and connection_string:
        result = await _DISCOVERY_DISPATCH[dialect](
            connection_string, db_schema, plugin
        )
        return normalize_schema(result)

    # Fallback for SQLite, Snowflake, or any unrecognized plugin
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
    for tname, cols in zip(table_names, descriptions):
        normalized_cols = [
            {
                "name": col.get("column_name", col.get("name", "")),
                "type": col.get("data_type", col.get("type", "")),
                "nullable": col.get("is_nullable", "YES") == "YES",
                "is_primary_key": bool(col.get("is_primary_key", False)),
            }
            for col in cols
        ]
        tables.append({"name": tname, "row_count": None, "columns": normalized_cols})

    return {
        "database_type": db_type,
        "database_name": str(db_name),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
    }
