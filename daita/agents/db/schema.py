"""
Schema discovery and normalization.

``discover_schema`` uses a dialect-keyed dispatch table instead of isinstance
checks, so adding new DB support never requires modifying this file.

The three per-dialect normalizers live in CatalogPlugin (catalog.py) and are
re-exported here for direct import convenience.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...plugins.catalog import CatalogPlugin
from .resolve import extract_sslmode

if TYPE_CHECKING:
    from ...plugins.base_db import BaseDatabasePlugin

logger = logging.getLogger(__name__)

# Re-export normalizers so tests/callers can import from this module
normalize_postgresql = CatalogPlugin.normalize_postgresql
normalize_mysql = CatalogPlugin.normalize_mysql
normalize_mongodb = CatalogPlugin.normalize_mongodb
normalize_schema = CatalogPlugin.normalize_discovery

# ---------------------------------------------------------------------------
# Dialect-keyed dispatch (Phase 2)
# Each entry: (catalog_method_name, kwargs_builder(connection_string, db_schema, plugin))
# ---------------------------------------------------------------------------

_CATALOG_DISPATCH: Dict[str, Any] = {
    "postgresql": (
        "discover_postgres",
        lambda cs, ds, plugin: {
            "connection_string": cs,
            "schema": ds or "public",
            "persist": False,
            "ssl_mode": extract_sslmode(cs),
        },
    ),
    "mysql": (
        "discover_mysql",
        lambda cs, ds, plugin: {
            "connection_string": cs,
            "schema": ds,
            "persist": False,
        },
    ),
    "mongodb": (
        "discover_mongodb",
        lambda cs, ds, plugin: {
            "connection_string": cs,
            "database": getattr(plugin, "database_name", ""),
            "persist": False,
        },
    ),
}


async def discover_schema(
    plugin: "BaseDatabasePlugin",
    connection_string: Optional[str],
    db_schema: Optional[str],
) -> Dict[str, Any]:
    """
    Discover the database schema via CatalogPlugin or plugin fallback.

    Returns a normalized schema dict (see :meth:`CatalogPlugin.normalize_discovery`).
    Uses dialect-keyed dispatch — no isinstance checks on concrete plugin types.
    """
    dialect = getattr(plugin, "sql_dialect", None)

    if dialect in _CATALOG_DISPATCH and connection_string:
        catalog = CatalogPlugin()
        method_name, kwargs_fn = _CATALOG_DISPATCH[dialect]
        kwargs = kwargs_fn(connection_string, db_schema, plugin)
        result = await getattr(catalog, method_name)(**kwargs)
        return CatalogPlugin.normalize_discovery(result["schema"])

    # Fallback for SQLite, Snowflake, or any unrecognized plugin
    return await discover_schema_fallback(plugin)


async def discover_schema_fallback(
    plugin: "BaseDatabasePlugin",
) -> Dict[str, Any]:
    """Fallback schema discovery using the plugin's ``tables()`` and ``describe()``."""
    table_names = await plugin.tables()
    descriptions = await asyncio.gather(*[plugin.describe(t) for t in table_names])

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
