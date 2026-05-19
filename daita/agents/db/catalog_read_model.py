"""DB-agent read model over the active CatalogPlugin store.

This module is intentionally on the ``from_db`` side of the boundary. Catalog
owns stores, profiles, search, inspection, relationships, and persistence; this
module turns those catalog facts into database-agent context such as summary
signals, metric candidates, counts, and display metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING

from .catalog_summary import build_db_summary
from .utils import plugin_database_name

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass(frozen=True)
class DBCatalogReadModel:
    """Catalog-backed facts shaped for DB-agent describe/context surfaces."""

    schema: Dict[str, Any] = field(default_factory=dict)
    catalog_summary: Dict[str, Any] = field(default_factory=dict)
    db_summary: Dict[str, Any] = field(default_factory=dict)
    store_id: str | None = None
    database_type: str = "unknown"
    database_name: str = "unknown"
    table_count: int = 0
    column_count: int = 0
    relationship_count: int = 0
    catalog_status: str = "unknown"


def build_db_catalog_read_model(
    agent: "Agent",
    *,
    summary: Dict[str, Any] | None = None,
    catalog_limit: int = 50,
) -> DBCatalogReadModel:
    """Return the active DB-agent read model from catalog-backed state."""
    schema = _catalog_schema_snapshot(agent)
    catalog_summary = _catalog_store_summary(agent, limit=catalog_limit)
    plugin = getattr(agent, "_db_plugin", None)
    tables = schema.get("tables", []) or []
    foreign_keys = schema.get("foreign_keys", []) or []
    db_summary = summary if summary is not None else build_db_summary(schema)

    return DBCatalogReadModel(
        schema=schema,
        catalog_summary=catalog_summary,
        db_summary=db_summary,
        store_id=getattr(agent, "_db_catalog_store_id", None),
        database_type=str(
            catalog_summary.get("database_type")
            or schema.get("database_type")
            or getattr(plugin, "sql_dialect", None)
            or "unknown"
        ),
        database_name=str(
            catalog_summary.get("database_name")
            or schema.get("database_name")
            or plugin_database_name(plugin)
            or "unknown"
        ),
        table_count=int(
            catalog_summary.get("table_count")
            or schema.get("table_count")
            or len(tables)
            or 0
        ),
        column_count=_column_count(schema, catalog_summary),
        relationship_count=len(foreign_keys),
        catalog_status=(
            "completed" if catalog_summary.get("store_id") or schema else "unknown"
        ),
    )


def db_summary_from_catalog(agent: "Agent") -> Dict[str, Any]:
    """Build the DB-specific summary for the active catalog store."""
    return build_db_catalog_read_model(agent).db_summary


def _catalog_schema_snapshot(agent: "Agent") -> Dict[str, Any]:
    catalog = vars(agent).get("_db_catalog")
    store_id = vars(agent).get("_db_catalog_store_id")
    if catalog is None or not store_id:
        return {}
    schema = catalog.get_schema(store_id)
    if schema is None:
        return {}
    return schema.to_dict() if hasattr(schema, "to_dict") else dict(schema)


def _catalog_store_summary(agent: "Agent", *, limit: int) -> Dict[str, Any]:
    catalog = vars(agent).get("_db_catalog")
    store_id = vars(agent).get("_db_catalog_store_id")
    if catalog is None or not store_id:
        return {}
    summary = catalog.summarize_store(store_id, limit=limit)
    if summary and not summary.get("error"):
        return summary
    return {}


def _column_count(schema: Dict[str, Any], catalog_summary: Dict[str, Any]) -> int:
    count = sum(
        len(table.get("columns", []) or []) for table in schema.get("tables", []) or []
    )
    if count:
        return count
    return sum(
        int(asset.get("column_count") or asset.get("field_count") or 0)
        for asset in catalog_summary.get("assets", []) or []
    )
