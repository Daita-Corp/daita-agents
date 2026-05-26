"""
Shared utilities for the analyst toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from inspect import iscoroutinefunction
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from .....core.tools import AgentTool
from ...catalog_profile import is_numeric_type
from ...query.metadata import identity_column

if TYPE_CHECKING:
    from .....plugins.base_db import BaseDatabasePlugin


@dataclass
class QueryResult:
    """Rows plus the DB guardrail metadata used to produce them."""

    rows: List[Dict[str, Any]]
    sql: str
    total_rows: int
    truncated: bool = False


@dataclass
class EntityProfileResult:
    """Profile matrix data shared by entity comparison/similarity tools."""

    profiles: Dict[Any, Dict[str, Any]]
    dimensions: List[str]
    query_results: List[QueryResult] = field(default_factory=list)
    skipped_dimensions: List[str] = field(default_factory=list)

    @property
    def source_truncated(self) -> bool:
        return any(result.truncated for result in self.query_results)

    @property
    def source_total_rows(self) -> int:
        return sum(result.total_rows for result in self.query_results)


@dataclass(frozen=True)
class AnalystCatalogContext:
    """Catalog context required by analyst tools that resolve DB structure."""

    catalog: Any
    store_id: Optional[str]
    database_type: str = "unknown"

    @classmethod
    def from_plugin(cls, plugin: Any) -> "AnalystCatalogContext":
        return cls(
            catalog=vars(plugin).get("_db_catalog") if plugin is not None else None,
            store_id=(
                vars(plugin).get("_db_catalog_store_id") if plugin is not None else None
            ),
            database_type=str(
                getattr(plugin, "sql_dialect", None)
                or getattr(plugin, "database_type", None)
                or "unknown"
            ),
        )

    @property
    def available(self) -> bool:
        return self.catalog is not None and bool(self.store_id)


# ---------------------------------------------------------------------------
# Lazy dependency helpers
# ---------------------------------------------------------------------------


def ensure_pandas():
    """Lazy import pandas, raising a helpful error if not installed."""
    try:
        import pandas as pd

        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for analyst tools. "
            "Install with: pip install 'daita-agents[data]'"
        )


def ensure_numpy():
    """Lazy import numpy, raising a helpful error if not installed."""
    try:
        import numpy as np

        return np
    except ImportError:
        raise ImportError(
            "numpy is required for analyst tools. "
            "Install with: pip install 'daita-agents[data]'"
        )


def make_analysis_tool(
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Any,
) -> AgentTool:
    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        category="analysis",
        source="analyst_toolkit",
    )


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------


def _catalog_table(
    context: AnalystCatalogContext,
    table: str,
) -> Optional[Dict[str, Any]]:
    if not context.available or not table:
        return None
    blocked_columns = getattr(context.catalog, "_db_blocked_columns", None)
    result = context.catalog.get_table_schema(
        context.store_id,
        table,
        limit=200,
        offset=0,
        blocked_columns=blocked_columns,
    )
    if not result.get("success"):
        return None
    table_def = dict(result.get("table") or {})
    columns = list(result.get("columns", []) or [])
    foreign_keys = result.get("foreign_keys", []) or []
    while result.get("truncated"):
        result = context.catalog.get_table_schema(
            context.store_id,
            table,
            limit=200,
            offset=len(columns),
            blocked_columns=blocked_columns,
            include_indexes=False,
            include_foreign_keys=False,
        )
        if not result.get("success"):
            break
        columns.extend(result.get("columns", []) or [])
    table_def["columns"] = columns
    table_def["foreign_keys"] = foreign_keys
    return table_def


def _asset_ref_matches(actual: Any, wanted: str) -> bool:
    actual_name = str(actual or "").lower()
    wanted_name = str(wanted or "").lower()
    return actual_name == wanted_name or actual_name.split(".")[-1] == wanted_name


def get_pk_column(
    context: AnalystCatalogContext,
    table: str,
) -> Optional[str]:
    """Return the best identity column name for a table, or None."""
    catalog_table = _catalog_table(context, table)
    if catalog_table is None:
        return None
    return identity_column(catalog_table, mode="declared_or_conventional")


def get_numeric_columns(
    context: AnalystCatalogContext,
    table: str,
) -> List[str]:
    """Return column names with numeric SQL types for a table."""
    catalog_table = _catalog_table(context, table)
    if catalog_table is None:
        return []
    return [
        col["name"]
        for col in catalog_table.get("columns", [])
        if is_numeric_type(col.get("type", "")) and not col.get("blocked_by_policy")
    ]


def _relationships_for_table(
    context: AnalystCatalogContext,
    table: str,
) -> List[Dict[str, Any]]:
    catalog_table = _catalog_table(context, table)
    if catalog_table is None:
        return []
    return catalog_table.get("foreign_keys", []) or []


def _fk_cols_for_table(
    context: AnalystCatalogContext,
    table: str,
) -> set:
    """Return the set of FK source columns in a table (surrogate keys — not metrics)."""
    return {
        rel["source_field"]
        for rel in _relationships_for_table(context, table)
        if _asset_ref_matches(rel.get("source_asset"), table)
        and rel.get("source_field")
    }


def infer_dimensions(
    context: AnalystCatalogContext,
    entity_table: str,
) -> List[Dict[str, Any]]:
    """
    Auto-generate aggregate SQL expressions from FK relationships.

    Walk 1-hop (direct FK children) and, when the child has no meaningful
    numeric columns, continue to 2-hop (grandchildren) so that metrics stored
    in junction/line-item tables (e.g. order_items) are reachable.

    PK and FK columns are excluded from metrics because their integer values
    are identifiers, not measures.

    Returns a list of dicts:
        1-hop: {expression, alias, child_table, fk_col}
        2-hop: {expression, alias, child_table, fk_col,
                grandchild_table, grandchild_fk_col, grandchild_alias, child_pk}
    """
    pk = get_pk_column(context, entity_table)
    if not pk:
        return []

    dims = []

    for fk in _relationships_for_table(context, entity_table):
        if not _asset_ref_matches(fk.get("target_asset"), entity_table):
            continue

        child = fk.get("source_asset")
        fk_col = fk.get("source_field")
        if not child or not fk_col:
            continue
        child_pk = get_pk_column(context, child)

        # Exclude PK and FK columns from metrics — they are identifiers, not measures
        child_excluded = _fk_cols_for_table(context, child) | (
            {child_pk} if child_pk else set()
        )
        numeric_cols = [
            col
            for col in get_numeric_columns(context, child)
            if col not in child_excluded
        ]

        # COUNT is always meaningful
        dims.append(
            {
                "expression": f"COUNT(c.{fk_col})",
                "alias": f"{child}_count",
                "aggregate": "count",
                "child_table": child,
                "fk_col": fk_col,
            }
        )

        for col in numeric_cols[:3]:
            dims.append(
                {
                    "expression": f"SUM(c.{col})",
                    "alias": f"{child}_{col}_sum",
                    "aggregate": "sum",
                    "metric_column": col,
                    "child_table": child,
                    "fk_col": fk_col,
                }
            )
            dims.append(
                {
                    "expression": f"AVG(c.{col})",
                    "alias": f"{child}_{col}_avg",
                    "aggregate": "avg",
                    "metric_column": col,
                    "child_table": child,
                    "fk_col": fk_col,
                }
            )

        # 2-hop: when child has no meaningful numeric columns, look for
        # grandchild tables (e.g. customers → orders → order_items)
        if not numeric_cols and child_pk:
            for fk2 in _relationships_for_table(context, child):
                if not _asset_ref_matches(fk2.get("target_asset"), child):
                    continue
                grandchild = fk2.get("source_asset")
                if not grandchild or grandchild.lower() == entity_table.lower():
                    continue  # skip back-reference

                gc_fk_col = fk2.get("source_field")
                if not gc_fk_col:
                    continue
                gc_alias = f"gc_{grandchild}"
                gc_pk = get_pk_column(context, grandchild)
                gc_excluded = _fk_cols_for_table(context, grandchild) | (
                    {gc_pk} if gc_pk else set()
                )
                gc_numeric = [
                    col
                    for col in get_numeric_columns(context, grandchild)
                    if col not in gc_excluded
                ]

                for col in gc_numeric[:3]:
                    dims.append(
                        {
                            "expression": f"SUM({gc_alias}.{col})",
                            "alias": f"{grandchild}_{col}_sum",
                            "aggregate": "sum",
                            "metric_column": col,
                            "child_table": child,
                            "fk_col": fk_col,
                            "grandchild_table": grandchild,
                            "grandchild_fk_col": gc_fk_col,
                            "grandchild_alias": gc_alias,
                            "child_pk": child_pk,
                        }
                    )
                    dims.append(
                        {
                            "expression": f"AVG({gc_alias}.{col})",
                            "alias": f"{grandchild}_{col}_avg",
                            "aggregate": "avg",
                            "metric_column": col,
                            "child_table": child,
                            "fk_col": fk_col,
                            "grandchild_table": grandchild,
                            "grandchild_fk_col": gc_fk_col,
                            "grandchild_alias": gc_alias,
                            "child_pk": child_pk,
                        }
                    )

    return dims


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------


async def safe_query(
    plugin: "BaseDatabasePlugin",
    sql: str,
    params: Optional[List[Any]] = None,
) -> QueryResult:
    """Execute through DB guardrails and preserve execution metadata."""
    normalized = sql.strip().rstrip(";")

    guarded_query = getattr(plugin, "_run_guarded_tool_query", None)
    if guarded_query is not None and iscoroutinefunction(guarded_query):
        result = await guarded_query(normalized, params or [])
        return QueryResult(
            rows=result.get("rows", []),
            sql=result.get("sql", normalized),
            total_rows=int(result.get("total_rows", 0)),
            truncated=bool(result.get("truncated", False)),
        )

    rows = await plugin.query(normalized, params or None)
    return QueryResult(rows=rows, sql=normalized, total_rows=len(rows), truncated=False)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def to_serializable(value: Any) -> Any:
    """Convert non-JSON-serialisable types (Decimal, datetime) to primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Identifier quoting
# ---------------------------------------------------------------------------


def quote_id(name: str, dialect: str) -> str:
    """Quote an identifier per dialect."""
    if dialect.lower() in ("mysql",):
        return f"`{name.replace('`', '``')}`"
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def quote_path(name: str, dialect: str) -> str:
    """Quote a possibly schema-qualified identifier path."""
    return ".".join(quote_id(part, dialect) for part in str(name).split("."))


def placeholder(dialect: str, index: int) -> str:
    """Return the parameter placeholder for a 1-based parameter index."""
    lowered = dialect.lower()
    if lowered == "postgresql":
        return f"${index}"
    if lowered == "mysql":
        return "%s"
    return "?"


def find_table(
    context: AnalystCatalogContext,
    table: str,
) -> Optional[Dict[str, Any]]:
    return _catalog_table(context, table)


def find_column(
    context: AnalystCatalogContext,
    table: str,
    column: str,
) -> Optional[Dict[str, Any]]:
    table_def = find_table(context, table)
    if table_def is None:
        return None
    wanted = (column or "").lower()
    for item in table_def.get("columns", []):
        if str(item.get("name", "")).lower() == wanted:
            return item
    return None


def source_metadata(result: QueryResult) -> Dict[str, Any]:
    """Compact metadata describing the source query behind an analyst result."""
    return {
        "source_sql": result.sql,
        "source_total_rows": result.total_rows,
        "source_truncated": result.truncated,
    }


def combined_source_metadata(results: Iterable[QueryResult]) -> Dict[str, Any]:
    items = list(results)
    return {
        "source_query_count": len(items),
        "source_total_rows": sum(item.total_rows for item in items),
        "source_truncated": any(item.truncated for item in items),
        "source_sql": [item.sql for item in items],
    }


async def build_entity_profiles(
    plugin: "BaseDatabasePlugin",
    catalog_context: AnalystCatalogContext,
    *,
    entity_table: str,
    id_column: str,
    dimensions: Optional[List[Dict[str, Any]]] = None,
    entity_ids: Optional[List[Any]] = None,
    candidate_limit: Optional[int] = None,
) -> EntityProfileResult:
    """Build aggregate profiles for entity-focused analyst tools."""
    dialect = getattr(plugin, "sql_dialect", "standard")
    if find_table(catalog_context, entity_table) is None:
        raise ValueError(f"Unknown entity table: {entity_table}")
    if find_column(catalog_context, entity_table, id_column) is None:
        raise ValueError(f"Unknown id column: {entity_table}.{id_column}")

    dims = dimensions or infer_dimensions(catalog_context, entity_table)
    child_tables = list(dict.fromkeys(d["child_table"] for d in dims))
    profiles: Dict[Any, Dict[str, Any]] = {eid: {} for eid in entity_ids or []}
    query_results: List[QueryResult] = []
    skipped: List[str] = []

    for child in child_tables:
        child_dims = [d for d in dims if d["child_table"] == child]
        fk_col = child_dims[0].get("fk_col")
        if not fk_col:
            skipped.extend(d.get("alias", "") for d in child_dims)
            continue
        _validate_profile_dimensions(catalog_context, child, child_dims)

        gc_joins: Dict[str, tuple] = {}
        for dim in child_dims:
            if dim.get("grandchild_table"):
                gc_alias = dim["grandchild_alias"]
                if gc_alias not in gc_joins:
                    gc_joins[gc_alias] = (
                        dim["grandchild_table"],
                        dim["grandchild_fk_col"],
                        dim.get("child_pk") or get_pk_column(catalog_context, child),
                    )

        exprs = ", ".join(
            f"{_dimension_expression(dim, dialect)} AS {quote_id(dim['alias'], dialect)}"
            for dim in child_dims
        )
        entity_alias = quote_id("e", dialect)
        child_alias = quote_id("c", dialect)
        sql = (
            f"SELECT {entity_alias}.{quote_id(id_column, dialect)} AS {quote_id('_entity_id', dialect)}, "
            f"{exprs} "
            f"FROM {quote_path(entity_table, dialect)} {entity_alias} "
            f"LEFT JOIN {quote_path(child, dialect)} {child_alias} "
            f"ON {entity_alias}.{quote_id(id_column, dialect)} = {child_alias}.{quote_id(fk_col, dialect)}"
        )
        for gc_alias, (gc_table, gc_fk_col, child_pk_col) in gc_joins.items():
            quoted_alias = quote_id(gc_alias, dialect)
            sql += (
                f" LEFT JOIN {quote_path(gc_table, dialect)} {quoted_alias}"
                f" ON {child_alias}.{quote_id(child_pk_col, dialect)} = "
                f"{quoted_alias}.{quote_id(gc_fk_col, dialect)}"
            )

        params: List[Any] = []
        if entity_ids:
            params = list(entity_ids)
            marks = ", ".join(
                placeholder(dialect, idx) for idx in range(1, len(params) + 1)
            )
            sql += f" WHERE {entity_alias}.{quote_id(id_column, dialect)} IN ({marks})"

        sql += f" GROUP BY {entity_alias}.{quote_id(id_column, dialect)}"
        if not entity_ids and candidate_limit:
            sql += (
                f" ORDER BY {entity_alias}.{quote_id(id_column, dialect)} "
                f"LIMIT {max(1, int(candidate_limit))}"
            )

        result = await safe_query(plugin, sql, params)
        query_results.append(result)
        for row in result.rows:
            eid = row.get("_entity_id")
            if eid not in profiles:
                profiles[eid] = {}
            for dim in child_dims:
                profiles[eid][dim["alias"]] = to_serializable(row.get(dim["alias"]))

    dim_names = list(dict.fromkeys(dim["alias"] for dim in dims))
    return EntityProfileResult(
        profiles=profiles,
        dimensions=dim_names,
        query_results=query_results,
        skipped_dimensions=[item for item in skipped if item],
    )


def _validate_profile_dimensions(
    catalog_context: AnalystCatalogContext,
    child_table: str,
    dimensions: List[Dict[str, Any]],
) -> None:
    if find_table(catalog_context, child_table) is None:
        raise ValueError(f"Unknown dimension table: {child_table}")
    for dim in dimensions:
        if (
            dim.get("metric_column")
            and find_column(
                catalog_context,
                dim.get("grandchild_table") or child_table,
                dim["metric_column"],
            )
            is None
        ):
            raise ValueError(
                f"Unknown metric column: {dim.get('grandchild_table') or child_table}.{dim['metric_column']}"
            )


def _dimension_expression(dim: Dict[str, Any], dialect: str) -> str:
    aggregate = str(dim.get("aggregate", "")).upper()
    child_alias = quote_id("c", dialect)
    if aggregate == "COUNT":
        return f"COUNT({child_alias}.{quote_id(dim['fk_col'], dialect)})"
    if aggregate in ("SUM", "AVG") and dim.get("metric_column"):
        alias = quote_id(dim.get("grandchild_alias") or "c", dialect)
        return f"{aggregate}({alias}.{quote_id(dim['metric_column'], dialect)})"
    return str(dim["expression"])
