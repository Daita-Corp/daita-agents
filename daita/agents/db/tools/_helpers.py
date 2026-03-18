"""
Shared utilities for the analyst toolkit.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....plugins.base_db import BaseDatabasePlugin

# ---------------------------------------------------------------------------
# SQL type constants
# ---------------------------------------------------------------------------

NUMERIC_TYPES = {
    "int", "integer", "bigint", "smallint", "tinyint",
    "float", "double", "real", "decimal", "numeric",
    "number", "money", "currency", "int4", "int8", "float4", "float8",
}

DATE_TYPES = {
    "date", "datetime", "timestamp", "timestamptz",
    "timestamp with time zone", "timestamp without time zone",
    "time", "timetz",
}


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


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

def get_pk_column(schema: Dict[str, Any], table: str) -> Optional[str]:
    """Return the primary key column name for a table, or None."""
    for t in schema.get("tables", []):
        if t["name"].lower() == table.lower():
            for col in t.get("columns", []):
                if col.get("is_primary_key"):
                    return col["name"]
    return None


def get_numeric_columns(schema: Dict[str, Any], table: str) -> List[str]:
    """Return column names with numeric SQL types for a table."""
    for t in schema.get("tables", []):
        if t["name"].lower() == table.lower():
            return [
                col["name"]
                for col in t.get("columns", [])
                if any(nt in col.get("type", "").lower() for nt in NUMERIC_TYPES)
            ]
    return []


def get_date_columns(schema: Dict[str, Any], table: str) -> List[str]:
    """Return column names with date/timestamp SQL types for a table."""
    for t in schema.get("tables", []):
        if t["name"].lower() == table.lower():
            return [
                col["name"]
                for col in t.get("columns", [])
                if any(dt in col.get("type", "").lower() for dt in DATE_TYPES)
            ]
    return []


def get_entity_tables(schema: Dict[str, Any]) -> List[str]:
    """Return tables that have a PK and are referenced by at least one FK."""
    referenced = {fk["target_table"] for fk in schema.get("foreign_keys", [])}
    result = []
    for t in schema.get("tables", []):
        if t["name"] in referenced and get_pk_column(schema, t["name"]) is not None:
            result.append(t["name"])
    return result


def _fk_cols_for_table(schema: Dict[str, Any], table: str) -> set:
    """Return the set of FK source columns in a table (surrogate keys — not metrics)."""
    return {
        fk["source_column"]
        for fk in schema.get("foreign_keys", [])
        if fk["source_table"].lower() == table.lower()
    }


def infer_dimensions(schema: Dict[str, Any], entity_table: str) -> List[Dict[str, Any]]:
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
    pk = get_pk_column(schema, entity_table)
    if not pk:
        return []

    dims = []

    for fk in schema.get("foreign_keys", []):
        if fk["target_table"].lower() != entity_table.lower():
            continue

        child = fk["source_table"]
        fk_col = fk["source_column"]
        child_pk = get_pk_column(schema, child)

        # Exclude PK and FK columns from metrics — they are identifiers, not measures
        child_excluded = (_fk_cols_for_table(schema, child) | ({child_pk} if child_pk else set()))
        numeric_cols = [
            col for col in get_numeric_columns(schema, child)
            if col not in child_excluded
        ]

        # COUNT is always meaningful
        dims.append({
            "expression": f"COUNT(c.{fk_col})",
            "alias": f"{child}_count",
            "child_table": child,
            "fk_col": fk_col,
        })

        for col in numeric_cols[:3]:
            dims.append({
                "expression": f"SUM(c.{col})",
                "alias": f"{child}_{col}_sum",
                "child_table": child,
                "fk_col": fk_col,
            })
            dims.append({
                "expression": f"AVG(c.{col})",
                "alias": f"{child}_{col}_avg",
                "child_table": child,
                "fk_col": fk_col,
            })

        # 2-hop: when child has no meaningful numeric columns, look for
        # grandchild tables (e.g. customers → orders → order_items)
        if not numeric_cols and child_pk:
            for fk2 in schema.get("foreign_keys", []):
                if fk2["target_table"].lower() != child.lower():
                    continue
                grandchild = fk2["source_table"]
                if grandchild.lower() == entity_table.lower():
                    continue  # skip back-reference

                gc_fk_col = fk2["source_column"]
                gc_alias = f"gc_{grandchild}"
                gc_pk = get_pk_column(schema, grandchild)
                gc_excluded = (
                    _fk_cols_for_table(schema, grandchild)
                    | ({gc_pk} if gc_pk else set())
                )
                gc_numeric = [
                    col for col in get_numeric_columns(schema, grandchild)
                    if col not in gc_excluded
                ]

                for col in gc_numeric[:3]:
                    dims.append({
                        "expression": f"SUM({gc_alias}.{col})",
                        "alias": f"{grandchild}_{col}_sum",
                        "child_table": child,
                        "fk_col": fk_col,
                        "grandchild_table": grandchild,
                        "grandchild_fk_col": gc_fk_col,
                        "grandchild_alias": gc_alias,
                        "child_pk": child_pk,
                    })
                    dims.append({
                        "expression": f"AVG({gc_alias}.{col})",
                        "alias": f"{grandchild}_{col}_avg",
                        "child_table": child,
                        "fk_col": fk_col,
                        "grandchild_table": grandchild,
                        "grandchild_fk_col": gc_fk_col,
                        "grandchild_alias": gc_alias,
                        "child_pk": child_pk,
                    })

    return dims


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

async def safe_query(
    plugin: "BaseDatabasePlugin",
    sql: str,
    max_rows: int = 10_000,
) -> List[Dict[str, Any]]:
    """Inject LIMIT if missing, then execute via plugin.query()."""
    normalized = sql.strip().rstrip(";")
    if re.search(r"\bLIMIT\b", normalized, re.IGNORECASE) is None:
        normalized = f"{normalized} LIMIT {max_rows}"
    return await plugin.query(normalized)


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
        return f"`{name}`"
    return f'"{name}"'
