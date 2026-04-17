"""PostgreSQL normalizer."""

import re
from typing import Any, Dict, List

from ._common import _normalize_relational


# Match the column list of a CREATE INDEX statement. Postgres appends optional
# `WHERE <predicate>` at the end for partial indexes, so grab the last fully
# parenthesized group that isn't followed by another open paren.
_INDEX_COL_LIST = re.compile(r"\(([^()]+)\)\s*(?:WHERE\b.*)?$", re.IGNORECASE | re.DOTALL)
_INDEX_USING = re.compile(r"\bUSING\s+(\w+)", re.IGNORECASE)


def _parse_pg_index(raw_idx: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a row from ``pg_indexes`` into the normalized index shape.

    Raw rows look like::

        {
            "tablename": "orders",
            "indexname": "orders_customer_idx",
            "indexdef":  "CREATE INDEX orders_customer_idx ON public.orders "
                         "USING btree (customer_id)"
        }

    We extract the index type, uniqueness, and column list from ``indexdef``
    because ``pg_indexes`` doesn't expose those fields in separate columns.
    """
    name = raw_idx.get("indexname", "")
    defn = raw_idx.get("indexdef", "") or ""

    unique = bool(re.search(r"\bCREATE\s+UNIQUE\s+INDEX\b", defn, re.IGNORECASE))

    type_match = _INDEX_USING.search(defn)
    idx_type = type_match.group(1).lower() if type_match else ""

    columns: List[str] = []
    col_match = _INDEX_COL_LIST.search(defn)
    if col_match:
        for piece in col_match.group(1).split(","):
            piece = piece.strip()
            if not piece:
                continue
            # Strip trailing ``ASC``/``DESC``/``NULLS FIRST``/``NULLS LAST``.
            # Expression indexes (``(lower(email))``) are kept verbatim —
            # agents can still reason about "column-like" expressions.
            col = re.split(r"\s+", piece, maxsplit=1)[0].strip('"')
            if col:
                columns.append(col)

    return {"name": name, "type": idx_type, "unique": unique, "columns": columns}


def normalize_postgresql(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin PostgreSQL discovery output.

    Also preserves declared indexes (dropped by the previous shared helper)
    so the graph persister can emit Index nodes + ``INDEXED_BY`` / ``COVERS``
    edges.
    """
    pk_set = {
        (pk["table_name"], pk["column_name"]) for pk in raw.get("primary_keys", [])
    }

    # Group indexes by table before attaching — one pass, O(indexes).
    idx_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for idx in raw.get("indexes", []):
        tname = idx.get("tablename")
        if not tname:
            continue
        idx_by_table.setdefault(tname, []).append(_parse_pg_index(idx))

    result = _normalize_relational(
        raw,
        "postgresql",
        is_primary_key_fn=lambda tname, c: (tname, c["column_name"]) in pk_set,
        default_schema="public",
    )
    for t in result.get("tables", []):
        t["indexes"] = idx_by_table.get(t["name"], [])
    return result
