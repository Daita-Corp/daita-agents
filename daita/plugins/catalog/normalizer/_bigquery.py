"""BigQuery normalizer."""

from typing import Any, Dict, List

from ._common import build_store_metadata


def normalize_bigquery(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize BigQuery discover output.

    BigQuery schemas are relational — columns are joined onto tables the same
    way as Postgres/MySQL. ``mode == 'REQUIRED'`` means non-nullable.
    Primary- and foreign-key constraints are declared as ``NOT ENFORCED`` in
    BigQuery and surfaced via ``INFORMATION_SCHEMA`` (populated by discovery).
    """
    col_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for col in raw.get("columns", []):
        col_by_table.setdefault(col["table_name"], []).append(col)

    pk_set = {
        (pk["table_name"], pk["column_name"])
        for pk in raw.get("primary_keys", [])
    }

    tables = []
    for t in raw.get("tables", []):
        tname = t["table_name"]
        cols = []
        for c in col_by_table.get(tname, []):
            col: Dict[str, Any] = {
                "name": c["column_name"],
                "type": c.get("data_type", ""),
                "nullable": c.get("is_nullable", "YES") == "YES",
                "is_primary_key": (tname, c["column_name"]) in pk_set,
            }
            if c.get("column_comment"):
                col["column_comment"] = c["column_comment"]
            cols.append(col)
        tables.append({"name": tname, "row_count": t.get("row_count"), "columns": cols})

    return {
        "database_type": "bigquery",
        "database_name": raw.get("dataset", ""),
        "tables": tables,
        "foreign_keys": raw.get("foreign_keys", []),
        "table_count": len(tables),
        "metadata": build_store_metadata(
            raw,
            extra={
                "location": raw.get("location"),
                "description": raw.get("description"),
            },
        ),
    }
