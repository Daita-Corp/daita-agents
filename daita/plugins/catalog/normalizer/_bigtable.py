"""Bigtable normalizer."""

from typing import Any, Dict

from ._common import build_store_metadata


def normalize_bigtable(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Bigtable discover output.

    Bigtable's schema is the rowkey plus one column per column family.
    Real cell-level columns are not queryable without reading rows.
    """
    tables = []
    for t in raw.get("tables", []):
        cols = [
            {
                "name": "row_key",
                "type": "bytes",
                "nullable": False,
                "is_primary_key": True,
            }
        ]
        for family in t.get("column_families", []):
            cols.append(
                {
                    "name": family,
                    "type": "column_family",
                    "nullable": True,
                    "is_primary_key": False,
                }
            )
        tables.append(
            {
                "name": t["table_name"],
                "row_count": t.get("row_count"),
                "columns": cols,
            }
        )

    return {
        "database_type": "bigtable",
        "database_name": raw.get("instance", ""),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
        "metadata": build_store_metadata(raw),
    }
