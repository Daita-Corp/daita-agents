"""Firestore normalizer."""

from typing import Any, Dict


def normalize_firestore(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Firestore discover output.

    Collections map to tables; sampled fields map to columns. Firestore's
    implicit document ID is exposed as the primary key column ``__name__``.
    """
    tables = []
    for coll in raw.get("collections", []):
        cols = [
            {
                "name": "__name__",
                "type": "string",
                "nullable": False,
                "is_primary_key": True,
                "column_comment": "Firestore document ID",
            }
        ]
        for f in coll.get("fields", []):
            cols.append(
                {
                    "name": f["field_name"],
                    "type": f["types"][0] if f.get("types") else "mixed",
                    "nullable": True,
                    "is_primary_key": False,
                }
            )
        tables.append(
            {
                "name": coll["collection_name"],
                "row_count": coll.get("document_count"),
                "columns": cols,
            }
        )

    return {
        "database_type": "firestore",
        "database_name": raw.get("database", "(default)"),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
        "metadata": {"project": raw.get("project", "")},
    }
