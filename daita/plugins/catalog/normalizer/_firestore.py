"""Firestore normalizer."""

from typing import Any, Dict, List

from ._common import build_store_metadata


def _normalize_firestore_indexes(
    raw_indexes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Translate Firestore index dicts into the canonical NormalizedIndex shape.

    Firestore's composite indexes map to ``type="composite"`` with the ordered
    field paths as ``columns``. Per-field directionality (ASCENDING /
    DESCENDING / ARRAY_CONFIG) is preserved in ``metadata.fields`` so nothing
    is lost for agents that need it.
    """
    out: List[Dict[str, Any]] = []
    for idx in raw_indexes:
        columns = [
            f["field_path"] for f in idx.get("fields", []) if f.get("field_path")
        ]
        out.append(
            {
                "name": idx.get("name", ""),
                "type": "composite",
                "columns": columns,
                "unique": False,
                "metadata": {
                    "query_scope": idx.get("query_scope", ""),
                    "state": idx.get("state", ""),
                    "fields": idx.get("fields", []),
                },
            }
        )
    return out


def normalize_firestore(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Firestore discover output.

    Collections map to tables; sampled fields map to columns. Firestore's
    implicit document ID is exposed as the primary key column ``__name__``.
    Declared composite indexes surface on each table via ``indexes``.
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
                "indexes": _normalize_firestore_indexes(coll.get("indexes", [])),
            }
        )

    return {
        "database_type": "firestore",
        "database_name": raw.get("database", "(default)"),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
        "metadata": build_store_metadata(raw),
    }
