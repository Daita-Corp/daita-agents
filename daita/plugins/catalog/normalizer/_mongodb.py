"""MongoDB normalizer."""

from typing import Any, Dict


def normalize_mongodb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin MongoDB discovery output."""
    tables = []
    for coll in raw.get("collections", []):
        cols = [
            {
                "name": f["field_name"],
                "type": f["types"][0] if f.get("types") else "mixed",
                "nullable": True,
                "is_primary_key": f["field_name"] == "_id",
            }
            for f in coll.get("fields", [])
        ]
        tables.append(
            {
                "name": coll["collection_name"],
                "row_count": coll.get("document_count"),
                "columns": cols,
            }
        )

    return {
        "database_type": "mongodb",
        "database_name": raw.get("database", ""),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
    }
