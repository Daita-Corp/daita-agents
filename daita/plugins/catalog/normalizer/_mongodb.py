"""MongoDB normalizer."""

from typing import Any, Dict

from ._common import build_store_metadata


def normalize_mongodb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin MongoDB discovery output.

    Propagates host/port into ``metadata`` so the persister can derive a
    collision-safe store identifier (``mongodb:<host>/<database>``). Two
    MongoDB deployments hosting the same database name no longer collide
    in the graph.
    """
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

    result: Dict[str, Any] = {
        "database_type": "mongodb",
        "database_name": raw.get("database", ""),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
    }
    metadata = build_store_metadata(raw)
    if metadata:
        result["metadata"] = metadata
    return result
