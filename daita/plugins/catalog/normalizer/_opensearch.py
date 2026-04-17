"""OpenSearch normalizer."""

from typing import Any, Dict

from ._common import build_store_metadata


def normalize_opensearch(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize OpenSearch discover output.

    Each index becomes a table, field mappings become columns.
    """
    host = raw.get("host", "")
    cluster_name = raw.get("cluster_name", host)

    tables = []
    for idx in raw.get("indices", []):
        cols = [
            {
                "name": field["field_name"],
                "type": field.get("type", "object"),
                "nullable": True,
                "is_primary_key": field["field_name"] == "_id",
            }
            for field in idx.get("fields", [])
        ]
        tables.append(
            {
                "name": idx["index_name"],
                "row_count": idx.get("doc_count"),
                "columns": cols,
            }
        )

    return {
        "database_type": "opensearch",
        "database_name": cluster_name,
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
        "metadata": build_store_metadata(raw, extra={"version": raw.get("version")}),
    }
