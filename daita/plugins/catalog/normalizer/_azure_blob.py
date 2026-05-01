"""Azure Blob Storage normalizer."""

from typing import Any, Dict


def normalize_azure_blob(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Azure Blob container discovery output."""
    container = raw.get("container", "")

    return {
        "database_type": "azure_blob",
        "database_name": container,
        "tables": [
            {
                "name": container,
                "row_count": raw.get("object_count"),
                "columns": [
                    {
                        "name": "name",
                        "type": "string",
                        "nullable": False,
                        "is_primary_key": True,
                    },
                    {
                        "name": "size_bytes",
                        "type": "number",
                        "nullable": False,
                        "is_primary_key": False,
                    },
                    {
                        "name": "last_modified",
                        "type": "timestamp",
                        "nullable": True,
                        "is_primary_key": False,
                    },
                    {
                        "name": "content_type",
                        "type": "string",
                        "nullable": True,
                        "is_primary_key": False,
                    },
                ],
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "account": raw.get("account", ""),
            "prefixes": raw.get("prefixes", {}),
            "content_types": raw.get("content_types", {}),
            "total_size_bytes": raw.get("total_size_bytes", 0),
            "properties": raw.get("properties", {}),
        },
    }
