"""GCS normalizer."""

from typing import Any, Dict


def normalize_gcs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize GCS discover output.

    Mirrors ``normalize_s3``: every GCS object has the same fixed shape
    (name, size, updated, storage_class), so that is the schema. Bucket-level
    analytics (prefixes, content types) go into metadata.
    """
    bucket = raw.get("bucket", "")

    columns = [
        {"name": "name", "type": "string", "nullable": False, "is_primary_key": True},
        {
            "name": "size_bytes",
            "type": "number",
            "nullable": False,
            "is_primary_key": False,
        },
        {
            "name": "updated",
            "type": "timestamp",
            "nullable": False,
            "is_primary_key": False,
        },
        {
            "name": "storage_class",
            "type": "string",
            "nullable": True,
            "is_primary_key": False,
        },
    ]

    return {
        "database_type": "gcs",
        "database_name": bucket,
        "tables": [
            {
                "name": bucket,
                "row_count": raw.get("object_count"),
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "project": raw.get("project", ""),
            "location": raw.get("location", ""),
            "storage_class": raw.get("storage_class", ""),
            "prefixes": raw.get("prefixes", {}),
            "content_types": raw.get("content_types", {}),
            "total_size_bytes": raw.get("total_size_bytes", 0),
            "versioning": raw.get("versioning", "Disabled"),
        },
    }
