"""S3 normalizer."""

from typing import Any, Dict


def normalize_s3(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize S3 discover output.

    The "schema" of an S3 bucket is the fixed set of fields every object has:
    key, size, last_modified, storage_class. Bucket-level analytics (prefix
    breakdown, content type distribution) go into metadata — they describe the
    bucket's contents, not a structural schema.
    """
    bucket = raw.get("bucket", "")

    columns = [
        {"name": "key", "type": "string", "nullable": False, "is_primary_key": True},
        {
            "name": "size_bytes",
            "type": "number",
            "nullable": False,
            "is_primary_key": False,
        },
        {
            "name": "last_modified",
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
        "database_type": "s3",
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
            "prefixes": raw.get("prefixes", {}),
            "content_types": raw.get("content_types", {}),
            "total_size_bytes": raw.get("total_size_bytes", 0),
            "versioning": raw.get("versioning", "Disabled"),
        },
    }
