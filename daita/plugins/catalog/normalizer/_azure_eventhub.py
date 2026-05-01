"""Azure Event Hubs normalizer."""

from typing import Any, Dict

_EVENT_ENVELOPE_COLUMNS = [
    {
        "name": "sequence_number",
        "type": "integer",
        "nullable": False,
        "is_primary_key": True,
    },
    {"name": "body", "type": "bytes", "nullable": False, "is_primary_key": False},
    {"name": "properties", "type": "map", "nullable": True, "is_primary_key": False},
    {
        "name": "enqueued_time",
        "type": "timestamp",
        "nullable": True,
        "is_primary_key": False,
    },
    {
        "name": "partition_id",
        "type": "string",
        "nullable": False,
        "is_primary_key": False,
    },
]


def normalize_azure_eventhub(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Event Hub discovery output."""
    eventhub = raw.get("eventhub", "")
    columns = list(_EVENT_ENVELOPE_COLUMNS)
    for group in raw.get("consumer_groups", []):
        columns.append(
            {
                "name": f"consumer_group:{group}",
                "type": "consumer_group",
                "nullable": True,
                "is_primary_key": False,
            }
        )

    return {
        "database_type": "eventhub",
        "database_name": eventhub,
        "tables": [{"name": eventhub, "row_count": None, "columns": columns}],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "namespace": raw.get("namespace", ""),
            "resource_name": raw.get("resource_name", ""),
            "partition_count": raw.get("partition_count"),
            "message_retention_days": raw.get("message_retention_days"),
            "status": raw.get("status", ""),
        },
    }
