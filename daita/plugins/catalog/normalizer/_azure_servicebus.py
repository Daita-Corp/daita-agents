"""Azure Service Bus normalizers."""

from typing import Any, Dict, List

_MESSAGE_COLUMNS: List[Dict[str, Any]] = [
    {"name": "message_id", "type": "string", "nullable": False, "is_primary_key": True},
    {"name": "body", "type": "bytes", "nullable": False, "is_primary_key": False},
    {"name": "properties", "type": "map", "nullable": True, "is_primary_key": False},
    {
        "name": "enqueued_time",
        "type": "timestamp",
        "nullable": True,
        "is_primary_key": False,
    },
]


def normalize_azure_servicebus_queue(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Service Bus queue discovery output."""
    queue = raw.get("queue", "")
    return {
        "database_type": "servicebus_queue",
        "database_name": queue,
        "tables": [
            {
                "name": queue,
                "row_count": raw.get("message_count"),
                "columns": _MESSAGE_COLUMNS,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "namespace": raw.get("namespace", ""),
            "resource_name": raw.get("resource_name", ""),
            "max_size_mb": raw.get("max_size_mb"),
            "dead_lettering_on_message_expiration": raw.get(
                "dead_lettering_on_message_expiration"
            ),
            "requires_duplicate_detection": raw.get("requires_duplicate_detection"),
        },
    }


def normalize_azure_servicebus_topic(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Service Bus topic discovery output."""
    topic = raw.get("topic", "")
    columns = list(_MESSAGE_COLUMNS)
    for sub in raw.get("subscriptions", []):
        columns.append(
            {
                "name": f"sub:{sub}",
                "type": "subscription",
                "nullable": True,
                "is_primary_key": False,
            }
        )

    return {
        "database_type": "servicebus_topic",
        "database_name": topic,
        "tables": [{"name": topic, "row_count": None, "columns": columns}],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "namespace": raw.get("namespace", ""),
            "resource_name": raw.get("resource_name", ""),
            "subscription_count": raw.get("subscription_count"),
            "max_size_mb": raw.get("max_size_mb"),
        },
    }
