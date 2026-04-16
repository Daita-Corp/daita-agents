"""Pub/Sub normalizers (topic + subscription)."""

from typing import Any, Dict, List


def normalize_pubsub_topic(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Pub/Sub topic discover output.

    Subscriptions become columns — this captures the routing topology in the
    same way SNS subscriptions do.
    """
    topic = raw.get("topic", "")

    columns: List[Dict[str, Any]] = [
        {
            "name": "message_id",
            "type": "string",
            "nullable": False,
            "is_primary_key": True,
        },
        {"name": "data", "type": "bytes", "nullable": False, "is_primary_key": False},
        {
            "name": "attributes",
            "type": "map",
            "nullable": True,
            "is_primary_key": False,
        },
        {
            "name": "publish_time",
            "type": "timestamp",
            "nullable": False,
            "is_primary_key": False,
        },
    ]
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
        "database_type": "pubsub_topic",
        "database_name": topic,
        "tables": [{"name": topic, "row_count": None, "columns": columns}],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "project": raw.get("project", ""),
            "resource_name": raw.get("resource_name", ""),
            "kms_key": raw.get("kms_key", ""),
            "message_retention": raw.get("message_retention", ""),
        },
    }


def normalize_pubsub_subscription(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Pub/Sub subscription discover output."""
    subscription = raw.get("subscription", "")

    return {
        "database_type": "pubsub_subscription",
        "database_name": subscription,
        "tables": [
            {
                "name": subscription,
                "row_count": None,
                "columns": [
                    {
                        "name": "topic",
                        "type": "string",
                        "nullable": False,
                        "is_primary_key": False,
                        "column_comment": raw.get("topic", ""),
                    }
                ],
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "project": raw.get("project", ""),
            "resource_name": raw.get("resource_name", ""),
            "topic": raw.get("topic", ""),
            "ack_deadline_seconds": raw.get("ack_deadline_seconds", 0),
            "push_endpoint": raw.get("push_endpoint", ""),
        },
    }
