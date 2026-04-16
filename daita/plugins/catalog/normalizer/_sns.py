"""SNS normalizer."""

from typing import Any, Dict, List


def normalize_sns(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize SNS discover output.

    The "schema" is the topic itself with subscriptions as columns,
    capturing the routing topology.
    """
    topic_name = raw.get("topic_name", "")

    columns: List[Dict[str, Any]] = [
        {
            "name": "TopicArn",
            "type": "string",
            "nullable": False,
            "is_primary_key": True,
        },
        {
            "name": "Message",
            "type": "string",
            "nullable": False,
            "is_primary_key": False,
        },
        {
            "name": "Subject",
            "type": "string",
            "nullable": True,
            "is_primary_key": False,
        },
        {
            "name": "MessageAttributes",
            "type": "map",
            "nullable": True,
            "is_primary_key": False,
        },
    ]

    for sub in raw.get("subscriptions", []):
        protocol = sub.get("protocol", "unknown")
        endpoint = sub.get("endpoint", "")
        columns.append(
            {
                "name": f"sub:{protocol}:{endpoint}",
                "type": "subscription",
                "nullable": True,
                "is_primary_key": False,
                "column_comment": f"protocol={protocol}",
            }
        )

    return {
        "database_type": "sns",
        "database_name": topic_name,
        "tables": [
            {
                "name": topic_name,
                "row_count": raw.get("subscription_count"),
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "is_fifo": raw.get("is_fifo", False),
            "display_name": raw.get("display_name", ""),
        },
    }
