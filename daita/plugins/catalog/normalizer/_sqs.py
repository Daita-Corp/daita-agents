"""SQS normalizer."""

from typing import Any, Dict, List


def normalize_sqs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize SQS discover output.

    The "schema" of an SQS queue combines standard message fields with
    message attributes and JSON body keys inferred from sampling.
    """
    queue_name = raw.get("queue_name", "")

    columns: List[Dict[str, Any]] = [
        {
            "name": "MessageId",
            "type": "string",
            "nullable": False,
            "is_primary_key": True,
        },
        {"name": "Body", "type": "string", "nullable": False, "is_primary_key": False},
        {
            "name": "MD5OfBody",
            "type": "string",
            "nullable": False,
            "is_primary_key": False,
        },
        {
            "name": "ReceiptHandle",
            "type": "string",
            "nullable": False,
            "is_primary_key": False,
        },
    ]

    for attr_name, types in raw.get("message_attributes", {}).items():
        dtype = types[0] if types else "String"
        columns.append(
            {
                "name": f"attr:{attr_name}",
                "type": dtype.lower(),
                "nullable": True,
                "is_primary_key": False,
                "column_comment": "message attribute",
            }
        )

    for key, dtype in raw.get("body_keys", {}).items():
        columns.append(
            {
                "name": f"body.{key}",
                "type": dtype,
                "nullable": True,
                "is_primary_key": False,
                "column_comment": "inferred from JSON body sample",
            }
        )

    return {
        "database_type": "sqs",
        "database_name": queue_name,
        "tables": [
            {
                "name": queue_name,
                "row_count": raw.get("approximate_message_count"),
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "is_fifo": raw.get("is_fifo", False),
            "visibility_timeout": raw.get("visibility_timeout", 30),
            "retention_seconds": raw.get("retention_seconds", 345600),
        },
    }
