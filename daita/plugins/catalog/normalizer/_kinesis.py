"""Kinesis normalizer."""

from typing import Any, Dict, List

from ._common import build_store_metadata


def normalize_kinesis(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Kinesis discover output.

    The stream is a single "table". Standard Kinesis record fields plus
    JSON payload keys inferred from sampling become the columns.
    """
    stream_name = raw.get("stream_name", "")

    columns: List[Dict[str, Any]] = [
        {
            "name": "SequenceNumber",
            "type": "string",
            "nullable": False,
            "is_primary_key": True,
        },
        {
            "name": "PartitionKey",
            "type": "string",
            "nullable": False,
            "is_primary_key": False,
        },
        {"name": "Data", "type": "blob", "nullable": False, "is_primary_key": False},
        {
            "name": "ApproximateArrivalTimestamp",
            "type": "timestamp",
            "nullable": False,
            "is_primary_key": False,
        },
    ]

    for key, dtype in raw.get("record_fields", {}).items():
        columns.append(
            {
                "name": f"data.{key}",
                "type": dtype,
                "nullable": True,
                "is_primary_key": False,
                "column_comment": "inferred from JSON record sample",
            }
        )

    return {
        "database_type": "kinesis",
        "database_name": stream_name,
        "tables": [
            {
                "name": stream_name,
                "row_count": None,  # Kinesis doesn't expose record count cheaply
                "columns": columns,
            }
        ],
        "foreign_keys": [],
        "table_count": 1,
        "metadata": build_store_metadata(
            raw,
            extra={
                "shard_count": raw.get("shard_count", 0),
                "retention_hours": raw.get("retention_hours", 24),
                "stream_mode": raw.get("stream_mode", "PROVISIONED"),
            },
        ),
    }
