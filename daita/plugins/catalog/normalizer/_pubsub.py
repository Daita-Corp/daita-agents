"""Pub/Sub normalizers (topic + subscription)."""

import json
from typing import Any, Dict, List, Optional


_DEFAULT_ENVELOPE_COLUMNS: List[Dict[str, Any]] = [
    {"name": "message_id", "type": "string", "nullable": False, "is_primary_key": True},
    {"name": "data", "type": "bytes", "nullable": False, "is_primary_key": False},
    {"name": "attributes", "type": "map", "nullable": True, "is_primary_key": False},
    {
        "name": "publish_time",
        "type": "timestamp",
        "nullable": False,
        "is_primary_key": False,
    },
]


def _columns_from_avro(definition: str) -> Optional[List[Dict[str, Any]]]:
    """Parse an Avro record schema into normalized columns.

    Returns None for non-record shapes or unparseable JSON — callers fall back
    to the generic Pub/Sub message envelope.
    """
    try:
        d = json.loads(definition)
    except Exception:
        return None
    if not isinstance(d, dict) or d.get("type") != "record":
        return None

    cols: List[Dict[str, Any]] = []
    for f in d.get("fields", []):
        field_type = f.get("type", "mixed")
        # Avro nullability is expressed as a union including "null".
        nullable = isinstance(field_type, list) and "null" in field_type
        col: Dict[str, Any] = {
            "name": f["name"],
            "type": str(field_type),
            "nullable": nullable,
            "is_primary_key": False,
        }
        if f.get("doc"):
            col["column_comment"] = f["doc"]
        cols.append(col)
    return cols or None


def normalize_pubsub_topic(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Pub/Sub topic discover output.

    When the topic binds an Avro schema via Pub/Sub Schema Registry, fields
    are parsed from the schema definition and surfaced as columns — agents
    see the real message shape rather than a generic envelope.

    Subscriptions remain represented as ``sub:<name>`` columns to preserve
    routing topology alongside message shape.
    """
    topic = raw.get("topic", "")
    schema = raw.get("schema")

    if schema and schema.get("type") == "AVRO":
        parsed = _columns_from_avro(schema.get("definition", ""))
    else:
        parsed = None

    columns: List[Dict[str, Any]] = list(parsed or _DEFAULT_ENVELOPE_COLUMNS)
    for sub in raw.get("subscriptions", []):
        columns.append(
            {
                "name": f"sub:{sub}",
                "type": "subscription",
                "nullable": True,
                "is_primary_key": False,
            }
        )

    # Non-lossy schema metadata on the table itself — agents that want the
    # raw contract can fetch ``schema_definition`` without re-calling discovery.
    table_metadata: Dict[str, Any] = {}
    if schema:
        table_metadata = {
            "schema_name": schema.get("name", ""),
            "schema_type": schema.get("type", ""),
            "schema_encoding": raw.get("schema_encoding", ""),
            "schema_revision": schema.get("revision_id", ""),
            "schema_definition": schema.get("definition", ""),
        }

    return {
        "database_type": "pubsub_topic",
        "database_name": topic,
        "tables": [
            {
                "name": topic,
                "row_count": None,
                "columns": columns,
                **({"metadata": table_metadata} if table_metadata else {}),
            }
        ],
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
