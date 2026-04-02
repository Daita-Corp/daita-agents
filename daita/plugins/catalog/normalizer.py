"""
Schema normalization functions.

Converts raw discover_* output into a uniform normalized shape consumed by
agents and the schema dispatch layer in daita.agents.db.schema.

Also provides store-level deduplication for infrastructure discovery.
"""

import re
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_discoverer import DiscoveredStore


def normalize_postgresql(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin PostgreSQL discovery output."""
    col_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for col in raw.get("columns", []):
        col_by_table.setdefault(col["table_name"], []).append(col)

    pk_set = {
        (pk["table_name"], pk["column_name"]) for pk in raw.get("primary_keys", [])
    }

    tables = []
    for t in raw.get("tables", []):
        tname = t["table_name"]
        cols = []
        for c in col_by_table.get(tname, []):
            col: Dict[str, Any] = {
                "name": c["column_name"],
                "type": c["data_type"],
                "nullable": c.get("is_nullable", "YES") == "YES",
                "is_primary_key": (tname, c["column_name"]) in pk_set,
            }
            if c.get("column_comment"):
                col["column_comment"] = c["column_comment"]
            cols.append(col)
        tables.append(
            {"name": tname, "row_count": t.get("row_count"), "columns": cols}
        )

    fks = [
        {
            "source_table": fk["source_table"],
            "source_column": fk["source_column"],
            "target_table": fk["target_table"],
            "target_column": fk["target_column"],
        }
        for fk in raw.get("foreign_keys", [])
    ]

    db_name = raw.get("schema", "public")
    return {
        "database_type": "postgresql",
        "database_name": db_name,
        "tables": tables,
        "foreign_keys": fks,
        "table_count": len(tables),
    }


def normalize_mysql(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin MySQL discovery output."""
    col_by_table: Dict[str, List[Dict[str, Any]]] = {}
    for col in raw.get("columns", []):
        col_by_table.setdefault(col["table_name"], []).append(col)

    tables = []
    for t in raw.get("tables", []):
        tname = t["table_name"]
        cols = []
        for c in col_by_table.get(tname, []):
            col: Dict[str, Any] = {
                "name": c["column_name"],
                "type": c.get("data_type", ""),
                "nullable": c.get("is_nullable", "YES") == "YES",
                "is_primary_key": c.get("column_key", "") == "PRI",
            }
            if c.get("column_comment"):
                col["column_comment"] = c["column_comment"]
            cols.append(col)
        tables.append(
            {"name": tname, "row_count": t.get("row_count"), "columns": cols}
        )

    fks = [
        {
            "source_table": fk["source_table"],
            "source_column": fk["source_column"],
            "target_table": fk["target_table"],
            "target_column": fk["target_column"],
        }
        for fk in raw.get("foreign_keys", [])
    ]

    db_name = raw.get("schema", "")
    return {
        "database_type": "mysql",
        "database_name": db_name,
        "tables": tables,
        "foreign_keys": fks,
        "table_count": len(tables),
    }


def normalize_mongodb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize CatalogPlugin MongoDB discovery output."""
    tables = []
    for coll in raw.get("collections", []):
        cols = [
            {
                "name": f["field_name"],
                "type": f["types"][0] if f.get("types") else "mixed",
                "nullable": True,
                "is_primary_key": f["field_name"] == "_id",
            }
            for f in coll.get("fields", [])
        ]
        tables.append(
            {
                "name": coll["collection_name"],
                "row_count": coll.get("document_count"),
                "columns": cols,
            }
        )

    return {
        "database_type": "mongodb",
        "database_name": raw.get("database", ""),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
    }


def normalize_discovery(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ``discover_*`` output to the normalized schema shape.

    Dispatches by ``raw["database_type"]``; returns *raw* unchanged for
    unrecognized types.

    Normalized shape::

        {
            "database_type": str,
            "database_name": str,
            "tables": [
                {
                    "name": str,
                    "row_count": int | None,
                    "columns": [
                        {"name": str, "type": str, "nullable": bool,
                         "is_primary_key": bool}
                    ],
                }
            ],
            "foreign_keys": [
                {"source_table": str, "source_column": str,
                 "target_table": str, "target_column": str}
            ],
            "table_count": int,
        }
    """
    db_type = raw.get("database_type", "unknown")
    if db_type == "postgresql":
        return normalize_postgresql(raw)
    if db_type == "mysql":
        return normalize_mysql(raw)
    if db_type == "mongodb":
        return normalize_mongodb(raw)
    if db_type == "dynamodb":
        return normalize_dynamodb(raw)
    if db_type == "s3":
        return normalize_s3(raw)
    if db_type == "apigateway":
        return normalize_apigateway(raw)
    return raw  # passthrough for unrecognized types


_DYNAMODB_TYPE_MAP = {
    "S": "string",
    "N": "number",
    "B": "binary",
    "SS": "string_set",
    "NS": "number_set",
    "BS": "binary_set",
    "L": "list",
    "M": "map",
    "BOOL": "boolean",
    "NULL": "null",
}


def normalize_dynamodb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DynamoDB discover output."""
    table_name = raw.get("table_name", "")
    key_schema = raw.get("key_schema", [])
    attribute_defs = raw.get("attribute_definitions", [])
    sampled = raw.get("sampled_attributes", {})

    # Build set of key attribute names for primary key detection
    key_names = {k["AttributeName"] for k in key_schema}

    # Build type map from attribute definitions
    defined_types = {
        a["AttributeName"]: _DYNAMODB_TYPE_MAP.get(a["AttributeType"], a["AttributeType"])
        for a in attribute_defs
    }

    # Start with defined attributes (key schema + attribute definitions)
    seen = set()
    columns = []

    # Key schema attributes first (marked as primary keys)
    for key in key_schema:
        name = key["AttributeName"]
        seen.add(name)
        columns.append({
            "name": name,
            "type": defined_types.get(name, "string"),
            "nullable": False,
            "is_primary_key": True,
            "column_comment": f"{'Partition' if key['KeyType'] == 'HASH' else 'Sort'} key",
        })

    # Remaining defined attributes
    for attr in attribute_defs:
        name = attr["AttributeName"]
        if name in seen:
            continue
        seen.add(name)
        columns.append({
            "name": name,
            "type": _DYNAMODB_TYPE_MAP.get(attr["AttributeType"], attr["AttributeType"]),
            "nullable": True,
            "is_primary_key": False,
        })

    # Sampled attributes not in definitions
    for attr_name, types in sampled.items():
        if attr_name in seen:
            continue
        seen.add(attr_name)
        # Pick the most common type, or first
        dtype = types[0] if types else "S"
        columns.append({
            "name": attr_name,
            "type": _DYNAMODB_TYPE_MAP.get(dtype, dtype),
            "nullable": True,
            "is_primary_key": False,
            "column_comment": "inferred from sample",
        })

    tables = [{
        "name": table_name,
        "row_count": raw.get("item_count"),
        "columns": columns,
    }]

    return {
        "database_type": "dynamodb",
        "database_name": table_name,
        "tables": tables,
        "foreign_keys": [],
        "table_count": 1,
    }


def normalize_s3(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize S3 discover output.

    The "schema" of an S3 bucket is the fixed set of fields every object has:
    key, size, last_modified, storage_class. Bucket-level analytics (prefix
    breakdown, content type distribution) go into metadata — they describe the
    bucket's contents, not a structural schema.
    """
    bucket = raw.get("bucket", "")

    # These fields exist on every S3 object — this is the real schema
    columns = [
        {"name": "key", "type": "string", "nullable": False, "is_primary_key": True},
        {"name": "size_bytes", "type": "number", "nullable": False, "is_primary_key": False},
        {"name": "last_modified", "type": "timestamp", "nullable": False, "is_primary_key": False},
        {"name": "storage_class", "type": "string", "nullable": True, "is_primary_key": False},
    ]

    tables = [{
        "name": bucket,
        "row_count": raw.get("object_count"),
        "columns": columns,
    }]

    return {
        "database_type": "s3",
        "database_name": bucket,
        "tables": tables,
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "prefixes": raw.get("prefixes", {}),
            "content_types": raw.get("content_types", {}),
            "total_size_bytes": raw.get("total_size_bytes", 0),
            "versioning": raw.get("versioning", "Disabled"),
        },
    }


def normalize_apigateway(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize API Gateway discover output.

    Each endpoint becomes a column. The column comment captures the
    authorization type and integration target (Lambda ARN, HTTP URL) —
    this is the key data for lineage edge inference.
    """
    api_name = raw.get("api_name", "")
    endpoints = raw.get("endpoints", [])

    columns = []
    integrations: Dict[str, Dict[str, str]] = {}
    for ep in endpoints:
        method = ep.get("method", "")
        path = ep.get("path", "/")
        col_name = f"{method} {path}"

        # Build column comment: authorization + integration target
        auth = ep.get("authorization", "NONE")
        integration_hint = ""
        if ep.get("integration_uri"):
            integration_hint = f" -> {ep['integration_uri']}"

        columns.append({
            "name": col_name,
            "type": "endpoint",
            "nullable": False,
            "is_primary_key": False,
            "column_comment": f"{auth}{integration_hint}",
        })

        # Collect full integration map for metadata
        if ep.get("integration_type") or ep.get("integration_uri"):
            integrations[col_name] = {
                "type": ep.get("integration_type", ""),
                "uri": ep.get("integration_uri", ""),
            }

    tables = [{
        "name": api_name,
        "row_count": len(endpoints),
        "columns": columns,
    }]

    return {
        "database_type": "apigateway",
        "database_name": api_name,
        "tables": tables,
        "foreign_keys": [],
        "table_count": 1,
        "metadata": {
            "endpoint": raw.get("endpoint", ""),
            "stage": raw.get("stage", ""),
            "region": raw.get("region", ""),
            "protocol_type": raw.get("protocol_type", ""),
            "authorizers": raw.get("authorizers", []),
            "integrations": integrations,
        },
    }


# ---------------------------------------------------------------------------
# Store-level deduplication
# ---------------------------------------------------------------------------

_ENV_PATTERNS = {
    "production": re.compile(r"(^|[\s_\-\.])prod(uction)?([\s_\-\.]|$)", re.IGNORECASE),
    "staging": re.compile(r"(^|[\s_\-\.])stag(ing|e)?([\s_\-\.]|$)", re.IGNORECASE),
    "development": re.compile(r"(^|[\s_\-\.])(dev(elopment)?|local)([\s_\-\.]|$)", re.IGNORECASE),
    "test": re.compile(r"(^|[\s_\-\.])test(ing)?([\s_\-\.]|$)", re.IGNORECASE),
}


def deduplicate_stores(stores: List["DiscoveredStore"]) -> List["DiscoveredStore"]:
    """Merge stores with matching fingerprints. Higher confidence wins for display fields."""
    seen: Dict[str, "DiscoveredStore"] = {}
    for store in stores:
        if store.id in seen:
            seen[store.id] = merge_store_sources(seen[store.id], store)
        else:
            seen[store.id] = store
    return list(seen.values())


def merge_store_sources(
    existing: "DiscoveredStore", new: "DiscoveredStore"
) -> "DiscoveredStore":
    """Merge two records for the same store from different discoverers."""
    # Higher confidence source wins for display fields
    winner = new if new.confidence > existing.confidence else existing
    loser = existing if winner is new else new

    # Track all sources that have seen this store
    seen_by = winner.metadata.get("seen_by", [winner.source])
    if loser.source not in seen_by:
        seen_by.append(loser.source)

    # Merge tags (union)
    merged_tags = list(set(winner.tags + loser.tags))

    # Use winner's metadata as base, merge loser's non-conflicting keys
    merged_metadata = {**loser.metadata, **winner.metadata}
    merged_metadata["seen_by"] = seen_by

    winner.tags = merged_tags
    winner.metadata = merged_metadata
    winner.last_seen = new.last_seen or existing.last_seen

    # Prefer non-None environment
    if not winner.environment and loser.environment:
        winner.environment = loser.environment

    return winner


def infer_environment(store: "DiscoveredStore") -> str:
    """Infer prod/staging/dev from name patterns, tags, cloud metadata."""
    # Check display name
    search_text = f"{store.display_name} {' '.join(store.tags)}"

    for env, pattern in _ENV_PATTERNS.items():
        if pattern.search(search_text):
            return env

    # Check cloud metadata tags
    cloud_env = store.metadata.get("environment") or store.metadata.get("env")
    if cloud_env:
        return cloud_env.lower()

    return "unknown"
