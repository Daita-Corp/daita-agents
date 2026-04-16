"""Memorystore (Redis) normalizer."""

from typing import Any, Dict


def normalize_memorystore(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Memorystore discover output.

    Redis is schemaless — we expose the instance-level metadata and leave
    ``tables`` empty, matching how other key-value stores are represented.
    """
    instance = raw.get("instance", "")

    return {
        "database_type": "memorystore",
        "database_name": instance,
        "tables": [],
        "foreign_keys": [],
        "table_count": 0,
        "metadata": {
            "project": raw.get("project", ""),
            "location": raw.get("location", ""),
            "resource_name": raw.get("resource_name", ""),
            "host": raw.get("host", ""),
            "port": raw.get("port", 0),
            "tier": raw.get("tier", ""),
            "redis_version": raw.get("redis_version", ""),
            "memory_size_gb": raw.get("memory_size_gb", 0),
            "state": raw.get("state", ""),
        },
    }
