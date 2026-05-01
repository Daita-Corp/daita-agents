"""Azure Cosmos DB normalizer."""

from typing import Any, Dict, List


def normalize_azure_cosmosdb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Cosmos DB SQL API databases and containers."""
    tables: List[Dict[str, Any]] = []
    for database in raw.get("databases", []):
        db_name = database.get("name", "")
        for container in database.get("containers", []):
            container_name = container.get("name", "")
            partition_paths = container.get("partition_key_paths", [])
            columns = [
                {
                    "name": "id",
                    "type": "string",
                    "nullable": False,
                    "is_primary_key": True,
                },
                {
                    "name": "_partition_key",
                    "type": "string",
                    "nullable": not bool(partition_paths),
                    "is_primary_key": False,
                    "column_comment": ",".join(partition_paths),
                },
                {
                    "name": "_ts",
                    "type": "integer",
                    "nullable": True,
                    "is_primary_key": False,
                },
                {
                    "name": "_etag",
                    "type": "string",
                    "nullable": True,
                    "is_primary_key": False,
                },
            ]
            tables.append(
                {
                    "name": f"{db_name}.{container_name}",
                    "row_count": None,
                    "columns": columns,
                    "metadata": {
                        "database": db_name,
                        "container": container_name,
                        "partition_key_paths": partition_paths,
                        "partition_key_kind": container.get("partition_key_kind", ""),
                        "indexing_mode": container.get("indexing_mode", ""),
                        "default_ttl": container.get("default_ttl"),
                    },
                }
            )

    return {
        "database_type": "cosmosdb",
        "database_name": raw.get("account", ""),
        "tables": tables,
        "foreign_keys": [],
        "table_count": len(tables),
        "metadata": {
            "subscription_id": raw.get("subscription_id", ""),
            "resource_group": raw.get("resource_group", ""),
            "database_count": len(raw.get("databases", [])),
        },
    }
