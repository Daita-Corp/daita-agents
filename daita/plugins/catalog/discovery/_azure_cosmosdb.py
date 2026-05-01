"""Azure Cosmos DB discovery."""

import logging
from typing import Any, Optional

from ._azure_common import azure_credential

logger = logging.getLogger(__name__)


async def discover_azure_cosmosdb(
    subscription_id: str,
    resource_group: str,
    account: str,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a Cosmos DB account's SQL databases and containers."""
    try:
        from azure.mgmt.cosmosdb import CosmosDBManagementClient
    except ImportError:
        raise ImportError(
            "azure-mgmt-cosmosdb is required. "
            "Install with: pip install 'daita-agents[azure]'"
        )

    client = CosmosDBManagementClient(azure_credential(tenant_id), subscription_id)

    databases: list[dict[str, Any]] = []
    try:
        dbs = list(client.sql_resources.list_sql_databases(resource_group, account))
    except Exception as exc:
        logger.warning("Cosmos SQL database list failed for %s: %s", account, exc)
        dbs = []

    for db in dbs:
        db_name = getattr(db, "name", "")
        db_resource = getattr(db, "resource", None)
        containers: list[dict[str, Any]] = []
        try:
            raw_containers = list(
                client.sql_resources.list_sql_containers(
                    resource_group, account, db_name
                )
            )
        except Exception as exc:
            logger.warning(
                "Cosmos container list failed for %s/%s: %s", account, db_name, exc
            )
            raw_containers = []

        for container in raw_containers:
            resource = getattr(container, "resource", None)
            partition_key = getattr(resource, "partition_key", None)
            indexing_policy = getattr(resource, "indexing_policy", None)
            containers.append(
                {
                    "name": getattr(container, "name", ""),
                    "id": getattr(container, "id", ""),
                    "partition_key_paths": list(
                        getattr(partition_key, "paths", None) or []
                    ),
                    "partition_key_kind": str(getattr(partition_key, "kind", "") or ""),
                    "indexing_mode": str(
                        getattr(indexing_policy, "indexing_mode", "") or ""
                    ),
                    "default_ttl": getattr(resource, "default_ttl", None),
                }
            )

        databases.append(
            {
                "name": db_name,
                "id": getattr(db, "id", ""),
                "resource_id": getattr(db_resource, "id", "") or db_name,
                "containers": containers,
            }
        )

    return {
        "database_type": "cosmosdb",
        "account": account,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "databases": databases,
    }
