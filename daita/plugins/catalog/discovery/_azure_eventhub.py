"""Azure Event Hubs discovery."""

import logging
from typing import Any, Optional

from ._azure_common import azure_credential

logger = logging.getLogger(__name__)


async def discover_azure_eventhub(
    subscription_id: str,
    resource_group: str,
    namespace: str,
    eventhub: str,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """Describe an Event Hub and its consumer groups."""
    try:
        from azure.mgmt.eventhub import EventHubManagementClient
    except ImportError:
        raise ImportError(
            "azure-mgmt-eventhub is required. "
            "Install with: pip install 'daita-agents[azure]'"
        )

    client = EventHubManagementClient(azure_credential(tenant_id), subscription_id)

    hub: Any = None
    try:
        hub = client.event_hubs.get(resource_group, namespace, eventhub)
    except Exception as exc:
        logger.warning("Event Hub get failed for %s/%s: %s", namespace, eventhub, exc)

    try:
        groups = list(
            getattr(client.consumer_groups, "list_by_event_hub")(
                resource_group, namespace, eventhub
            )
        )
    except Exception as exc:
        logger.debug("Event Hub consumer group list failed for %s: %s", eventhub, exc)
        groups = []

    return {
        "database_type": "eventhub",
        "namespace": namespace,
        "eventhub": eventhub,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "resource_name": getattr(hub, "id", "") if hub else "",
        "partition_count": getattr(hub, "partition_count", None) if hub else None,
        "message_retention_days": (
            getattr(hub, "message_retention_in_days", None) if hub else None
        ),
        "status": str(getattr(hub, "status", "") or "") if hub else "",
        "consumer_groups": [getattr(group, "name", "") for group in groups],
    }
