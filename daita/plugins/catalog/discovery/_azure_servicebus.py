"""Azure Service Bus discovery."""

import logging
from typing import Any, Optional

from ._azure_common import azure_credential

logger = logging.getLogger(__name__)


async def discover_azure_servicebus_queue(
    subscription_id: str,
    resource_group: str,
    namespace: str,
    queue: str,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a Service Bus queue."""
    try:
        from azure.mgmt.servicebus import ServiceBusManagementClient
    except ImportError:
        raise ImportError(
            "azure-mgmt-servicebus is required. "
            "Install with: pip install 'daita-agents[azure]'"
        )

    client = ServiceBusManagementClient(azure_credential(tenant_id), subscription_id)
    q: Any = None
    try:
        q = client.queues.get(resource_group, namespace, queue)
    except Exception as exc:
        logger.warning(
            "Service Bus queue get failed for %s/%s: %s", namespace, queue, exc
        )

    return {
        "database_type": "servicebus_queue",
        "namespace": namespace,
        "queue": queue,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "resource_name": getattr(q, "id", "") if q else "",
        "message_count": getattr(q, "message_count", None) if q else None,
        "max_size_mb": getattr(q, "max_size_in_megabytes", None) if q else None,
        "dead_lettering_on_message_expiration": (
            getattr(q, "dead_lettering_on_message_expiration", None) if q else None
        ),
        "requires_duplicate_detection": (
            getattr(q, "requires_duplicate_detection", None) if q else None
        ),
    }


async def discover_azure_servicebus_topic(
    subscription_id: str,
    resource_group: str,
    namespace: str,
    topic: str,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a Service Bus topic and subscriptions."""
    try:
        from azure.mgmt.servicebus import ServiceBusManagementClient
    except ImportError:
        raise ImportError(
            "azure-mgmt-servicebus is required. "
            "Install with: pip install 'daita-agents[azure]'"
        )

    client = ServiceBusManagementClient(azure_credential(tenant_id), subscription_id)
    t: Any = None
    try:
        t = client.topics.get(resource_group, namespace, topic)
    except Exception as exc:
        logger.warning(
            "Service Bus topic get failed for %s/%s: %s", namespace, topic, exc
        )

    try:
        subscriptions = list(
            client.subscriptions.list_by_topic(resource_group, namespace, topic)
        )
    except Exception as exc:
        logger.debug(
            "Service Bus subscription list failed for %s/%s: %s", namespace, topic, exc
        )
        subscriptions = []

    return {
        "database_type": "servicebus_topic",
        "namespace": namespace,
        "topic": topic,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "resource_name": getattr(t, "id", "") if t else "",
        "subscription_count": getattr(t, "subscription_count", None) if t else None,
        "max_size_mb": getattr(t, "max_size_in_megabytes", None) if t else None,
        "subscriptions": [getattr(sub, "name", "") for sub in subscriptions],
    }
