"""Azure API Management discovery."""

import logging
from typing import Any, Optional

from ._azure_common import azure_credential

logger = logging.getLogger(__name__)


async def discover_azure_apim(
    subscription_id: str,
    resource_group: str,
    service: str,
    api_id: str,
    tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    """Describe an API Management API and its operations."""
    try:
        from azure.mgmt.apimanagement import ApiManagementClient
    except ImportError:
        raise ImportError(
            "azure-mgmt-apimanagement is required. "
            "Install with: pip install 'daita-agents[azure]'"
        )

    client = ApiManagementClient(azure_credential(tenant_id), subscription_id)
    api: Any = None
    try:
        api = client.api.get(resource_group, service, api_id)
    except Exception as exc:
        logger.warning(
            "API Management API get failed for %s/%s: %s", service, api_id, exc
        )

    try:
        operations = list(
            client.api_operation.list_by_api(resource_group, service, api_id)
        )
    except Exception as exc:
        logger.debug(
            "API Management operation list failed for %s/%s: %s", service, api_id, exc
        )
        operations = []

    return {
        "database_type": "azure_apim",
        "service": service,
        "api_id": api_id,
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "resource_name": getattr(api, "id", "") if api else "",
        "display_name": getattr(api, "display_name", "") if api else "",
        "path": getattr(api, "path", "") if api else "",
        "protocols": list(getattr(api, "protocols", None) or []) if api else [],
        "operations": [
            {
                "name": getattr(op, "name", ""),
                "display_name": getattr(op, "display_name", ""),
                "method": getattr(op, "method", ""),
                "url_template": getattr(op, "url_template", ""),
                "description": getattr(op, "description", ""),
            }
            for op in operations
        ],
    }
