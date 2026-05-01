"""Azure Blob Storage container discovery."""

import logging
from typing import Any, Optional

from ._azure_common import azure_credential

logger = logging.getLogger(__name__)


async def discover_azure_blob(
    account: str,
    container: str,
    account_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    max_keys: int = 100,
) -> dict[str, Any]:
    """Sample an Azure Blob container's contents to infer structure."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError(
            "azure-storage-blob is required. "
            "Install with: pip install 'daita-agents[azure]'"
        )

    logger.debug("discover_azure_blob: sampling %s/%s", account, container)

    credential = azure_credential(tenant_id=tenant_id)
    service = BlobServiceClient(
        account_url=account_url or f"https://{account}.blob.core.windows.net",
        credential=credential,
    )
    container_client = service.get_container_client(container)

    blobs: list[Any] = []
    try:
        for blob in container_client.list_blobs():
            blobs.append(blob)
            if len(blobs) >= max_keys:
                break
    except Exception as exc:
        logger.warning(
            "Azure Blob list_blobs failed for %s/%s: %s", account, container, exc
        )

    prefixes: dict[str, int] = {}
    content_types: dict[str, int] = {}
    total_size = 0
    for blob in blobs:
        name = getattr(blob, "name", "") or ""
        total_size += getattr(blob, "size", 0) or 0
        if "/" in name:
            top, _ = name.split("/", 1)
            prefixes[top] = prefixes.get(top, 0) + 1

        content_settings = getattr(blob, "content_settings", None)
        content_type = getattr(content_settings, "content_type", "") or ""
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "unknown"
        content_types[content_type or ext] = (
            content_types.get(content_type or ext, 0) + 1
        )

    properties: dict[str, Any] = {}
    try:
        props = container_client.get_container_properties()
        properties = {
            "lease_state": str(getattr(props, "lease", {}).get("state", "") or ""),
            "public_access": str(getattr(props, "public_access", "") or ""),
            "last_modified": str(getattr(props, "last_modified", "") or ""),
        }
    except Exception as exc:
        logger.debug(
            "Azure Blob get_container_properties failed for %s: %s", container, exc
        )

    return {
        "database_type": "azure_blob",
        "account": account,
        "container": container,
        "object_count": len(blobs),
        "total_size_bytes": total_size,
        "prefixes": prefixes,
        "content_types": content_types,
        "max_keys_sampled": max_keys,
        "properties": properties,
    }
