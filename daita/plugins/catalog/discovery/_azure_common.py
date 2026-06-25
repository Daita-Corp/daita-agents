"""Shared helpers for Azure catalog discovery."""

import os
from typing import Any, Optional

_AZURE_INSTALL_HINT = (
    "azure-identity is required. Install with: pip install 'daita-agents[azure]'"
)


def csv_env(name: str) -> list[str]:
    """Parse a comma-separated env var into a list, filtering empty entries."""
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def azure_credential(tenant_id: Optional[str] = None) -> Any:
    """Resolve an Azure credential using the standard Azure SDK chain."""
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise ImportError(_AZURE_INSTALL_HINT)

    kwargs = {"tenant_id": tenant_id} if tenant_id else {}
    return DefaultAzureCredential(**kwargs)


def resource_group_from_id(resource_id: str) -> str:
    """Extract the resource group segment from an Azure resource ID."""
    parts = [part for part in resource_id.split("/") if part]
    lowered = [part.lower() for part in parts]
    try:
        return parts[lowered.index("resourcegroups") + 1]
    except (ValueError, IndexError):
        return ""


def azure_location(value: Any) -> Optional[str]:
    """Normalize Azure location values without forcing empty strings."""
    location = getattr(value, "location", None) or ""
    return location or None
