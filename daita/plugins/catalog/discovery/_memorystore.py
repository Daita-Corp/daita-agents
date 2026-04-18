"""Memorystore (Redis) instance discovery."""

import logging
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


async def discover_memorystore(
    project: str,
    location: str,
    instance: str,
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a Memorystore Redis instance.

    Redis has no schema in the relational sense, so the "tables" list is empty.
    Instance-level properties (tier, version, endpoint) go into metadata.
    """
    try:
        from google.cloud import redis_v1
    except ImportError:
        raise ImportError(
            "google-cloud-redis is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_memorystore: %s/%s/%s", project, location, instance)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    client = redis_v1.CloudRedisClient(credentials=creds)
    name = client.instance_path(project, location, instance)

    host = ""
    port = 0
    tier = ""
    version = ""
    memory_size = 0
    state = ""
    try:
        inst = client.get_instance(name=name)
        host = inst.host or ""
        port = inst.port or 0
        tier = redis_v1.Instance.Tier(inst.tier).name if inst.tier else ""
        version = inst.redis_version or ""
        memory_size = inst.memory_size_gb or 0
        state = redis_v1.Instance.State(inst.state).name if inst.state else ""
    except Exception as exc:
        logger.warning("Memorystore get_instance failed for %s: %s", name, exc)

    return {
        "database_type": "memorystore",
        "project": project,
        "location": location,
        "instance": instance,
        "resource_name": name,
        "host": host,
        "port": port,
        "tier": tier,
        "redis_version": version,
        "memory_size_gb": memory_size,
        "state": state,
    }
