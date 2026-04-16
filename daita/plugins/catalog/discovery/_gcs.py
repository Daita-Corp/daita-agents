"""GCS bucket structure discovery."""

import logging
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


async def discover_gcs(
    bucket: str,
    project: Optional[str] = None,
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
    max_keys: int = 100,
) -> dict[str, Any]:
    """Sample a GCS bucket's contents to infer its structure.

    Returns a raw result dict with prefix structure, content type stats, and
    versioning configuration — the GCS analogue of ``discover_s3``.
    """
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_gcs: sampling bucket %s in %s", bucket, project)

    creds, default_project = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    client = storage.Client(project=project or default_project, credentials=creds)
    bucket_obj = client.bucket(bucket)

    # Sample objects
    blobs: list[Any] = []
    try:
        blobs = list(client.list_blobs(bucket_or_name=bucket, max_results=max_keys))
    except Exception as exc:
        logger.warning("GCS list_blobs failed for %s: %s", bucket, exc)

    prefixes: dict[str, int] = {}
    content_types: dict[str, int] = {}
    total_size = 0
    for blob in blobs:
        name = blob.name or ""
        total_size += blob.size or 0

        if "/" in name:
            top, _ = name.split("/", 1)
            prefixes[top] = prefixes.get(top, 0) + 1

        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "unknown"
        content_types[ext] = content_types.get(ext, 0) + 1

    # Versioning / location / storage class come from the bucket reload
    versioning = "Disabled"
    location = ""
    storage_class = ""
    try:
        bucket_obj.reload()
        versioning = "Enabled" if bucket_obj.versioning_enabled else "Disabled"
        location = bucket_obj.location or ""
        storage_class = bucket_obj.storage_class or ""
    except Exception as exc:
        logger.debug("GCS bucket.reload failed for %s: %s", bucket, exc)

    return {
        "database_type": "gcs",
        "bucket": bucket,
        "project": project or default_project,
        "location": location,
        "storage_class": storage_class,
        "object_count": len(blobs),
        "total_size_bytes": total_size,
        "prefixes": prefixes,
        "content_types": content_types,
        "versioning": versioning,
        "max_keys_sampled": max_keys,
    }
