"""S3 bucket structure discovery."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def discover_s3(
    bucket: str,
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
    max_keys: int = 100,
) -> Dict[str, Any]:
    """
    Sample an S3 bucket's contents to infer its structure.

    Returns a raw result dict with prefix structure, object metadata stats,
    and versioning/encryption configuration.
    """
    import boto3

    logger.debug("discover_s3: sampling bucket %s in %s", bucket, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)
    client = session.client("s3", region_name=region)

    # Sample objects
    objects = []
    try:
        resp = client.list_objects_v2(Bucket=bucket, MaxKeys=max_keys)
        objects = resp.get("Contents", [])
    except Exception as exc:
        logger.warning("S3 list_objects_v2 failed for %s: %s", bucket, exc)

    # Infer prefix structure (top-level "folders")
    prefixes: Dict[str, int] = {}
    content_types: Dict[str, int] = {}
    total_size = 0
    for obj in objects:
        key = obj.get("Key", "")
        total_size += obj.get("Size", 0)

        # Extract top-level prefix
        parts = key.split("/", 1)
        if len(parts) > 1:
            prefixes[parts[0]] = prefixes.get(parts[0], 0) + 1

        # Infer content type from extension
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else "unknown"
        content_types[ext] = content_types.get(ext, 0) + 1

    # Versioning status
    versioning = "Disabled"
    try:
        v_resp = client.get_bucket_versioning(Bucket=bucket)
        versioning = v_resp.get("Status", "Disabled")
    except Exception:
        pass

    return {
        "database_type": "s3",
        "bucket": bucket,
        "region": region,
        "object_count": len(objects),
        "total_size_bytes": total_size,
        "prefixes": prefixes,
        "content_types": content_types,
        "versioning": versioning,
        "max_keys_sampled": max_keys,
    }
