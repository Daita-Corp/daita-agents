"""OpenSearch domain discovery."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def discover_opensearch(
    host: str,
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Connect to an OpenSearch domain and extract index mappings.

    Returns a raw result dict with indices and their field mappings.
    """
    try:
        from opensearchpy import (
            AWSV4SignerAuth,
            OpenSearch,
            RequestsHttpConnection,
        )
    except ImportError:
        raise ImportError(
            "opensearch-py is required. Install with: pip install 'daita-agents[opensearch]'"
        )

    import boto3

    logger.debug("discover_opensearch: connecting to %s in %s", host, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)
    credentials = session.get_credentials().get_frozen_credentials()
    auth = AWSV4SignerAuth(credentials, region, "es")

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    # Get cluster info
    cluster_info: Dict[str, Any] = {}
    try:
        cluster_info = client.info()
    except Exception as exc:
        logger.warning("OpenSearch info() failed for %s: %s", host, exc)

    # List indices (exclude system indices)
    indices: List[Dict[str, Any]] = []
    try:
        mappings = client.indices.get_mapping(index="*")
        stats = {}
        try:
            stats = client.indices.stats(index="*")["indices"]
        except Exception:
            pass

        for index_name, index_data in mappings.items():
            if index_name.startswith("."):
                continue  # skip system indices

            properties = index_data.get("mappings", {}).get("properties", {})

            fields = []
            for field_name, field_info in properties.items():
                fields.append(
                    {
                        "field_name": field_name,
                        "type": field_info.get("type", "object"),
                        "index": field_info.get("index", True),
                    }
                )

            index_stats = stats.get(index_name, {}).get("total", {})
            doc_count = index_stats.get("docs", {}).get("count")
            size_bytes = index_stats.get("store", {}).get("size_in_bytes")

            indices.append(
                {
                    "index_name": index_name,
                    "doc_count": doc_count,
                    "size_bytes": size_bytes,
                    "fields": fields,
                }
            )
    except Exception as exc:
        logger.warning("OpenSearch mapping retrieval failed for %s: %s", host, exc)

    version = cluster_info.get("version", {}).get("number", "")

    return {
        "database_type": "opensearch",
        "host": host,
        "region": region,
        "cluster_name": cluster_info.get("cluster_name", ""),
        "version": version,
        "indices": indices,
        "index_count": len(indices),
    }
