"""DynamoDB table schema discovery."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def discover_dynamodb(
    table_name: str,
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
    sample_size: int = 20,
) -> Dict[str, Any]:
    """
    Describe a DynamoDB table's schema and sample items to infer attribute patterns.

    Returns a raw result dict with key schema, attribute definitions, GSIs,
    and attributes inferred from sampled items.
    """
    import boto3

    logger.debug("discover_dynamodb: describing table %s in %s", table_name, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)
    client = session.client("dynamodb", region_name=region)

    # Describe table structure
    desc = client.describe_table(TableName=table_name)["Table"]

    key_schema = desc.get("KeySchema", [])
    attribute_defs = desc.get("AttributeDefinitions", [])
    gsis = desc.get("GlobalSecondaryIndexes", [])
    lsis = desc.get("LocalSecondaryIndexes", [])

    # Sample items to discover attributes beyond the key schema
    sampled_attributes: Dict[str, set] = {}
    try:
        scan_resp = client.scan(TableName=table_name, Limit=sample_size)
        for item in scan_resp.get("Items", []):
            for attr_name, attr_value in item.items():
                # DynamoDB returns {"S": "val"}, {"N": "123"}, etc.
                dtype = next(iter(attr_value.keys()), "S")
                if attr_name not in sampled_attributes:
                    sampled_attributes[attr_name] = set()
                sampled_attributes[attr_name].add(dtype)
    except Exception as exc:
        logger.warning("DynamoDB scan failed for %s: %s", table_name, exc)

    return {
        "database_type": "dynamodb",
        "table_name": table_name,
        "region": region,
        "key_schema": key_schema,
        "attribute_definitions": attribute_defs,
        "item_count": desc.get("ItemCount", 0),
        "table_size_bytes": desc.get("TableSizeBytes", 0),
        "table_status": desc.get("TableStatus", ""),
        "billing_mode": desc.get("BillingModeSummary", {}).get(
            "BillingMode", "PROVISIONED"
        ),
        "global_secondary_indexes": [
            {
                "index_name": g["IndexName"],
                "key_schema": g.get("KeySchema", []),
                "projection": g.get("Projection", {}).get("ProjectionType", ""),
                "item_count": g.get("ItemCount", 0),
            }
            for g in gsis
        ],
        "local_secondary_indexes": [
            {
                "index_name": l["IndexName"],
                "key_schema": l.get("KeySchema", []),
            }
            for l in lsis
        ],
        "sampled_attributes": {
            k: list(v) for k, v in sampled_attributes.items()
        },
        "sample_size": sample_size,
    }
