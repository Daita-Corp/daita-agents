"""Kinesis Data Stream discovery."""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def discover_kinesis(
    stream_name: str,
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
    sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Describe a Kinesis stream and sample records to infer schema.

    Returns a raw result dict with stream metadata and inferred record fields.
    """
    import boto3

    logger.debug("discover_kinesis: inspecting stream %s in %s", stream_name, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)
    client = session.client("kinesis", region_name=region)

    # Stream summary
    summary = {}
    try:
        summary = client.describe_stream_summary(StreamName=stream_name)[
            "StreamDescriptionSummary"
        ]
    except Exception as exc:
        logger.warning("Kinesis describe_stream_summary failed for %s: %s", stream_name, exc)

    # Sample records from the first shard to infer schema
    record_fields: Dict[str, str] = {}
    sampled_count = 0
    try:
        # Get first shard
        shards_resp = client.list_shards(StreamName=stream_name, MaxResults=1)
        shards = shards_resp.get("Shards", [])
        if shards:
            shard_id = shards[0]["ShardId"]
            iterator_resp = client.get_shard_iterator(
                StreamName=stream_name,
                ShardId=shard_id,
                ShardIteratorType="TRIM_HORIZON",
            )
            shard_iterator = iterator_resp["ShardIterator"]

            records_resp = client.get_records(
                ShardIterator=shard_iterator,
                Limit=sample_size,
            )
            for record in records_resp.get("Records", []):
                sampled_count += 1
                try:
                    data = json.loads(record["Data"])
                    if isinstance(data, dict):
                        for key, value in data.items():
                            record_fields[key] = type(value).__name__
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception as exc:
        logger.warning("Kinesis record sampling failed for %s: %s", stream_name, exc)

    return {
        "database_type": "kinesis",
        "stream_name": stream_name,
        "region": region,
        "arn": summary.get("StreamARN", ""),
        "shard_count": summary.get("OpenShardCount", 0),
        "retention_hours": summary.get("RetentionPeriodHours", 24),
        "status": summary.get("StreamStatus", ""),
        "stream_mode": summary.get("StreamModeDetails", {}).get("StreamMode", "PROVISIONED"),
        "consumer_count": summary.get("ConsumerCount", 0),
        "record_fields": record_fields,
        "sampled_count": sampled_count,
        "sample_size": sample_size,
    }
