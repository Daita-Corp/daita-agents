"""SQS queue discovery."""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def discover_sqs(
    queue_url: str,
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
    sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Inspect an SQS queue's attributes and peek at messages to infer schema.

    Returns a raw result dict with queue attributes and sampled message
    attribute keys/types.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required. Install with: pip install 'daita-agents[aws]'"
        )

    logger.debug("discover_sqs: inspecting queue %s in %s", queue_url, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)
    client = session.client("sqs", region_name=region)

    # Queue attributes
    attrs = {}
    try:
        attrs = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["All"])[
            "Attributes"
        ]
    except Exception as exc:
        logger.warning("SQS get_queue_attributes failed for %s: %s", queue_url, exc)

    queue_arn = attrs.get("QueueArn", "")
    queue_name = (
        queue_arn.rsplit(":", 1)[-1] if queue_arn else queue_url.rsplit("/", 1)[-1]
    )

    # Peek at messages to infer message attribute schema (does not delete)
    message_attributes: Dict[str, List[str]] = {}
    body_keys: Dict[str, str] = {}
    try:
        resp = client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=min(sample_size, 10),
            VisibilityTimeout=0,  # peek only — immediately visible again
            MessageAttributeNames=["All"],
        )
        for msg in resp.get("Messages", []):
            # Collect message attributes
            for attr_name, attr_val in msg.get("MessageAttributes", {}).items():
                if attr_name not in message_attributes:
                    message_attributes[attr_name] = []
                dtype = attr_val.get("DataType", "String")
                if dtype not in message_attributes[attr_name]:
                    message_attributes[attr_name].append(dtype)

            # Try to parse JSON body to infer payload fields
            try:
                body = json.loads(msg.get("Body", "{}"))
                if isinstance(body, dict):
                    for key, value in body.items():
                        body_keys[key] = type(value).__name__
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception as exc:
        logger.warning("SQS message sampling failed for %s: %s", queue_url, exc)

    return {
        "database_type": "sqs",
        "queue_name": queue_name,
        "queue_url": queue_url,
        "region": region,
        "arn": queue_arn,
        "approximate_message_count": int(attrs.get("ApproximateNumberOfMessages", 0)),
        "visibility_timeout": int(attrs.get("VisibilityTimeout", 30)),
        "max_message_size": int(attrs.get("MaximumMessageSize", 262144)),
        "retention_seconds": int(attrs.get("MessageRetentionPeriod", 345600)),
        "is_fifo": queue_name.endswith(".fifo"),
        "message_attributes": message_attributes,
        "body_keys": body_keys,
        "sample_size": sample_size,
    }
