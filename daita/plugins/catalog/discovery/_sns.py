"""SNS topic discovery."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def discover_sns(
    topic_arn: str,
    region: str = "us-east-1",
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspect an SNS topic's attributes and list its subscriptions.

    Returns a raw result dict with topic attributes and subscription details.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required. Install with: pip install 'daita-agents[aws]'"
        )

    logger.debug("discover_sns: inspecting topic %s in %s", topic_arn, region)

    kwargs: Dict[str, Any] = {}
    if profile_name:
        kwargs["profile_name"] = profile_name
    session = boto3.Session(**kwargs)
    client = session.client("sns", region_name=region)

    topic_name = topic_arn.rsplit(":", 1)[-1]

    # Topic attributes
    attrs = {}
    try:
        attrs = client.get_topic_attributes(TopicArn=topic_arn)["Attributes"]
    except Exception as exc:
        logger.warning("SNS get_topic_attributes failed for %s: %s", topic_arn, exc)

    # List subscriptions
    subscriptions: List[Dict[str, str]] = []
    try:
        paginator = client.get_paginator("list_subscriptions_by_topic")
        for page in paginator.paginate(TopicArn=topic_arn):
            for sub in page.get("Subscriptions", []):
                subscriptions.append({
                    "subscription_arn": sub.get("SubscriptionArn", ""),
                    "protocol": sub.get("Protocol", ""),
                    "endpoint": sub.get("Endpoint", ""),
                })
    except Exception as exc:
        logger.warning("SNS list_subscriptions failed for %s: %s", topic_arn, exc)

    return {
        "database_type": "sns",
        "topic_name": topic_name,
        "topic_arn": topic_arn,
        "region": region,
        "display_name": attrs.get("DisplayName", ""),
        "subscription_count": int(attrs.get("SubscriptionsConfirmed", 0)),
        "subscriptions_pending": int(attrs.get("SubscriptionsPending", 0)),
        "is_fifo": topic_name.endswith(".fifo"),
        "subscriptions": subscriptions,
    }
