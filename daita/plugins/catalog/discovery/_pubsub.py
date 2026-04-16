"""Pub/Sub topic and subscription discovery."""

import logging
from typing import Any, Optional

from ._gcp_common import gcp_credentials

logger = logging.getLogger(__name__)


async def discover_pubsub_topic(
    project: str,
    topic: str,
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a Pub/Sub topic including its attached subscriptions."""
    try:
        from google.cloud import pubsub_v1
    except ImportError:
        raise ImportError(
            "google-cloud-pubsub is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_pubsub_topic: %s/%s", project, topic)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    publisher = pubsub_v1.PublisherClient(credentials=creds)
    topic_path = publisher.topic_path(project, topic)

    kms_key = ""
    message_retention = ""
    try:
        t = publisher.get_topic(request={"topic": topic_path})
        kms_key = t.kms_key_name or ""
        message_retention = (
            str(t.message_retention_duration) if t.message_retention_duration else ""
        )
    except Exception as exc:
        logger.debug("Pub/Sub get_topic failed for %s: %s", topic_path, exc)

    subscriptions: list[str] = []
    try:
        subscriptions = [
            s.split("/")[-1]
            for s in publisher.list_topic_subscriptions(request={"topic": topic_path})
        ]
    except Exception as exc:
        logger.debug(
            "Pub/Sub list_topic_subscriptions failed for %s: %s", topic_path, exc
        )

    return {
        "database_type": "pubsub_topic",
        "project": project,
        "topic": topic,
        "resource_name": topic_path,
        "kms_key": kms_key,
        "message_retention": message_retention,
        "subscriptions": subscriptions,
    }


async def discover_pubsub_subscription(
    project: str,
    subscription: str,
    credentials_path: Optional[str] = None,
    impersonate_service_account: Optional[str] = None,
) -> dict[str, Any]:
    """Describe a Pub/Sub subscription."""
    try:
        from google.cloud import pubsub_v1
    except ImportError:
        raise ImportError(
            "google-cloud-pubsub is required. "
            "Install with: pip install 'daita-agents[gcp]'"
        )

    logger.debug("discover_pubsub_subscription: %s/%s", project, subscription)

    creds, _ = gcp_credentials(
        credentials_path=credentials_path,
        impersonate_service_account=impersonate_service_account,
    )
    subscriber = pubsub_v1.SubscriberClient(credentials=creds)
    sub_path = subscriber.subscription_path(project, subscription)

    topic = ""
    ack_deadline = 0
    push_endpoint = ""
    try:
        s = subscriber.get_subscription(request={"subscription": sub_path})
        topic = s.topic.split("/")[-1] if s.topic else ""
        ack_deadline = s.ack_deadline_seconds or 0
        push_endpoint = (s.push_config.push_endpoint if s.push_config else "") or ""
    except Exception as exc:
        logger.debug("Pub/Sub get_subscription failed for %s: %s", sub_path, exc)

    return {
        "database_type": "pubsub_subscription",
        "project": project,
        "subscription": subscription,
        "resource_name": sub_path,
        "topic": topic,
        "ack_deadline_seconds": ack_deadline,
        "push_endpoint": push_endpoint,
    }
