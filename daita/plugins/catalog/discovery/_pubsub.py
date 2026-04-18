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
    schema_ref = ""
    schema_encoding = ""
    try:
        t = publisher.get_topic(request={"topic": topic_path})
        kms_key = t.kms_key_name or ""
        message_retention = (
            str(t.message_retention_duration) if t.message_retention_duration else ""
        )
        ss = getattr(t, "schema_settings", None)
        if ss and ss.schema and ss.schema != "_deleted-schema_":
            schema_ref = ss.schema
            schema_encoding = ss.encoding.name if ss.encoding else ""
    except Exception as exc:
        logger.debug("Pub/Sub get_topic failed for %s: %s", topic_path, exc)

    schema = _resolve_pubsub_schema(creds, schema_ref) if schema_ref else None

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
        "schema_encoding": schema_encoding,
        "schema": schema,
    }


def _resolve_pubsub_schema(creds: Any, schema_name: str) -> Optional[dict[str, Any]]:
    """Fetch the full schema definition from Pub/Sub Schema Registry.

    Returns a dict shaped ``{name, type, definition, revision_id}`` or None on
    failure. Type is ``PROTOCOL_BUFFER`` or ``AVRO``. Failure paths are
    intentionally quiet — discovery degrades gracefully to "no schema".
    """
    try:
        from google.cloud import pubsub_v1
        from google.pubsub_v1 import types as pstypes
    except ImportError:
        return None

    try:
        client = pubsub_v1.SchemaServiceClient(credentials=creds)
        s = client.get_schema(
            request={"name": schema_name, "view": pstypes.SchemaView.FULL}
        )
        return {
            "name": s.name,
            "type": s.type_.name if s.type_ else "",
            "definition": s.definition or "",
            "revision_id": s.revision_id or "",
        }
    except Exception as exc:
        logger.debug("Pub/Sub get_schema failed for %s: %s", schema_name, exc)
        return None


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
