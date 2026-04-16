"""Pub/Sub profilers (topic and subscription)."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_pubsub_subscription, normalize_pubsub_topic
from ._common import _dict_to_normalized_schema


class PubSubTopicProfiler(BaseProfiler):
    """Profiles Pub/Sub topics."""

    def supports(self, store_type: str) -> bool:
        return store_type == "pubsub_topic"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_pubsub_topic

        hint = store.connection_hint
        result = await discover_pubsub_topic(
            project=hint.get("project", ""),
            topic=hint.get("topic", ""),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(
            normalize_pubsub_topic(result), store_id=store.id
        )


class PubSubSubscriptionProfiler(BaseProfiler):
    """Profiles Pub/Sub subscriptions."""

    def supports(self, store_type: str) -> bool:
        return store_type == "pubsub_subscription"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_pubsub_subscription

        hint = store.connection_hint
        result = await discover_pubsub_subscription(
            project=hint.get("project", ""),
            subscription=hint.get("subscription", ""),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(
            normalize_pubsub_subscription(result), store_id=store.id
        )
