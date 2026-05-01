"""Azure Service Bus profilers."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import (
    normalize_azure_servicebus_queue,
    normalize_azure_servicebus_topic,
)
from ._common import _dict_to_normalized_schema


class AzureServiceBusQueueProfiler(BaseProfiler):
    """Profiles Service Bus queues."""

    def supports(self, store_type: str) -> bool:
        return store_type == "servicebus_queue"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_azure_servicebus_queue

        hint = store.connection_hint
        result = await discover_azure_servicebus_queue(
            subscription_id=hint.get("subscription_id", ""),
            resource_group=hint.get("resource_group", ""),
            namespace=hint.get("namespace", ""),
            queue=hint.get("queue", ""),
            tenant_id=hint.get("tenant_id"),
        )
        return _dict_to_normalized_schema(
            normalize_azure_servicebus_queue(result), store_id=store.id
        )


class AzureServiceBusTopicProfiler(BaseProfiler):
    """Profiles Service Bus topics."""

    def supports(self, store_type: str) -> bool:
        return store_type == "servicebus_topic"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_azure_servicebus_topic

        hint = store.connection_hint
        result = await discover_azure_servicebus_topic(
            subscription_id=hint.get("subscription_id", ""),
            resource_group=hint.get("resource_group", ""),
            namespace=hint.get("namespace", ""),
            topic=hint.get("topic", ""),
            tenant_id=hint.get("tenant_id"),
        )
        return _dict_to_normalized_schema(
            normalize_azure_servicebus_topic(result), store_id=store.id
        )
