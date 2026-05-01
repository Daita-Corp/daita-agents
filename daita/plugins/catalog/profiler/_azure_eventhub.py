"""Azure Event Hubs profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_azure_eventhub
from ._common import _dict_to_normalized_schema


class AzureEventHubProfiler(BaseProfiler):
    """Profiles Event Hubs and consumer groups."""

    def supports(self, store_type: str) -> bool:
        return store_type == "eventhub"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_azure_eventhub

        hint = store.connection_hint
        result = await discover_azure_eventhub(
            subscription_id=hint.get("subscription_id", ""),
            resource_group=hint.get("resource_group", ""),
            namespace=hint.get("namespace", ""),
            eventhub=hint.get("eventhub", ""),
            tenant_id=hint.get("tenant_id"),
        )
        return _dict_to_normalized_schema(
            normalize_azure_eventhub(result), store_id=store.id
        )
