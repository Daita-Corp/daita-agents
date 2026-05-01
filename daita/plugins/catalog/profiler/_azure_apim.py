"""Azure API Management profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_azure_apim
from ._common import _dict_to_normalized_schema


class AzureAPIMProfiler(BaseProfiler):
    """Profiles API Management APIs and operations."""

    def supports(self, store_type: str) -> bool:
        return store_type == "azure_apim"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_azure_apim

        hint = store.connection_hint
        result = await discover_azure_apim(
            subscription_id=hint.get("subscription_id", ""),
            resource_group=hint.get("resource_group", ""),
            service=hint.get("service", ""),
            api_id=hint.get("api_id", ""),
            tenant_id=hint.get("tenant_id"),
        )
        return _dict_to_normalized_schema(
            normalize_azure_apim(result), store_id=store.id
        )
