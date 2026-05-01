"""Azure Cosmos DB profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_azure_cosmosdb
from ._common import _dict_to_normalized_schema


class AzureCosmosDBProfiler(BaseProfiler):
    """Profiles Cosmos DB SQL API databases and containers."""

    def supports(self, store_type: str) -> bool:
        return store_type == "cosmosdb"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_azure_cosmosdb

        hint = store.connection_hint
        result = await discover_azure_cosmosdb(
            subscription_id=hint.get("subscription_id", ""),
            resource_group=hint.get("resource_group", ""),
            account=hint.get("account", ""),
            tenant_id=hint.get("tenant_id"),
        )
        return _dict_to_normalized_schema(
            normalize_azure_cosmosdb(result), store_id=store.id
        )
