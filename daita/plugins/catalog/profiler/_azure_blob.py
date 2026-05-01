"""Azure Blob Storage profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_azure_blob
from ._common import _dict_to_normalized_schema


class AzureBlobProfiler(BaseProfiler):
    """Profiles Azure Blob containers by sampling blobs."""

    def supports(self, store_type: str) -> bool:
        return store_type == "azure_blob"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_azure_blob

        hint = store.connection_hint
        result = await discover_azure_blob(
            account=hint.get("account", ""),
            container=hint.get("container", ""),
            account_url=hint.get("account_url"),
            tenant_id=hint.get("tenant_id"),
        )
        return _dict_to_normalized_schema(
            normalize_azure_blob(result), store_id=store.id
        )
