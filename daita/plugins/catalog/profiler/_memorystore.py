"""Memorystore profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_memorystore
from ._common import _dict_to_normalized_schema


class MemorystoreProfiler(BaseProfiler):
    """Profiles Memorystore Redis instances."""

    def supports(self, store_type: str) -> bool:
        return store_type == "memorystore"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_memorystore

        hint = store.connection_hint
        result = await discover_memorystore(
            project=hint.get("project", ""),
            location=hint.get("location", ""),
            instance=hint.get("instance", ""),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(
            normalize_memorystore(result), store_id=store.id
        )
