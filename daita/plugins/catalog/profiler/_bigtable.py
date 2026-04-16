"""Bigtable profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_bigtable
from ._common import _dict_to_normalized_schema


class BigtableProfiler(BaseProfiler):
    """Profiles Bigtable instances by listing tables and column families."""

    def supports(self, store_type: str) -> bool:
        return store_type == "bigtable"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_bigtable

        hint = store.connection_hint
        result = await discover_bigtable(
            project=hint.get("project", ""),
            instance=hint.get("instance", ""),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(normalize_bigtable(result), store_id=store.id)
