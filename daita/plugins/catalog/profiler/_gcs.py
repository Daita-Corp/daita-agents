"""GCS profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_gcs
from ._common import _dict_to_normalized_schema


class GCSProfiler(BaseProfiler):
    """Profiles GCS buckets by sampling objects."""

    def supports(self, store_type: str) -> bool:
        return store_type == "gcs"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_gcs

        hint = store.connection_hint
        result = await discover_gcs(
            bucket=hint.get("bucket", ""),
            project=hint.get("project"),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(normalize_gcs(result), store_id=store.id)
