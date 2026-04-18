"""BigQuery profiler.

Metadata-only: lists tables and their schemas. Never issues SELECT —
that is the responsibility of ``BigQueryPlugin`` (runtime operations).
"""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_bigquery
from ._common import _dict_to_normalized_schema


class BigQueryProfiler(BaseProfiler):
    """Profiles BigQuery datasets by listing tables and their schemas."""

    def supports(self, store_type: str) -> bool:
        return store_type == "bigquery"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        from ..discovery import discover_bigquery

        hint = store.connection_hint
        result = await discover_bigquery(
            project=hint.get("project", ""),
            dataset=hint.get("dataset", ""),
            credentials_path=hint.get("credentials_path"),
            impersonate_service_account=hint.get("impersonate_service_account"),
        )
        return _dict_to_normalized_schema(normalize_bigquery(result), store_id=store.id)
