"""OpenSearch profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_opensearch
from ._common import _dict_to_normalized_schema


class OpenSearchProfiler(BaseProfiler):
    """Profiles OpenSearch domains by extracting index mappings."""

    def supports(self, store_type: str) -> bool:
        return store_type == "opensearch"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Connect to an OpenSearch domain and extract index mappings."""
        try:
            from opensearchpy import OpenSearch
        except ImportError:
            raise ImportError(
                "opensearch-py is required. Install with: pip install 'daita-agents[opensearch]'"
            )

        from ..discovery import discover_opensearch

        conn_hint = store.connection_hint
        result = await discover_opensearch(
            host=conn_hint.get("host", ""),
            region=conn_hint.get("region", "us-east-1"),
        )

        normalized = normalize_opensearch(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
