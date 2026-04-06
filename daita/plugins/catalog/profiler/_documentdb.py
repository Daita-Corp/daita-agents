"""DocumentDB profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_documentdb
from ._common import _dict_to_normalized_schema


class DocumentDBProfiler(BaseProfiler):
    """Profiles DocumentDB clusters by sampling documents (MongoDB-compatible)."""

    def supports(self, store_type: str) -> bool:
        return store_type == "documentdb"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Connect to a DocumentDB cluster and infer schema from samples."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError(
                "motor is required. Install with: pip install 'daita-agents[mongodb]'"
            )

        from ..discovery import discover_documentdb

        conn_hint = store.connection_hint
        result = await discover_documentdb(
            host=conn_hint.get("host", ""),
            port=conn_hint.get("port", 27017),
        )

        normalized = normalize_documentdb(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
