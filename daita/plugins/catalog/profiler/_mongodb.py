"""MongoDB profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_mongodb
from ._common import _dict_to_normalized_schema


class MongoDBProfiler(BaseProfiler):
    """Profiles MongoDB databases using motor."""

    def supports(self, store_type: str) -> bool:
        return store_type == "mongodb"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Connect to a MongoDB store and extract its schema."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError(
                "motor is required. Install with: pip install 'daita-agents[mongodb]'"
            )

        from ..discovery import discover_mongodb

        conn_hint = store.connection_hint
        connection_string = conn_hint.get("connection_string", "")
        database = conn_hint.get("database", "")

        result = await discover_mongodb(
            connection_string=connection_string,
            database=database,
        )

        normalized = normalize_mongodb(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
