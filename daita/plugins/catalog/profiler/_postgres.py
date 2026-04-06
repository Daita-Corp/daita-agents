"""PostgreSQL profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_postgresql
from ._common import _dict_to_normalized_schema


class PostgresProfiler(BaseProfiler):
    """Profiles PostgreSQL databases using asyncpg."""

    def supports(self, store_type: str) -> bool:
        return store_type in ("postgresql", "postgres")

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Connect to a PostgreSQL store and extract its schema."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required. Install with: pip install 'daita-agents[postgresql]'"
            )

        from ..discovery import discover_postgres

        conn_hint = store.connection_hint
        connection_string = conn_hint.get("connection_string", "")
        schema = conn_hint.get("schema", "public")
        ssl_mode = conn_hint.get("ssl_mode", "verify-full")

        result = await discover_postgres(
            connection_string=connection_string,
            schema=schema,
            ssl_mode=ssl_mode,
        )

        normalized = normalize_postgresql(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
