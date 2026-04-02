"""MySQL/MariaDB profiler."""

from ..base_discoverer import DiscoveredStore
from ..base_profiler import BaseProfiler, NormalizedSchema
from ..normalizer import normalize_mysql
from ._common import _dict_to_normalized_schema


class MySQLProfiler(BaseProfiler):
    """Profiles MySQL/MariaDB databases using aiomysql."""

    def supports(self, store_type: str) -> bool:
        return store_type == "mysql"

    async def profile(self, store: DiscoveredStore) -> NormalizedSchema:
        """Connect to a MySQL store and extract its schema."""
        try:
            import aiomysql
        except ImportError:
            raise ImportError(
                "aiomysql is required. Install with: pip install 'daita-agents[mysql]'"
            )

        from ..discovery import discover_mysql

        conn_hint = store.connection_hint
        connection_string = conn_hint.get("connection_string", "")
        schema = conn_hint.get("schema")

        result = await discover_mysql(
            connection_string=connection_string,
            schema=schema,
        )

        normalized = normalize_mysql(result)
        return _dict_to_normalized_schema(normalized, store_id=store.id)
