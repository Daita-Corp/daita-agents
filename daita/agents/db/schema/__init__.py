"""Schema discovery, navigation, caching, and summarization for ``from_db``."""

from .cache import (
    cache_key,
    clear_schema_snapshot,
    detect_drift,
    load_cached_schema,
    load_schema_snapshot,
    save_schema_cache,
)
from .discovery import (
    NUMERIC_TYPES,
    discover_schema,
    discover_schema_fallback,
    is_numeric_type,
    normalize_schema,
)
from .metadata import (
    column_name,
    matching_tables,
    normalize_identifier,
    schema_table_columns,
    short_table_name,
    split_identifier,
    table_name,
)
from .navigation import (
    describe_relationships,
    find_join_paths,
    inspect_table,
    list_tables,
    search_schema,
    should_register_schema_navigation,
)
from .sampling import PII_COLUMN_PATTERNS, sample_numeric_columns
from .summary import build_db_summary, suggested_questions

__all__ = [
    "NUMERIC_TYPES",
    "PII_COLUMN_PATTERNS",
    "build_db_summary",
    "cache_key",
    "clear_schema_snapshot",
    "column_name",
    "describe_relationships",
    "detect_drift",
    "discover_schema",
    "discover_schema_fallback",
    "find_join_paths",
    "inspect_table",
    "is_numeric_type",
    "list_tables",
    "load_cached_schema",
    "load_schema_snapshot",
    "matching_tables",
    "normalize_schema",
    "normalize_identifier",
    "sample_numeric_columns",
    "save_schema_cache",
    "schema_table_columns",
    "search_schema",
    "should_register_schema_navigation",
    "short_table_name",
    "split_identifier",
    "suggested_questions",
    "table_name",
]
