"""
Legacy DB query tool-view names.

These names are presentation compatibility for generic chat/LLM tool surfaces.
Runtime planning should use capabilities and evidence contracts instead.
"""

DB_PLAN_TOOL_VIEW = "db_plan_query"
DB_COMPILE_TOOL_VIEW = "db_compile_and_query"
DB_VALIDATE_SQL_TOOL_VIEW = "db_validate_sql"
DB_QUERY_TOOL_VIEW = "db_query"
DB_COUNT_TOOL_VIEW = "db_count"
DB_SAMPLE_TOOL_VIEW = "db_sample"
DB_EXECUTE_TOOL_VIEW = "db_execute"
CATALOG_SEARCH_TOOL_VIEW = "catalog_search_schema"
CATALOG_INSPECT_TOOL_VIEW = "catalog_inspect_table"
CATALOG_RELATIONSHIP_TOOL_VIEW = "catalog_find_join_paths"
MONGO_FIND_TOOL_VIEW = "db_find"
MONGO_AGGREGATE_TOOL_VIEW = "db_aggregate"

SQL_QUERY_TOOL_VIEWS = (
    DB_COMPILE_TOOL_VIEW,
    DB_PLAN_TOOL_VIEW,
    DB_QUERY_TOOL_VIEW,
    DB_VALIDATE_SQL_TOOL_VIEW,
    DB_COUNT_TOOL_VIEW,
    DB_SAMPLE_TOOL_VIEW,
    DB_EXECUTE_TOOL_VIEW,
)

MONGO_QUERY_TOOL_VIEWS = (
    MONGO_FIND_TOOL_VIEW,
    MONGO_AGGREGATE_TOOL_VIEW,
)

CATALOG_QUERY_TOOL_VIEWS = (
    CATALOG_SEARCH_TOOL_VIEW,
    CATALOG_INSPECT_TOOL_VIEW,
    CATALOG_RELATIONSHIP_TOOL_VIEW,
)
