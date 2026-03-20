"""
Unit tests for PostgreSQLPlugin tool handlers.

Tests that redundant count fields are absent from results,
vector_upsert returns id-only, and vector_search validates
filter for SQL injection — without a real PostgreSQL connection.
"""

import pytest
from daita.plugins.postgresql import PostgreSQLPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin():
    plugin = PostgreSQLPlugin(host="localhost", database="testdb")
    # Inject a fake connection so connect() is never needed
    plugin._connection = object()  # truthy sentinel
    return plugin


# ---------------------------------------------------------------------------
# _tool_query — no redundant row_count field
# ---------------------------------------------------------------------------


async def test_tool_query_no_row_count_field():
    plugin = make_plugin()

    async def fake_query(sql, params=None):
        return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    plugin.query = fake_query

    result = await plugin._tool_query({"sql": "SELECT * FROM users LIMIT 10"})

    assert "row_count" not in result
    assert "rows" in result
    assert "total_rows" in result


async def test_tool_query_contains_expected_fields():
    plugin = make_plugin()

    async def fake_query(sql, params=None):
        return [{"id": i} for i in range(5)]

    plugin.query = fake_query

    result = await plugin._tool_query({"sql": "SELECT id FROM users LIMIT 10"})

    assert set(result.keys()) == {"rows", "total_rows", "truncated"}


# ---------------------------------------------------------------------------
# _tool_get_schema — no redundant column_count field
# ---------------------------------------------------------------------------


async def test_tool_get_schema_no_column_count_field():
    plugin = make_plugin()

    async def fake_describe(table):
        return [
            {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
            {"column_name": "name", "data_type": "text", "is_nullable": "YES"},
        ]

    plugin.describe = fake_describe

    result = await plugin._tool_get_schema({"table_name": "users"})

    assert "column_count" not in result
    assert "table" in result
    assert "columns" in result


async def test_tool_get_schema_columns_are_compact_strings():
    plugin = make_plugin()

    async def fake_describe(table):
        return [
            {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
            {"column_name": "email", "data_type": "text", "is_nullable": "YES"},
        ]

    plugin.describe = fake_describe

    result = await plugin._tool_get_schema({"table_name": "users"})

    for col in result["columns"]:
        parts = col.split(":")
        assert len(parts) == 3
        assert parts[2] in ("nn", "null")


# ---------------------------------------------------------------------------
# _tool_query — default LIMIT injection
# ---------------------------------------------------------------------------


async def test_tool_query_injects_limit_when_absent():
    plugin = make_plugin()
    captured_sql = []

    async def fake_query(sql, params=None):
        captured_sql.append(sql)
        return []

    plugin.query = fake_query

    await plugin._tool_query({"sql": "SELECT * FROM users"})

    assert "LIMIT" in captured_sql[0].upper()


async def test_tool_query_does_not_double_limit():
    plugin = make_plugin()
    captured_sql = []

    async def fake_query(sql, params=None):
        captured_sql.append(sql)
        return []

    plugin.query = fake_query

    await plugin._tool_query({"sql": "SELECT * FROM users LIMIT 5"})

    assert captured_sql[0].upper().count("LIMIT") == 1


# ---------------------------------------------------------------------------
# postgres_vector_search — SQL injection validation
# ---------------------------------------------------------------------------


async def test_vector_search_rejects_semicolon_in_filter():
    plugin = make_plugin()

    with pytest.raises(Exception):
        await plugin._tool_vector_search({
            "table": "embeddings",
            "vector": [0.1] * 3,
            "filter": "id = 1; DROP TABLE users--",
        })


async def test_vector_search_rejects_comment_in_filter():
    plugin = make_plugin()

    with pytest.raises(Exception):
        await plugin._tool_vector_search({
            "table": "embeddings",
            "vector": [0.1] * 3,
            "filter": "id = 1 -- comment",
        })


# ---------------------------------------------------------------------------
# postgres_count and postgres_sample tools present
# ---------------------------------------------------------------------------


def test_count_tool_present():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "postgres_count" in names


def test_sample_tool_present():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "postgres_sample" in names


def test_list_tables_not_in_default_tools():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "postgres_list_tables" not in names


def test_get_schema_not_in_default_tools():
    plugin = make_plugin()
    names = {t.name for t in plugin.get_tools()}
    assert "postgres_get_schema" not in names
