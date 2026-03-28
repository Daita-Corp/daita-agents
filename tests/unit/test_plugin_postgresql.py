"""
Unit tests for PostgreSQLPlugin and BaseDatabasePlugin.

Tests real behavior — SQL normalization, truncation logic, column compaction,
LIMIT injection edge cases, specific exception types from injection validation,
and connection state helpers — without requiring a live PostgreSQL connection.
"""

import pytest
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.plugins.base_db import BaseDatabasePlugin
from daita.core.exceptions import ValidationError, PluginError
from daita.core.exceptions import ConnectionError as DaitaConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plugin(**kwargs):
    plugin = PostgreSQLPlugin(host="localhost", database="testdb", **kwargs)
    plugin._connection = object()  # truthy sentinel — skips connect()
    return plugin


# ---------------------------------------------------------------------------
# BaseDatabasePlugin._normalize_sql  (static method, no mock needed)
# ---------------------------------------------------------------------------


class TestNormalizeSql:
    def test_strips_trailing_semicolon(self):
        assert BaseDatabasePlugin._normalize_sql("SELECT 1;") == "SELECT 1"

    def test_strips_trailing_whitespace(self):
        assert BaseDatabasePlugin._normalize_sql("SELECT 1  ") == "SELECT 1"

    def test_strips_semicolon_then_whitespace(self):
        assert BaseDatabasePlugin._normalize_sql("SELECT 1 ;  ") == "SELECT 1"

    def test_preserves_internal_semicolons(self):
        # Only trailing semicolons are stripped
        sql = "SELECT ';' AS ch"
        assert BaseDatabasePlugin._normalize_sql(sql) == sql

    def test_no_op_when_clean(self):
        assert BaseDatabasePlugin._normalize_sql("SELECT 1") == "SELECT 1"

    def test_empty_string(self):
        assert BaseDatabasePlugin._normalize_sql("") == ""


# ---------------------------------------------------------------------------
# BaseDatabasePlugin._compact_column  (static method, no mock needed)
# ---------------------------------------------------------------------------


class TestCompactColumn:
    def test_not_nullable_column(self):
        col = {"column_name": "id", "data_type": "integer", "is_nullable": "NO"}
        assert BaseDatabasePlugin._compact_column(col) == "id:integer:nn"

    def test_nullable_column(self):
        col = {"column_name": "email", "data_type": "text", "is_nullable": "YES"}
        assert BaseDatabasePlugin._compact_column(col) == "email:text:null"

    def test_not_null_string_variant(self):
        col = {"column_name": "x", "data_type": "boolean", "is_nullable": "NOT NULL"}
        assert BaseDatabasePlugin._compact_column(col) == "x:boolean:nn"

    def test_zero_string_variant(self):
        col = {"column_name": "x", "data_type": "int", "is_nullable": "0"}
        assert BaseDatabasePlugin._compact_column(col) == "x:int:nn"

    def test_alternate_key_names(self):
        # Some drivers return 'name' and 'type' instead of 'column_name'/'data_type'
        col = {"name": "price", "type": "numeric", "is_nullable": "YES"}
        assert BaseDatabasePlugin._compact_column(col) == "price:numeric:null"

    def test_missing_nullable_defaults_to_null(self):
        col = {"column_name": "x", "data_type": "text"}
        result = BaseDatabasePlugin._compact_column(col)
        assert result.endswith(":null")


# ---------------------------------------------------------------------------
# BaseDatabasePlugin._truncate_result  (static method, no mock needed)
# ---------------------------------------------------------------------------


class TestTruncateResult:
    def test_no_truncation_when_under_limits(self):
        rows = [{"id": i} for i in range(10)]
        result = BaseDatabasePlugin._truncate_result(rows)
        assert result["rows"] == rows
        assert result["total_rows"] == 10
        assert result["truncated"] is False

    def test_row_count_cap_enforced(self):
        rows = [{"id": i} for i in range(300)]
        result = BaseDatabasePlugin._truncate_result(rows, max_rows=200)
        assert len(result["rows"]) == 200
        assert result["total_rows"] == 300
        assert result["truncated"] is True

    def test_char_cap_enforced(self):
        # Each row is ~50 chars; 100 rows ~5000 chars; cap at 200 chars
        rows = [{"value": "x" * 40} for _ in range(100)]
        result = BaseDatabasePlugin._truncate_result(rows, max_rows=200, max_chars=200)
        assert len(result["rows"]) < 100
        assert result["total_rows"] == 100
        assert result["truncated"] is True

    def test_char_cap_preserves_complete_rows(self):
        # Binary search should never cut mid-row
        rows = [{"id": i, "name": f"user_{i}"} for i in range(50)]
        result = BaseDatabasePlugin._truncate_result(rows, max_rows=200, max_chars=500)
        # All rows in result should be valid dicts
        for row in result["rows"]:
            assert isinstance(row, dict)
            assert "id" in row

    def test_empty_rows(self):
        result = BaseDatabasePlugin._truncate_result([])
        assert result["rows"] == []
        assert result["total_rows"] == 0
        assert result["truncated"] is False

    def test_total_rows_always_reflects_original_count(self):
        rows = [{"id": i} for i in range(5)]
        result = BaseDatabasePlugin._truncate_result(rows, max_rows=3)
        assert result["total_rows"] == 5  # original count, not truncated count


# ---------------------------------------------------------------------------
# BaseDatabasePlugin._validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    def test_raises_when_not_connected(self):
        plugin = PostgreSQLPlugin(host="localhost", database="testdb")
        # _connection, _pool, _client are all None — not connected
        with pytest.raises(ValidationError, match="not connected"):
            plugin._validate_connection()

    def test_no_error_when_connection_set(self):
        plugin = PostgreSQLPlugin(host="localhost", database="testdb")
        plugin._connection = object()
        plugin._validate_connection()  # should not raise

    def test_no_error_when_pool_set(self):
        plugin = PostgreSQLPlugin(host="localhost", database="testdb")
        plugin._pool = object()
        plugin._validate_connection()  # should not raise


# ---------------------------------------------------------------------------
# BaseDatabasePlugin._handle_connection_error
# ---------------------------------------------------------------------------


class TestHandleConnectionError:
    def test_import_error_raises_plugin_error(self):
        plugin = make_plugin()
        with pytest.raises(PluginError) as exc_info:
            plugin._handle_connection_error(
                ImportError("asyncpg not installed"), "connect"
            )
        assert exc_info.value.retry_hint == "permanent"

    def test_generic_error_raises_daita_connection_error(self):
        plugin = make_plugin()
        with pytest.raises(DaitaConnectionError):
            plugin._handle_connection_error(
                RuntimeError("connection refused"), "query"
            )

    def test_error_message_includes_operation(self):
        plugin = make_plugin()
        with pytest.raises(DaitaConnectionError, match="query"):
            plugin._handle_connection_error(RuntimeError("timeout"), "query")


# ---------------------------------------------------------------------------
# BaseDatabasePlugin.is_connected property
# ---------------------------------------------------------------------------


class TestIsConnected:
    def test_false_when_all_none(self):
        plugin = PostgreSQLPlugin(host="localhost", database="testdb")
        assert plugin.is_connected is False

    def test_true_when_connection_set(self):
        plugin = make_plugin()
        assert plugin.is_connected is True

    def test_true_when_pool_set(self):
        plugin = PostgreSQLPlugin(host="localhost", database="testdb")
        plugin._pool = object()
        assert plugin.is_connected is True

    def test_true_when_client_set(self):
        plugin = PostgreSQLPlugin(host="localhost", database="testdb")
        plugin._client = object()
        assert plugin.is_connected is True


# ---------------------------------------------------------------------------
# _tool_query — LIMIT injection with real SQL patterns
# ---------------------------------------------------------------------------


class TestToolQueryLimitInjection:
    async def _run(self, sql, rows=None, **plugin_kwargs):
        plugin = make_plugin(**plugin_kwargs)
        captured = []

        async def fake_query(s, params=None):
            captured.append(s)
            return rows or []

        plugin.query = fake_query
        await plugin._tool_query({"sql": sql})
        return captured[0]

    async def test_adds_limit_when_absent(self):
        sent = await self._run("SELECT * FROM users")
        assert "LIMIT" in sent.upper()

    async def test_no_double_limit_when_present(self):
        sent = await self._run("SELECT * FROM users LIMIT 5")
        assert sent.upper().count("LIMIT") == 1

    async def test_no_double_limit_lowercase(self):
        sent = await self._run("select * from users limit 10")
        assert sent.upper().count("LIMIT") == 1

    async def test_strips_trailing_semicolon_before_limit_check(self):
        # SQL with semicolon should still get LIMIT injected cleanly
        sent = await self._run("SELECT * FROM orders;")
        assert "LIMIT" in sent.upper()
        assert ";" not in sent

    async def test_cte_without_limit_gets_limit_injected(self):
        # CTEs are common LLM outputs — make sure LIMIT is injected
        cte = "WITH recent AS (SELECT * FROM logs) SELECT * FROM recent"
        sent = await self._run(cte)
        assert "LIMIT" in sent.upper()

    async def test_union_without_limit_gets_limit_injected(self):
        union_sql = "SELECT id FROM a UNION SELECT id FROM b"
        sent = await self._run(union_sql)
        assert "LIMIT" in sent.upper()


# ---------------------------------------------------------------------------
# _tool_query — result shape and truncation integration
# ---------------------------------------------------------------------------


class TestToolQueryResultShape:
    async def test_exact_keys_in_result(self):
        plugin = make_plugin()

        async def fake_query(sql, params=None):
            return [{"id": 1}, {"id": 2}]

        plugin.query = fake_query
        result = await plugin._tool_query({"sql": "SELECT id FROM t LIMIT 10"})
        assert set(result.keys()) == {"rows", "total_rows", "truncated"}

    async def test_no_row_count_field(self):
        plugin = make_plugin()

        async def fake_query(sql, params=None):
            return [{"id": 1}]

        plugin.query = fake_query
        result = await plugin._tool_query({"sql": "SELECT id FROM t LIMIT 10"})
        assert "row_count" not in result

    async def test_rows_and_total_rows_match_on_small_result(self):
        plugin = make_plugin()
        data = [{"id": i} for i in range(5)]

        async def fake_query(sql, params=None):
            return data

        plugin.query = fake_query
        result = await plugin._tool_query({"sql": "SELECT id FROM t LIMIT 10"})
        assert len(result["rows"]) == 5
        assert result["total_rows"] == 5
        assert result["truncated"] is False

    async def test_truncated_flag_set_when_over_row_cap(self):
        plugin = make_plugin()
        # Return 300 rows; _truncate_result caps at 200
        data = [{"id": i} for i in range(300)]

        async def fake_query(sql, params=None):
            return data

        plugin.query = fake_query
        result = await plugin._tool_query({"sql": "SELECT id FROM t LIMIT 300"})
        assert result["truncated"] is True
        assert result["total_rows"] == 300
        assert len(result["rows"]) == 200


# ---------------------------------------------------------------------------
# _tool_get_schema — compact column format
# ---------------------------------------------------------------------------


class TestToolGetSchema:
    async def test_result_has_table_and_columns_keys(self):
        plugin = make_plugin()

        async def fake_describe(table):
            return [{"column_name": "id", "data_type": "integer", "is_nullable": "NO"}]

        plugin.describe = fake_describe
        result = await plugin._tool_get_schema({"table_name": "users"})
        assert "table" in result
        assert "columns" in result
        assert "column_count" not in result

    async def test_columns_are_compact_strings(self):
        plugin = make_plugin()

        async def fake_describe(table):
            return [
                {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                {"column_name": "email", "data_type": "text", "is_nullable": "YES"},
            ]

        plugin.describe = fake_describe
        result = await plugin._tool_get_schema({"table_name": "users"})
        assert result["columns"][0] == "id:integer:nn"
        assert result["columns"][1] == "email:text:null"

    async def test_table_name_preserved_in_result(self):
        plugin = make_plugin()

        async def fake_describe(table):
            return []

        plugin.describe = fake_describe
        result = await plugin._tool_get_schema({"table_name": "orders"})
        assert result["table"] == "orders"


# ---------------------------------------------------------------------------
# vector_search — SQL injection validation raises ValidationError specifically
# ---------------------------------------------------------------------------


class TestVectorSearchInjectionValidation:
    """
    The key improvement over old tests: assert ValidationError specifically,
    not bare Exception. A bare Exception would pass even if an unrelated
    AttributeError or TypeError was thrown.

    vector_search() checks `if self._pool is None: await self.connect()` before
    validation, so we set _pool to skip the connect path.
    """

    def make_vs_plugin(self):
        plugin = make_plugin()
        plugin._pool = object()  # skips auto-connect in vector_search
        return plugin

    async def test_rejects_semicolon_in_filter(self):
        plugin = self.make_vs_plugin()
        with pytest.raises(ValidationError, match="';'"):
            await plugin._tool_vector_search(
                {
                    "table": "embeddings",
                    "vector_column": "emb",
                    "query_vector": [0.1, 0.2, 0.3],
                    "filter": "id = 1; DROP TABLE users--",
                }
            )

    async def test_rejects_line_comment_in_filter(self):
        plugin = self.make_vs_plugin()
        with pytest.raises(ValidationError, match="comment"):
            await plugin._tool_vector_search(
                {
                    "table": "embeddings",
                    "vector_column": "emb",
                    "query_vector": [0.1, 0.2, 0.3],
                    "filter": "id = 1 -- comment",
                }
            )

    async def test_rejects_block_comment_in_filter(self):
        plugin = self.make_vs_plugin()
        with pytest.raises(ValidationError, match="comment"):
            await plugin._tool_vector_search(
                {
                    "table": "embeddings",
                    "vector_column": "emb",
                    "query_vector": [0.1, 0.2, 0.3],
                    "filter": "id = 1 /* evil */",
                }
            )

    async def test_rejects_subquery_in_filter(self):
        plugin = self.make_vs_plugin()
        with pytest.raises(ValidationError, match="subquery"):
            await plugin._tool_vector_search(
                {
                    "table": "embeddings",
                    "vector_column": "emb",
                    "query_vector": [0.1, 0.2, 0.3],
                    "filter": "id IN (SELECT id FROM secrets)",
                }
            )

    async def test_raises_value_error_for_invalid_distance_type(self):
        plugin = self.make_vs_plugin()
        with pytest.raises(ValueError, match="distance_type"):
            await plugin._tool_vector_search(
                {
                    "table": "embeddings",
                    "vector_column": "emb",
                    "query_vector": [0.1, 0.2, 0.3],
                    "distance_type": "hamming",
                }
            )


# ---------------------------------------------------------------------------
# Tool registration (secondary — existence + correct names)
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_expected_default_tools_present(self):
        plugin = make_plugin()
        names = {t.name for t in plugin.get_tools()}
        assert "postgres_query" in names
        assert "postgres_count" in names
        assert "postgres_sample" in names
        assert "postgres_inspect" in names

    def test_list_tables_not_in_default_tools(self):
        plugin = make_plugin()
        names = {t.name for t in plugin.get_tools()}
        assert "postgres_list_tables" not in names

    def test_get_schema_not_in_default_tools(self):
        plugin = make_plugin()
        names = {t.name for t in plugin.get_tools()}
        assert "postgres_get_schema" not in names

    def test_write_tools_absent_when_read_only(self):
        plugin = make_plugin(read_only=True)
        names = {t.name for t in plugin.get_tools()}
        assert "postgres_execute" not in names
        assert "postgres_vector_upsert" not in names

    def test_write_tools_present_when_not_read_only(self):
        plugin = make_plugin(read_only=False)
        names = {t.name for t in plugin.get_tools()}
        assert "postgres_execute" in names
