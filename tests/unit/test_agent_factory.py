"""
Tests for daita/agents/db/ — Agent.from_db() builder.

All tests use mocked DB connections; no real databases are required.
Run with: pytest tests/unit/test_agent_factory.py -v
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from daita.agents.db.cache import (
    cache_key as _db_cache_key,
    detect_drift as _db_detect_drift,
    load_cached_schema as _db_load_cached_schema,
    save_schema_cache as _db_save_schema_cache,
)
from daita.agents.db.prompt import build_prompt as _db_build_prompt, infer_domain as _infer_domain
from daita.agents.db.resolve import resolve_plugin as _db_resolve_plugin
from daita.agents.db.schema import normalize_schema as _db_normalize_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_normalized_schema(tables=None, fks=None, db_type="postgresql", db_name="public"):
    """Build a minimal normalized schema dict for tests."""
    tables = tables or []
    return {
        "database_type": db_type,
        "database_name": db_name,
        "tables": tables,
        "foreign_keys": fks or [],
        "table_count": len(tables),
    }


def _table(name, columns=None, row_count=None):
    columns = columns or [{"name": "id", "type": "integer", "nullable": False, "is_primary_key": True}]
    return {"name": name, "columns": columns, "row_count": row_count}


# ---------------------------------------------------------------------------
# Plugin resolution tests
# ---------------------------------------------------------------------------


class TestResolveDbPlugin:
    # Patch at the source module — resolve_plugin imports lazily inside function bodies,
    # so `daita.agents.db.builder.Foo` doesn't exist at module level.
    def test_resolve_postgresql_string(self):
        with patch("daita.plugins.postgresql.PostgreSQLPlugin") as MockPg:
            plugin, created = _db_resolve_plugin("postgresql://user:pass@host/db")
        assert created is True
        MockPg.assert_called_once_with(connection_string="postgresql://user:pass@host/db", read_only=True)

    def test_resolve_postgres_scheme(self):
        with patch("daita.plugins.postgresql.PostgreSQLPlugin") as MockPg:
            plugin, created = _db_resolve_plugin("postgres://user:pass@host/db")
        assert created is True
        MockPg.assert_called_once_with(connection_string="postgres://user:pass@host/db", read_only=True)

    def test_resolve_mysql_string(self):
        with patch("daita.plugins.mysql.MySQLPlugin") as MockMy:
            plugin, created = _db_resolve_plugin("mysql://user:pass@host/db")
        assert created is True
        MockMy.assert_called_once_with(connection_string="mysql://user:pass@host/db", read_only=True)

    def test_resolve_mongodb_string(self):
        with patch("daita.plugins.mongodb.MongoDBPlugin") as MockMongo:
            plugin, created = _db_resolve_plugin("mongodb://host/mydb")
        assert created is True
        MockMongo.assert_called_once_with(connection_string="mongodb://host/mydb", read_only=True)

    def test_resolve_mongodb_no_database(self):
        with pytest.raises(ValueError, match="database name"):
            _db_resolve_plugin("mongodb://host/")

    def test_resolve_sqlite_uri(self):
        with patch("daita.plugins.sqlite.SQLitePlugin") as MockSq:
            plugin, created = _db_resolve_plugin("sqlite:///path/to/db.sqlite")
        assert created is True
        MockSq.assert_called_once_with(path="/path/to/db.sqlite", read_only=True)

    def test_resolve_sqlite_bare_path_db(self):
        with patch("daita.plugins.sqlite.SQLitePlugin") as MockSq:
            plugin, created = _db_resolve_plugin("./data.db")
        assert created is True
        MockSq.assert_called_once_with(path="./data.db", read_only=True)

    def test_resolve_sqlite_bare_path_sqlite(self):
        with patch("daita.plugins.sqlite.SQLitePlugin") as MockSq:
            _db_resolve_plugin("myfile.sqlite")
        MockSq.assert_called_once_with(path="myfile.sqlite", read_only=True)

    def test_resolve_sqlite_memory(self):
        with patch("daita.plugins.sqlite.SQLitePlugin") as MockSq:
            _db_resolve_plugin(":memory:")
        MockSq.assert_called_once_with(path=":memory:", read_only=True)

    def test_resolve_snowflake_rejected(self):
        with pytest.raises(ValueError, match="Snowflake"):
            _db_resolve_plugin("snowflake://account/db")

    def test_resolve_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unsupported scheme"):
            _db_resolve_plugin("redis://host/0")

    def test_resolve_db_plugin_instance(self):
        from daita.plugins.base_db import BaseDatabasePlugin

        mock_plugin = MagicMock(spec=BaseDatabasePlugin)
        plugin, created = _db_resolve_plugin(mock_plugin)
        assert plugin is mock_plugin
        assert created is False


# ---------------------------------------------------------------------------
# Schema normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeDbSchema:
    def test_normalize_postgresql(self):
        raw = {
            "database_type": "postgresql",
            "schema": "public",
            "tables": [{"table_name": "orders", "row_count": 1000}],
            "columns": [
                {
                    "table_name": "orders",
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": "NO",
                },
                {
                    "table_name": "orders",
                    "column_name": "total",
                    "data_type": "numeric",
                    "is_nullable": "YES",
                },
            ],
            "primary_keys": [{"table_name": "orders", "column_name": "id"}],
            "foreign_keys": [
                {
                    "source_table": "orders",
                    "source_column": "customer_id",
                    "target_table": "customers",
                    "target_column": "id",
                }
            ],
            "table_count": 1,
        }
        result = _db_normalize_schema(raw)
        assert result["database_type"] == "postgresql"
        assert result["database_name"] == "public"
        assert len(result["tables"]) == 1
        tbl = result["tables"][0]
        assert tbl["name"] == "orders"
        assert tbl["row_count"] == 1000
        id_col = next(c for c in tbl["columns"] if c["name"] == "id")
        assert id_col["is_primary_key"] is True
        assert id_col["nullable"] is False
        total_col = next(c for c in tbl["columns"] if c["name"] == "total")
        assert total_col["is_primary_key"] is False
        assert total_col["nullable"] is True
        assert len(result["foreign_keys"]) == 1
        assert result["foreign_keys"][0]["source_table"] == "orders"

    def test_normalize_mysql(self):
        raw = {
            "database_type": "mysql",
            "schema": "mydb",
            "tables": [{"table_name": "users", "row_count": 500}],
            "columns": [
                {
                    "table_name": "users",
                    "column_name": "id",
                    "data_type": "int",
                    "is_nullable": "NO",
                    "column_key": "PRI",
                },
                {
                    "table_name": "users",
                    "column_name": "email",
                    "data_type": "varchar",
                    "is_nullable": "YES",
                    "column_key": "",
                },
            ],
            "foreign_keys": [],
            "table_count": 1,
        }
        result = _db_normalize_schema(raw)
        assert result["database_type"] == "mysql"
        assert result["database_name"] == "mydb"
        tbl = result["tables"][0]
        id_col = next(c for c in tbl["columns"] if c["name"] == "id")
        assert id_col["is_primary_key"] is True
        email_col = next(c for c in tbl["columns"] if c["name"] == "email")
        assert email_col["is_primary_key"] is False

    def test_normalize_mongodb(self):
        raw = {
            "database_type": "mongodb",
            "database": "analytics",
            "collections": [
                {
                    "collection_name": "events",
                    "document_count": 50000,
                    "fields": [
                        {"field_name": "_id", "types": ["ObjectId"], "sample_count": 100},
                        {"field_name": "user_id", "types": ["str"], "sample_count": 100},
                    ],
                }
            ],
            "collection_count": 1,
        }
        result = _db_normalize_schema(raw)
        assert result["database_type"] == "mongodb"
        assert result["database_name"] == "analytics"
        assert len(result["tables"]) == 1
        tbl = result["tables"][0]
        assert tbl["name"] == "events"
        assert tbl["row_count"] == 50000
        id_col = next(c for c in tbl["columns"] if c["name"] == "_id")
        assert id_col["is_primary_key"] is True
        user_col = next(c for c in tbl["columns"] if c["name"] == "user_id")
        assert user_col["is_primary_key"] is False
        assert result["foreign_keys"] == []


# ---------------------------------------------------------------------------
# Domain inference tests
# ---------------------------------------------------------------------------


class TestInferDomain:
    def test_infer_ecommerce(self):
        schema = _make_normalized_schema(
            tables=[_table("orders"), _table("customers"), _table("products")]
        )
        assert _infer_domain(schema) == "e-commerce"

    def test_infer_crm(self):
        schema = _make_normalized_schema(
            tables=[_table("contacts"), _table("leads"), _table("opportunities")]
        )
        assert _infer_domain(schema) == "CRM"

    def test_infer_general_below_threshold(self):
        schema = _make_normalized_schema(
            tables=[_table("foo"), _table("bar"), _table("baz")]
        )
        assert _infer_domain(schema) == "general-purpose"

    def test_infer_general_single_match(self):
        # Only one keyword match — below threshold of 2
        schema = _make_normalized_schema(tables=[_table("orders"), _table("misc")])
        result = _infer_domain(schema)
        # With only one e-commerce keyword, should fall back to general-purpose
        assert result == "general-purpose"

    def test_infer_empty_schema(self):
        schema = _make_normalized_schema(tables=[])
        assert _infer_domain(schema) == "general-purpose"

    def test_infer_hr(self):
        schema = _make_normalized_schema(
            tables=[_table("employees"), _table("departments"), _table("payroll")]
        )
        assert _infer_domain(schema) == "HR"


# ---------------------------------------------------------------------------
# Prompt generation tests
# ---------------------------------------------------------------------------


class TestBuildDbPrompt:
    def _cols(self, names):
        return [
            {"name": n, "type": "text", "nullable": True, "is_primary_key": i == 0}
            for i, n in enumerate(names)
        ]

    def test_prompt_small_schema_has_column_types(self):
        tables = [
            _table("orders", self._cols(["id", "total", "status"]), row_count=5000),
            _table("customers", self._cols(["id", "email"]), row_count=1000),
        ]
        schema = _make_normalized_schema(tables=tables)
        prompt = _db_build_prompt(schema, "e-commerce", None)
        # Full detail tier: should have markdown table headers
        assert "| Column | Type | PK | Nullable |" in prompt
        assert "orders" in prompt
        assert "customers" in prompt
        assert "5,000 rows" in prompt

    def test_prompt_large_schema_summary_only(self):
        tables = [_table(f"table_{i}", self._cols(["id", "name"]), row_count=1000) for i in range(100)]
        schema = _make_normalized_schema(tables=tables)
        schema["table_count"] = 100
        prompt = _db_build_prompt(schema, "general-purpose", None)
        # Summary tier: no markdown table headers
        assert "| Column | Type | PK | Nullable |" not in prompt
        assert "Columns:" not in prompt
        # Should have summary lines
        assert "1K rows" in prompt

    def test_prompt_medium_schema_column_names_only(self):
        tables = [_table(f"tbl_{i}", self._cols(["id", "name"])) for i in range(40)]
        schema = _make_normalized_schema(tables=tables)
        schema["table_count"] = 40
        prompt = _db_build_prompt(schema, "general-purpose", None)
        assert "Columns:" in prompt
        assert "| Column | Type | PK | Nullable |" not in prompt

    def test_prompt_includes_fk(self):
        tables = [_table("orders"), _table("customers")]
        fks = [
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ]
        schema = _make_normalized_schema(tables=tables, fks=fks)
        prompt = _db_build_prompt(schema, "e-commerce", None)
        assert "orders.customer_id" in prompt
        assert "customers.id" in prompt

    def test_prompt_no_fk_message(self):
        schema = _make_normalized_schema(tables=[_table("foo")])
        prompt = _db_build_prompt(schema, "general-purpose", None)
        assert "No foreign key relationships discovered." in prompt

    def test_prompt_with_user_prompt(self):
        schema = _make_normalized_schema(tables=[_table("foo")])
        prompt = _db_build_prompt(schema, "general-purpose", "Focus on sales metrics only.")
        assert "Focus on sales metrics only." in prompt

    def test_prompt_empty_database(self):
        schema = _make_normalized_schema(tables=[])
        prompt = _db_build_prompt(schema, "general-purpose", None)
        assert "empty" in prompt.lower()

    def test_prompt_header_includes_domain_and_type(self):
        schema = _make_normalized_schema(tables=[_table("t")], db_type="postgresql")
        prompt = _db_build_prompt(schema, "financial", None)
        assert "financial" in prompt
        assert "postgresql" in prompt


# ---------------------------------------------------------------------------
# Integration tests (fully mocked)
# ---------------------------------------------------------------------------


class TestFromDbIntegration:
    """End-to-end from_db() tests using mocked plugins and CatalogPlugin."""

    def _pg_raw_schema(self):
        return {
            "database_type": "postgresql",
            "schema": "public",
            "tables": [{"table_name": "users", "row_count": 100}],
            "columns": [
                {
                    "table_name": "users",
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": "NO",
                }
            ],
            "primary_keys": [{"table_name": "users", "column_name": "id"}],
            "foreign_keys": [],
            "table_count": 1,
        }

    async def test_from_db_end_to_end(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        raw = self._pg_raw_schema()
        normalized = _db_normalize_schema(raw)

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_plugin.disconnect = AsyncMock()

        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        # Patch: plugin resolution, schema discovery, and Agent construction.
        # Agent is imported lazily inside from_db() via `from .agent import Agent`,
        # so we patch it at daita.agents.agent.Agent.
        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        # Plugin connect was called
        mock_plugin.connect.assert_awaited_once()
        # add_plugin was called with our plugin
        mock_agent.add_plugin.assert_called_once_with(mock_plugin)
        # _db_schema stored on agent
        assert mock_agent._db_schema["database_type"] == "postgresql"
        assert mock_agent._db_schema["tables"][0]["name"] == "users"
        # _db_plugin stored on agent
        assert mock_agent._db_plugin is mock_plugin

    async def test_from_db_connect_failure_raises_agent_error(self):
        from daita.agents.db import from_db
        from daita.core.exceptions import AgentError
        import daita.agents.db.builder as fac

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_plugin.disconnect = AsyncMock()

        with patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)):
            with pytest.raises(AgentError, match="Failed to connect to database"):
                await from_db("postgresql://localhost/testdb")

        # Disconnect should have been called for cleanup
        mock_plugin.disconnect.assert_awaited_once()

    async def test_from_db_discovery_failure_raises_agent_error(self):
        from daita.agents.db import from_db
        from daita.core.exceptions import AgentError
        import daita.agents.db.builder as fac

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_plugin.disconnect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(side_effect=RuntimeError("schema error"))),
        ):
            with pytest.raises(AgentError, match="Schema discovery failed"):
                await from_db("postgresql://localhost/testdb")

        mock_plugin.disconnect.assert_awaited_once()

    async def test_from_db_forwards_kwargs_to_agent(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        raw = self._pg_raw_schema()
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=_db_normalize_schema(raw))),
            patch("daita.agents.agent.Agent", return_value=mock_agent) as MockAgent,
        ):
            await from_db(
                "postgresql://localhost/testdb",
                name="my agent",
                model="gpt-4o",
                api_key="sk-test",
                llm_provider="openai",
            )

        call_kwargs = MockAgent.call_args[1]
        assert call_kwargs["name"] == "my agent"
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["llm_provider"] == "openai"

    async def test_from_db_plugin_instance_not_disconnected_on_error(self):
        """When user passes a plugin instance, we don't clean it up on error."""
        from daita.agents.db import from_db
        from daita.core.exceptions import AgentError
        from daita.plugins.base_db import BaseDatabasePlugin
        import daita.agents.db.builder as fac

        mock_plugin = MagicMock(spec=BaseDatabasePlugin)
        mock_plugin.connect = AsyncMock(side_effect=RuntimeError("fail"))
        mock_plugin.disconnect = AsyncMock()

        with patch.object(fac, "resolve_plugin", return_value=(mock_plugin, False)):
            with pytest.raises(AgentError):
                await from_db(mock_plugin)

        # We did NOT create the plugin, so we must not disconnect it
        mock_plugin.disconnect.assert_not_awaited()

    async def test_from_db_schema_stored_with_table_names(self):
        """Verify that _db_schema on the agent contains the discovered table names."""
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        raw = self._pg_raw_schema()
        normalized = _db_normalize_schema(raw)

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb")

        assert mock_agent._db_schema["tables"][0]["name"] == "users"


# ---------------------------------------------------------------------------
# Lineage integration tests
# ---------------------------------------------------------------------------


class TestFromDbLineage:
    def _schema_with_fks(self):
        return _make_normalized_schema(
            tables=[_table("orders"), _table("customers")],
            fks=[
                {
                    "source_table": "orders",
                    "source_column": "customer_id",
                    "target_table": "customers",
                    "target_column": "id",
                }
            ],
        )

    def _base_mocks(self):
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        return mock_plugin, mock_agent

    async def test_lineage_true_seeds_fks(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = self._schema_with_fks()
        mock_plugin, mock_agent = self._base_mocks()
        mock_lineage = MagicMock()
        mock_lineage.register_flow = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch("daita.plugins.lineage.LineagePlugin", return_value=mock_lineage),
        ):
            await from_db("postgresql://localhost/testdb", lineage=True)

        mock_agent.add_plugin.assert_any_call(mock_lineage)
        mock_lineage.register_flow.assert_awaited_once()
        call_kwargs = mock_lineage.register_flow.call_args[1]
        assert call_kwargs["source_id"] == "table:orders"
        assert call_kwargs["target_id"] == "table:customers"
        assert mock_agent._db_lineage is mock_lineage

    async def test_lineage_plugin_instance_used_directly(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = self._schema_with_fks()
        mock_plugin, mock_agent = self._base_mocks()
        custom_lineage = MagicMock()
        custom_lineage.register_flow = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb", lineage=custom_lineage)

        mock_agent.add_plugin.assert_any_call(custom_lineage)
        assert mock_agent._db_lineage is custom_lineage

    async def test_lineage_none_skipped(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = self._schema_with_fks()
        mock_plugin, mock_agent = self._base_mocks()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb")

        # add_plugin called only once (for the DB plugin), not for lineage
        assert mock_agent.add_plugin.call_count == 1

    async def test_lineage_entity_id_format(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[_table("invoices"), _table("accounts")],
            fks=[
                {
                    "source_table": "invoices",
                    "source_column": "account_id",
                    "target_table": "accounts",
                    "target_column": "id",
                }
            ],
        )
        mock_plugin, mock_agent = self._base_mocks()
        mock_lineage = MagicMock()
        mock_lineage.register_flow = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch("daita.plugins.lineage.LineagePlugin", return_value=mock_lineage),
        ):
            await from_db("postgresql://localhost/testdb", lineage=True)

        call_kwargs = mock_lineage.register_flow.call_args[1]
        assert call_kwargs["source_id"].startswith("table:")
        assert call_kwargs["target_id"].startswith("table:")

    async def test_lineage_no_fks_no_register_calls(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")], fks=[])
        mock_plugin, mock_agent = self._base_mocks()
        mock_lineage = MagicMock()
        mock_lineage.register_flow = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch("daita.plugins.lineage.LineagePlugin", return_value=mock_lineage),
        ):
            await from_db("postgresql://localhost/testdb", lineage=True)

        mock_lineage.register_flow.assert_not_awaited()


# ---------------------------------------------------------------------------
# Memory integration tests
# ---------------------------------------------------------------------------


class TestFromDbMemory:
    def _base_mocks(self):
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        return mock_plugin, mock_agent

    async def test_memory_true_creates_with_workspace(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[_table("orders"), _table("customers"), _table("products")]
        )
        mock_plugin, mock_agent = self._base_mocks()
        mock_memory = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch("daita.plugins.memory.MemoryPlugin", return_value=mock_memory) as MockMem,
        ):
            await from_db("postgresql://localhost/testdb", memory=True)

        MockMem.assert_called_once()
        call_kwargs = MockMem.call_args[1]
        assert "workspace" in call_kwargs
        assert call_kwargs["workspace"]  # non-empty
        mock_agent.add_plugin.assert_any_call(mock_memory)
        assert mock_agent._db_memory is mock_memory

    async def test_memory_plugin_instance_used_directly(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        custom_memory = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb", memory=custom_memory)

        mock_agent.add_plugin.assert_any_call(custom_memory)
        assert mock_agent._db_memory is custom_memory

    async def test_memory_none_skipped(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb")

        assert mock_agent.add_plugin.call_count == 1  # only DB plugin


# ---------------------------------------------------------------------------
# Schema caching tests
# ---------------------------------------------------------------------------


class TestSchemaCache:
    def test_cache_key_redacts_password(self):
        key1 = _db_cache_key("postgresql://user:secret1@host/db")
        key2 = _db_cache_key("postgresql://user:secret2@host/db")
        assert key1 == key2

    def test_cache_key_different_hosts(self):
        key1 = _db_cache_key("postgresql://user:pass@host1/db")
        key2 = _db_cache_key("postgresql://user:pass@host2/db")
        assert key1 != key2

    def test_cache_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        schema = _make_normalized_schema(tables=[_table("users")])
        cache_key = "testkey123"

        _db_save_schema_cache(cache_key, schema)
        result = _db_load_cached_schema(cache_key, ttl=3600)

        assert result is not None
        loaded_schema, is_expired = result
        assert loaded_schema == schema
        assert is_expired is False

    async def test_cache_hit_skips_discovery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("orders"), _table("customers"), _table("products")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_discover = AsyncMock(return_value=schema)

        # Pre-populate cache
        source = "postgresql://user:pass@host/db"
        cache_key = _db_cache_key(source)
        _db_save_schema_cache(cache_key, schema)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", mock_discover),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=3600)

        mock_discover.assert_not_awaited()

    async def test_cache_expired_triggers_rediscovery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("orders"), _table("customers"), _table("products")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_discover = AsyncMock(return_value=schema)

        source = "postgresql://user:pass@host/db"
        cache_key = _db_cache_key(source)
        cache_dir = tmp_path / ".daita" / "schema_cache"
        cache_dir.mkdir(parents=True)
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        (cache_dir / f"{cache_key}.json").write_text(
            json.dumps({"schema": schema, "cached_at": old_ts, "cache_key": cache_key})
        )

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", mock_discover),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=0)

        mock_discover.assert_awaited_once()

    def test_drift_detection_added_table(self):
        old = _make_normalized_schema(tables=[_table("users"), _table("orders")])
        new = _make_normalized_schema(tables=[_table("users"), _table("orders"), _table("products")])
        drift = _db_detect_drift(old, new)
        assert drift is not None
        assert "products" in drift["added_tables"]
        assert drift["removed_tables"] == []

    def test_drift_detection_removed_column(self):
        old_cols = [
            {"name": "id", "type": "integer", "nullable": False, "is_primary_key": True},
            {"name": "email", "type": "text", "nullable": True, "is_primary_key": False},
        ]
        new_cols = [
            {"name": "id", "type": "integer", "nullable": False, "is_primary_key": True},
        ]
        old = _make_normalized_schema(tables=[{"name": "users", "columns": old_cols, "row_count": None}])
        new = _make_normalized_schema(tables=[{"name": "users", "columns": new_cols, "row_count": None}])
        drift = _db_detect_drift(old, new)
        assert drift is not None
        change = next(c for c in drift["column_changes"] if c["table"] == "users")
        assert "email" in change["removed_columns"]

    def test_drift_detection_no_change(self):
        schema = _make_normalized_schema(tables=[_table("users"), _table("orders")])
        drift = _db_detect_drift(schema, schema)
        assert drift is None

    async def test_expired_cache_fallback_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("orders"), _table("customers"), _table("products")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        source = "postgresql://user:pass@host/db"
        cache_key = _db_cache_key(source)
        cache_dir = tmp_path / ".daita" / "schema_cache"
        cache_dir.mkdir(parents=True)
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        (cache_dir / f"{cache_key}.json").write_text(
            json.dumps({"schema": schema, "cached_at": old_ts, "cache_key": cache_key})
        )

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(side_effect=RuntimeError("DB down"))),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=0)

        assert mock_agent._db_schema == schema


# ---------------------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------------------


class TestFromDbAuditLog:
    def _base_mocks(self):
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        return mock_plugin, mock_agent

    async def test_audit_log_initialised_empty(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(return_value={"result": "ok", "tool_calls": []})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        assert hasattr(agent, "_db_audit_log")
        assert agent._db_audit_log == []

    async def test_audit_log_accumulates_across_runs(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()

        tool_calls_run1 = [{"tool": "postgres_query", "arguments": {"sql": "SELECT 1"}, "result": {}}]
        tool_calls_run2 = [{"tool": "postgres_query", "arguments": {"sql": "SELECT 2"}, "result": {}}]
        mock_agent.run = AsyncMock(side_effect=[
            {"result": "first", "tool_calls": tool_calls_run1},
            {"result": "second", "tool_calls": tool_calls_run2},
        ])

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        await agent.run("first question")
        await agent.run("second question")

        assert len(agent._db_audit_log) == 2
        assert agent._db_audit_log[0]["prompt"] == "first question"
        assert agent._db_audit_log[0]["tool_calls"] == tool_calls_run1
        assert agent._db_audit_log[1]["prompt"] == "second question"
        assert agent._db_audit_log[1]["tool_calls"] == tool_calls_run2

    async def test_audit_log_entry_has_timestamp(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(return_value={"result": "ok", "tool_calls": []})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        await agent.run("test")

        entry = agent._db_audit_log[0]
        assert "timestamp" in entry
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(entry["timestamp"])
        assert dt.tzinfo is not None

    async def test_run_return_value_is_string(self):
        """Wrapping run() must return the plain string result, not the detailed dict."""
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(
            return_value={"result": "42 rows found", "tool_calls": [], "tokens": {}, "cost": 0.0}
        )

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        result = await agent.run("anything")
        assert result == "42 rows found"


# ---------------------------------------------------------------------------
# ConversationHistory integration tests
# ---------------------------------------------------------------------------


class TestFromDbHistory:
    def _base_mocks(self):
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_agent._tool_call_history = []
        return mock_plugin, mock_agent

    async def test_history_true_creates_conversation_history(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db
        from daita.agents.conversation import ConversationHistory

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(return_value={"result": "ok", "tool_calls": []})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", history=True)

        assert hasattr(agent, "_db_history")
        assert isinstance(agent._db_history, ConversationHistory)

    async def test_history_true_auto_injected_into_run(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        original_run = AsyncMock(return_value={"result": "ok", "tool_calls": []})
        mock_agent.run = original_run

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", history=True)

        await agent.run("first question")
        await agent.run("second question")

        # The underlying run() should have received the history kwarg both times
        calls = original_run.call_args_list
        assert all("history" in c.kwargs for c in calls)
        h0 = calls[0].kwargs["history"]
        h1 = calls[1].kwargs["history"]
        assert h0 is h1  # same ConversationHistory object

    async def test_history_custom_instance_used_directly(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db
        from daita.agents.conversation import ConversationHistory

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(return_value={"result": "ok", "tool_calls": []})
        custom_history = ConversationHistory(max_turns=5)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", history=custom_history)

        assert agent._db_history is custom_history

    async def test_history_none_no_db_history_attr(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(return_value={"result": "ok", "tool_calls": []})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        # MagicMock auto-generates attributes on access; check __dict__ for explicit assignment
        assert "_db_history" not in mock_agent.__dict__

    async def test_per_call_override_suppresses_auto_injection(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        original_run = AsyncMock(return_value={"result": "ok", "tool_calls": []})
        mock_agent.run = original_run

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", history=True)

        # Explicitly passing history=None should override auto-injection
        await agent.run("standalone question", history=None)

        call_kwargs = original_run.call_args.kwargs
        assert "history" in call_kwargs
        assert call_kwargs["history"] is None


# ---------------------------------------------------------------------------
# Analyst toolkit tests
# ---------------------------------------------------------------------------


def _analyst_schema():
    """Minimal schema with customers → orders FK for analyst tool tests."""
    return _make_normalized_schema(
        tables=[
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "type": "integer", "nullable": False, "is_primary_key": True},
                    {"name": "name", "type": "varchar", "nullable": True, "is_primary_key": False},
                ],
                "row_count": 100,
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "type": "integer", "nullable": False, "is_primary_key": True},
                    {"name": "customer_id", "type": "integer", "nullable": False, "is_primary_key": False},
                    {"name": "total", "type": "numeric", "nullable": True, "is_primary_key": False},
                ],
                "row_count": 500,
            },
        ],
        fks=[
            {
                "source_table": "orders",
                "source_column": "customer_id",
                "target_table": "customers",
                "target_column": "id",
            }
        ],
    )


def _mock_plugin_with_query(rows):
    """Return a mock plugin whose query() returns rows."""
    p = MagicMock()
    p.query = AsyncMock(return_value=rows)
    return p


class TestRegisterAnalystTools:
    """register_analyst_tools registers all 6 tool names."""

    def test_all_six_tools_registered(self):
        from daita.agents.db.tools import register_analyst_tools
        from daita.core.tools import ToolRegistry

        schema = _analyst_schema()
        plugin = _mock_plugin_with_query([])

        # Build a minimal agent-like object with a real ToolRegistry
        class FakeAgent:
            tool_registry = ToolRegistry()

        agent = FakeAgent()
        register_analyst_tools(agent, plugin, schema)

        registered = set(agent.tool_registry.tool_names)
        expected = {
            "pivot_table",
            "correlate",
            "detect_anomalies",
            "compare_entities",
            "find_similar",
            "forecast_trend",
        }
        assert expected.issubset(registered)


class TestPivotTableTool:
    async def test_basic_pivot(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.pivot_table import create_pivot_table_tool

        rows = [
            {"category": "A", "month": "Jan", "revenue": 100},
            {"category": "A", "month": "Feb", "revenue": 200},
            {"category": "B", "month": "Jan", "revenue": 150},
            {"category": "B", "month": "Feb", "revenue": 50},
        ]
        plugin = _mock_plugin_with_query(rows)
        tool = create_pivot_table_tool(plugin, _analyst_schema())

        result = await tool.handler({
            "sql": "SELECT * FROM orders",
            "rows": "category",
            "columns": "month",
            "values": "revenue",
            "aggfunc": "sum",
        })

        assert result["success"] is True
        assert result["row_count"] > 0
        assert "pivot" in result

    async def test_missing_sql_returns_error(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.pivot_table import create_pivot_table_tool

        plugin = _mock_plugin_with_query([])
        tool = create_pivot_table_tool(plugin, _analyst_schema())
        result = await tool.handler({"rows": "a", "columns": "b", "values": "c"})
        assert result["success"] is False
        assert "sql" in result["error"]

    async def test_graceful_degradation_without_pandas(self):
        from daita.agents.db.tools.pivot_table import create_pivot_table_tool

        plugin = _mock_plugin_with_query([])
        tool = create_pivot_table_tool(plugin, _analyst_schema())

        with patch("daita.agents.db.tools.pivot_table.ensure_pandas", side_effect=ImportError("pandas not found")):
            result = await tool.handler({
                "sql": "SELECT 1",
                "rows": "a",
                "columns": "b",
                "values": "c",
            })
        assert result["success"] is False
        assert "pandas" in result["error"].lower() or "not found" in result["error"].lower()


class TestCorrelateTool:
    async def test_basic_correlation(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.correlate import create_correlate_tool

        rows = [{"x": i, "y": i * 2, "z": -i} for i in range(1, 21)]
        plugin = _mock_plugin_with_query(rows)
        tool = create_correlate_tool(plugin, _analyst_schema())

        result = await tool.handler({"sql": "SELECT x, y, z FROM t"})

        assert result["success"] is True
        assert len(result["correlations"]) > 0
        # x and y should be perfectly correlated
        xy = next((p for p in result["correlations"] if set([p["column_a"], p["column_b"]]) == {"x", "y"}), None)
        assert xy is not None
        assert abs(xy["correlation"]) > 0.99

    async def test_missing_sql_returns_error(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.correlate import create_correlate_tool

        plugin = _mock_plugin_with_query([])
        tool = create_correlate_tool(plugin, _analyst_schema())
        result = await tool.handler({})
        assert result["success"] is False


class TestDetectAnomaliesTool:
    async def test_zscore_detects_outlier(self):
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from daita.agents.db.tools.detect_anomalies import create_detect_anomalies_tool

        # Values mostly 10, with one extreme outlier at 1000
        rows = [{"val": 10 + i % 3, "id": i} for i in range(50)]
        rows.append({"val": 1000, "id": 999})
        plugin = _mock_plugin_with_query(rows)
        tool = create_detect_anomalies_tool(plugin, _analyst_schema())

        result = await tool.handler({"sql": "SELECT * FROM orders", "column": "val"})

        assert result["success"] is True
        assert result["anomaly_count"] >= 1
        ids = [r["id"] for r in result["anomalies"]]
        assert 999 in ids

    async def test_iqr_method(self):
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from daita.agents.db.tools.detect_anomalies import create_detect_anomalies_tool

        rows = [{"val": 5, "id": i} for i in range(40)]
        rows.append({"val": 200, "id": 999})
        plugin = _mock_plugin_with_query(rows)
        tool = create_detect_anomalies_tool(plugin, _analyst_schema())

        result = await tool.handler({"sql": "SELECT * FROM t", "column": "val", "method": "iqr"})
        assert result["success"] is True
        assert result["anomaly_count"] >= 1

    async def test_graceful_degradation_without_numpy(self):
        from daita.agents.db.tools.detect_anomalies import create_detect_anomalies_tool

        plugin = _mock_plugin_with_query([])
        tool = create_detect_anomalies_tool(plugin, _analyst_schema())

        with patch("daita.agents.db.tools.detect_anomalies.ensure_numpy", side_effect=ImportError("numpy not found")):
            result = await tool.handler({"sql": "SELECT 1", "column": "val"})
        assert result["success"] is False


class TestHelpersInferDimensions:
    def test_infer_dimensions_with_fk_schema(self):
        from daita.agents.db.tools._helpers import infer_dimensions

        schema = _analyst_schema()
        dims = infer_dimensions(schema, "customers")

        aliases = [d["alias"] for d in dims]
        assert "orders_count" in aliases
        # Should include at least the count
        assert len(dims) >= 1

    def test_infer_dimensions_no_fk(self):
        from daita.agents.db.tools._helpers import infer_dimensions

        schema = _make_normalized_schema(
            tables=[_table("standalone")],
            fks=[],
        )
        dims = infer_dimensions(schema, "standalone")
        assert dims == []

    def test_get_pk_column(self):
        from daita.agents.db.tools._helpers import get_pk_column

        schema = _analyst_schema()
        assert get_pk_column(schema, "customers") == "id"
        assert get_pk_column(schema, "orders") == "id"
        assert get_pk_column(schema, "nonexistent") is None

    def test_get_numeric_columns(self):
        from daita.agents.db.tools._helpers import get_numeric_columns

        schema = _analyst_schema()
        numeric = get_numeric_columns(schema, "orders")
        assert "total" in numeric
        # id is integer — also numeric
        assert "id" in numeric

    def test_to_serializable(self):
        from decimal import Decimal
        from datetime import date
        from daita.agents.db.tools._helpers import to_serializable

        assert to_serializable(Decimal("3.14")) == pytest.approx(3.14)
        assert to_serializable(date(2024, 1, 15)) == "2024-01-15"
        assert to_serializable("plain") == "plain"
        assert to_serializable(42) == 42


class TestForecastTrendTool:
    async def test_linear_trend_up(self):
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from daita.agents.db.tools.forecast_trend import create_forecast_trend_tool

        rows = [
            {"month": f"2024-{str(i).zfill(2)}-01", "revenue": 1000 + i * 200}
            for i in range(1, 13)
        ]
        plugin = _mock_plugin_with_query(rows)
        tool = create_forecast_trend_tool(plugin, _analyst_schema())

        result = await tool.handler({
            "sql": "SELECT month, revenue FROM orders",
            "date_column": "month",
            "metric_column": "revenue",
            "periods": 3,
        })

        assert result["success"] is True
        assert result["trend"]["direction"] == "up"
        assert len(result["forecast"]) == 3
        assert result["trend"]["r_squared"] > 0.9

    async def test_insufficient_data(self):
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from daita.agents.db.tools.forecast_trend import create_forecast_trend_tool

        plugin = _mock_plugin_with_query([{"d": "2024-01-01", "v": 100}])
        tool = create_forecast_trend_tool(plugin, _analyst_schema())

        result = await tool.handler({
            "sql": "SELECT d, v FROM t",
            "date_column": "d",
            "metric_column": "v",
        })
        assert result["success"] is False
        assert "2 data points" in result["error"]


class TestToolkitParam:
    """toolkit=None disables analyst tool registration."""

    async def test_toolkit_none_skips_registration(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import ToolRegistry

        schema = _analyst_schema()
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        real_registry = ToolRegistry()
        mock_agent.tool_registry = real_registry

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch.object(fac, "sample_numeric_columns", AsyncMock()),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb", toolkit=None)

        analyst_tools = {
            "pivot_table", "correlate", "detect_anomalies",
            "compare_entities", "find_similar", "forecast_trend",
        }
        registered = set(real_registry.tool_names)
        assert analyst_tools.isdisjoint(registered), (
            f"Expected no analyst tools but found: {analyst_tools & registered}"
        )
