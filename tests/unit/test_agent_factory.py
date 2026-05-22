"""
Tests for daita/agents/db/ — Agent.from_db() builder.

All tests use mocked DB connections; no real databases are required.
Run with: pytest tests/unit/test_agent_factory.py -v
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from daita.agents.db.catalog_freshness import (
    catalog_profile_key as _db_catalog_profile_key,
    detect_profile_drift as _db_detect_profile_drift,
    load_catalog_profile_snapshot as _db_load_catalog_profile_snapshot,
)
from daita.agents.db.catalog_profile import normalize_schema as _db_normalize_schema
from daita.agents.db.prompt import (
    build_prompt as _db_build_prompt,
    infer_domain as _infer_domain,
)
from daita.agents.db.resolve import resolve_plugin as _db_resolve_plugin
from daita.agents.db.catalog_summary import build_db_summary as _db_build_summary
from daita.plugins.catalog import CatalogPlugin
from daita.agents.db.monitors import normalize_monitor_definition
from daita.agents.db.findings import normalize_finding
from daita.agents.db.memory import (
    DBMemory,
    DBMemoryRecord,
    calibrate_db_memory,
    normalize_db_memory_record,
    recall_db_memory_context,
)
from daita.agents.db.config.policies import SchemaPromptPolicy, ToolResultPolicy
from daita.plugins.catalog.base_profiler import NormalizedSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_normalized_schema(
    tables=None, fks=None, db_type="postgresql", db_name="public"
):
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
    columns = columns or [
        {"name": "id", "type": "integer", "nullable": False, "is_primary_key": True}
    ]
    return {"name": name, "columns": columns, "row_count": row_count}


def _agent_with_catalog(schema, **attrs):
    store_id = schema.get("store_id") or "test-store"
    catalog_schema = dict(schema)
    catalog_schema["store_id"] = store_id
    catalog = CatalogPlugin()
    catalog._schemas[store_id] = NormalizedSchema.from_dict(catalog_schema)
    return SimpleNamespace(
        _db_catalog=catalog,
        _db_catalog_store_id=store_id,
        **attrs,
    )


async def _attach_plugin_catalog(plugin, schema):
    catalog = CatalogPlugin()
    registered = await catalog.register_schema(
        schema, store_type=schema.get("database_type", "unknown")
    )
    plugin._db_catalog = catalog
    plugin._db_catalog_store_id = registered["store_id"]
    return catalog, registered["store_id"]


class TestFromDbJoinPathNavigation:
    async def test_catalog_inspect_table_uses_catalog_state(self):
        catalog = CatalogPlugin()
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "users",
                    columns=[
                        {"name": "id", "type": "integer"},
                        {"name": "email", "type": "text"},
                    ],
                )
            ]
        )
        registered = await catalog.register_schema(schema, store_type="postgresql")

        result = catalog.get_table_schema(registered["store_id"], "users")

        assert result["success"] is True
        assert result["columns"][0]["name"] == "id"
        assert result["columns"][1]["name"] == "email"

    async def test_catalog_find_join_path_returns_sql_ready_predicates(self):
        catalog = CatalogPlugin()
        schema = _make_normalized_schema(
            tables=[
                _table("operations"),
                _table("api_keys"),
                _table("users"),
            ],
            fks=[
                {
                    "source_table": "operations",
                    "source_column": "api_key_id",
                    "target_table": "api_keys",
                    "target_column": "api_key_id",
                },
                {
                    "source_table": "api_keys",
                    "source_column": "created_by",
                    "target_table": "users",
                    "target_column": "user_id",
                },
            ],
        )
        registered = await catalog.register_schema(schema, store_type="postgresql")

        result = catalog.find_relationship_paths(
            registered["store_id"],
            from_assets=["operations"],
            to_assets=["users"],
        )

        assert result["success"] is True
        assert result["reachable"] is True
        assert result["path_count"] == 1
        path = result["paths"][0]
        assert path["tables"] == ["operations", "api_keys", "users"]
        assert [join["predicate"] for join in path["joins"]] == [
            "operations.api_key_id = api_keys.api_key_id",
            "api_keys.created_by = users.user_id",
        ]

    async def test_catalog_find_join_path_warns_for_membership_bridge_paths(self):
        catalog = CatalogPlugin()
        schema = _make_normalized_schema(
            tables=[
                _table("operations"),
                _table("organization"),
                _table("organization_members"),
                _table("users"),
            ],
            fks=[
                {
                    "source_table": "operations",
                    "source_column": "organization_id",
                    "target_table": "organization",
                    "target_column": "organization_id",
                },
                {
                    "source_table": "organization_members",
                    "source_column": "organization_id",
                    "target_table": "organization",
                    "target_column": "organization_id",
                },
                {
                    "source_table": "organization_members",
                    "source_column": "user_id",
                    "target_table": "users",
                    "target_column": "user_id",
                },
            ],
        )
        registered = await catalog.register_schema(schema, store_type="postgresql")

        result = catalog.find_relationship_paths(
            registered["store_id"],
            from_assets=["operations"],
            to_assets=["users"],
        )

        assert result["reachable"] is True
        assert result["paths"][0]["tables"] == [
            "operations",
            "organization",
            "organization_members",
            "users",
        ]
        assert result["paths"][0]["warnings"]

    async def test_catalog_find_join_path_reports_unknown_tables_with_candidates(self):
        catalog = CatalogPlugin()
        schema = _make_normalized_schema(tables=[_table("users")])
        registered = await catalog.register_schema(schema, store_type="postgresql")

        result = catalog.find_relationship_paths(
            registered["store_id"],
            from_assets=["user"],
            to_assets=["users"],
        )

        assert result["success"] is False
        assert result["reachable"] is False
        assert result["errors"][0]["error"] == "asset not found"
        assert result["errors"][0]["candidates"]


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
        MockPg.assert_called_once_with(
            connection_string="postgresql://user:pass@host/db", read_only=True
        )

    def test_resolve_postgres_scheme(self):
        with patch("daita.plugins.postgresql.PostgreSQLPlugin") as MockPg:
            plugin, created = _db_resolve_plugin("postgres://user:pass@host/db")
        assert created is True
        MockPg.assert_called_once_with(
            connection_string="postgres://user:pass@host/db", read_only=True
        )

    def test_resolve_mysql_string(self):
        with patch("daita.plugins.mysql.MySQLPlugin") as MockMy:
            plugin, created = _db_resolve_plugin("mysql://user:pass@host/db")
        assert created is True
        MockMy.assert_called_once_with(
            connection_string="mysql://user:pass@host/db", read_only=True
        )

    def test_resolve_mongodb_string(self):
        with patch("daita.plugins.mongodb.MongoDBPlugin") as MockMongo:
            plugin, created = _db_resolve_plugin("mongodb://host/mydb")
        assert created is True
        MockMongo.assert_called_once_with(
            connection_string="mongodb://host/mydb", read_only=True
        )

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
                        {
                            "field_name": "_id",
                            "types": ["ObjectId"],
                            "sample_count": 100,
                        },
                        {
                            "field_name": "user_id",
                            "types": ["str"],
                            "sample_count": 100,
                        },
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
        tables = [
            _table(f"table_{i}", self._cols(["id", "name"]), row_count=1000)
            for i in range(100)
        ]
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
        prompt = _db_build_prompt(
            schema, "general-purpose", "Focus on sales metrics only."
        )
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

    def test_prompt_budget_uses_retrieval_for_wide_schemas(self):
        from daita.agents.db.prompt import build_prompt_result

        tables = [
            _table(
                f"wide_{i}",
                self._cols([f"col_{i}_{j}" for j in range(15)]),
            )
            for i in range(25)
        ]
        schema = _make_normalized_schema(tables=tables)
        schema["table_count"] = 25

        result = build_prompt_result(schema, "analytics", None)

        assert result.strategy == "retrieval"
        assert result.column_count == 375
        assert "Columns: col_0_0" not in result.prompt
        assert "| Column | Type | PK | Nullable |" not in result.prompt
        assert "- wide_0" in result.prompt

    def test_prompt_policies_are_config_importable(self):
        from daita.agents.db.config.policies import (
            SchemaPromptPolicy as PublicSchemaPromptPolicy,
        )
        from daita.agents.db.config.policies import (
            ToolResultPolicy as PublicToolResultPolicy,
        )

        assert PublicSchemaPromptPolicy is SchemaPromptPolicy
        assert PublicToolResultPolicy is ToolResultPolicy

    def test_invalid_policy_values_fail_fast(self):
        with pytest.raises(ValueError, match="max_inline_schema_tokens"):
            SchemaPromptPolicy(max_inline_schema_tokens=0)
        with pytest.raises(ValueError, match="max_result_tokens"):
            ToolResultPolicy(max_result_tokens=0)


class TestDbSummary:
    def test_summary_derives_health_questions_and_metrics(self):
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[
                        {"name": "id", "type": "integer", "is_primary_key": True},
                        {"name": "customer_id", "type": "integer"},
                        {"name": "total_amount", "type": "numeric"},
                        {"name": "updated_at", "type": "timestamp"},
                    ],
                    row_count=100,
                ),
                _table(
                    "customers",
                    columns=[
                        {"name": "id", "type": "integer", "is_primary_key": True},
                        {"name": "email", "type": "text"},
                    ],
                    row_count=20,
                ),
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

        summary = _db_build_summary(schema)

        assert "orders" in summary["fact_tables"]
        assert "customers" in summary["entity_tables"]
        assert summary["candidate_metrics"][0]["column"] == "total_amount"
        assert "suggested_questions" not in summary
        assert "suggested_monitors" not in summary

    def test_monitor_definition_validation_rejects_incomplete_definition(self):
        with pytest.raises(ValueError, match="requires sql"):
            normalize_monitor_definition(
                {
                    "name": "orders freshness",
                    "type": "freshness",
                    "severity": "warning",
                    "entity": {"table": "orders", "column": "updated_at"},
                    "threshold": {"max_age_hours": 24},
                }
            )


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
        # add_plugin was called with the DB plugin and the catalog plugin.
        assert mock_agent.add_plugin.call_args_list[0].args[0] is mock_plugin
        # _db_plugin stored on agent
        assert mock_agent._db_plugin is mock_plugin
        assert mock_agent._db_catalog is not None
        assert mock_agent._db_catalog_store_id
        catalog_schema = mock_agent._db_catalog.get_schema(
            mock_agent._db_catalog_store_id
        )
        assert catalog_schema.database_type == "postgresql"
        assert catalog_schema.tables[0].name == "users"

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

    async def test_from_db_discovery_failure_raises_agent_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        from daita.agents.db import from_db
        from daita.core.exceptions import AgentError
        import daita.agents.db.builder as fac

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_plugin.disconnect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(
                fac,
                "discover_schema",
                AsyncMock(side_effect=RuntimeError("schema error")),
            ),
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
            patch.object(
                fac,
                "discover_schema",
                AsyncMock(return_value=_db_normalize_schema(raw)),
            ),
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
        assert call_kwargs["temperature"] == 0

    async def test_from_db_preserves_explicit_temperature(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        raw = self._pg_raw_schema()
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(
                fac,
                "discover_schema",
                AsyncMock(return_value=_db_normalize_schema(raw)),
            ),
            patch("daita.agents.agent.Agent", return_value=mock_agent) as MockAgent,
        ):
            await from_db("postgresql://localhost/testdb", temperature=0.2)

        assert MockAgent.call_args[1]["temperature"] == 0.2

    async def test_from_db_uses_internal_prompt_budgeting(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(
            tables=[
                _table(
                    f"wide_{i}",
                    [{"name": f"col_{j}", "type": "text"} for j in range(15)],
                )
                for i in range(25)
            ]
        )
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

        assert mock_agent._db_prompt_metadata["strategy"] == "retrieval"

    async def test_from_db_catalog_prompt_policy_override_is_preserved(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(tables=[_table("orders")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(
                "postgresql://localhost/testdb",
                schema_prompt_policy=SchemaPromptPolicy(),
            )

        assert mock_agent._db_prompt_metadata["strategy"] == "full"

    async def test_from_db_tool_result_policy_is_attached(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(tables=[_table("orders")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        policy = ToolResultPolicy(max_rows_inline=3)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb", tool_result_policy=policy)

        assert mock_agent._db_tool_result_policy is policy

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

    async def test_from_db_catalog_stores_discovered_table_names(self):
        """Verify that catalog state contains the discovered table names."""
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

        catalog_schema = mock_agent._db_catalog.get_schema(
            mock_agent._db_catalog_store_id
        )
        assert catalog_schema.tables[0].name == "users"

    async def test_from_db_does_not_sample_values_by_default(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        normalized = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[{"name": "amount", "type": "numeric"}],
                )
            ]
        )

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
            patch.object(fac, "sample_numeric_columns", AsyncMock()) as sample,
        ):
            await from_db("postgresql://localhost/testdb")

        sample.assert_not_awaited()

    async def test_from_db_attaches_db_only_describe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[
                        {"name": "id", "type": "integer"},
                        {"name": "total", "type": "numeric"},
                    ],
                ),
                _table("customers"),
            ],
            fks=[
                {
                    "source_table": "orders",
                    "source_column": "customer_id",
                    "target_table": "customers",
                    "target_column": "id",
                }
            ],
            db_type="postgresql",
            db_name="warehouse",
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_plugin.sql_dialect = "postgresql"
        mock_plugin.database_name = "warehouse"

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db(
                "postgresql://localhost/testdb",
                name="Warehouse",
                query_default_limit=25,
                query_max_rows=100,
                query_max_chars=1000,
                query_timeout=12,
                allowed_tables=["orders"],
                blocked_tables=["payments"],
                blocked_columns=["email"],
            )

        description = agent.describe()
        assert description["kind"] == "database"
        assert description["name"] == "Warehouse"
        assert {"sql", "schema", "analyst_tools", "audit"}.issubset(
            description["capabilities"]
        )
        assert description["db"]["database_type"] == "postgresql"
        assert description["db"]["database_name"] == "warehouse"
        assert description["db"]["mode"] == "analyst"
        assert description["db"]["table_count"] == 2
        assert description["db"]["column_count"] == 3
        assert description["db"]["relationship_count"] == 1
        assert description["db"]["drift_status"] == "none"
        assert description["db"]["metric_count"] == 1
        assert "suggested_question_count" not in description["db"]
        assert "suggested_monitor_count" not in description["db"]
        assert description["db"]["finding_count"] == 0
        assert description["db"]["open_finding_count"] == 0
        assert description["db"]["quality_status"] in {"ok", "warning"}
        assert "summary" in description["db"]
        assert description["db"]["query_policy"] == {
            "read_only": True,
            "default_limit": 25,
            "max_rows": 100,
            "max_chars": 1000,
            "timeout": 12,
            "has_table_allowlist": True,
            "has_table_blocklist": True,
            "has_column_blocklist": True,
        }

    async def test_from_db_attaches_db_context_without_profile_surface(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(
            tables=[_table("orders")],
            db_type="postgresql",
            db_name="warehouse",
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        assert agent.db.schema["store_id"] == agent._db_catalog_store_id
        assert isinstance(agent.db.schema.get("tables"), list)
        assert agent.db.plugin is mock_plugin
        assert agent.db.mode == "analyst"
        assert agent.db.drift is None
        assert agent.db.memory is None
        assert agent.db.lineage is None
        assert agent.db.history is None
        assert agent.db.quality is None
        assert agent.db.summary is agent._db_summary
        assert not hasattr(agent.db, "suggested_questions")
        assert not hasattr(agent.db, "suggest_monitors")
        assert agent.db.monitor_events == []
        assert agent.db.findings.all == []
        assert agent.db.findings.open == []
        assert agent.db.findings.resolved == []
        assert not hasattr(agent.db, "profile")
        assert not hasattr(agent.db, "describe")
        assert not hasattr(agent.db, "diagram")

    async def test_from_db_registers_explicit_monitors_as_local_watches(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[
                        {"name": "id", "type": "integer", "is_primary_key": True},
                        {"name": "updated_at", "type": "timestamp"},
                        {"name": "total_amount", "type": "numeric"},
                    ],
                    row_count=100,
                )
            ],
            db_type="postgresql",
            db_name="warehouse",
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        monitor = {
            "name": "orders freshness",
            "type": "freshness",
            "severity": "warning",
            "entity": {"table": "orders", "column": "updated_at"},
            "sql": 'SELECT MAX("updated_at") AS latest FROM "orders"',
            "threshold": {"max_age_hours": 24},
            "interval": "1h",
        }
        registered = agent.db.register_monitors([monitor])

        assert registered
        assert registered[0]["watch_name"].startswith("db:")
        assert len(agent._watches) == len(registered)
        assert agent._watches[0].source._plugin is mock_plugin
        assert agent._watches[0].source._condition == registered[0]["sql"]

    async def test_from_db_registers_custom_monitor_handler(self):
        from daita.agents.db import from_db
        from daita.core.watch import WatchEvent
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(tables=[_table("orders")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        seen = []

        async def handler(event, monitor):
            seen.append((event.value, monitor["name"]))

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        monitor = {
            "name": "orders non-empty",
            "type": "row_count",
            "severity": "info",
            "entity": {"table": "orders"},
            "sql": 'SELECT COUNT(*) AS row_count FROM "orders"',
            "threshold": {"min_rows": 1},
            "interval": "1h",
        }
        agent.db.register_monitors([monitor], handler=handler)
        event = WatchEvent(
            value=0,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling",
        )

        await agent._watches[0].handler(event)

        assert seen == [(0, "orders non-empty")]
        assert agent.db.findings.open[0]["title"] == "orders non-empty"
        assert (
            agent.db.monitor_events[0]["finding_id"] == agent.db.findings.open[0]["id"]
        )

    async def test_from_db_default_monitor_handler_records_local_events(self):
        from daita.agents.db import from_db
        from daita.core.watch import WatchEvent
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(tables=[_table("orders")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        monitor = {
            "name": "orders non-empty",
            "type": "row_count",
            "severity": "info",
            "entity": {"table": "orders"},
            "sql": 'SELECT COUNT(*) AS row_count FROM "orders"',
            "threshold": {"min_rows": 1},
            "interval": "1h",
        }
        agent.db.register_monitors([monitor])
        event = WatchEvent(
            value=0,
            previous_value=2,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling",
        )

        await agent._watches[0].handler(event)

        assert agent.db.monitor_events[0]["monitor"]["name"] == "orders non-empty"
        assert agent.db.monitor_events[0]["value"] == 0
        assert agent.db.monitor_events[0]["previous_value"] == 2
        assert (
            agent.db.monitor_events[0]["finding_id"] == agent.db.findings.open[0]["id"]
        )
        assert agent.db.findings.open[0]["title"] == "orders non-empty"
        assert agent.db.findings.open[0]["kind"] == "db_monitor.row_count"
        assert agent.db.findings.open[0]["observed"]["value"] == 0

    async def test_from_db_resolve_event_marks_active_finding_resolved(self):
        from daita.agents.db import from_db
        from daita.core.watch import WatchEvent
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(tables=[_table("orders")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        monitor = {
            "name": "orders non-empty",
            "type": "row_count",
            "severity": "info",
            "entity": {"table": "orders"},
            "sql": 'SELECT COUNT(*) AS row_count FROM "orders"',
            "threshold": {"min_rows": 1},
            "interval": "1h",
        }
        agent.db.register_monitors([monitor])
        opened = WatchEvent(
            value=0,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling",
        )
        resolved = WatchEvent(
            value=2,
            previous_value=0,
            triggered_at=datetime.now(timezone.utc),
            source_type="polling",
            resolved=True,
        )

        await agent._watches[0].handler(opened)
        opened_id = agent.db.findings.open[0]["id"]
        await agent._watches[0].handler(resolved)

        assert agent.db.findings.open == []
        assert agent.db.findings.resolved[0]["id"] == opened_id
        assert agent.db.findings.resolved[0]["observed"]["resolved"] is True

    async def test_db_findings_contract_is_json_safe_and_exportable(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        normalized = _make_normalized_schema(tables=[_table("orders")])
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=normalized)),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        finding = agent.db.findings.add(
            {
                "title": "Revenue dropped",
                "severity": "critical",
                "status": "open",
                "kind": "metric_regression",
                "source": {"type": "manual"},
                "entity": {"table": "orders", "column": "total"},
                "observed": {"value": datetime(2026, 5, 6, tzinfo=timezone.utc)},
            }
        )

        assert finding["id"].startswith("fnd_")
        assert finding["observed"]["value"] == "2026-05-06T00:00:00+00:00"
        assert agent.db.findings.last()["title"] == "Revenue dropped"
        assert (
            json.loads(agent.db.findings.export_json())[0]["title"] == "Revenue dropped"
        )

    def test_finding_validation_rejects_bad_status(self):
        with pytest.raises(ValueError, match="Unsupported finding status"):
            normalize_finding(
                {
                    "title": "Bad finding",
                    "severity": "warning",
                    "status": "ignored",
                    "kind": "db_observation",
                }
            )

    async def test_from_db_audit_context(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(
                fac,
                "discover_schema",
                AsyncMock(return_value=_make_normalized_schema()),
            ),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        assert agent.db.audit.entries == []
        assert agent.db.audit.last() is None

        agent._db_audit_log.append({"prompt": "hello", "tool_calls": []})

        assert agent.db.audit.entries == [{"prompt": "hello", "tool_calls": []}]
        assert agent.db.audit.last() == {"prompt": "hello", "tool_calls": []}
        assert agent.db.audit.export_json() == '[{"prompt": "hello", "tool_calls": []}]'

    def test_db_run_context_is_compact_and_policy_aware(self):
        from daita.agents.db.runtime.run_context import build_db_run_context

        agent = _agent_with_catalog(
            _make_normalized_schema(
                tables=[_table("orders"), _table("customers")],
                fks=[
                    {
                        "source_table": "orders",
                        "source_column": "customer_id",
                        "target_table": "customers",
                        "target_column": "id",
                    }
                ],
                db_type="postgresql",
                db_name="warehouse",
            ),
            tool_registry=SimpleNamespace(tool_names=["postgres_query", "pivot_table"]),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=25,
                query_max_rows=100,
                query_max_chars=1000,
                query_timeout=12,
                allowed_tables={"orders"},
                blocked_tables={"payments"},
                blocked_columns={"email"},
                sql_dialect="postgresql",
            ),
        )

        context = build_db_run_context(agent, max_chars=700)

        assert len(context) <= 700
        assert context.startswith("<db_runtime_context>")
        assert context.endswith("</db_runtime_context>")
        assert "tables=2" in context
        assert "relationships=1" in context
        assert "default_limit=25" in context
        assert "analyst_tools" in context
        assert "Data health:" in context
        assert "Candidate metrics:" in context
        assert "Memory: disabled" in context

    async def test_db_context_run_augments_prompt_but_audit_keeps_original(self):
        from daita.agents.db.runtime.audit import make_audited_run
        from daita.agents.db.runtime.run_context import make_db_context_run

        agent = _agent_with_catalog(
            _make_normalized_schema(
                tables=[_table("orders")], db_type="postgresql", db_name="warehouse"
            ),
            tool_registry=SimpleNamespace(tool_names=["postgres_query"]),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=50,
                query_max_rows=200,
                query_max_chars=50000,
                query_timeout=30,
            ),
            _db_audit_log=[],
            _tool_call_history=[],
        )
        seen = {}

        async def original_run(prompt, **kwargs):
            seen["prompt"] = prompt
            seen["kwargs"] = kwargs
            return {"result": "ok", "tool_calls": []}

        wrapped = make_audited_run(agent, make_db_context_run(agent, original_run))
        result = await wrapped("How much revenue?", detailed=True)

        assert result["result"] == "ok"
        assert seen["prompt"].startswith("<db_runtime_context>")
        assert "User question:\nHow much revenue?" in seen["prompt"]
        assert seen["kwargs"]["detailed"] is True
        assert result["from_db_metrics"]["llm_call_count"] is None
        assert result["from_db_metrics"]["tool_call_count"] == 0
        assert result["from_db_metrics"]["selected_tools"] == []
        assert agent._db_audit_log[0]["prompt"] == "How much revenue?"

    async def test_db_context_run_replaces_none_tools_with_selected_profile(self):
        from daita.agents.db.runtime.run_context import make_db_context_run

        seen = {}

        async def original_run(prompt, **kwargs):
            seen["kwargs"] = kwargs
            return {"result": "ok", "tool_calls": []}

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("sales")]),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_compile_and_query",
                    "db_plan_query",
                    "db_query",
                    "db_count",
                    "catalog_search_schema",
                ]
            ),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=50,
                query_max_rows=200,
                query_max_chars=50000,
                query_timeout=30,
            ),
        )

        wrapped = make_db_context_run(agent, original_run)
        await wrapped("What were sales last month?", tools=None)

        assert seen["kwargs"]["tools"] == ["db_compile_and_query"]

    async def test_db_context_run_preserves_explicit_tool_list(self):
        from daita.agents.db.runtime.run_context import make_db_context_run

        seen = {}

        async def original_run(prompt, **kwargs):
            seen["kwargs"] = kwargs
            return {"result": "ok", "tool_calls": []}

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("sales")]),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_compile_and_query",
                    "db_plan_query",
                    "db_query",
                    "catalog_search_schema",
                ]
            ),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=50,
                query_max_rows=200,
                query_max_chars=50000,
                query_timeout=30,
            ),
        )

        wrapped = make_db_context_run(agent, original_run)
        await wrapped("What were sales last month?", tools=["db_query"])

        assert seen["kwargs"]["tools"] == ["db_query"]

    async def test_stream_audit_records_tool_arguments_and_result_metadata(self):
        from daita.agents.db.runtime.audit import make_audited_stream
        from daita.core.streaming import AgentEvent, EventType

        agent = SimpleNamespace(_db_audit_log=[])

        async def original_stream(prompt, **kwargs):
            yield AgentEvent(
                type=EventType.TOOL_CALL,
                tool_name="postgres_query",
                tool_args={"sql": "SELECT COUNT(*) FROM orders"},
            )
            yield AgentEvent(
                type=EventType.TOOL_RESULT,
                tool_name="postgres_query",
                result={"rows": [{"count": 12}]},
            )
            yield AgentEvent(type=EventType.COMPLETE, final_result="done")

        wrapped = make_audited_stream(agent, original_stream)
        events = [event async for event in wrapped("Count orders")]

        assert events[-1].type == EventType.COMPLETE
        assert agent._db_audit_log == [
            {
                "timestamp": agent._db_audit_log[0]["timestamp"],
                "prompt": "Count orders",
                "tool_calls": [
                    {
                        "tool": "postgres_query",
                        "arguments": {"sql": "SELECT COUNT(*) FROM orders"},
                        "result": {"row_count": 1},
                    }
                ],
            }
        ]

    async def test_stream_audit_keeps_arguments_on_error(self):
        from daita.agents.db.runtime.audit import make_audited_stream
        from daita.core.streaming import AgentEvent, EventType

        agent = SimpleNamespace(_db_audit_log=[])

        async def original_stream(prompt, **kwargs):
            yield AgentEvent(
                type=EventType.TOOL_CALL,
                tool_name="postgres_query",
                tool_args={"sql": "SELECT * FROM blocked_table"},
            )
            yield AgentEvent(type=EventType.ERROR, error="blocked table")

        wrapped = make_audited_stream(agent, original_stream)
        events = [event async for event in wrapped("Show blocked table")]

        assert events[-1].type == EventType.ERROR
        assert agent._db_audit_log[0]["tool_calls"] == [
            {
                "tool": "postgres_query",
                "arguments": {"sql": "SELECT * FROM blocked_table"},
            }
        ]
        assert agent._db_audit_log[0]["error"] == "blocked table"


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
        mock_agent.run = AsyncMock(return_value="[]")
        return mock_plugin, mock_agent

    async def test_lineage_true_attaches_without_seeding_schema_fks(self):
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
        mock_lineage.register_flow.assert_not_awaited()
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
        custom_lineage.register_flow.assert_not_awaited()
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

        # add_plugin called for DB + catalog, not lineage.
        assert mock_agent.add_plugin.call_count == 2

    async def test_lineage_graph_backend_is_exposed_for_query_tracing(self):
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
        mock_lineage._graph_backend = MagicMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch("daita.plugins.lineage.LineagePlugin", return_value=mock_lineage),
        ):
            await from_db("postgresql://localhost/testdb", lineage=True)

        mock_lineage.register_flow.assert_not_awaited()
        assert mock_agent._db_query_graph_backend is mock_lineage._graph_backend
        assert mock_plugin._db_query_graph_backend is mock_lineage._graph_backend

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
        from daita.core.tools import ToolRegistry

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_agent.run = AsyncMock(return_value="[]")
        mock_agent.tool_registry = ToolRegistry()
        return mock_plugin, mock_agent

    async def test_memory_true_creates_with_workspace(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[_table("orders"), _table("customers"), _table("products")]
        )
        mock_plugin, mock_agent = self._base_mocks()
        mock_memory = MagicMock()
        mock_memory.backend = None
        mock_memory.recall = AsyncMock(return_value=[])
        mock_memory.remember = AsyncMock(return_value={"status": "ok"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch(
                "daita.plugins.memory.MemoryPlugin", return_value=mock_memory
            ) as MockMem,
        ):
            await from_db("postgresql://localhost/testdb", memory=True)

        MockMem.assert_called_once()
        call_kwargs = MockMem.call_args[1]
        assert "workspace" in call_kwargs
        assert call_kwargs["workspace"]  # non-empty
        mock_agent.add_plugin.assert_any_call(mock_memory)
        assert mock_agent._db_memory is mock_memory
        assert mock_agent._db_memory_semantics.plugin is mock_memory

    async def test_memory_plugin_instance_used_directly(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        custom_memory = MagicMock()
        custom_memory.backend = None
        custom_memory.recall = AsyncMock(return_value=[])
        custom_memory.remember = AsyncMock(return_value={"status": "ok"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db("postgresql://localhost/testdb", memory=custom_memory)

        mock_agent.add_plugin.assert_any_call(custom_memory)
        assert mock_agent._db_memory is custom_memory
        assert mock_agent._db_memory_semantics.plugin is custom_memory

    async def test_from_db_removes_generic_memory_write_tools(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db
        from daita.core.tools import AgentTool

        async def fail_if_called(args):
            raise AssertionError("generic memory write tool should not be callable")

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        for name in ("remember", "update_memory", "recall"):
            mock_agent.tool_registry.register(
                AgentTool(
                    name=name,
                    description=f"{name} test tool",
                    parameters={},
                    handler=fail_if_called,
                    source="plugin",
                    category="memory",
                )
            )
        custom_memory = MagicMock()
        custom_memory.backend = None
        custom_memory.recall = AsyncMock(return_value=[])
        custom_memory.remember = AsyncMock(return_value={"status": "ok"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        assert "remember" not in agent.tool_registry.tool_names
        assert "update_memory" not in agent.tool_registry.tool_names
        assert "db_remember" in agent.tool_registry.tool_names
        assert "recall" in agent.tool_registry.tool_names

    async def test_db_remember_tool_stores_validated_db_memory_record(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        custom_memory = MagicMock()
        custom_memory.backend = None
        custom_memory.recall = AsyncMock(return_value=[])
        custom_memory.remember = AsyncMock(return_value={"chunk_id": "mem-1"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        result = await agent.tool_registry.execute(
            "db_remember",
            {
                "kind": "business_rule",
                "key": "business_rule:revenue_refunds",
                "text": "Revenue excludes refunded orders.",
                "metadata": {"metric": "revenue"},
                "importance": 0.8,
            },
        )

        assert result["success"] is True
        assert result["kind"] == "business_rule"
        assert result["category"] == "db_semantics"
        custom_memory.remember.assert_awaited_once()
        stored_content = custom_memory.remember.await_args.args[0]
        assert "Revenue excludes refunded orders." in stored_content
        assert '"kind": "business_rule"' in stored_content

        invalid = await agent.tool_registry.execute(
            "db_remember",
            {"kind": "knowledge", "key": "x", "text": "too vague"},
        )

        assert invalid["success"] is False
        assert "Unsupported DB memory kind" in invalid["error"]

    async def test_db_remember_tool_updates_existing_record_by_key(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        backend = MagicMock()
        backend.list_by_category = AsyncMock(
            return_value=[
                {
                    "chunk_id": "old-1",
                    "content": "old content",
                    "metadata": {
                        "db_memory": {
                            "kind": "metric_definition",
                            "key": "metric:revenue",
                        }
                    },
                },
                {
                    "chunk_id": "other-1",
                    "content": "other content",
                    "metadata": {
                        "db_memory": {
                            "kind": "metric_definition",
                            "key": "metric:margin",
                        }
                    },
                },
            ]
        )
        backend.delete_chunks = AsyncMock()
        backend.remember = AsyncMock(return_value={"chunk_id": "new-1"})
        custom_memory = SimpleNamespace(backend=backend)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        result = await agent.tool_registry.execute(
            "db_remember",
            {
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded orders.",
            },
        )

        assert result["success"] is True
        assert result["status"] == "updated"
        assert result["updated"] == 1
        backend.list_by_category.assert_awaited_once_with(
            category="db_semantics", limit=1000
        )
        backend.delete_chunks.assert_awaited_once_with(["old-1"])
        backend.remember.assert_awaited_once()

    async def test_db_remember_tool_appends_when_backend_cannot_delete_existing_key(
        self,
    ):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        backend = MagicMock()
        backend.list_by_category = AsyncMock(
            return_value=[
                {
                    "chunk_id": "old-1",
                    "content": 'DB memory record:\n{"key": "business_rule:refunds"}',
                    "metadata": {},
                }
            ]
        )
        del backend.delete_chunks
        backend.remember = AsyncMock(return_value={"chunk_id": "new-1"})
        custom_memory = SimpleNamespace(backend=backend)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        result = await agent.tool_registry.execute(
            "db_remember",
            {
                "kind": "business_rule",
                "key": "business_rule:refunds",
                "text": "Refunded orders are excluded from revenue.",
            },
        )

        assert result["success"] is True
        assert result["status"] == "stored"
        assert result["updated"] == 0
        assert result["stored"]["upsert_fallback"] == "append"
        backend.remember.assert_awaited_once()

    async def test_db_remember_tool_rejects_pii_values(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        custom_memory = MagicMock()
        custom_memory.backend = None
        custom_memory.recall = AsyncMock(return_value=[])
        custom_memory.remember = AsyncMock(return_value={"chunk_id": "mem-1"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        result = await agent.tool_registry.execute(
            "db_remember",
            {
                "kind": "business_rule",
                "key": "business_rule:vip_customer",
                "text": "VIP customer email is jane@example.com.",
            },
        )

        assert result["success"] is False
        assert "PII values" in result["error"]
        custom_memory.remember.assert_not_awaited()

    async def test_db_remember_tool_rejects_sensitive_metadata_keys(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        custom_memory = MagicMock()
        custom_memory.backend = None
        custom_memory.recall = AsyncMock(return_value=[])
        custom_memory.remember = AsyncMock(return_value={"chunk_id": "mem-1"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        result = await agent.tool_registry.execute(
            "db_remember",
            {
                "kind": "data_contract_note",
                "key": "contract:users",
                "text": "Users must have verified contact info.",
                "metadata": {"email": "jane@example.com"},
            },
        )

        assert result["success"] is False
        assert "sensitive field" in result["error"]
        custom_memory.remember.assert_not_awaited()

    async def test_db_remember_tool_allows_schema_level_pii_column_mentions(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        custom_memory = MagicMock()
        custom_memory.backend = None
        custom_memory.recall = AsyncMock(return_value=[])
        custom_memory.remember = AsyncMock(return_value={"chunk_id": "mem-1"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", memory=custom_memory)

        result = await agent.tool_registry.execute(
            "db_remember",
            {
                "kind": "schema_interpretation",
                "key": "schema:users.email",
                "text": "users.email is a contact column and should not be used as an entity key.",
                "metadata": {"table": "users", "column": "email"},
            },
        )

        assert result["success"] is True
        custom_memory.remember.assert_awaited_once()

    async def test_from_db_run_does_not_offer_generic_memory_write_tools(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import AgentTool
        from daita.llm.mock import MockLLMProvider

        async def fail_if_called(args):
            raise AssertionError("generic memory write tool should not run")

        class FakeDbPlugin:
            read_only = True
            sql_dialect = "postgresql"

            async def connect(self):
                return None

            async def disconnect(self):
                return None

            def get_tools(self):
                return []

        class FakeMemoryPlugin:
            backend = None

            def get_tools(self):
                return [
                    AgentTool(
                        name="remember",
                        description="Generic memory write",
                        parameters={},
                        handler=fail_if_called,
                        source="plugin",
                        category="memory",
                    ),
                    AgentTool(
                        name="update_memory",
                        description="Generic memory update",
                        parameters={},
                        handler=fail_if_called,
                        source="plugin",
                        category="memory",
                    ),
                    AgentTool(
                        name="recall",
                        description="Generic memory read",
                        parameters={},
                        handler=lambda args: [],
                        source="plugin",
                        category="memory",
                    ),
                ]

            async def recall(self, *args, **kwargs):
                return []

            async def remember(self, *args, **kwargs):
                raise AssertionError("generic memory write method should not run")

        schema = _make_normalized_schema(tables=[_table("orders")])
        db_plugin = FakeDbPlugin()
        memory_plugin = FakeMemoryPlugin()
        llm = MockLLMProvider(delay=0)

        with (
            patch.object(fac, "resolve_plugin", return_value=(db_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch.object(
                fac, "_register_db_facade_tools", lambda *args, **kwargs: None
            ),
        ):
            agent = await from_db(
                "postgresql://localhost/testdb",
                llm_provider=llm,
                memory=memory_plugin,
            )

        result = await agent.run(
            "Please remember that revenue excludes refunds and update_memory if needed.",
            detailed=True,
        )

        assert result["tool_calls"] == []
        assert "remember" not in agent.tool_registry.tool_names
        assert "update_memory" not in agent.tool_registry.tool_names
        assert "db_remember" in agent.tool_registry.tool_names
        selected = agent._db_last_context_metadata["selected_tools"]
        assert "remember" not in selected
        assert "update_memory" not in selected
        assert "db_remember" in selected
        offered_tool_names = {
            tool.get("function", {}).get("name", tool.get("name"))
            for tool in llm.call_history[-1]["tools"]
        }
        assert "remember" not in offered_tool_names
        assert "update_memory" not in offered_tool_names
        assert "db_remember" in offered_tool_names

    async def test_from_db_run_does_not_execute_hallucinated_generic_memory_write(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import AgentTool
        from daita.llm.mock import MockLLMProvider

        class HallucinatedRememberLLM(MockLLMProvider):
            async def _generate_impl(self, messages, tools=None, **kwargs):
                self.call_history.append(
                    {
                        "messages": messages,
                        "tools": tools,
                        "params": kwargs,
                    }
                )
                if tools and len(self.call_history) == 1:
                    return {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-remember",
                                "name": "remember",
                                "arguments": {
                                    "content": "Revenue excludes refunded orders.",
                                    "category": "knowledge",
                                },
                            }
                        ],
                    }
                if tools:
                    return {"content": "done", "tool_calls": None}
                return "done"

        async def fail_if_called(args):
            raise AssertionError("generic memory write tool should not run")

        class FakeDbPlugin:
            read_only = True
            sql_dialect = "postgresql"

            async def connect(self):
                return None

            async def disconnect(self):
                return None

            def get_tools(self):
                return []

        class FakeMemoryPlugin:
            backend = None

            def get_tools(self):
                return [
                    AgentTool(
                        name="remember",
                        description="Generic memory write",
                        parameters={},
                        handler=fail_if_called,
                        source="plugin",
                        category="memory",
                    )
                ]

            async def recall(self, *args, **kwargs):
                return []

            async def remember(self, *args, **kwargs):
                raise AssertionError("generic memory write method should not run")

        schema = _make_normalized_schema(tables=[_table("orders")])
        llm = HallucinatedRememberLLM(delay=0)

        with (
            patch.object(fac, "resolve_plugin", return_value=(FakeDbPlugin(), True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch.object(
                fac, "_register_db_facade_tools", lambda *args, **kwargs: None
            ),
        ):
            agent = await from_db(
                "postgresql://localhost/testdb",
                llm_provider=llm,
                memory=FakeMemoryPlugin(),
            )

        result = await agent.run(
            "Remember that revenue excludes refunds.", detailed=True
        )

        assert result["result"] == "done"
        assert result["tool_calls"][0]["tool"] == "remember"
        assert result["tool_calls"][0]["result"] == {
            "error": "Tool 'remember' not found"
        }

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

        assert mock_agent.add_plugin.call_count == 2  # DB plugin + catalog

    async def test_memory_calibration_stores_structured_unit_conventions(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[
                        {"name": "total_cents", "type": "numeric"},
                    ],
                )
            ]
        )
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "table": "orders",
                        "column": "total_cents",
                        "unit": "cents",
                        "confidence": "high",
                        "reason": "column name contains cents",
                    }
                ]
            )
        )
        mock_memory = MagicMock()
        mock_memory.backend = None
        mock_memory.recall = AsyncMock(return_value=[])
        mock_memory.remember = AsyncMock(return_value={"status": "ok"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(
                "postgresql://localhost/testdb",
                memory=mock_memory,
                calibrate_memory=True,
            )

        categories = [
            call.kwargs["category"] for call in mock_memory.remember.call_args_list
        ]
        contents = [call.args[0] for call in mock_memory.remember.call_args_list]

        assert "db_semantics" in categories
        assert "db_cache_marker" in categories
        assert any("orders.total_cents is stored as cents" in c for c in contents)
        assert all("numeric_column_units" not in c for c in contents)

    async def test_memory_calibration_skips_when_exact_marker_exists(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[{"name": "total_cents", "type": "numeric"}],
                )
            ]
        )
        mock_plugin, mock_agent = self._base_mocks()
        original_run = mock_agent.run
        mock_memory = MagicMock()
        backend = MagicMock()
        backend.list_by_category = AsyncMock(
            return_value=[
                {
                    "content": "DB exact cache marker: numeric_unit_calibration:postgresql://localhost/testdb"
                }
            ]
        )
        backend.recall = AsyncMock(return_value=[])
        mock_memory.backend = backend
        mock_memory.recall = AsyncMock(return_value=[])
        mock_memory.remember = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
            patch.object(
                fac,
                "catalog_profile_key",
                return_value="postgresql://localhost/testdb",
            ),
        ):
            await from_db(
                "postgresql://localhost/testdb",
                memory=mock_memory,
                calibrate_memory=True,
            )

        original_run.assert_not_awaited()
        backend.recall.assert_not_awaited()
        mock_memory.recall.assert_not_awaited()
        mock_memory.remember.assert_not_awaited()

    async def test_db_memory_recall_uses_single_semantic_category_and_filters_kind(
        self,
    ):
        backend = MagicMock()
        backend.recall = AsyncMock(
            return_value=[
                {
                    "chunk_id": "1",
                    "content": 'DB memory record:\n{"kind": "unit_convention", "text": "orders.total_cents is stored as cents"}',
                },
                {
                    "chunk_id": "2",
                    "content": 'DB memory record:\n{"kind": "business_rule", "text": "exclude refunds"}',
                },
            ]
        )
        plugin = SimpleNamespace(backend=backend)
        db_memory = DBMemory(plugin)

        results = await db_memory.recall(
            "How much revenue?",
            kinds=["unit_convention"],
            limit=5,
        )

        backend.recall.assert_awaited_once()
        assert backend.recall.call_args.kwargs["category"] == "db_semantics"
        assert [r["chunk_id"] for r in results] == ["1"]

    async def test_db_memory_stores_metric_definitions_and_business_rules(self):
        backend = MagicMock()
        backend.remember = AsyncMock(side_effect=lambda *args, **kwargs: kwargs)
        db_memory = DBMemory(SimpleNamespace(backend=backend))

        await db_memory.remember_many(
            [
                DBMemoryRecord(
                    kind="metric_definition",
                    key="metric:revenue",
                    text="Revenue excludes refunded orders.",
                    metadata={"metric": "revenue"},
                ),
                {
                    "kind": "business_rule",
                    "key": "rule:refunds",
                    "text": "Refunded orders must be excluded from revenue.",
                    "metadata": {"table": "orders"},
                },
            ]
        )

        assert backend.remember.await_count == 2
        first_call = backend.remember.await_args_list[0]
        second_call = backend.remember.await_args_list[1]
        assert first_call.kwargs["category"] == "db_semantics"
        assert first_call.kwargs["index_content"] == "Revenue excludes refunded orders."
        assert (
            first_call.kwargs["extra_metadata"]["db_memory"]["kind"]
            == "metric_definition"
        )
        assert (
            second_call.kwargs["extra_metadata"]["db_memory"]["kind"] == "business_rule"
        )

    async def test_db_memory_recalls_relevant_business_rules(self):
        backend = MagicMock()
        backend.recall = AsyncMock(
            return_value=[
                {
                    "chunk_id": "rule-1",
                    "content": (
                        'DB memory record:\n{"kind": "business_rule", '
                        '"text": "Revenue excludes refunded orders."}'
                    ),
                },
                {
                    "chunk_id": "metric-1",
                    "content": (
                        'DB memory record:\n{"kind": "metric_definition", '
                        '"text": "Revenue is SUM(total_amount)."}'
                    ),
                },
            ]
        )
        db_memory = DBMemory(SimpleNamespace(backend=backend))

        results = await db_memory.recall(
            "How should revenue be calculated?",
            kinds=["business_rule"],
            limit=5,
        )

        assert [r["chunk_id"] for r in results] == ["rule-1"]
        backend.recall.assert_awaited_once()
        assert backend.recall.call_args.kwargs["category"] == "db_semantics"

    async def test_row_level_questions_do_not_recall_memory_context(self):
        class FailingIfCalledMemory:
            async def recall(self, *args, **kwargs):
                raise AssertionError("row-level prompt should not recall DB memory")

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("customers")]),
            _db_memory_semantics=FailingIfCalledMemory(),
        )

        snippets = await recall_db_memory_context(
            agent, "Show me the email for customer_id 1"
        )

        assert snippets == []
        assert agent._db_last_memory_recall_decision["reason"] == "row_level_prompt"

    async def test_direct_count_question_skips_db_memory_recall(self):
        class FailingIfCalledMemory:
            async def recall(self, *args, **kwargs):
                raise AssertionError("direct count prompt should not recall DB memory")

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            _db_memory_semantics=FailingIfCalledMemory(),
        )

        snippets = await recall_db_memory_context(agent, "How many orders this month?")

        assert snippets == []
        assert (
            agent._db_last_memory_recall_decision["reason"]
            == "direct_schema_matched_query"
        )

    async def test_semantic_metric_question_recalls_db_memory(self):
        class FakeDBMemory:
            async def recall(self, *args, **kwargs):
                return [
                    {
                        "content": 'DB memory record:\n{"text": "Revenue excludes refunds."}'
                    }
                ]

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            _db_memory_semantics=FakeDBMemory(),
        )

        snippets = await recall_db_memory_context(
            agent, "What business rule should I use to calculate revenue?"
        )

        assert snippets == ["Revenue excludes refunds."]
        assert agent._db_last_memory_recall_decision["reason"] == "semantic_prompt"

    async def test_semantic_language_overrides_identifier_skip(self):
        class FakeDBMemory:
            async def recall(self, *args, **kwargs):
                return [
                    {
                        "content": 'DB memory record:\n{"text": "Customer IDs are internal only."}'
                    }
                ]

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("customers")]),
            _db_memory_semantics=FakeDBMemory(),
        )

        snippets = await recall_db_memory_context(agent, "What does customer_id mean?")

        assert snippets == ["Customer IDs are internal only."]
        assert agent._db_last_memory_recall_decision["reason"] == "semantic_prompt"

    async def test_db_memory_marker_lookup_uses_exact_category_listing(self):
        backend = MagicMock()
        backend.list_by_category = AsyncMock(
            return_value=[
                {"content": "DB exact cache marker: numeric_unit_calibration:abc"}
            ]
        )
        backend.recall = AsyncMock(return_value=[])
        db_memory = DBMemory(SimpleNamespace(backend=backend))

        assert await db_memory.has_marker("numeric_unit_calibration:abc") is True

        backend.list_by_category.assert_awaited_once_with(
            category="db_cache_marker",
            limit=1000,
        )
        backend.recall.assert_not_awaited()

    async def test_db_context_run_includes_recalled_db_memory(self):
        from daita.agents.db.runtime.run_context import make_db_context_run

        class FakeDBMemory:
            async def recall(self, *args, **kwargs):
                return [
                    {
                        "content": 'DB memory record:\n{"text": "orders.total_cents is stored as cents"}'
                    }
                ]

        captured = {}

        async def original_run(prompt, **kwargs):
            captured["prompt"] = prompt
            return "ok"

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            tool_registry=SimpleNamespace(tool_names=["postgres_query"]),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=50,
                query_max_rows=200,
                query_max_chars=50000,
                query_timeout=30,
            ),
            _db_memory=object(),
            _db_memory_semantics=FakeDBMemory(),
        )

        wrapped = make_db_context_run(agent, original_run)
        await wrapped("How much revenue did we make?")

        assert (
            "Memory: relevant=orders.total_cents is stored as cents"
            in captured["prompt"]
        )

    async def test_db_context_run_omits_memory_for_row_level_questions(self):
        from daita.agents.db.runtime.run_context import make_db_context_run

        class FailingIfCalledMemory:
            async def recall(self, *args, **kwargs):
                raise AssertionError("row-level prompt should not recall DB memory")

        captured = {}

        async def original_run(prompt, **kwargs):
            captured["prompt"] = prompt
            return "ok"

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("customers")]),
            tool_registry=SimpleNamespace(tool_names=["postgres_query"]),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=50,
                query_max_rows=200,
                query_max_chars=50000,
                query_timeout=30,
            ),
            _db_memory=object(),
            _db_memory_semantics=FailingIfCalledMemory(),
        )

        wrapped = make_db_context_run(agent, original_run)
        await wrapped("Look up customer_id 1 email")

        assert "Memory: enabled;" in captured["prompt"]
        assert "relevant=" not in captured["prompt"]

    async def test_memory_recall_failure_does_not_break_agent_run(self):
        from daita.agents.db.runtime.run_context import make_db_context_run

        class BrokenDBMemory:
            async def recall(self, *args, **kwargs):
                raise RuntimeError("memory backend unavailable")

        captured = {}

        async def original_run(prompt, **kwargs):
            captured["prompt"] = prompt
            return "ok"

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            tool_registry=SimpleNamespace(tool_names=["postgres_query"]),
            _db_drift=None,
            _db_plugin=SimpleNamespace(
                read_only=True,
                query_default_limit=50,
                query_max_rows=200,
                query_max_chars=50000,
                query_timeout=30,
            ),
            _db_memory=object(),
            _db_memory_semantics=BrokenDBMemory(),
        )

        wrapped = make_db_context_run(agent, original_run)
        result = await wrapped("How is revenue defined?")

        assert result == "ok"
        assert "Memory: enabled;" in captured["prompt"]
        assert "relevant=" not in captured["prompt"]

    async def test_unit_calibration_marker_prevents_repeated_calibration(self):
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[{"name": "total_cents", "type": "numeric"}],
                )
            ]
        )
        agent = SimpleNamespace(
            run=AsyncMock(
                return_value=json.dumps(
                    [
                        {
                            "table": "orders",
                            "column": "total_cents",
                            "unit": "cents",
                            "confidence": "high",
                        }
                    ]
                )
            )
        )

        class RecordingDBMemory:
            def __init__(self):
                self.markers = set()
                self.records = []

            async def has_marker(self, key):
                return key in self.markers

            async def remember_many(self, records):
                self.records.extend(records)
                return [{"ok": True} for _ in records]

            async def mark(self, key):
                self.markers.add(key)
                return {"ok": True}

        db_memory = RecordingDBMemory()

        await calibrate_db_memory(
            agent,
            schema,
            db_memory,
            marker_key="numeric_unit_calibration:test",
        )
        await calibrate_db_memory(
            agent,
            schema,
            db_memory,
            marker_key="numeric_unit_calibration:test",
        )

        agent.run.assert_awaited_once()
        assert len(db_memory.records) == 1
        assert db_memory.records[0].key == "unit_convention:orders.total_cents"

    def test_db_memory_record_validation(self):
        record = normalize_db_memory_record(
            {
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded orders.",
                "importance": 2,
            }
        )

        assert record.importance == 1.0
        assert record.category == "db_semantics"

        with pytest.raises(ValueError, match="Unsupported DB memory kind"):
            normalize_db_memory_record({"kind": "row", "key": "x", "text": "bad"})


# ---------------------------------------------------------------------------
# Catalog profile reuse tests
# ---------------------------------------------------------------------------


def _write_catalog_schema(schema, *, key="postgresql:default", last_seen=None):
    catalog_dir = Path(".daita")
    catalog_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(schema)
    now = datetime.now(timezone.utc).isoformat()
    payload.setdefault("first_seen", last_seen or now)
    payload.setdefault("last_seen", last_seen or now)
    (catalog_dir / "catalog.json").write_text(json.dumps({key: payload}, indent=2))


class TestCatalogProfileFreshness:
    def test_catalog_profile_key_redacts_password(self):
        key1 = _db_catalog_profile_key("postgresql://user:secret1@host/db")
        key2 = _db_catalog_profile_key("postgresql://user:secret2@host/db")
        assert key1 == key2

    def test_catalog_profile_key_different_hosts(self):
        key1 = _db_catalog_profile_key("postgresql://user:pass@host1/db")
        key2 = _db_catalog_profile_key("postgresql://user:pass@host2/db")
        assert key1 != key2

    def test_catalog_snapshot_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        schema = _make_normalized_schema(tables=[_table("users")])
        profile_key = "testkey123"

        _write_catalog_schema(schema)
        result = _db_load_catalog_profile_snapshot(
            profile_key, catalog_keys=["postgresql:default"], ttl=3600
        )

        assert result is not None
        loaded_schema, is_expired = result
        assert loaded_schema["tables"] == schema["tables"]
        assert loaded_schema["database_type"] == schema["database_type"]
        assert loaded_schema["last_seen"]
        assert is_expired is False

    async def test_catalog_profile_hit_skips_discovery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[_table("orders"), _table("customers"), _table("products")]
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_discover = AsyncMock(return_value=schema)

        source = "postgresql://user:pass@host/db"
        _write_catalog_schema(schema)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", mock_discover),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=3600)

        mock_discover.assert_not_awaited()

    async def test_catalog_snapshot_skips_discovery_by_default(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("orders"), _table("customers")])
        source = "postgresql://user:pass@host/db"
        _write_catalog_schema(schema)

        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_discover = AsyncMock(return_value=_make_normalized_schema())

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", mock_discover),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=None)

        mock_discover.assert_not_awaited()
        catalog_schema = mock_agent._db_catalog.get_schema(
            mock_agent._db_catalog_store_id
        )
        assert [table.name for table in catalog_schema.tables] == [
            table["name"] for table in schema["tables"]
        ]

    async def test_stale_catalog_profile_triggers_rediscovery(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[_table("orders"), _table("customers"), _table("products")]
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_discover = AsyncMock(return_value=schema)

        source = "postgresql://user:pass@host/db"
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        _write_catalog_schema(schema, last_seen=old_ts)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", mock_discover),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=0)

        mock_discover.assert_awaited_once()

    def test_drift_detection_added_table(self):
        old = _make_normalized_schema(tables=[_table("users"), _table("orders")])
        new = _make_normalized_schema(
            tables=[_table("users"), _table("orders"), _table("products")]
        )
        drift = _db_detect_profile_drift(old, new)
        assert drift is not None
        assert "products" in drift["added_tables"]
        assert drift["removed_tables"] == []

    def test_drift_detection_removed_column(self):
        old_cols = [
            {
                "name": "id",
                "type": "integer",
                "nullable": False,
                "is_primary_key": True,
            },
            {
                "name": "email",
                "type": "text",
                "nullable": True,
                "is_primary_key": False,
            },
        ]
        new_cols = [
            {
                "name": "id",
                "type": "integer",
                "nullable": False,
                "is_primary_key": True,
            },
        ]
        old = _make_normalized_schema(
            tables=[{"name": "users", "columns": old_cols, "row_count": None}]
        )
        new = _make_normalized_schema(
            tables=[{"name": "users", "columns": new_cols, "row_count": None}]
        )
        drift = _db_detect_profile_drift(old, new)
        assert drift is not None
        change = next(c for c in drift["column_changes"] if c["table"] == "users")
        assert "email" in change["removed_columns"]

    def test_drift_detection_no_change(self):
        schema = _make_normalized_schema(tables=[_table("users"), _table("orders")])
        drift = _db_detect_profile_drift(schema, schema)
        assert drift is None

    async def test_stale_catalog_profile_fallback_on_failure(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(
            tables=[_table("orders"), _table("customers"), _table("products")]
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()

        source = "postgresql://user:pass@host/db"
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        _write_catalog_schema(schema, last_seen=old_ts)

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(
                fac, "discover_schema", AsyncMock(side_effect=RuntimeError("DB down"))
            ),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            await from_db(source, cache_ttl=0)

        catalog_schema = mock_agent._db_catalog.get_schema(
            mock_agent._db_catalog_store_id
        )
        assert [table.name for table in catalog_schema.tables] == [
            table["name"] for table in schema["tables"]
        ]


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

        tool_calls_run1 = [
            {"tool": "postgres_query", "arguments": {"sql": "SELECT 1"}, "result": {}}
        ]
        tool_calls_run2 = [
            {"tool": "postgres_query", "arguments": {"sql": "SELECT 2"}, "result": {}}
        ]
        mock_agent.run = AsyncMock(
            side_effect=[
                {"result": "first", "tool_calls": tool_calls_run1},
                {"result": "second", "tool_calls": tool_calls_run2},
            ]
        )

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
        assert agent._db_audit_log[0]["tool_calls"] == [
            {
                "tool": "postgres_query",
                "arguments": {"sql": "SELECT 1"},
                "result": {"result_type": "dict"},
            }
        ]
        assert agent._db_audit_log[1]["prompt"] == "second question"
        assert agent._db_audit_log[1]["tool_calls"] == [
            {
                "tool": "postgres_query",
                "arguments": {"sql": "SELECT 2"},
                "result": {"result_type": "dict"},
            }
        ]

    async def test_audit_log_redacts_raw_db_rows(self):
        import daita.agents.db.builder as fac
        from daita.agents.db import from_db

        schema = _make_normalized_schema(tables=[_table("users")])
        mock_plugin, mock_agent = self._base_mocks()
        mock_agent.run = AsyncMock(
            return_value={
                "result": "ok",
                "tool_calls": [
                    {
                        "tool": "postgres_query",
                        "arguments": {
                            "sql": "SELECT email FROM users WHERE id = $1",
                            "params": ["secret@example.com"],
                        },
                        "result": {
                            "rows": [{"email": "secret@example.com"}],
                            "total_rows": 1,
                            "truncated": False,
                            "sql": "SELECT email FROM users WHERE id = $1 LIMIT 50",
                        },
                    }
                ],
            }
        )

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        await agent.run("show email")

        serialized = json.dumps(agent._db_audit_log)
        assert "secret@example.com" not in serialized
        assert agent._db_audit_log[0]["tool_calls"] == [
            {
                "tool": "postgres_query",
                "arguments": {
                    "sql": "SELECT email FROM users WHERE id = $1",
                    "param_count": 1,
                },
                "result": {
                    "sql": "SELECT email FROM users WHERE id = $1 LIMIT 50",
                    "total_rows": 1,
                    "truncated": False,
                    "row_count": 1,
                },
            }
        ]

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
            return_value={
                "result": "42 rows found",
                "tool_calls": [],
                "tokens": {},
                "cost": 0.0,
            }
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
            agent = await from_db(
                "postgresql://localhost/testdb", history=custom_history
            )

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
                    {
                        "name": "id",
                        "type": "integer",
                        "nullable": False,
                        "is_primary_key": True,
                    },
                    {
                        "name": "name",
                        "type": "varchar",
                        "nullable": True,
                        "is_primary_key": False,
                    },
                ],
                "row_count": 100,
            },
            {
                "name": "orders",
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "nullable": False,
                        "is_primary_key": True,
                    },
                    {
                        "name": "customer_id",
                        "type": "integer",
                        "nullable": False,
                        "is_primary_key": False,
                    },
                    {
                        "name": "total",
                        "type": "numeric",
                        "nullable": True,
                        "is_primary_key": False,
                    },
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


def _analyst_context(schema=None):
    from daita.agents.db.tools.analyst._helpers import AnalystCatalogContext

    schema = schema or _analyst_schema()
    store_id = schema.get("store_id") or "analyst-test-store"
    catalog_schema = dict(schema)
    catalog_schema["store_id"] = store_id
    catalog = CatalogPlugin()
    catalog._schemas[store_id] = NormalizedSchema.from_dict(catalog_schema)
    return AnalystCatalogContext(
        catalog=catalog,
        store_id=store_id,
        database_type=schema.get("database_type", "postgresql"),
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


class TestFromDbQueryFacadeTools:
    async def test_sql_facade_tools_delegate_to_plugin_guardrails(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": [{"x": 1}]})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": [{"x": 2}]})
        schema = _make_normalized_schema(db_type="postgresql")

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

        assert set(tools) == {
            "db_compile_and_query",
            "db_plan_query",
            "db_validate_sql",
            "db_query",
            "db_count",
            "db_sample",
        }
        result = await tools["db_query"].handler({"sql": "SELECT 1"})
        assert result == {"rows": [{"x": 1}]}
        plugin._tool_query.assert_awaited_once_with({"sql": "SELECT 1"})

    async def test_sql_facade_plan_query_enriches_intent_and_records_run_state(self):
        from daita.agents.db.runtime.state import DbRunState, set_db_run_state
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        state = DbRunState()
        set_db_run_state(plugin, state)
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[
                        {"name": "operation_id", "type": "text"},
                        {"name": "api_key_id", "type": "text"},
                        {"name": "total_tokens", "type": "integer"},
                        {"name": "cost", "type": "numeric"},
                    ],
                ),
                _table(
                    "api_keys",
                    columns=[
                        {"name": "api_key_id", "type": "text"},
                        {"name": "created_by", "type": "text"},
                    ],
                ),
                _table(
                    "users",
                    columns=[
                        {"name": "user_id", "type": "text"},
                        {"name": "email", "type": "text"},
                    ],
                ),
            ],
            fks=[
                {
                    "source_table": "operations",
                    "source_column": "api_key_id",
                    "target_table": "api_keys",
                    "target_column": "api_key_id",
                },
                {
                    "source_table": "api_keys",
                    "source_column": "created_by",
                    "target_table": "users",
                    "target_column": "user_id",
                },
            ],
        )
        await _attach_plugin_catalog(plugin, schema)

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        result = await tools["db_plan_query"].handler(
            {
                "goal": "Show operations by user",
                "required_fields": ["total tokens", "cost", "user email"],
                "candidate_tables": ["operations", "users"],
                "required_joins": [
                    {"from_tables": ["operations"], "to_tables": ["users"]}
                ],
                "limit": 10,
                "include_diagnostics": True,
            }
        )

        assert result["ok"] is True
        assert result["route"] == "join_query"
        assert "operations" in result["resolved_tables"]
        assert "users" in result["resolved_tables"]
        assert result["field_candidates"]["total tokens"][0]["column"] == "total_tokens"
        assert any(path.get("reachable") for path in result["join_paths"])
        assert state.summary()["planned_query_count"] == 1
        assert "total tokens" in state.required_answer_fields

    async def test_sql_facade_preflights_missing_column_before_query(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "users",
                    columns=[
                        {"name": "user_id", "type": "text"},
                        {"name": "email", "type": "text"},
                    ],
                )
            ]
        )

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

        result = await tools["db_query"].handler(
            {"sql": "SELECT u.username FROM users u"}
        )

        assert result["error"] == "SQL preflight failed against known schema"
        assert result["repair_required"] is True
        assert result["do_not_retry_same_sql"] is True
        assert result["missing_columns"] == [
            {"table": "users", "column": "username", "reason": "column not found"}
        ]
        assert result["available_columns"] == {"users": ["email", "user_id"]}
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_records_preflight_failures_in_run_state(self):
        from daita.agents.db.runtime.state import DbRunState, set_db_run_state
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        state = DbRunState()
        set_db_run_state(plugin, state)
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "users",
                    columns=[
                        {"name": "user_id", "type": "text"},
                        {"name": "email", "type": "text"},
                    ],
                )
            ]
        )
        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

        await tools["db_query"].handler({"sql": "SELECT u.username FROM users u"})

        assert state.summary()["failed_sql_count"] == 1
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_blocks_repeated_invalid_sql_without_generic_error(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "users",
                    columns=[
                        {"name": "user_id", "type": "text"},
                        {"name": "email", "type": "text"},
                    ],
                )
            ]
        )

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        args = {"sql": "SELECT u.username FROM users u"}

        first = await tools["db_query"].handler(args)
        second = await tools["db_query"].handler(args)

        assert first["error"] == "SQL preflight failed against known schema"
        assert second["blocked_repeat"] is True
        assert second["repair_required"] is True
        assert "error" not in second
        plugin._tool_query.assert_not_awaited()

    async def test_validate_sql_tool_does_not_execute_query(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(tables=[_table("users")])

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        result = await tools["db_validate_sql"].handler(
            {"sql": "SELECT users.id FROM users"}
        )

        assert result["ok"] is True
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_normalizes_trailing_semicolon_before_guardrails(self):
        from daita.agents.db.tools.query import create_db_query_tools
        from daita.plugins.base_db import BaseDatabasePlugin

        class GuardedPlugin(BaseDatabasePlugin):
            sql_dialect = "postgresql"

            async def connect(self):
                return None

            async def disconnect(self):
                return None

        plugin = GuardedPlugin(read_only=True)
        plugin._tool_query = AsyncMock(return_value={"rows": [{"agent_id": "a1"}]})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[
                        {"name": "operation_id", "type": "uuid"},
                        {"name": "agent_id", "type": "text"},
                        {"name": "created_at", "type": "timestamp"},
                    ],
                )
            ]
        )
        sql = """
        SELECT operations.agent_id, COUNT(operations.operation_id) AS operation_count
        FROM operations
        WHERE operations.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY operations.agent_id
        ORDER BY operation_count DESC
        LIMIT 10;
        """
        normalized_sql = BaseDatabasePlugin._normalize_sql(sql)

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

        validation = await tools["db_validate_sql"].handler({"sql": sql})
        result = await tools["db_query"].handler({"sql": sql})

        assert validation["ok"] is True
        assert result == {"rows": [{"agent_id": "a1"}]}
        plugin._tool_query.assert_awaited_once_with({"sql": normalized_sql})

    async def test_sql_facade_still_rejects_internal_semicolon(self):
        from daita.agents.db.tools.query import create_db_query_tools
        from daita.core.exceptions import ValidationError
        from daita.plugins.base_db import BaseDatabasePlugin

        class GuardedPlugin(BaseDatabasePlugin):
            sql_dialect = "postgresql"

            async def connect(self):
                return None

            async def disconnect(self):
                return None

        plugin = GuardedPlugin(read_only=True)
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[{"name": "operation_id", "type": "uuid"}],
                )
            ]
        )
        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

        with pytest.raises(ValidationError, match="multiple statements"):
            await tools["db_validate_sql"].handler(
                {"sql": "SELECT operation_id FROM operations; SELECT 1"}
            )
        with pytest.raises(ValidationError, match="multiple statements"):
            await tools["db_query"].handler(
                {"sql": "SELECT operation_id FROM operations; SELECT 1"}
            )
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_allows_unaliased_join_table_references(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[
                        {"name": "agent_id", "type": "text"},
                        {"name": "total_tokens", "type": "integer"},
                        {"name": "timestamp", "type": "timestamp"},
                    ],
                ),
                _table(
                    "agents",
                    columns=[
                        {"name": "agent_id", "type": "text"},
                        {"name": "agent_name", "type": "text"},
                    ],
                ),
            ]
        )

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        sql = """
        SELECT agents.agent_id,
               agents.agent_name,
               SUM(operations.total_tokens) AS total_tokens_used
        FROM operations
        JOIN agents ON operations.agent_id = agents.agent_id
        WHERE operations.timestamp >= NOW() - INTERVAL '1 month'
        GROUP BY agents.agent_id, agents.agent_name
        ORDER BY total_tokens_used DESC
        LIMIT 10
        """
        result = await tools["db_validate_sql"].handler({"sql": sql})

        assert result["ok"] is True
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_blocks_metric_drift_from_required_plan_fields(self):
        from daita.agents.db.runtime.state import DbRunState, set_db_run_state
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        set_db_run_state(plugin, DbRunState())
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[
                        {"name": "agent_id", "type": "text"},
                        {"name": "latency_ms", "type": "integer"},
                        {"name": "total_tokens", "type": "integer"},
                    ],
                )
            ]
        )

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        await tools["db_plan_query"].handler(
            {
                "goal": "Show total tokens by agent",
                "required_fields": ["total tokens"],
                "candidate_tables": ["operations"],
                "aggregations": ["SUM(total_tokens) AS total_tokens_used"],
            }
        )
        sql = """
        SELECT operations.agent_id,
               SUM(operations.latency_ms) AS total_tokens_used
        FROM operations
        GROUP BY operations.agent_id
        """
        result = await tools["db_validate_sql"].handler({"sql": sql})

        assert (
            result["error"] == "SQL does not preserve required fields from query plan"
        )
        assert result["repair_required"] is True
        assert result["preflight_failed"] is True
        assert result["suggested_next_tool"] == "db_plan_query"
        assert result["required_field_warnings"] == [
            {
                "required_field": "total tokens",
                "expected_columns": ["total_tokens"],
                "reason": "required field not referenced by SQL",
            }
        ]
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_allows_count_alias_for_required_count_metric(self):
        from daita.agents.db.runtime.state import DbRunState, set_db_run_state
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        set_db_run_state(plugin, DbRunState())
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[
                        {"name": "operation_id", "type": "uuid"},
                        {"name": "agent_id", "type": "text"},
                        {"name": "created_at", "type": "timestamp"},
                    ],
                ),
                _table(
                    "agents",
                    columns=[
                        {"name": "agent_id", "type": "text"},
                        {"name": "agent_name", "type": "text"},
                        {"name": "total_operations", "type": "integer"},
                    ],
                ),
            ]
        )

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        await tools["db_plan_query"].handler(
            {
                "goal": "Find the agent with the most operations this month",
                "required_fields": ["agent_id", "agent_name", "total_operations"],
                "candidate_tables": ["agents", "operations"],
                "aggregations": ["COUNT(operations.operation_id) AS total_operations"],
                "grouping": ["agents.agent_id", "agents.agent_name"],
            }
        )
        sql = """
        SELECT agents.agent_id,
               agents.agent_name,
               COUNT(operations.operation_id) AS total_operations
        FROM agents
        JOIN operations ON agents.agent_id = operations.agent_id
        WHERE operations.created_at >= date_trunc('month', current_date)
        GROUP BY agents.agent_id, agents.agent_name
        ORDER BY total_operations DESC
        LIMIT 1
        """
        result = await tools["db_validate_sql"].handler({"sql": sql})

        assert result["ok"] is True
        plugin._tool_query.assert_not_awaited()

    async def test_sql_facade_plan_warns_on_missing_aggregation_column(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin.sql_dialect = "postgresql"
        plugin.read_only = True
        plugin._tool_query = AsyncMock(return_value={"rows": []})
        plugin._tool_count = AsyncMock(return_value={"count": 3})
        plugin._tool_sample = AsyncMock(return_value={"rows": []})
        schema = _make_normalized_schema(
            tables=[
                _table(
                    "operations",
                    columns=[
                        {"name": "operation_id", "type": "uuid"},
                        {"name": "agent_id", "type": "text"},
                    ],
                )
            ]
        )

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}
        result = await tools["db_plan_query"].handler(
            {
                "goal": "Find which agent has the most operations",
                "required_fields": ["total_operations"],
                "candidate_tables": ["operations"],
                "aggregations": [
                    "SUM(operations.total_operations) AS total_operations"
                ],
                "include_diagnostics": True,
            }
        )

        assert result["plan_warnings"] == [
            {
                "type": "unknown_aggregation_column",
                "aggregation": "SUM(operations.total_operations) AS total_operations",
                "table": "operations",
                "column": "total_operations",
                "suggested_next_tool": "catalog_inspect_table",
                "guidance": (
                    "For count-style questions, count stable rows such as "
                    "COUNT(*) or COUNT(primary_key) and alias the result, "
                    "instead of summing a similarly named column that is not "
                    "present in the fact table."
                ),
            }
        ]

    async def test_mongo_facade_uses_document_store_capabilities(self):
        from daita.agents.db.tools.query import create_db_query_tools

        plugin = MagicMock()
        plugin._tool_find = AsyncMock(return_value={"documents": [{"x": 1}]})
        plugin._tool_aggregate = AsyncMock(return_value={"documents": [{"x": 2}]})
        plugin._tool_count = AsyncMock(return_value={"count": 2})
        schema = _make_normalized_schema(db_type="mongodb")

        tools = {tool.name: tool for tool in create_db_query_tools(plugin, schema)}

        assert set(tools) == {"db_find", "db_aggregate", "db_count"}
        await tools["db_find"].handler({"collection": "events"})
        plugin._tool_find.assert_awaited_once_with({"collection": "events"})

    async def test_from_db_hides_provider_query_duplicates(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import AgentTool, ToolRegistry

        async def query_handler(args):
            return {"rows": []}

        mock_plugin = MagicMock()
        mock_plugin.sql_dialect = "postgresql"
        mock_plugin.read_only = True
        mock_plugin.connect = AsyncMock()
        mock_plugin._tool_query = AsyncMock(return_value={"rows": []})
        mock_plugin._tool_count = AsyncMock(return_value={"count": 0})
        mock_plugin._tool_sample = AsyncMock(return_value={"rows": []})
        mock_plugin.get_tools.return_value = [
            AgentTool(
                name="postgres_query",
                description="Provider query",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=query_handler,
                category="database",
                source="plugin",
            ),
            AgentTool(
                name="postgres_count",
                description="Provider count",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=query_handler,
                category="database",
                source="plugin",
            ),
        ]
        mock_agent = MagicMock()
        mock_agent.add_plugin = lambda plugin: mock_agent.tool_registry.register_many(
            plugin.get_tools()
        )
        mock_agent.tool_registry = ToolRegistry()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(
                fac,
                "discover_schema",
                AsyncMock(return_value=_make_normalized_schema(db_type="postgresql")),
            ),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb")

        assert "db_query" in agent.tool_registry.tool_names
        assert "db_compile_and_query" in agent.tool_registry.tool_names
        assert "db_count" in agent.tool_registry.tool_names
        assert "postgres_query" not in agent.tool_registry.tool_names
        assert "postgres_count" not in agent.tool_registry.tool_names

    async def test_from_db_hides_mongo_write_provider_tools(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import AgentTool, ToolRegistry

        async def handler(args):
            return {}

        mock_plugin = MagicMock()
        mock_plugin.read_only = False
        mock_plugin.connect = AsyncMock()
        mock_plugin._tool_find = AsyncMock(return_value={"documents": []})
        mock_plugin._tool_aggregate = AsyncMock(return_value={"results": []})
        mock_plugin._tool_count = AsyncMock(return_value={"count": 0})
        mock_plugin.get_tools.return_value = [
            AgentTool(
                name="mongodb_find",
                description="Provider find",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=handler,
                category="database",
                source="plugin",
            ),
            AgentTool(
                name="mongodb_insert",
                description="Provider insert",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=handler,
                category="database",
                source="plugin",
            ),
        ]
        mock_agent = MagicMock()
        mock_agent.add_plugin = lambda plugin: mock_agent.tool_registry.register_many(
            plugin.get_tools()
        )
        mock_agent.tool_registry = ToolRegistry()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(
                fac,
                "discover_schema",
                AsyncMock(return_value=_make_normalized_schema(db_type="mongodb")),
            ),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("mongodb://localhost/testdb", read_only=False)

        assert "db_find" in agent.tool_registry.tool_names
        assert "mongodb_find" not in agent.tool_registry.tool_names
        assert "mongodb_insert" not in agent.tool_registry.tool_names


class TestDbResultCompaction:
    def test_compacts_large_db_rows_for_llm_context(self):
        from daita.agents.db.runtime.result_compaction import (
            compact_tool_result_for_context,
        )

        result = {
            "rows": [
                {
                    "id": i,
                    "status": "ok",
                    "workflow_context": {"large": "x" * 2000},
                    "notes": "n" * 500,
                }
                for i in range(30)
            ],
            "total_rows": 30,
            "truncated": False,
            "sql": "SELECT * FROM runs",
        }

        compacted = compact_tool_result_for_context("db_query", result)

        assert compacted["row_count"] == 30
        assert compacted["returned_rows"] <= 20
        assert compacted["truncated"] is True
        assert "workflow_context" in compacted["omitted_columns"]
        assert "rows" not in compacted
        assert len(compacted["rows_preview"][0]["notes"]) <= 300

    def test_non_db_tool_results_are_unchanged(self):
        from daita.agents.db.runtime.result_compaction import (
            compact_tool_result_for_context,
        )

        result = {"rows": [{"x": "y" * 1000}]}

        assert compact_tool_result_for_context("send_slack", result) is result

    def test_compacts_mongo_aggregate_results_shape(self):
        from daita.agents.db.runtime.result_compaction import (
            compact_tool_result_for_context,
        )

        compacted = compact_tool_result_for_context(
            "db_aggregate",
            {"results": [{"bucket": i, "payload": "x" * 500} for i in range(25)]},
        )

        assert "results" not in compacted
        assert compacted["row_count"] == 25
        assert compacted["returned_rows"] <= 20
        assert compacted["truncated"] is True
        assert "rows_preview" in compacted

    def test_join_path_compaction_preserves_predicates(self):
        from daita.agents.db.runtime.result_compaction import (
            compact_tool_result_for_context,
        )

        compacted = compact_tool_result_for_context(
            "catalog_find_join_paths",
            {
                "success": True,
                "path_count": 1,
                "paths": [
                    {
                        "tables": ["operations", "api_keys", "users"],
                        "joins": [
                            {
                                "left_table": "operations",
                                "left_column": "api_key_id",
                                "right_table": "api_keys",
                                "right_column": "api_key_id",
                                "predicate": "operations.api_key_id = api_keys.api_key_id",
                                "extra": "x" * 2000,
                            }
                        ],
                        "predicate": "operations.api_key_id = api_keys.api_key_id",
                        "confidence": 0.9,
                        "warnings": ["creator path"],
                    }
                ],
            },
        )

        path = compacted["paths"][0]
        assert path["predicate"] == "operations.api_key_id = api_keys.api_key_id"
        assert path["joins"][0]["predicate"] == (
            "operations.api_key_id = api_keys.api_key_id"
        )
        assert "extra" not in path["joins"][0]


class TestFromDbToolProfiles:
    def test_catalog_schema_tools_are_not_terminal(self):
        from daita.agents.db.runtime.run_context import TERMINAL_DB_TOOLS

        assert "db_compile_and_query" in TERMINAL_DB_TOOLS
        assert "db_query" in TERMINAL_DB_TOOLS
        assert "catalog_inspect_table" not in TERMINAL_DB_TOOLS
        assert "catalog_find_join_paths" not in TERMINAL_DB_TOOLS

    def test_clear_data_prompt_with_schema_match_starts_with_compile_only(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            _make_normalized_schema(
                tables=[
                    _table(
                        "sales",
                        columns=[
                            {"name": "sale_id", "type": "integer"},
                            {"name": "revenue", "type": "numeric"},
                            {"name": "created_at", "type": "timestamp"},
                        ],
                    )
                ]
            ),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_plan_query",
                    "db_compile_and_query",
                    "db_query",
                    "db_validate_sql",
                    "db_count",
                    "db_sample",
                    "catalog_search_schema",
                    "catalog_inspect_table",
                    "catalog_find_join_paths",
                    "pivot_table",
                    "forecast_trend",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(agent, "What were sales last month?")

        assert selected == ["db_compile_and_query"]

    def test_data_prompt_without_schema_match_adds_schema_search_only(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_plan_query",
                    "db_compile_and_query",
                    "db_query",
                    "db_count",
                    "catalog_search_schema",
                    "catalog_inspect_table",
                    "catalog_find_join_paths",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(agent, "What were sales last month?")

        assert selected == ["db_compile_and_query", "catalog_search_schema"]

    def test_explicit_sql_prompt_keeps_manual_sql_tools(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_compile_and_query",
                    "db_plan_query",
                    "db_query",
                    "db_validate_sql",
                    "catalog_search_schema",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent, "Run this SQL: SELECT COUNT(*) FROM orders"
        )

        assert selected == [
            "db_plan_query",
            "db_query",
            "db_validate_sql",
            "db_compile_and_query",
        ]

    def test_join_prompt_keeps_relationship_navigation_tools(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders"), _table("customers")]),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_compile_and_query",
                    "db_plan_query",
                    "db_query",
                    "catalog_search_schema",
                    "catalog_inspect_table",
                    "catalog_find_join_paths",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent, "Show sales by customer and find the join path"
        )

        assert selected == [
            "db_plan_query",
            "db_query",
            "db_compile_and_query",
            "catalog_search_schema",
            "catalog_inspect_table",
            "catalog_find_join_paths",
        ]

    def test_schema_prompt_selects_schema_tools_only(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            {"tables": []},
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_query",
                    "db_count",
                    "catalog_search_schema",
                    "catalog_inspect_table",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent, "What can you tell me about my traces table?"
        )

        assert selected == ["catalog_search_schema", "catalog_inspect_table"]

    def test_explicit_validate_sql_mention_is_preserved(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            tool_registry=SimpleNamespace(
                tool_names=["db_plan_query", "db_query", "db_validate_sql"]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent, "Use db_validate_sql for this orders query"
        )

        assert selected == ["db_plan_query", "db_query", "db_validate_sql"]

    def test_analysis_prompt_adds_matching_analyst_tool(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            {"tables": []},
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_query",
                    "catalog_search_schema",
                    "detect_anomalies",
                    "forecast_trend",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent, "Forecast the revenue trend for next quarter"
        )

        assert "forecast_trend" in selected
        assert "detect_anomalies" not in selected

    def test_write_prompt_adds_execute_when_available(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            {"tables": []},
            tool_registry=SimpleNamespace(
                tool_names=["db_query", "db_execute", "catalog_search_schema"]
            ),
        )

        selected = select_db_tools_for_prompt(agent, "Update customer status")

        assert "db_execute" in selected

    def test_explicit_tool_name_mentions_are_preserved(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            {"tables": []},
            tool_registry=SimpleNamespace(
                tool_names=["db_query", "dq_detect_anomaly", "detect_anomalies"]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent, "Use dq_detect_anomaly on daily_metrics revenue"
        )

        assert "dq_detect_anomaly" in selected

    def test_strict_explicit_tool_request_uses_only_named_tool(self):
        from types import SimpleNamespace
        from daita.agents.db.config.tool_profiles import select_db_tools_for_prompt

        agent = _agent_with_catalog(
            _make_normalized_schema(tables=[_table("orders")]),
            tool_registry=SimpleNamespace(
                tool_names=[
                    "db_compile_and_query",
                    "db_plan_query",
                    "db_query",
                    "db_count",
                    "catalog_search_schema",
                ]
            ),
        )

        selected = select_db_tools_for_prompt(
            agent,
            "Use db_query exactly once to answer which customer has top revenue",
        )

        assert selected == ["db_query"]


class TestPivotTableTool:
    async def test_basic_pivot(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.analyst.pivot_table import create_pivot_table_tool

        rows = [
            {"category": "A", "month": "Jan", "revenue": 100},
            {"category": "A", "month": "Feb", "revenue": 200},
            {"category": "B", "month": "Jan", "revenue": 150},
            {"category": "B", "month": "Feb", "revenue": 50},
        ]
        plugin = _mock_plugin_with_query(rows)
        tool = create_pivot_table_tool(plugin, _analyst_context())

        result = await tool.handler(
            {
                "sql": "SELECT * FROM orders",
                "rows": "category",
                "columns": "month",
                "values": "revenue",
                "aggfunc": "sum",
            }
        )

        assert result["success"] is True
        assert result["row_count"] > 0
        assert "pivot" in result

    async def test_missing_sql_returns_error(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.analyst.pivot_table import create_pivot_table_tool

        plugin = _mock_plugin_with_query([])
        tool = create_pivot_table_tool(plugin, _analyst_context())
        result = await tool.handler({"rows": "a", "columns": "b", "values": "c"})
        assert result["success"] is False
        assert "sql" in result["error"]

    async def test_graceful_degradation_without_pandas(self):
        from daita.agents.db.tools.analyst.pivot_table import create_pivot_table_tool

        plugin = _mock_plugin_with_query([])
        tool = create_pivot_table_tool(plugin, _analyst_context())

        with patch(
            "daita.agents.db.tools.analyst.pivot_table.ensure_pandas",
            side_effect=ImportError("pandas not found"),
        ):
            result = await tool.handler(
                {
                    "sql": "SELECT 1",
                    "rows": "a",
                    "columns": "b",
                    "values": "c",
                }
            )
        assert result["success"] is False
        assert (
            "pandas" in result["error"].lower()
            or "not found" in result["error"].lower()
        )


class TestCorrelateTool:
    async def test_basic_correlation(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.analyst.correlate import create_correlate_tool

        rows = [{"x": i, "y": i * 2, "z": -i} for i in range(1, 21)]
        plugin = _mock_plugin_with_query(rows)
        tool = create_correlate_tool(plugin, _analyst_context())

        result = await tool.handler({"sql": "SELECT x, y, z FROM t"})

        assert result["success"] is True
        assert len(result["correlations"]) > 0
        # x and y should be perfectly correlated
        xy = next(
            (
                p
                for p in result["correlations"]
                if set([p["column_a"], p["column_b"]]) == {"x", "y"}
            ),
            None,
        )
        assert xy is not None
        assert abs(xy["correlation"]) > 0.99

    async def test_missing_sql_returns_error(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        from daita.agents.db.tools.analyst.correlate import create_correlate_tool

        plugin = _mock_plugin_with_query([])
        tool = create_correlate_tool(plugin, _analyst_context())
        result = await tool.handler({})
        assert result["success"] is False


class TestDetectAnomaliesTool:
    async def test_zscore_detects_outlier(self):
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from daita.agents.db.tools.analyst.detect_anomalies import (
            create_detect_anomalies_tool,
        )

        # Values mostly 10, with one extreme outlier at 1000
        rows = [{"val": 10 + i % 3, "id": i} for i in range(50)]
        rows.append({"val": 1000, "id": 999})
        plugin = _mock_plugin_with_query(rows)
        tool = create_detect_anomalies_tool(plugin, _analyst_context())

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

        from daita.agents.db.tools.analyst.detect_anomalies import (
            create_detect_anomalies_tool,
        )

        rows = [{"val": 5, "id": i} for i in range(40)]
        rows.append({"val": 200, "id": 999})
        plugin = _mock_plugin_with_query(rows)
        tool = create_detect_anomalies_tool(plugin, _analyst_context())

        result = await tool.handler(
            {"sql": "SELECT * FROM t", "column": "val", "method": "iqr"}
        )
        assert result["success"] is True
        assert result["anomaly_count"] >= 1

    async def test_graceful_degradation_without_numpy(self):
        from daita.agents.db.tools.analyst.detect_anomalies import (
            create_detect_anomalies_tool,
        )

        plugin = _mock_plugin_with_query([])
        tool = create_detect_anomalies_tool(plugin, _analyst_context())

        with patch(
            "daita.agents.db.tools.analyst.detect_anomalies.ensure_numpy",
            side_effect=ImportError("numpy not found"),
        ):
            result = await tool.handler({"sql": "SELECT 1", "column": "val"})
        assert result["success"] is False


class TestHelpersInferDimensions:
    def test_infer_dimensions_uses_catalog_relationships(self):
        from daita.agents.db.tools.analyst._helpers import infer_dimensions

        dims = infer_dimensions(_analyst_context(), "customers")

        aliases = [d["alias"] for d in dims]
        assert "orders_count" in aliases
        assert "orders_total_sum" in aliases

    def test_infer_dimensions_with_fk_schema(self):
        from daita.agents.db.tools.analyst._helpers import infer_dimensions

        dims = infer_dimensions(_analyst_context(), "customers")

        aliases = [d["alias"] for d in dims]
        assert "orders_count" in aliases
        # Should include at least the count
        assert len(dims) >= 1

    def test_infer_dimensions_no_fk(self):
        from daita.agents.db.tools.analyst._helpers import infer_dimensions

        dims = infer_dimensions(
            _analyst_context(
                _make_normalized_schema(
                    tables=[_table("standalone")],
                    fks=[],
                )
            ),
            "standalone",
        )
        assert dims == []

    def test_get_pk_column(self):
        from daita.agents.db.tools.analyst._helpers import get_pk_column

        context = _analyst_context()
        assert get_pk_column(context, "customers") == "id"
        assert get_pk_column(context, "orders") == "id"
        assert get_pk_column(context, "nonexistent") is None

    def test_get_numeric_columns(self):
        from daita.agents.db.tools.analyst._helpers import (
            find_column,
            get_numeric_columns,
        )

        numeric = get_numeric_columns(_analyst_context(), "orders")
        assert "total" in numeric
        # id is integer — also numeric
        assert "id" in numeric

        wide_columns = [
            {"name": f"feature_{idx:03d}", "type": "text"} for idx in range(240)
        ]
        wide_columns.append({"name": "late_metric", "type": "numeric"})
        wide_columns.append({"name": "blocked_metric", "type": "numeric"})
        context = _analyst_context(
            _make_normalized_schema(
                tables=[_table("wide_events", columns=wide_columns)],
            )
        )
        context.catalog._db_blocked_columns = {"blocked_metric"}

        assert find_column(context, "wide_events", "late_metric") is not None
        wide_numeric = get_numeric_columns(context, "wide_events")
        assert "late_metric" in wide_numeric
        assert "blocked_metric" not in wide_numeric

    def test_to_serializable(self):
        from decimal import Decimal
        from datetime import date
        from daita.agents.db.tools.analyst._helpers import to_serializable

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

        from daita.agents.db.tools.analyst.forecast_trend import (
            create_forecast_trend_tool,
        )

        rows = [
            {"month": f"2024-{str(i).zfill(2)}-01", "revenue": 1000 + i * 200}
            for i in range(1, 13)
        ]
        plugin = _mock_plugin_with_query(rows)
        tool = create_forecast_trend_tool(plugin, _analyst_context())

        result = await tool.handler(
            {
                "sql": "SELECT month, revenue FROM orders",
                "date_column": "month",
                "metric_column": "revenue",
                "periods": 3,
            }
        )

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

        from daita.agents.db.tools.analyst.forecast_trend import (
            create_forecast_trend_tool,
        )

        plugin = _mock_plugin_with_query([{"d": "2024-01-01", "v": 100}])
        tool = create_forecast_trend_tool(plugin, _analyst_context())

        result = await tool.handler(
            {
                "sql": "SELECT d, v FROM t",
                "date_column": "d",
                "metric_column": "v",
            }
        )
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
            "pivot_table",
            "correlate",
            "detect_anomalies",
            "compare_entities",
            "find_similar",
            "forecast_trend",
        }
        registered = set(real_registry.tool_names)
        assert analyst_tools.isdisjoint(
            registered
        ), f"Expected no analyst tools but found: {analyst_tools & registered}"


class TestFromDbModePresets:
    async def test_simple_mode_skips_analyst_tools(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import ToolRegistry

        schema = _analyst_schema()
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_agent.tool_registry = ToolRegistry()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db("postgresql://localhost/testdb", mode="simple")

        assert agent._db_mode == "simple"
        assert "pivot_table" not in agent.tool_registry.tool_names
        assert mock_plugin.query_max_rows == 100
        assert mock_plugin.query_max_chars == 25000

    async def test_mode_explicit_toolkit_override_wins(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac
        from daita.core.tools import ToolRegistry

        schema = _analyst_schema()
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.add_plugin = MagicMock()
        mock_agent.tool_registry = ToolRegistry()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch("daita.agents.agent.Agent", return_value=mock_agent),
        ):
            agent = await from_db(
                "postgresql://localhost/testdb",
                mode="simple",
                toolkit="analyst",
            )

        assert agent._db_mode == "simple"
        assert "pivot_table" in agent.tool_registry.tool_names

    async def test_data_team_mode_registers_quality_tools(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        schema = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[
                        {"name": "id", "type": "integer", "is_primary_key": True},
                        {"name": "total", "type": "numeric"},
                        {"name": "updated_at", "type": "timestamp"},
                    ],
                )
            ]
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_plugin.sql_dialect = "postgresql"

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch(
                "daita.core.graph.backend.auto_select_backend", return_value=MagicMock()
            ),
        ):
            agent = await from_db("postgresql://localhost/testdb", mode="data_team")

        assert agent.db.mode == "data_team"
        assert agent.db.quality is not None
        assert "dq_profile" in agent.tool_registry.tool_names
        description = agent.describe()
        assert "data_quality" in description["capabilities"]
        assert description["db"]["quality_enabled"] is True
        assert description["db"]["query_policy"]["timeout"] == 60

    async def test_data_team_mode_can_enable_db_memory_semantics(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        schema = _make_normalized_schema(
            tables=[
                _table(
                    "orders",
                    columns=[{"name": "total_cents", "type": "numeric"}],
                )
            ]
        )
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()
        mock_plugin.sql_dialect = "postgresql"
        mock_memory = MagicMock()
        mock_memory.backend = None
        mock_memory.recall = AsyncMock(return_value=[])
        mock_memory.remember = AsyncMock(return_value={"status": "ok"})

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
            patch(
                "daita.core.graph.backend.auto_select_backend", return_value=MagicMock()
            ),
        ):
            agent = await from_db(
                "postgresql://localhost/testdb",
                mode="data_team",
                memory=mock_memory,
            )

        assert agent.db.mode == "data_team"
        assert agent.db.memory is mock_memory
        assert agent.db.memory_semantics.plugin is mock_memory

    async def test_mode_quality_override_wins(self):
        from daita.agents.db import from_db
        import daita.agents.db.builder as fac

        schema = _analyst_schema()
        mock_plugin = MagicMock()
        mock_plugin.connect = AsyncMock()

        with (
            patch.object(fac, "resolve_plugin", return_value=(mock_plugin, True)),
            patch.object(fac, "discover_schema", AsyncMock(return_value=schema)),
        ):
            agent = await from_db(
                "postgresql://localhost/testdb",
                mode="data_team",
                quality=False,
            )

        assert agent.db.mode == "data_team"
        assert agent.db.quality is None
        assert "dq_profile" not in agent.tool_registry.tool_names

    async def test_invalid_mode_raises_value_error(self):
        from daita.agents.db import from_db

        with pytest.raises(ValueError, match="Unknown from_db mode"):
            await from_db("postgresql://localhost/testdb", mode="nope")
