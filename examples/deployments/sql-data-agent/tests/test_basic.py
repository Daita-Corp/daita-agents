"""
Tests for the SQL Data Agent.

Tool functions are tested directly (no LLM required).
Integration tests that need a real database or LLM are skipped automatically
when the relevant environment variables are not set.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_agent_can_be_created(self):
        from agents.sql_agent import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "SQL Data Agent"

    def test_agent_tool_registry_populated(self):
        from agents.sql_agent import create_agent, inspect_schema, run_query

        # Verify the tool objects exist and have the expected names
        assert inspect_schema.name == "inspect_schema"
        assert run_query.name == "run_query"
        # Verify create_agent runs without error
        agent = create_agent()
        assert agent is not None


# ---------------------------------------------------------------------------
# Tool unit tests — run_query safety checks (no DB required)
# ---------------------------------------------------------------------------


class TestRunQuerySafety:
    async def test_rejects_insert(self):
        from agents.sql_agent import run_query

        result = await run_query.execute({"sql": "INSERT INTO users VALUES (1, 'alice')"})
        assert "error" in result
        assert "SELECT" in result["error"]

    async def test_rejects_update(self):
        from agents.sql_agent import run_query

        result = await run_query.execute({"sql": "UPDATE users SET name = 'x' WHERE id = 1"})
        assert "error" in result

    async def test_rejects_drop(self):
        from agents.sql_agent import run_query

        result = await run_query.execute({"sql": "DROP TABLE users"})
        assert "error" in result

    async def test_rejects_delete(self):
        from agents.sql_agent import run_query

        result = await run_query.execute({"sql": "DELETE FROM users"})
        assert "error" in result

    async def test_missing_database_url(self, monkeypatch):
        from agents.sql_agent import run_query

        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = await run_query.execute({"sql": "SELECT 1"})
        assert "error" in result
        assert "DATABASE_URL" in result["error"]


# ---------------------------------------------------------------------------
# Integration tests — require a real PostgreSQL database
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping database integration tests",
)
class TestInspectSchema:
    async def test_returns_tables(self):
        from agents.sql_agent import inspect_schema

        result = await inspect_schema.execute({})
        assert "error" not in result
        assert "tables" in result
        assert isinstance(result["tables"], list)

    async def test_table_filter(self):
        from agents.sql_agent import inspect_schema

        result = await inspect_schema.execute({"table_filter": "order"})
        assert "error" not in result
        # All returned tables should contain "order" in their name
        for table in result["tables"]:
            assert "order" in table["table"].lower()

    async def test_columns_have_required_fields(self):
        from agents.sql_agent import inspect_schema

        result = await inspect_schema.execute({})
        for table in result["tables"]:
            for col in table["columns"]:
                assert "name" in col
                assert "type" in col
                assert "nullable" in col


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping database integration tests",
)
class TestRunQuery:
    async def test_select_one(self):
        from agents.sql_agent import run_query

        result = await run_query.execute({"sql": "SELECT 1 AS value"})
        assert "error" not in result
        assert result["row_count"] == 1
        assert result["rows"][0]["value"] == 1

    async def test_max_rows_clamped(self):
        from agents.sql_agent import run_query

        result = await run_query.execute({"sql": "SELECT generate_series(1, 10) AS n", "max_rows": 3})
        assert "error" not in result
        assert result["row_count"] <= 3
        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# LLM integration tests — require OPENAI_API_KEY and DATABASE_URL
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("DATABASE_URL")),
    reason="OPENAI_API_KEY and DATABASE_URL both required for agent integration tests",
)
class TestAgentIntegration:
    async def test_schema_question(self):
        from agents.sql_agent import create_agent

        agent = create_agent()
        await agent.start()
        try:
            result = await agent.run("What tables are in this database?")
            assert isinstance(result, str)
            assert len(result) > 10
        finally:
            await agent.stop()
