"""
Tests for the Database Health Monitor.

Agent creation and tool registration are tested without any external services.
Integration tests that need a real database are skipped automatically when
DATABASE_URL is not set.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Agent creation and tool registration
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_agent_can_be_created(self):
        from agents.monitor_agent import create_agent

        agent = create_agent()
        assert agent is not None
        assert agent.name == "DB Health Monitor"

    def test_watch_state_is_not_agent_owned(self):
        from agents.monitor_agent import create_agent

        agent = create_agent()
        assert not hasattr(agent, "_watches")

    def test_diagnostic_tools_registered(self):
        from agents.monitor_agent import create_agent

        agent = create_agent()
        names = set(agent.tool_names)
        assert "get_slow_queries" in names
        assert "get_connection_stats" in names
        assert "get_table_bloat" in names

    def test_agent_tools_project_before_setup(self):
        from agents.monitor_agent import create_agent

        agent = create_agent()
        names = set(agent.health["tools"]["names"])
        assert {"get_slow_queries", "get_connection_stats", "get_table_bloat"} <= names


# ---------------------------------------------------------------------------
# Tool unit tests — no DB required
# ---------------------------------------------------------------------------


class TestToolSafety:
    async def test_slow_queries_missing_db_url(self, monkeypatch):
        from agents.monitor_agent import get_slow_queries

        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = await get_slow_queries.execute({})
        assert "error" in result
        assert "DATABASE_URL" in result["error"]

    async def test_table_bloat_missing_db_url(self, monkeypatch):
        from agents.monitor_agent import get_table_bloat

        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = await get_table_bloat.execute({})
        assert "error" in result
        assert "DATABASE_URL" in result["error"]

    async def test_connection_stats_missing_db_url(self, monkeypatch):
        from agents.monitor_agent import get_connection_stats

        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = await get_connection_stats.execute({})
        assert "error" in result
        assert "DATABASE_URL" in result["error"]


# ---------------------------------------------------------------------------
# Integration tests — require a real PostgreSQL database
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping database integration tests",
)
class TestIntegration:
    async def test_slow_queries_returns_list(self):
        from agents.monitor_agent import get_slow_queries

        result = await get_slow_queries.execute({"min_duration_seconds": 0})
        assert "error" not in result
        assert "slow_queries" in result
        assert isinstance(result["slow_queries"], list)

    async def test_table_bloat_returns_list(self):
        from agents.monitor_agent import get_table_bloat

        result = await get_table_bloat.execute({"min_dead_tuples": 0})
        assert "error" not in result
        assert "bloated_tables" in result
        assert isinstance(result["bloated_tables"], list)

    async def test_connection_stats_returns_fields(self):
        from agents.monitor_agent import get_connection_stats

        result = await get_connection_stats.execute({})
        assert "error" not in result
        assert "active" in result
        assert "idle" in result
        assert "total" in result
        assert "max_connections" in result
        assert "utilization_pct" in result
