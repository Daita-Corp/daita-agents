"""Local smoke tests for the data-team-agent deployment template."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.data_team_agent import (  # noqa: E402
    MONITOR_ID,
    create_data_team_agent,
    default_paths,
    ensure_pending_orders_monitor,
    pending_orders_observation,
    seed_local_sqlite,
)


def test_imports_and_monitor_observation_shape() -> None:
    observation = pending_orders_observation()

    assert observation["kind"] == "metric_sql"
    assert observation["capability_owner"] == "sqlite"
    assert observation["source_scope"] == ["orders"]


def test_seed_local_sqlite_creates_copyable_fixture(tmp_path) -> None:
    db_path = asyncio.run(seed_local_sqlite(tmp_path / "team_sales.sqlite"))

    with sqlite3.connect(db_path) as connection:
        customer_count = connection.execute("select count(*) from customers").fetchone()
        order_count = connection.execute("select count(*) from orders").fetchone()

    assert customer_count == (5,)
    assert order_count == (6,)


def test_create_agent_registers_local_runtime_plugins_and_monitor(tmp_path) -> None:
    async def check() -> None:
        paths = default_paths(tmp_path)
        await seed_local_sqlite(paths.db_path)
        agent = await create_data_team_agent(paths, use_live_llm=False)
        try:
            inspection = await agent.describe()
            monitor = await ensure_pending_orders_monitor(agent)
            monitor_inspection = await agent.inspect_monitor(MONITOR_ID)
        finally:
            await agent.stop()

        assert inspection.profile == "data_team"
        assert {"catalog", "sqlite", "data_quality", "lineage", "memory"} <= set(
            inspection.plugin_ids
        )
        assert "data_quality:quality.profile" in inspection.capability_ids
        assert "lineage:lineage.trace" in inspection.capability_ids
        assert "memory:memory.semantic.write" in inspection.capability_ids
        assert monitor.id == MONITOR_ID
        assert monitor_inspection is not None
        assert monitor_inspection.monitor.observation_plan["kind"] == "metric_sql"
        assert paths.runtime_store_path.exists()

    asyncio.run(check())
