"""Live-gated integration tests for ``Agent.from_db`` over SQLite.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_from_db_live_sqlite.py \
        -m "requires_llm and integration" -v -s

These tests pass live LLM configuration into ``Agent.from_db`` so the suite is
ready for model-backed DB planning/synthesis. The current DB runtime path is
deterministic, so assertions focus on runtime-owned tasks, evidence, plugins,
and diagnostics rather than generic chat ``tool_calls``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import DbIntentKind
from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus

load_dotenv(Path.cwd() / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


def _require_live_openai() -> dict[str, object]:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db tests")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm_provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


async def _seed_sales_db(path: Path) -> None:
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            total REAL NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO customers (id, name, region) VALUES
            (1, 'Ada', 'NA'),
            (2, 'Linus', 'EU'),
            (3, 'Grace', 'NA');
        INSERT INTO orders (customer_id, total, status) VALUES
            (1, 120.00, 'complete'),
            (1, 80.00, 'pending'),
            (2, 50.00, 'complete'),
            (3, 175.00, 'complete');
        """)
    await plugin.disconnect()


def _memory_plugin(tmp_path: Path) -> MemoryPlugin:
    embedder = MockEmbeddingProvider(dim=8)
    plugin = MemoryPlugin(workspace="from-db-live-memory", embedder=embedder)
    plugin.backend = LocalMemoryBackend(
        workspace="from-db-live-memory",
        agent_id="from-db-live-memory",
        scope="project",
        base_dir=tmp_path,
        embedder=embedder,
    )
    plugin.environment = "local"
    return plugin


def _evidence_kinds(result) -> set[str]:
    return {item.kind for item in result.evidence}


def _task_capabilities(result) -> list[str]:
    tasks = result.diagnostics.get("execution", {}).get("tasks", [])
    return [task.get("capability_id", "") for task in tasks]


async def test_from_db_live_query_uses_runtime_tasks_and_evidence(tmp_path):
    db_path = tmp_path / "sales.sqlite"
    await _seed_sales_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbQuery",
        cache_ttl=0,
        **_require_live_openai(),
    )

    try:
        result = await agent.run_detailed("How many customers are there?")
        inspection = await agent.describe()
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.DATA_QUERY
    assert result.answer == "The count is 3."
    assert {"schema.asset_profile", "query.plan", "sql.validation", "query.result"} <= (
        _evidence_kinds(result)
    )
    assert {"db.schema.inspect", "db.sql.validate", "db.sql.execute_read"} <= set(
        _task_capabilities(result)
    )
    assert result.diagnostics["verification"]["passed"] is True
    assert result.diagnostics["execution"]["task_count"] == 3
    assert inspection.operation_count == 1
    assert not hasattr(agent, "tool_names")


async def test_from_db_live_catalog_assisted_join_records_relationship_path(tmp_path):
    db_path = tmp_path / "sales.sqlite"
    await _seed_sales_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbJoin",
        cache_ttl=0,
        **_require_live_openai(),
    )

    try:
        result = await agent.run_detailed(
            "Join orders to customers using their relationship"
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.CATALOG_ASSISTED_DATA_QUERY
    assert result.answer == "Returned 4 rows."
    assert {
        "catalog.source_registered",
        "schema.search_result",
        "schema.relationship_path",
        "query.plan",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(result)
    assert "catalog.relationship_paths.find" in _task_capabilities(result)
    assert result.diagnostics["verification"]["passed"] is True


async def test_from_db_live_plugins_register_and_execute_runtime_capabilities(tmp_path):
    db_path = tmp_path / "sales.sqlite"
    await _seed_sales_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbPlugins",
        mode="data_team",
        quality=True,
        lineage=True,
        memory=_memory_plugin(tmp_path),
        cache_ttl=0,
        **_require_live_openai(),
    )

    try:
        inspection = await agent.describe()
        quality = await agent.run_detailed(
            "Profile the customers table",
            mode="quality.check",
        )
        lineage = await agent.run_detailed(
            "Trace lineage for orders",
            mode="lineage.trace",
        )
        memory = await agent.run_detailed(
            "Remember that revenue excludes tax",
            mode="memory.update",
            metadata={"category": "db_semantics"},
        )
    finally:
        await agent.stop()

    assert inspection.plugin_ids == (
        "catalog",
        "sqlite",
        "data_quality",
        "lineage",
        "memory",
    )
    assert "data_quality:quality.profile" in inspection.capability_ids
    assert "lineage:lineage.trace" in inspection.capability_ids
    assert "memory:memory.semantic.write" in inspection.capability_ids
    assert quality.status is OperationStatus.SUCCEEDED
    assert lineage.status is OperationStatus.SUCCEEDED
    assert memory.status is OperationStatus.SUCCEEDED
    assert "quality.profile" in _evidence_kinds(quality)
    assert "lineage.trace" in _evidence_kinds(lineage)
    assert "memory.semantic.write" in _evidence_kinds(memory)


async def test_from_db_live_resolves_non_descriptive_prompt_without_looping(tmp_path):
    db_path = tmp_path / "sales.sqlite"
    await _seed_sales_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbAmbiguous",
        cache_ttl=0,
        **_require_live_openai(),
    )

    try:
        resolved = await agent.run_detailed("show me customers")
        bounded_fallback = await agent.run_detailed("customers")
    finally:
        await agent.stop()

    assert resolved.status is OperationStatus.SUCCEEDED
    assert resolved.intent.kind is DbIntentKind.DATA_QUERY
    assert resolved.answer == "Returned 3 rows."
    assert {"query.plan", "sql.validation", "query.result"} <= _evidence_kinds(resolved)
    assert resolved.diagnostics["execution"]["task_count"] == 3

    assert bounded_fallback.status is OperationStatus.SUCCEEDED
    assert bounded_fallback.intent.kind is DbIntentKind.CONVERSATIONAL
    assert (
        bounded_fallback.answer == "The DB operation completed with verified evidence."
    )
    assert _evidence_kinds(bounded_fallback) == {"schema.asset_profile"}
    assert bounded_fallback.diagnostics["execution"]["task_count"] == 1
