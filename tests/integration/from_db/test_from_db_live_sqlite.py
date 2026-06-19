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
from daita.agents.conversation import ConversationHistory
from daita.db import DbIntentKind
from daita.db.session_context import db_session_context_from_request
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
    assert {
        "schema.asset_profile",
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(result)
    assert {"db.schema.inspect", "db.sql.validate", "db.sql.execute_read"} <= set(
        _task_capabilities(result)
    )
    assert result.diagnostics["verification"]["passed"] is True
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
            "Join orders to customers using their relationship and return records"
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
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(result)
    assert "catalog.relationship_paths.find" in _task_capabilities(result)
    assert result.diagnostics["verification"]["passed"] is True


async def test_from_db_live_grounds_completed_orders_to_observed_status(tmp_path):
    db_path = tmp_path / "sales.sqlite"
    await _seed_sales_db(db_path)
    agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbValueGrounding",
        cache_ttl=0,
        **_require_live_openai(),
    )

    try:
        result = await agent.run_detailed("Show completed orders by status")
    finally:
        await agent.stop()

    query_result = next(item for item in result.evidence if item.kind == "query.result")
    sql_validation = next(
        item for item in result.evidence if item.kind == "sql.validation"
    )
    statuses = {row.get("status") for row in query_result.payload.get("rows", [])}

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.DATA_QUERY
    assert statuses == {"complete"}
    assert (
        len(query_result.payload["rows"]) == 3
        or query_result.payload["rows"][0].get("completed_orders") == 3
    )
    assert "'complete'" in sql_validation.payload["sql"]
    assert "'completed'" not in sql_validation.payload["sql"]
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
    assert {
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
        "query.result",
    } <= _evidence_kinds(resolved)

    assert bounded_fallback.status is OperationStatus.SUCCEEDED
    assert bounded_fallback.intent.kind is DbIntentKind.CONVERSATIONAL
    assert (
        bounded_fallback.answer == "The DB operation completed with verified evidence."
    )
    assert "schema.asset_profile" in _evidence_kinds(bounded_fallback)
    assert "query.result" not in _evidence_kinds(bounded_fallback)


async def test_from_db_live_stateful_and_stateless_session_context(tmp_path):
    db_path = tmp_path / "sales.sqlite"
    await _seed_sales_db(db_path)
    live_openai = _require_live_openai()
    stateful_agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbStateful",
        stateful=True,
        cache_ttl=0,
        **live_openai,
    )
    stateless_agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbStateless",
        cache_ttl=0,
        **live_openai,
    )
    explicit_history_agent = await Agent.from_db(
        str(db_path),
        name="LiveFromDbExplicitHistory",
        cache_ttl=0,
        **live_openai,
    )
    explicit_history = ConversationHistory(
        session_id="live-explicit-history",
        workspace="LiveFromDbExplicitHistory",
    )

    try:
        stateful_first = await stateful_agent.run_detailed(
            "What columns are in customers?",
            mode="schema.query",
        )
        stateful_second = await stateful_agent.run_detailed(
            "Please detail them.",
            mode="schema.query",
        )
        stateless_first = await stateless_agent.run_detailed(
            "What columns are in customers?",
            mode="schema.query",
        )
        stateless_second = await stateless_agent.run_detailed(
            "Please detail them.",
            mode="schema.query",
        )
        explicit_override = await explicit_history_agent.run_detailed(
            "How many customers are there?",
            history=explicit_history,
            session_id="live-explicit-session",
        )
    finally:
        await stateful_agent.stop()
        await stateless_agent.stop()
        await explicit_history_agent.stop()

    assert stateful_first.status is OperationStatus.SUCCEEDED
    assert stateful_second.status is OperationStatus.SUCCEEDED
    assert stateful_first.request.session_id == stateful_second.request.session_id
    assert stateful_agent._default_history is not None
    assert stateful_agent._default_history.turn_count == 2
    assert "customers" in (stateful_second.answer or "")
    assert "orders" not in (stateful_second.answer or "")
    stateful_context = db_session_context_from_request(stateful_second.request)
    assert stateful_context is not None
    assert "customers" in stateful_context.referents.tables
    assert (
        "runtime.evidence"
        in stateful_context.diagnostics["referent_sources"]["tables"].values()
    )
    assert "conversation_messages" not in (
        stateful_second.request.session_context or {}
    )

    assert stateless_first.status is OperationStatus.SUCCEEDED
    assert stateless_second.status is OperationStatus.SUCCEEDED
    assert stateless_first.request.session_id is None
    assert stateless_second.request.session_id is None
    assert stateless_agent._default_history is None
    stateless_context = db_session_context_from_request(stateless_second.request)
    assert stateless_context is not None
    assert stateless_context.referents.tables == ()
    assert "customers" in (stateless_second.answer or "")
    assert "orders" in (stateless_second.answer or "")

    assert explicit_override.status is OperationStatus.SUCCEEDED
    assert explicit_override.request.session_id == "live-explicit-session"
    assert explicit_history.turn_count == 1
    assert explicit_history.messages[-2:] == [
        {"role": "user", "content": "How many customers are there?"},
        {"role": "assistant", "content": explicit_override.answer or ""},
    ]
