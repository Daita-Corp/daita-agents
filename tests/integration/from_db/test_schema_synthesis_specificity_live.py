"""Live OpenAI regression tests for schema-query synthesis specificity.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_schema_synthesis_specificity_live.py \
        -m "requires_llm and integration" -v -s
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import DbRequest, DbRuntime
from daita.db.llm_service import db_llm_service_from_config
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus

load_dotenv(Path.cwd() / ".env")

pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


def _require_live_openai() -> str:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live OpenAI DB tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")
    return os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini")


def _live_openai_kwargs(model: str) -> dict[str, object]:
    return {
        "llm_provider": "openai",
        "model": model,
        "api_key": os.environ["OPENAI_API_KEY"],
        "temperature": 0,
    }


async def _seed_specificity_schema(sqlite: SQLitePlugin) -> None:
    await sqlite.execute_script("""
        CREATE TABLE operations (
            operation_id INTEGER PRIMARY KEY,
            organization_id INTEGER NOT NULL,
            api_key_id INTEGER,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE runtime_operations (
            runtime_operation_id INTEGER PRIMARY KEY,
            operation_id INTEGER,
            worker_id TEXT NOT NULL,
            started_at TEXT NOT NULL
        );
        CREATE TABLE runtime_monitors (
            monitor_id INTEGER PRIMARY KEY,
            operation_id INTEGER,
            name TEXT NOT NULL,
            enabled INTEGER NOT NULL
        );
        CREATE TABLE audit_logs (
            audit_id INTEGER PRIMARY KEY,
            actor_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO operations (
            operation_id, organization_id, api_key_id, status, timestamp, created_at
        ) VALUES (1, 10, 99, 'succeeded', '2026-06-19T12:00:00Z', '2026-06-19T12:00:00Z');
        INSERT INTO runtime_operations (
            runtime_operation_id, operation_id, worker_id, started_at
        ) VALUES (1, 1, 'worker-alpha', '2026-06-19T12:01:00Z');
        INSERT INTO runtime_monitors (monitor_id, operation_id, name, enabled)
        VALUES (1, 1, 'ops-health', 1);
        INSERT INTO audit_logs (audit_id, actor_id, event_type, created_at)
        VALUES (1, 42, 'operation_created', '2026-06-19T12:00:00Z');
        """)


async def _runtime_for_specificity_schema(tmp_path, model: str) -> DbRuntime:
    db_path = tmp_path / "schema-specificity-live.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await _seed_specificity_schema(sqlite)
    return DbRuntime(
        plugins=(CatalogPlugin(auto_persist=False), sqlite),
        db_llm_service=db_llm_service_from_config(
            model=model,
            llm_provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
        ),
    )


def _synthesis_diagnostics(result) -> dict[str, object]:
    synthesis = next(
        item for item in result.evidence if item.kind == "answer.synthesis"
    )
    diagnostics = synthesis.payload["diagnostics"]
    assert diagnostics["mode"] == "llm", diagnostics.get("fallback_reason")
    assert diagnostics["provider"] == "openai"
    return diagnostics


async def test_live_openai_schema_query_for_operations_table_excludes_unrelated_tables(
    tmp_path,
):
    model = _require_live_openai()
    runtime = await _runtime_for_specificity_schema(tmp_path, model)

    try:
        result = await runtime.run(
            DbRequest(
                "can you tell me about the operations table?",
                mode="schema.query",
            )
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.answer

    diagnostics = _synthesis_diagnostics(result)
    assert diagnostics["context"]["schema_answer_scope"]["mode"] == "asset"
    assert diagnostics["context"]["schema_answer_scope"]["selected_table_names"] == [
        "operations"
    ]

    answer = result.answer.lower()
    assert "operations" in answer
    assert "operation_id" in answer
    assert "runtime_operations" not in answer
    assert "runtime_monitors" not in answer
    assert "audit_logs" not in answer


async def test_live_openai_schema_query_disambiguates_near_collision_table(tmp_path):
    model = _require_live_openai()
    runtime = await _runtime_for_specificity_schema(tmp_path, model)

    try:
        result = await runtime.run(
            DbRequest(
                "tell me about the runtime_operations table",
                mode="schema.query",
            )
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    diagnostics = _synthesis_diagnostics(result)
    assert diagnostics["context"]["schema_answer_scope"]["mode"] == "asset"
    assert diagnostics["context"]["schema_answer_scope"]["selected_table_names"] == [
        "runtime_operations"
    ]

    answer = result.answer.lower()
    assert "runtime_operations" in answer
    assert "runtime_operation_id" in answer
    assert "runtime_monitors" not in answer
    assert "audit_logs" not in answer


async def test_live_openai_database_wide_schema_prompt_keeps_broad_scope(tmp_path):
    model = _require_live_openai()
    runtime = await _runtime_for_specificity_schema(tmp_path, model)

    try:
        result = await runtime.run(DbRequest("what tables exist?", mode="schema.query"))
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    diagnostics = _synthesis_diagnostics(result)
    assert diagnostics["context"]["schema_answer_scope"]["mode"] == "database"
    selected = set(
        diagnostics["context"]["schema_answer_scope"]["selected_table_names"]
    )
    assert {
        "operations",
        "runtime_operations",
        "runtime_monitors",
        "audit_logs",
    } <= selected

    answer = result.answer.lower()
    assert "operations" in answer
    assert "runtime_operations" in answer
    assert "runtime_monitors" in answer
    assert "audit_logs" in answer


async def test_live_openai_missing_table_prompt_does_not_dump_full_schema(tmp_path):
    model = _require_live_openai()
    runtime = await _runtime_for_specificity_schema(tmp_path, model)

    try:
        result = await runtime.run(
            DbRequest("tell me about the opertion table", mode="schema.query")
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    diagnostics = _synthesis_diagnostics(result)
    assert diagnostics["context"]["schema_answer_scope"]["mode"] == "ambiguous"

    answer = result.answer.lower()
    assert "operations" in answer
    assert "runtime_monitors:" not in answer
    assert "audit_logs:" not in answer


async def test_live_openai_agent_from_db_schema_query_uses_same_scope(tmp_path):
    model = _require_live_openai()
    db_path = tmp_path / "schema-specificity-agent-live.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await _seed_specificity_schema(sqlite)
    await sqlite.disconnect()
    agent = await Agent.from_db(
        str(db_path),
        name="LiveSchemaSpecificityAgent",
        cache_ttl=3600,
        **_live_openai_kwargs(model),
    )

    try:
        result = await agent.run_detailed(
            "can you tell me about the operations table?",
            mode="schema.query",
        )
    finally:
        await agent.stop()

    assert result.status is OperationStatus.SUCCEEDED
    diagnostics = _synthesis_diagnostics(result)
    assert diagnostics["context"]["schema_answer_scope"]["mode"] == "asset"
    assert diagnostics["context"]["schema_answer_scope"]["selected_table_names"] == [
        "operations"
    ]

    answer = (result.answer or "").lower()
    assert "operations" in answer
    assert "operation_id" in answer
    assert "runtime_operations" not in answer
    assert "runtime_monitors" not in answer
    assert "audit_logs" not in answer
