"""Shared helpers for live ``Agent.from_db`` production contract tests."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import (
    DbAgent,
    DbLLMConfig,
    DbRuntime,
    DbRuntimeConfig,
    DbRuntimeOptions,
    DbSourceOptions,
)
from daita.db.llm_service import db_llm_service_from_config
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.catalog.persistence import catalog_profile_key
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus, SQLiteRuntimeStore

load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.cwd() / ".env.local", override=False)

DEFAULT_LIVE_OPENAI_MODEL = "gpt-5.4-mini"
FULL_LOOP_EVIDENCE = frozenset(
    {
        "planner.decision",
        "planner.compilation",
        "query.result",
        "verification.result",
        "answer.synthesis",
    }
)
UNSAFE_SQL_VERBS = frozenset(
    {"DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE"}
)
WRITE_EVIDENCE_KINDS = frozenset({"write.execution"})
WRITE_CAPABILITY_IDS = frozenset(
    {
        "db.sql.execute_write",
        "db.write.execute",
        "write.execute",
        "write.execution",
    }
)
SQL_EVIDENCE_KINDS = (
    "sql.validation",
    "query.result",
    "query.plan.validation",
)
PLANNED_SQL_EVIDENCE_KINDS = ("query.plan.proposal",)
PLANNER_ARTIFACTS = {
    "planner.decision": "planner_decisions.json",
    "planner.compilation": "planner_compilations.json",
}
RICH_POSTGRES_SEED_SQL = """
DROP TABLE IF EXISTS monitor_actions;
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS runtime_monitors;
DROP TABLE IF EXISTS runtime_operations;
DROP TABLE IF EXISTS operations;
DROP TABLE IF EXISTS support_tickets;
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS regions;

CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    region_code TEXT NOT NULL UNIQUE,
    region_name TEXT NOT NULL
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region_id INTEGER NOT NULL REFERENCES regions(region_id),
    tier TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    sku TEXT NOT NULL UNIQUE
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL,
    total NUMERIC(10, 2) NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE refunds (
    refund_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    amount NUMERIC(10, 2) NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(product_id)
);

CREATE TABLE operations (
    operation_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    api_key_id TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE runtime_operations (
    runtime_operation_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
    worker_id TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE TABLE runtime_monitors (
    monitor_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
    name TEXT NOT NULL,
    enabled INTEGER NOT NULL
);

CREATE TABLE audit_logs (
    audit_id INTEGER PRIMARY KEY,
    actor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE monitor_actions (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    note TEXT NOT NULL
);

INSERT INTO regions (region_id, region_code, region_name) VALUES
    (1, 'NA', 'North America'),
    (2, 'EU', 'Europe'),
    (3, 'APAC', 'Asia Pacific');

INSERT INTO customers (
    customer_id, name, region_id, tier, email, phone
) VALUES
    (1, 'Ada Lovelace', 1, 'enterprise', 'ada@example.com', '+1-555-0101'),
    (2, 'Linus Torvalds', 2, 'startup', 'linus@example.com', '+358-555-0102'),
    (3, 'Grace Hopper', 1, 'enterprise', 'grace@example.com', '+1-555-0103'),
    (4, 'Katherine Johnson', 3, 'midmarket', 'katherine@example.com', '+61-555-0104');

INSERT INTO products (product_id, category, sku) VALUES
    (10, 'analytics', 'AN-100'),
    (11, 'automation', 'AU-200'),
    (12, 'governance', 'GV-300');

INSERT INTO orders (
    order_id, customer_id, status, total, created_at, updated_at
) VALUES
    (100, 1, 'complete', 120.00, '2026-01-02T10:00:00Z', '2026-01-02T10:00:00Z'),
    (101, 1, 'pending', 80.00, '2026-01-03T11:00:00Z', '2026-01-03T11:30:00Z'),
    (102, 2, 'complete', 50.00, '2026-01-04T09:00:00Z', '2026-01-04T09:05:00Z'),
    (103, 3, 'complete', 175.00, '2026-01-05T13:00:00Z', '2026-01-05T13:00:00Z'),
    (104, 4, 'pending', 210.00, '2026-01-06T15:00:00Z', '2026-01-06T15:20:00Z');

INSERT INTO refunds (refund_id, order_id, amount, reason) VALUES
    (1000, 102, 10.00, 'courtesy_credit'),
    (1001, 103, 25.00, 'late_delivery');

INSERT INTO support_tickets (
    ticket_id, customer_id, status, severity, product_id
) VALUES
    (2000, 1, 'open', 'high', 10),
    (2001, 2, 'closed', 'low', 11),
    (2002, 3, 'open', 'low', 12),
    (2003, 4, 'closed', 'high', 10);

INSERT INTO operations (
    operation_id, organization_id, api_key_id, status, timestamp
) VALUES
    ('op_live_1', 'org_001', 'key_redacted_1', 'succeeded', '2026-01-07T10:00:00Z'),
    ('op_live_2', 'org_001', 'key_redacted_2', 'running', '2026-01-07T10:05:00Z');

INSERT INTO runtime_operations (
    runtime_operation_id, operation_id, worker_id, started_at
) VALUES
    ('rt_op_1', 'op_live_1', 'worker_alpha', '2026-01-07T10:00:01Z'),
    ('rt_op_2', 'op_live_2', 'worker_beta', '2026-01-07T10:05:01Z');

INSERT INTO runtime_monitors (
    monitor_id, operation_id, name, enabled
) VALUES
    ('mon_pending_orders', 'op_live_2', 'pending orders monitor', 1);

INSERT INTO audit_logs (audit_id, actor_id, event_type, created_at) VALUES
    (3000, 'user_001', 'operation.created', '2026-01-07T09:59:59Z'),
    (3001, 'worker_alpha', 'operation.completed', '2026-01-07T10:00:30Z');

INSERT INTO monitor_actions (id, status, note) VALUES
    (1, 'pending', 'fixture row for approval and resume tests');
"""


def require_live_openai_kwargs() -> dict[str, object]:
    """Return live OpenAI kwargs or skip when the live gate is unavailable."""
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db tests")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm": DbLLMConfig(
            provider="openai",
            model=os.environ.get("OPENAI_TEST_MODEL", DEFAULT_LIVE_OPENAI_MODEL),
            api_key=api_key,
            temperature=0,
        )
    }


async def seed_rich_sqlite_schema(db_path: Path) -> Path:
    """Create the shared rich SQLite fixture used by live from_db tests."""
    plugin = SQLitePlugin(path=str(db_path))
    await plugin.execute_script("""
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS monitor_actions;
        DROP TABLE IF EXISTS audit_logs;
        DROP TABLE IF EXISTS runtime_monitors;
        DROP TABLE IF EXISTS runtime_operations;
        DROP TABLE IF EXISTS operations;
        DROP TABLE IF EXISTS support_tickets;
        DROP TABLE IF EXISTS refunds;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS regions;

        CREATE TABLE regions (
            region_id INTEGER PRIMARY KEY,
            region_code TEXT NOT NULL UNIQUE,
            region_name TEXT NOT NULL
        );

        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region_id INTEGER NOT NULL REFERENCES regions(region_id),
            tier TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            sku TEXT NOT NULL UNIQUE
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            status TEXT NOT NULL,
            total REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE refunds (
            refund_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(order_id),
            amount REAL NOT NULL,
            reason TEXT NOT NULL
        );

        CREATE TABLE support_tickets (
            ticket_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            product_id INTEGER NOT NULL REFERENCES products(product_id)
        );

        CREATE TABLE operations (
            operation_id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            api_key_id TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE runtime_operations (
            runtime_operation_id TEXT PRIMARY KEY,
            operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            worker_id TEXT NOT NULL,
            started_at TEXT NOT NULL
        );

        CREATE TABLE runtime_monitors (
            monitor_id TEXT PRIMARY KEY,
            operation_id TEXT NOT NULL REFERENCES operations(operation_id),
            name TEXT NOT NULL,
            enabled INTEGER NOT NULL
        );

        CREATE TABLE audit_logs (
            audit_id INTEGER PRIMARY KEY,
            actor_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE monitor_actions (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            note TEXT NOT NULL
        );

        INSERT INTO regions (region_id, region_code, region_name) VALUES
            (1, 'NA', 'North America'),
            (2, 'EU', 'Europe'),
            (3, 'APAC', 'Asia Pacific');

        INSERT INTO customers (
            customer_id, name, region_id, tier, email, phone
        ) VALUES
            (1, 'Ada Lovelace', 1, 'enterprise', 'ada@example.com', '+1-555-0101'),
            (2, 'Linus Torvalds', 2, 'startup', 'linus@example.com', '+358-555-0102'),
            (3, 'Grace Hopper', 1, 'enterprise', 'grace@example.com', '+1-555-0103'),
            (4, 'Katherine Johnson', 3, 'midmarket', 'katherine@example.com', '+61-555-0104');

        INSERT INTO products (product_id, category, sku) VALUES
            (10, 'analytics', 'AN-100'),
            (11, 'automation', 'AU-200'),
            (12, 'governance', 'GV-300');

        INSERT INTO orders (
            order_id, customer_id, status, total, created_at, updated_at
        ) VALUES
            (100, 1, 'complete', 120.00, '2026-01-02T10:00:00Z', '2026-01-02T10:00:00Z'),
            (101, 1, 'pending', 80.00, '2026-01-03T11:00:00Z', '2026-01-03T11:30:00Z'),
            (102, 2, 'complete', 50.00, '2026-01-04T09:00:00Z', '2026-01-04T09:05:00Z'),
            (103, 3, 'complete', 175.00, '2026-01-05T13:00:00Z', '2026-01-05T13:00:00Z'),
            (104, 4, 'pending', 210.00, '2026-01-06T15:00:00Z', '2026-01-06T15:20:00Z');

        INSERT INTO refunds (refund_id, order_id, amount, reason) VALUES
            (1000, 102, 10.00, 'courtesy_credit'),
            (1001, 103, 25.00, 'late_delivery');

        INSERT INTO support_tickets (
            ticket_id, customer_id, status, severity, product_id
        ) VALUES
            (2000, 1, 'open', 'high', 10),
            (2001, 2, 'closed', 'low', 11),
            (2002, 3, 'open', 'low', 12),
            (2003, 4, 'closed', 'high', 10);

        INSERT INTO operations (
            operation_id, organization_id, api_key_id, status, timestamp
        ) VALUES
            ('op_live_1', 'org_001', 'key_redacted_1', 'succeeded', '2026-01-07T10:00:00Z'),
            ('op_live_2', 'org_001', 'key_redacted_2', 'running', '2026-01-07T10:05:00Z');

        INSERT INTO runtime_operations (
            runtime_operation_id, operation_id, worker_id, started_at
        ) VALUES
            ('rt_op_1', 'op_live_1', 'worker_alpha', '2026-01-07T10:00:01Z'),
            ('rt_op_2', 'op_live_2', 'worker_beta', '2026-01-07T10:05:01Z');

        INSERT INTO runtime_monitors (
            monitor_id, operation_id, name, enabled
        ) VALUES
            ('mon_pending_orders', 'op_live_2', 'pending orders monitor', 1);

        INSERT INTO audit_logs (audit_id, actor_id, event_type, created_at) VALUES
            (3000, 'user_001', 'operation.created', '2026-01-07T09:59:59Z'),
            (3001, 'worker_alpha', 'operation.completed', '2026-01-07T10:00:30Z');

        INSERT INTO monitor_actions (id, status, note) VALUES
            (1, 'pending', 'fixture row for approval and resume tests');
    """)
    await plugin.disconnect()
    return db_path


def require_live_postgres_enabled() -> None:
    """Skip unless the release Postgres gate is explicitly enabled."""
    if os.environ.get("DAITA_EVAL_POSTGRES") != "1":
        pytest.skip("Set DAITA_EVAL_POSTGRES=1 to run live Postgres from_db tests")


async def seed_rich_postgres_schema(url: str) -> str:
    """Create the shared rich Postgres fixture used by live from_db tests."""
    asyncpg = pytest.importorskip(
        "asyncpg",
        reason="asyncpg required: pip install 'daita-agents[postgresql]'",
    )
    deadline = time.time() + 30
    last_error: Exception | None = None
    while time.time() < deadline:
        connection = None
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(RICH_POSTGRES_SEED_SQL)
            return url
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
        finally:
            if connection is not None:
                await connection.close()
    raise RuntimeError(f"Could not seed Postgres test database: {last_error}")


async def create_live_sqlite_from_db_agent(
    db_path: Path,
    *,
    runtime_path: Path,
    name: str = "LiveFromDbProductionContract",
    read_only: bool | None = None,
    allowed_tables: tuple[str, ...] | list[str] | None = None,
    blocked_tables: tuple[str, ...] | list[str] | None = None,
    blocked_columns: tuple[str, ...] | list[str] | None = None,
    stateful: bool = False,
):
    """Build a live OpenAI ``Agent.from_db`` over SQLite with persisted state."""
    if read_only is False:
        return await _create_writable_sqlite_runtime_agent(
            db_path,
            runtime_path=runtime_path,
            name=name,
        )
    return await Agent.from_db(
        str(db_path),
        name=name,
        source_options=DbSourceOptions(
            cache_ttl=0,
            read_only=read_only,
            allowed_tables=allowed_tables,
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        ),
        runtime=DbRuntimeOptions(store="sqlite", store_path=runtime_path),
        stateful=stateful,
        **require_live_openai_kwargs(),
    )


async def _create_writable_sqlite_runtime_agent(
    db_path: Path,
    *,
    runtime_path: Path,
    name: str,
) -> DbAgent:
    """Build the direct runtime used by write-governance contract tests."""
    llm = require_live_openai_kwargs()["llm"]
    assert isinstance(llm, DbLLMConfig)
    source_options = DbSourceOptions(read_only=False, cache_ttl=0)
    source_plugin = SQLitePlugin(path=str(db_path), **source_options.to_dict())
    profile_key = catalog_profile_key(str(db_path))
    runtime = DbRuntime(
        source=str(db_path),
        config=DbRuntimeConfig(
            source_options=source_options,
            plugins=(CatalogPlugin(auto_persist=False), source_plugin),
            metadata={
                "from_db_options": {
                    "catalog_profile_key": profile_key,
                    "catalog_store_id": f"from_db:{profile_key}",
                    "catalog_keys": [f"from_db:{profile_key}"],
                    "llm": llm.safe_metadata(),
                    "source_options": source_options.to_dict(),
                }
            },
        ),
        store=SQLiteRuntimeStore(runtime_path),
        db_llm_service=db_llm_service_from_config(llm, agent_id=name),
    )
    await runtime.setup(agent_id=name)
    return DbAgent(runtime=runtime, name=name)


async def create_live_postgres_from_db_agent(
    url: str,
    *,
    runtime_path: Path,
    name: str = "LiveFromDbPostgresProductionContract",
    read_only: bool | None = None,
    allowed_tables: tuple[str, ...] | list[str] | None = None,
    blocked_tables: tuple[str, ...] | list[str] | None = None,
    blocked_columns: tuple[str, ...] | list[str] | None = None,
):
    """Build a live OpenAI ``Agent.from_db`` over Postgres with persisted state."""
    return await Agent.from_db(
        url,
        name=name,
        source_options=DbSourceOptions(
            cache_ttl=0,
            read_only=read_only,
            allowed_tables=allowed_tables,
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        ),
        runtime=DbRuntimeOptions(store="sqlite", store_path=runtime_path),
        **require_live_openai_kwargs(),
    )


def evidence_kinds(result_or_snapshot: Any) -> set[str]:
    """Return evidence kinds from a ``DbOperationResult`` or ``OperationSnapshot``."""
    return {item.kind for item in _evidence_items(result_or_snapshot)}


def task_capabilities(result_or_snapshot: Any) -> list[str]:
    """Return task capability IDs from a result diagnostics payload or snapshot."""
    tasks = getattr(result_or_snapshot, "tasks", None)
    if tasks is not None:
        return [task.capability_id for task in tasks]

    diagnostics = getattr(result_or_snapshot, "diagnostics", {}) or {}
    execution = diagnostics.get("execution") if isinstance(diagnostics, dict) else {}
    task_payloads = (
        execution.get("task_refs", []) if isinstance(execution, dict) else []
    )
    return [
        str(task.get("capability_id") or "")
        for task in task_payloads
        if isinstance(task, dict)
    ]


def latest_evidence(
    result_or_snapshot: Any,
    kind: str,
    *,
    accepted_only: bool = True,
):
    """Return the latest evidence item of ``kind`` or ``None``."""
    for item in reversed(_evidence_items(result_or_snapshot)):
        if item.kind != kind:
            continue
        if accepted_only and not item.accepted:
            continue
        return item
    return None


def query_rows(result_or_snapshot: Any) -> list[dict[str, Any]]:
    """Return rows from raw latest accepted ``query.result`` evidence."""
    _require_raw_evidence_surface(result_or_snapshot, helper="query_rows")
    query_result = latest_evidence(result_or_snapshot, "query.result")
    assert query_result is not None, "Expected query.result evidence"
    rows = query_result.payload.get("rows") or []
    assert isinstance(rows, list), rows
    return [dict(row) for row in rows if isinstance(row, dict)]


def row_values(result_or_snapshot: Any) -> set[Any]:
    """Return scalar values from raw latest accepted ``query.result`` rows."""
    _require_raw_evidence_surface(result_or_snapshot, helper="row_values")
    values: set[Any] = set()
    for row in query_rows(result_or_snapshot):
        values.update(row.values())
    return values


def sql_from_result(result_or_snapshot: Any) -> str:
    """Extract SQL from an ``OperationSnapshot`` or other raw evidence surface."""
    _require_raw_evidence_surface(result_or_snapshot, helper="sql_from_result")
    diagnostics = getattr(result_or_snapshot, "diagnostics", {}) or {}
    if isinstance(diagnostics, dict):
        execution = diagnostics.get("execution")
        if isinstance(execution, dict):
            planned_sql = execution.get("planned_sql")
            if isinstance(planned_sql, str) and planned_sql.strip():
                return planned_sql.strip()

    for kind in SQL_EVIDENCE_KINDS:
        evidence = latest_evidence(result_or_snapshot, kind)
        if evidence is None:
            continue
        sql = _first_sql_value(evidence.payload)
        if sql:
            return sql
    for kind in PLANNED_SQL_EVIDENCE_KINDS:
        evidence = latest_evidence(result_or_snapshot, kind)
        if evidence is None:
            continue
        sql = _first_sql_value(evidence.payload)
        if sql:
            return sql
    for kind in ("planner.compilation",):
        evidence = latest_evidence(result_or_snapshot, kind)
        if evidence is None:
            continue
        sql = _first_sql_value(evidence.payload)
        if sql:
            return sql
    sql = _sql_from_task_inputs(result_or_snapshot)
    if sql:
        return sql
    return ""


def all_sql_strings(result_or_snapshot: Any) -> list[str]:
    """Extract all SQL strings from a snapshot or other raw evidence surface."""
    _require_raw_evidence_surface(result_or_snapshot, helper="all_sql_strings")
    values: list[str] = []

    diagnostics = getattr(result_or_snapshot, "diagnostics", {}) or {}
    values.extend(_all_sql_values(diagnostics))

    for evidence in _evidence_items(result_or_snapshot):
        values.extend(_all_sql_values(evidence.payload))

    tasks = getattr(result_or_snapshot, "tasks", None)
    if tasks is not None:
        for task in tasks:
            values.extend(_all_sql_values(getattr(task, "input", None)))

    if isinstance(diagnostics, dict):
        execution = diagnostics.get("execution")
        task_payloads = (
            execution.get("tasks", []) if isinstance(execution, dict) else []
        )
        for task in task_payloads:
            if isinstance(task, dict):
                values.extend(_all_sql_values(task.get("input")))

    return _ordered_unique_nonempty(values)


def diagnostic_text(result_or_snapshot: Any) -> str:
    """Return searchable JSON text for diagnostics, evidence, tasks, and policy."""
    payload: dict[str, Any] = {
        "diagnostics": getattr(result_or_snapshot, "diagnostics", {}) or {},
        "evidence": [
            (
                item.to_dict()
                if hasattr(item, "to_dict")
                else getattr(item, "__dict__", {})
            )
            for item in _evidence_items(result_or_snapshot)
        ],
        "task_statuses": _task_statuses(result_or_snapshot),
    }
    policy_decisions = getattr(result_or_snapshot, "policy_decisions", None)
    if policy_decisions is not None:
        payload["policy_decisions"] = [
            (
                item.to_dict()
                if hasattr(item, "to_dict")
                else getattr(item, "__dict__", {})
            )
            for item in policy_decisions
        ]
    events = getattr(result_or_snapshot, "events", None)
    if events is not None:
        payload["events"] = [
            (
                item.to_dict()
                if hasattr(item, "to_dict")
                else getattr(item, "__dict__", {})
            )
            for item in events
        ]
    governance_audits = getattr(result_or_snapshot, "governance_audit_records", None)
    if governance_audits is not None:
        payload["governance_audit_records"] = [
            (
                item.to_dict()
                if hasattr(item, "to_dict")
                else getattr(item, "__dict__", {})
            )
            for item in governance_audits
        ]
    return json.dumps(payload, sort_keys=True, default=str)


def assert_successful_prompt_run(result: Any, *, snapshot: Any | None = None) -> None:
    """Assert the normal successful prompt-run contract."""
    assert result.status is OperationStatus.SUCCEEDED
    assert result.operation_id
    assert result.answer
    verification = result.diagnostics.get("verification", {})
    assert verification.get("passed") is True
    if snapshot is not None:
        assert snapshot.operation.id == result.operation_id
        assert snapshot.operation.status is OperationStatus.SUCCEEDED
        assert snapshot.tasks
        assert snapshot.evidence


def assert_loop_evidence(result_or_snapshot: Any) -> None:
    """Assert the live LLM loop evidence contract for successful db.run prompts."""
    missing = FULL_LOOP_EVIDENCE - evidence_kinds(result_or_snapshot)
    assert not missing, f"Missing loop evidence: {sorted(missing)}"


def assert_synthesized_answer(
    snapshot_or_raw: Any,
    *,
    public_result: Any | None = None,
) -> None:
    """Validate raw synthesis and, when supplied, its redacted public projection."""
    _require_raw_evidence_surface(snapshot_or_raw, helper="assert_synthesized_answer")
    synthesis = latest_evidence(snapshot_or_raw, "answer.synthesis")
    assert synthesis is not None
    answer = synthesis.payload.get("answer")
    assert isinstance(answer, str) and answer.strip()
    if public_result is None:
        return

    assert public_result.answer == answer
    public_synthesis = latest_evidence(public_result, "answer.synthesis")
    assert public_synthesis is not None
    assert public_synthesis.payload.get("redacted") is True
    assert "answer" not in public_synthesis.payload
    assert "answer_facts" not in public_synthesis.payload


def assert_scalar_answer_fact(
    result_or_snapshot: Any,
    *,
    value: Any,
    label: str | None = None,
    aggregate_kind: str | None = None,
) -> None:
    """Assert raw synthesis evidence preserves a primary scalar fact."""
    _require_raw_evidence_surface(
        result_or_snapshot,
        helper="assert_scalar_answer_fact",
    )
    synthesis = latest_evidence(result_or_snapshot, "answer.synthesis")
    assert synthesis is not None
    facts = synthesis.payload.get("answer_facts") or {}
    assert isinstance(facts, dict)
    primary = facts.get("primary_scalar") or {}
    assert isinstance(primary, dict)
    assert _same_scalar_value(primary.get("value"), value)
    if label is not None:
        assert primary.get("label") == label
    if aggregate_kind is not None:
        assert primary.get("aggregate_kind") == aggregate_kind
    answer = str(synthesis.payload.get("answer") or "")
    assert re.search(rf"\b{re.escape(str(value))}\b", answer), answer


def _same_scalar_value(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    try:
        return float(str(actual).replace(",", "")) == float(
            str(expected).replace(",", "")
        )
    except (TypeError, ValueError):
        return str(actual) == str(expected)


def assert_no_unexpected_write_execution(
    result_or_snapshot: Any,
    *,
    allowed_capabilities: set[str] | None = None,
    allowed_evidence_kinds: set[str] | None = None,
) -> None:
    """Assert the operation did not execute unexpected write capabilities."""
    allowed_capabilities = allowed_capabilities or set()
    allowed_evidence_kinds = allowed_evidence_kinds or set()
    executed_writes = [
        capability
        for capability in task_capabilities(result_or_snapshot)
        if _is_write_capability(capability) and capability not in allowed_capabilities
    ]
    write_evidence = [
        kind
        for kind in evidence_kinds(result_or_snapshot)
        if kind in WRITE_EVIDENCE_KINDS and kind not in allowed_evidence_kinds
    ]
    assert not executed_writes, f"Unexpected write capabilities: {executed_writes}"
    assert not write_evidence, f"Unexpected write evidence: {write_evidence}"


def assert_sql_is_read_only(sql: str) -> None:
    """Assert SQL is a read-only statement and contains no destructive verbs."""
    normalized = sql.strip().rstrip(";")
    assert normalized, "Expected SQL to be present"
    assert re.match(r"(?is)^\s*(select|with)\b", normalized), sql
    unsafe = sorted(
        verb
        for verb in UNSAFE_SQL_VERBS
        if re.search(rf"(?i)\b{re.escape(verb)}\b", normalized)
    )
    assert not unsafe, f"SQL contains unsafe verbs {unsafe}: {sql}"


def write_failure_artifacts(
    base_dir: Path,
    *,
    result: Any | None = None,
    snapshot: Any | None = None,
) -> Path:
    """Write operation debugging artifacts for failed live assertions."""
    operation_id = _operation_id(result=result, snapshot=snapshot)
    artifact_dir = base_dir / operation_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if result is not None:
        _write_json(artifact_dir / "result.json", _result_to_dict(result))
        try:
            _write_json(artifact_dir / "telemetry.json", result.telemetry)
        except Exception as exc:  # noqa: BLE001
            _write_json(artifact_dir / "telemetry.json", {"error": repr(exc)})
    if snapshot is not None:
        _write_json(artifact_dir / "operation_snapshot.json", snapshot.to_dict())

    source = snapshot if snapshot is not None else result
    if source is not None:
        evidence = [item.to_dict() for item in _evidence_items(source)]
        _write_json(artifact_dir / "evidence.json", evidence)
        _write_json(artifact_dir / "task_statuses.json", _task_statuses(source))
        for kind, filename in PLANNER_ARTIFACTS.items():
            _write_evidence_artifact(
                artifact_dir / filename,
                source,
                kind,
            )
        (artifact_dir / "sql.txt").write_text(
            sql_from_result(source) + "\n",
            encoding="utf-8",
        )
    return artifact_dir


def _evidence_items(result_or_snapshot: Any) -> tuple[Any, ...]:
    evidence = getattr(result_or_snapshot, "evidence", ())
    return tuple(evidence or ())


def _require_raw_evidence_surface(source: Any, *, helper: str) -> None:
    """Reject caller-facing projections for helpers that inspect raw facts."""
    operation = getattr(source, "operation", None)
    tasks = getattr(source, "tasks", None)
    evidence = _evidence_items(source)
    if operation is not None and tasks is not None:
        return

    diagnostics = getattr(source, "diagnostics", {}) or {}
    execution = diagnostics.get("execution") if isinstance(diagnostics, dict) else {}
    if isinstance(execution, dict) and isinstance(execution.get("tasks"), list):
        return

    if any(
        isinstance(getattr(item, "payload", None), dict)
        and getattr(item, "payload").get("redacted") is not True
        for item in evidence
    ):
        return

    raise AssertionError(
        f"{helper} requires an OperationSnapshot or another raw evidence surface"
    )


def _first_sql_value(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("sql", "planned_sql", "selected_sql", "query", "statement"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            nested = _first_sql_value(value)
            if nested:
                return nested
    if isinstance(payload, (list, tuple)):
        for value in payload:
            nested = _first_sql_value(value)
            if nested:
                return nested
    return ""


def _all_sql_values(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if (
                key in {"sql", "planned_sql", "selected_sql", "query", "statement"}
                and isinstance(value, str)
                and _looks_like_sql(value)
            ):
                values.append(value.strip())
            else:
                values.extend(_all_sql_values(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_all_sql_values(value))
    return values


def _looks_like_sql(value: str) -> bool:
    return bool(
        re.match(
            r"(?is)^\s*(select|with|update|delete|insert|drop|alter|truncate|create|explain)\b",
            value,
        )
    )


def _ordered_unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        unique.append(stripped)
    return unique


def _sql_from_task_inputs(result_or_snapshot: Any) -> str:
    tasks = getattr(result_or_snapshot, "tasks", None)
    if tasks is not None:
        for task in tasks:
            sql = _first_sql_value(getattr(task, "input", None))
            if sql:
                return sql

    diagnostics = getattr(result_or_snapshot, "diagnostics", {}) or {}
    execution = diagnostics.get("execution") if isinstance(diagnostics, dict) else {}
    task_payloads = execution.get("tasks", []) if isinstance(execution, dict) else []
    for task in task_payloads:
        if not isinstance(task, dict):
            continue
        sql = _first_sql_value(task.get("input"))
        if sql:
            return sql
    return ""


def _is_write_capability(capability: str) -> bool:
    capability = str(capability or "")
    if capability in WRITE_CAPABILITY_IDS:
        return True
    return any(capability.endswith(f":{item}") for item in WRITE_CAPABILITY_IDS)


def _operation_id(*, result: Any | None, snapshot: Any | None) -> str:
    if result is not None and getattr(result, "operation_id", None):
        return str(result.operation_id)
    if snapshot is not None and getattr(snapshot, "operation", None) is not None:
        return str(snapshot.operation.id)
    return "unknown-operation"


def _result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "operation_id": result.operation_id,
        "status": result.status.value,
        "answer": result.answer,
        "warnings": list(result.warnings),
        "intent": {
            "kind": result.intent.kind.value,
            "confidence": result.intent.confidence,
            "diagnostics": result.intent.diagnostics,
        },
        "contract": {
            "operation_type": result.contract.operation_type,
            "required_capabilities": list(result.contract.required_capabilities),
            "required_evidence": list(result.contract.required_evidence),
        },
        "request": {
            "prompt": result.request.prompt,
            "mode": result.request.mode,
            "session_id": result.request.session_id,
        },
        "diagnostics": result.diagnostics,
        "evidence": [item.to_dict() for item in result.evidence],
    }


def _task_statuses(result_or_snapshot: Any) -> list[dict[str, Any]]:
    tasks = getattr(result_or_snapshot, "tasks", None)
    if tasks is not None:
        return [
            {
                "id": task.id,
                "capability_id": task.capability_id,
                "executor_id": task.executor_id,
                "status": task.status.value,
            }
            for task in tasks
        ]
    diagnostics = getattr(result_or_snapshot, "diagnostics", {}) or {}
    execution = diagnostics.get("execution") if isinstance(diagnostics, dict) else {}
    tasks = execution.get("task_refs", []) if isinstance(execution, dict) else []
    return [dict(task) for task in tasks if isinstance(task, dict)]


def _write_evidence_artifact(path: Path, result_or_snapshot: Any, kind: str) -> None:
    items = [
        item.to_dict()
        for item in _evidence_items(result_or_snapshot)
        if item.kind == kind
    ]
    if items:
        _write_json(path, items)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return repr(value)
