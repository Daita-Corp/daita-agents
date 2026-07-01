"""Shared helpers for live ``Agent.from_db`` production contract tests."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from daita.agents.agent import Agent
from daita.db import DbRuntimeOptions
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus

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


def require_live_openai_kwargs() -> dict[str, object]:
    """Return live OpenAI kwargs or skip when the live gate is unavailable."""
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db tests")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm_provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", DEFAULT_LIVE_OPENAI_MODEL),
        "api_key": api_key,
        "temperature": 0,
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
            (2003, 4, 'closed', 'critical', 10);

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


async def create_live_sqlite_from_db_agent(
    db_path: Path,
    *,
    runtime_path: Path,
    name: str = "LiveFromDbProductionContract",
):
    """Build a live OpenAI ``Agent.from_db`` over SQLite with persisted state."""
    return await Agent.from_db(
        str(db_path),
        name=name,
        cache_ttl=0,
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
    task_payloads = execution.get("tasks", []) if isinstance(execution, dict) else []
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


def sql_from_result(result_or_snapshot: Any) -> str:
    """Extract the planned or executed SQL from a result or snapshot."""
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


def assert_synthesized_answer(result_or_snapshot: Any) -> None:
    """Assert that ``db.answer.synthesize`` produced accepted answer evidence."""
    synthesis = latest_evidence(result_or_snapshot, "answer.synthesis")
    assert synthesis is not None
    answer = synthesis.payload.get("answer")
    assert isinstance(answer, str) and answer.strip()
    if hasattr(result_or_snapshot, "answer"):
        assert result_or_snapshot.answer == answer


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
    tasks = execution.get("tasks", []) if isinstance(execution, dict) else []
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
