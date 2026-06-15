"""Live-gated integration tests for durable ``Agent.from_db`` monitors.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_from_db_live_monitors.py \
        -m "requires_llm and requires_db and integration" -v -s

The tests below intentionally use explicit ``DbMonitor`` definitions. Prompt
managed monitor commands are covered separately as control-plane behavior; these
cases exercise the persisted scheduler, live PostgreSQL observation, live LLM
analysis/report synthesis, governance, approval resume, and delivery capability
paths owned by ``DbRuntime``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
import time
from pathlib import Path
from typing import Any, Mapping

import pytest
from dotenv import load_dotenv

asyncpg = pytest.importorskip(
    "asyncpg",
    reason="asyncpg required: pip install 'daita-agents[postgresql]'",
)

from daita.agents.agent import Agent
from daita.db import DbAgent, DbMonitor, DbMonitorScheduler, DbRuntime, DbRuntimeConfig
from daita.db.llm_service import db_llm_service_from_config
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.postgresql import PostgreSQLPlugin
from daita.runtime import (
    AccessMode,
    ApprovalStatus,
    Capability,
    Evidence,
    EvidenceSchema,
    OperationStatus,
    RiskLevel,
    RuntimeEventType,
    SQLiteRuntimeStore,
    Task,
    TaskStatus,
)

from tests.integration._harness import start_container

load_dotenv(Path.cwd() / ".env")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
]

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_USER = "daita"
POSTGRES_PASSWORD = "daita_test_pw"
POSTGRES_DB = "daita_from_db_monitor_test"

NOW = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)

SEED_SQL = """
DROP TABLE IF EXISTS monitor_actions;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    total NUMERIC(10, 2) NOT NULL,
    status TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE monitor_actions (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    note TEXT
);

INSERT INTO customers (customer_id, name, region) VALUES
    (1, 'Ada', 'NA'),
    (2, 'Linus', 'EU'),
    (3, 'Grace', 'NA');

INSERT INTO orders (order_id, customer_id, total, status, updated_at) VALUES
    (1, 1, 120.00, 'complete', '2026-06-14T11:30:00Z'),
    (2, 1, 80.00, 'pending', '2026-06-14T11:45:00Z'),
    (3, 2, 50.00, 'complete', '2026-06-14T11:40:00Z'),
    (4, 3, 175.00, 'pending', '2026-06-14T11:50:00Z');

INSERT INTO monitor_actions (id, status, note) VALUES
    (1, 'pending', 'ready for monitor approval test');
"""


@pytest.fixture(scope="module")
def live_openai_kwargs() -> dict[str, Any]:
    if os.environ.get("DAITA_RUN_LIVE_LLM") != "1":
        pytest.skip("Set DAITA_RUN_LIVE_LLM=1 to run live from_db monitor tests")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return {
        "llm_provider": "openai",
        "model": os.environ.get("OPENAI_TEST_MODEL", "gpt-5.4-mini"),
        "api_key": api_key,
        "temperature": 0,
    }


@pytest.fixture(scope="module")
def postgres_container(live_openai_kwargs):
    container = start_container(
        POSTGRES_IMAGE,
        container_port=5432,
        env={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DB,
        },
        tag_prefix="daita-from-db-monitor-pg",
    )
    try:
        yield container
    finally:
        container.remove()


@pytest.fixture(scope="module")
def seeded_postgres_url(postgres_container) -> str:
    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{postgres_container.host}:{postgres_container.host_port}/{POSTGRES_DB}"
    )
    asyncio.run(_seed_postgres(url))
    return url


async def test_live_monitor_metric_observation_records_runtime_tasks(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorObservation",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "pending_order_observer",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    try:
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        tick_snapshot = await agent.runtime.store.inspect_operation(run.operation_id)
        state = await agent.runtime.monitor_store.load_monitor_state(monitor.id)
    finally:
        await agent.stop()

    assert run.status == "succeeded"
    assert run.triggered is False
    assert run.summary["reason"] == "no_match"
    assert state is not None
    assert state.cursor["last_observation_fingerprint"]
    assert state.last_tick_operation_id == run.operation_id
    assert [task.capability_id for task in tick_snapshot.tasks] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert {
        "sql.validation",
        "query.result",
        "monitor.observation",
        "monitor.trigger_decision",
    } <= _evidence_kinds(tick_snapshot.evidence)
    assert _started_before(
        tick_snapshot.events,
        "db.sql.validate",
        "db.sql.execute_read",
    )


async def test_live_monitor_threshold_trigger_runs_investigation_with_analysis_synthesis(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorInvestigation",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "pending_order_investigation",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
        action_plan=_investigation_action(),
    )

    try:
        await _execute(
            seeded_postgres_url,
            "update monitor_actions set status = 'pending' where id = 1",
        )
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        action_id = run.summary["triggered_operation_id"]
        action_snapshot = await agent.runtime.store.inspect_operation(action_id)
    finally:
        await agent.stop()

    synthesis = _latest(action_snapshot.evidence, "analysis.synthesis")
    action_result = _latest(action_snapshot.evidence, "monitor.action_result")

    assert run.status == "triggered"
    assert run.summary["action_status"] == "succeeded"
    assert action_snapshot.operation.status is OperationStatus.SUCCEEDED
    assert {
        "monitor.action_plan",
        "analysis.plan",
        "analysis.plan.validation",
        "analysis.synthesis",
        "monitor.action_result",
    } <= _evidence_kinds(action_snapshot.evidence)
    assert synthesis.accepted is True
    assert synthesis.payload["diagnostics"]["mode"] in {
        "llm",
        "deterministic_fallback",
    }
    assert action_result.payload["status"] == "succeeded"
    assert {
        item["kind"] for item in action_result.payload["cited_tick_evidence_refs"]
    } == {
        "monitor.observation",
        "monitor.trigger_decision",
    }


async def test_live_monitor_scheduled_report_reads_live_db_uses_llm_and_delivers(
    seeded_postgres_url,
    live_openai_kwargs,
):
    delivery_plugin = MonitorDeliveryProbePlugin()
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorReportDelivery",
        plugins=(delivery_plugin,),
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "daily_order_report",
        observation_plan=_pending_orders_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            delivery_intent={
                "delivery_kind": "slack",
                "capability_id": "slack.summary.send",
                "capability_owner": "monitor_delivery_probe",
                "target": {"channel": "#ops"},
                "format": "markdown",
            }
        ),
    )

    try:
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        action_id = run.summary["triggered_operation_id"]
        action_snapshot = await agent.runtime.store.inspect_operation(action_id)
        rerun_payload = await agent.runtime.execute_monitor_delivery(
            action_id,
            monitor_id=monitor.id,
            monitor_name=monitor.name,
            monitor_run_id=run.id,
            tick_operation_id=run.operation_id,
            report_evidence_id=run.summary["report_evidence_id"],
        )
    finally:
        await agent.stop()

    synthesis = _latest(action_snapshot.evidence, "analysis.synthesis")
    report = _latest(action_snapshot.evidence, "monitor.report")
    delivery_result = _latest(action_snapshot.evidence, "monitor.delivery_result")
    plugin_result = _latest(action_snapshot.evidence, "slack.operation.result")
    task_capabilities = [task.capability_id for task in action_snapshot.tasks]

    assert run.summary["action_status"] == "succeeded"
    assert run.summary["delivery_status"] == "succeeded"
    assert synthesis.payload["diagnostics"]["mode"] == "llm"
    assert report.payload["delivery_intent"]["target"]["channel"] == "#ops"
    assert task_capabilities.count("db.sql.validate") >= 2
    assert task_capabilities.count("db.sql.execute_read") >= 2
    assert "slack.summary.send" in task_capabilities
    assert delivery_plugin.executor.calls == 1
    assert delivery_result.accepted is True
    assert delivery_result.payload["status"] == "succeeded"
    assert (
        delivery_result.payload["idempotency_key"] == rerun_payload["idempotency_key"]
    )
    assert plugin_result.id in {
        item["id"] for item in delivery_result.payload["plugin_result_evidence_refs"]
    }


async def test_live_monitor_write_proposal_requires_approval_and_resumes(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorWriteApproval",
        read_only=False,
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "approved_monitor_write",
        source_scope=("monitor_actions",),
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
        action_plan=_write_proposal_action(
            "update monitor_actions set status = 'approved' where id = 1"
        ),
    )

    try:
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        action_id = run.summary["triggered_operation_id"]
        before = await _fetchval(
            seeded_postgres_url,
            "select status from monitor_actions where id = 1",
        )
        approvals = await agent.list_monitor_approvals(monitor_id=monitor.id)
        assert len(approvals) == 1
        await agent.approve_monitor_approval(str(approvals[0]["approval_id"]))
        resumed = await agent.runtime.resume_operation(action_id)
        after = await _fetchval(
            seeded_postgres_url,
            "select status from monitor_actions where id = 1",
        )
        inspection = await agent.inspect_monitor(monitor.id)
    finally:
        await agent.stop()

    proposal = _latest(resumed.evidence, "monitor.write_proposal")
    write_execution = _latest(resumed.evidence, "monitor.write_execution")
    action_result = _latest(resumed.evidence, "monitor.action_result")

    assert before == "pending"
    assert after == "approved"
    assert proposal.payload["status"] == "executed"
    assert write_execution.accepted is True
    assert action_result.payload["status"] == "succeeded"
    assert inspection is not None
    finalized = next(item for item in inspection.runs if item.id == run.id)
    assert finalized.summary["action_status"] == "succeeded"


async def test_live_monitor_destructive_write_is_denied_before_execution(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorDestructiveWrite",
        read_only=False,
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "destructive_monitor_write",
        source_scope=("monitor_actions",),
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
        action_plan=_write_proposal_action("delete from monitor_actions where id = 1"),
    )

    try:
        await _execute(
            seeded_postgres_url,
            "update monitor_actions set status = 'pending' where id = 1",
        )
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        action_snapshot = await agent.runtime.store.inspect_operation(
            run.summary["triggered_operation_id"]
        )
        row_count = await _fetchval(
            seeded_postgres_url,
            "select count(*)::int from monitor_actions where id = 1",
        )
    finally:
        await agent.stop()

    action_result = _latest(action_snapshot.evidence, "monitor.action_result")

    assert row_count == 1
    assert action_result.payload["status"] == "blocked"
    assert action_result.payload["block_reason"] == "deny_destructive_operations"
    assert "write.execution" not in _evidence_kinds(action_snapshot.evidence)
    assert any(
        decision.policy_id == "deny_destructive_operations"
        for decision in action_snapshot.policy_decisions
    )


async def test_live_monitor_consecutive_match_and_cooldown_state_is_durable(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorCooldown",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "durable_cooldown_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2, "consecutive_matches": 2},
        budgets={"cooldown_seconds": 300},
        action_plan=_investigation_action(),
    )

    try:
        await agent.create_monitor(monitor)
        first = (await agent.runtime.tick_monitors(now=NOW))[0]
        second = (await agent.runtime.tick_monitors(now=NOW + timedelta(seconds=1)))[0]
        third = (await agent.runtime.tick_monitors(now=NOW + timedelta(seconds=2)))[0]
        state = await agent.runtime.monitor_store.load_monitor_state(monitor.id)
    finally:
        await agent.stop()

    assert first.status == "succeeded"
    assert first.triggered is False
    assert first.summary["consecutive_matches"] == 1
    assert second.status == "triggered"
    assert second.triggered is True
    assert second.summary["consecutive_matches"] == 2
    assert third.status == "skipped"
    assert third.summary["reason"] == "cooldown"
    assert state is not None
    assert state.last_triggered_operation_id == second.summary["triggered_operation_id"]
    assert state.cooldown_until is not None


async def test_live_monitor_sql_observation_scope_guard_blocks_cross_table_read(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorScopeGuard",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "scoped_order_monitor",
        source_scope=("orders",),
        observation_plan={
            "kind": "metric_sql",
            "metric": "customer_count",
            "sql": "select count(*)::int as customer_count from customers",
            "value_path": "rows.0.customer_count",
            "source_scope": ["orders"],
            "capability_owner": "postgresql",
        },
        trigger={"path": "customer_count", "gt": 0},
    )

    try:
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        tick_snapshot = await agent.runtime.store.inspect_operation(run.operation_id)
        state = await agent.runtime.monitor_store.load_monitor_state(monitor.id)
    finally:
        await agent.stop()

    assert run.status == "blocked"
    assert run.summary["reason"] == "observation_source_scope_blocked"
    assert state is not None
    assert state.consecutive_failures == 1
    observation = _latest(tick_snapshot.evidence, "monitor.observation")
    assert observation.accepted is False
    assert observation.payload["reason"] == "observation_source_scope_blocked"
    assert [task.capability_id for task in tick_snapshot.tasks] == ["db.sql.validate"]


async def test_live_monitor_tick_lease_prevents_duplicate_action(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorLease",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "leased_pending_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
        action_plan=_investigation_action(),
    )

    try:
        await agent.create_monitor(monitor)
        assert await agent.runtime.monitor_store.claim_monitor_tick_lease(
            monitor.id,
            lease_id="external-holder",
            now=NOW.isoformat(),
            expires_at=(NOW + timedelta(minutes=5)).isoformat(),
        )
        skipped = (await agent.runtime.tick_monitors(now=NOW + timedelta(seconds=1)))[0]
        await agent.runtime.monitor_store.release_monitor_tick_lease(
            monitor.id,
            lease_id="external-holder",
        )
        scheduler_a = DbMonitorScheduler(runtime=agent.runtime, scheduler_id="a")
        scheduler_b = DbMonitorScheduler(runtime=agent.runtime, scheduler_id="b")
        results = await asyncio.gather(
            scheduler_a.run_once(now=NOW + timedelta(minutes=10)),
            scheduler_b.run_once(now=NOW + timedelta(minutes=10)),
        )
        runs = [result.run for batch in results for result in batch]
    finally:
        await agent.stop()

    assert skipped.status == "skipped"
    assert skipped.summary["reason"] == "lease_lost"
    triggered = [run for run in runs if run.triggered]
    lease_lost = [run for run in runs if run.summary["reason"] == "lease_lost"]
    assert len(triggered) == 1
    assert len(lease_lost) <= 1


async def test_live_prompt_monitor_control_plane_audits_without_runtime_tasks(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorCommands",
        cache_ttl=0,
        **live_openai_kwargs,
    )

    try:
        created = await agent.run_detailed(
            "Monitor pending orders every 15 minutes. If pending orders exceed 500."
        )
        listed = await agent.run_detailed("List active monitors.")
        inspected = await agent.run_detailed("Inspect monitor pending_orders.")
        paused = await agent.run_detailed("Pause pending_orders monitor.")
        resumed = await agent.run_detailed("Resume pending_orders monitor.")
        deleted = await agent.run_detailed("Delete pending_orders monitor.")
        operations = await agent.runtime.store.list_operations()
        tasks = await agent.runtime.store.list_tasks()
    finally:
        await agent.stop()

    assert created.status is OperationStatus.SUCCEEDED
    assert listed.status is OperationStatus.SUCCEEDED
    assert inspected.status is OperationStatus.SUCCEEDED
    assert paused.status is OperationStatus.SUCCEEDED
    assert resumed.status is OperationStatus.SUCCEEDED
    assert deleted.status is OperationStatus.SUCCEEDED
    assert [operation.operation_type for operation in operations] == [
        "monitor.create",
        "monitor.list",
        "monitor.inspect",
        "monitor.pause",
        "monitor.resume",
        "monitor.delete",
    ]
    assert tasks == []


async def test_live_monitor_resume_after_runtime_restart_with_persistent_store(
    tmp_path,
    seeded_postgres_url,
    live_openai_kwargs,
):
    store_path = tmp_path / "from_db_monitor_restart.sqlite"
    await _execute(
        seeded_postgres_url,
        "update monitor_actions set status = 'pending' where id = 1",
    )
    first_agent = await _runtime_agent_with_sqlite_store(
        seeded_postgres_url,
        store_path,
        live_openai_kwargs,
        read_only=False,
    )
    monitor = _monitor(
        "restart_write_monitor",
        source_scope=("monitor_actions",),
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
        action_plan=_write_proposal_action(
            "update monitor_actions set status = 'restarted' where id = 1"
        ),
    )

    try:
        await first_agent.create_monitor(monitor)
        run = (await first_agent.runtime.tick_monitors(now=NOW))[0]
        child_id = run.summary["triggered_operation_id"]
        approvals = await first_agent.list_monitor_approvals(monitor_id=monitor.id)
        assert len(approvals) == 1
    finally:
        await first_agent.stop()

    second_agent = await _runtime_agent_with_sqlite_store(
        seeded_postgres_url,
        store_path,
        live_openai_kwargs,
        read_only=False,
    )
    try:
        inspection_before = await second_agent.inspect_monitor(monitor.id)
        approvals_after_restart = await second_agent.list_monitor_approvals(
            monitor_id=monitor.id
        )
        await second_agent.approve_monitor_approval(
            str(approvals_after_restart[0]["approval_id"])
        )
        resumed = await second_agent.runtime.resume_operation(child_id)
        status_after = await _fetchval(
            seeded_postgres_url,
            "select status from monitor_actions where id = 1",
        )
        inspection_after = await second_agent.inspect_monitor(monitor.id)
    finally:
        await second_agent.stop()

    assert inspection_before is not None
    assert [item.id for item in inspection_before.runs] == [run.id]
    assert len(approvals_after_restart) == 1
    assert status_after == "restarted"
    assert _latest(resumed.evidence, "monitor.write_execution").accepted is True
    assert inspection_after is not None
    finalized = next(item for item in inspection_after.runs if item.id == run.id)
    assert finalized.summary["action_status"] == "succeeded"


async def test_live_monitor_governed_delivery_requests_approval_and_resumes_once(
    seeded_postgres_url,
    live_openai_kwargs,
):
    delivery_plugin = MonitorDeliveryProbePlugin(requires_approval=True)
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorGovernedDelivery",
        plugins=(delivery_plugin,),
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "governed_delivery_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(
            delivery_intent={
                "delivery_kind": "slack",
                "capability_id": "slack.summary.send",
                "capability_owner": "monitor_delivery_probe",
                "target": {"channel": "#ops"},
                "format": "markdown",
                "requires_approval": True,
            }
        ),
        policy={"governed_delivery": True},
    )

    try:
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        child_id = run.summary["triggered_operation_id"]
        before = await agent.runtime.store.inspect_operation(child_id)
        approvals = await agent.list_monitor_approvals(monitor_id=monitor.id)
        await agent.approve_monitor_approval(str(approvals[0]["approval_id"]))
        resumed = await agent.runtime.resume_operation(child_id)
        inspection = await agent.inspect_monitor(monitor.id)
    finally:
        await agent.stop()

    blocked_result = _latest(before.evidence, "monitor.delivery_result")
    delivered_result = _latest(resumed.evidence, "monitor.delivery_result")
    delivery_task = next(
        task for task in before.tasks if task.capability_id == "slack.summary.send"
    )

    assert delivery_plugin.executor.calls == 1
    assert run.summary["delivery_status"] == "blocked"
    assert blocked_result.payload["block_reason"] == "governance_approval_required"
    assert len(approvals) == 1
    assert approvals[0]["status"] == ApprovalStatus.PENDING.value
    assert approvals[0]["context"]["kind"] == "monitor.delivery"
    assert delivery_task.status is TaskStatus.PENDING
    assert delivered_result.accepted is True
    assert delivered_result.payload["status"] == "succeeded"
    assert inspection is not None
    finalized = next(item for item in inspection.runs if item.id == run.id)
    assert finalized.summary["delivery_status"] == "succeeded"


async def test_live_monitor_repeated_scheduler_ticks_do_not_duplicate_actions(
    seeded_postgres_url,
    live_openai_kwargs,
):
    ticks = int(os.environ.get("DAITA_MONITOR_LIVE_SOAK_TICKS", "50"))
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorSoak",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    monitor = _monitor(
        "soak_pending_observer",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )

    try:
        await agent.create_monitor(monitor)
        runs = [
            (await agent.runtime.tick_monitors(now=NOW + timedelta(seconds=index)))[0]
            for index in range(ticks)
        ]
        inspection = await agent.inspect_monitor(monitor.id)
        operations = await agent.runtime.store.list_operations()
        tasks = await agent.runtime.store.list_tasks()
    finally:
        await agent.stop()

    assert len(runs) == ticks
    assert all(run.status == "succeeded" for run in runs)
    assert all(run.triggered is False for run in runs)
    assert inspection is not None
    assert len(inspection.runs) == ticks
    assert (
        sum(1 for operation in operations if operation.operation_type == "monitor.tick")
        == ticks
    )
    assert [task.capability_id for task in tasks].count("db.sql.validate") == ticks
    assert [task.capability_id for task in tasks].count("db.sql.execute_read") == ticks


async def test_live_monitor_llm_provider_failure_falls_back_for_report_synthesis(
    seeded_postgres_url,
    live_openai_kwargs,
):
    broken_llm_kwargs = {
        **live_openai_kwargs,
        "model": "daita-definitely-not-a-real-monitor-model",
    }
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorLLMFallback",
        cache_ttl=0,
        **broken_llm_kwargs,
    )
    monitor = _monitor(
        "llm_failure_report",
        observation_plan=_pending_orders_observation(),
        trigger={"type": "scheduled"},
        action_plan=_scheduled_report_action(),
    )

    try:
        await agent.create_monitor(monitor)
        run = (await agent.runtime.tick_monitors(now=NOW))[0]
        action_snapshot = await agent.runtime.store.inspect_operation(
            run.summary["triggered_operation_id"]
        )
    finally:
        await agent.stop()

    synthesis = _latest(action_snapshot.evidence, "analysis.synthesis")

    assert run.summary["action_status"] == "succeeded"
    assert synthesis.payload["diagnostics"]["mode"] == "deterministic_fallback"
    assert synthesis.payload["diagnostics"]["fallback_reason"]
    assert (
        "db_llm_service_unavailable"
        not in synthesis.payload["diagnostics"]["fallback_reason"]
    )


async def test_live_multi_monitor_scheduler_mixed_states(
    seeded_postgres_url,
    live_openai_kwargs,
):
    agent = await Agent.from_db(
        seeded_postgres_url,
        name="LiveFromDbMonitorMixedScheduler",
        cache_ttl=0,
        **live_openai_kwargs,
    )
    active = _monitor(
        "mixed_active_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gt": 10},
    )
    paused = _monitor(
        "mixed_paused_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
    )
    cooling = _monitor(
        "mixed_cooling_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
    )
    backing_off = _monitor(
        "mixed_backoff_monitor",
        observation_plan=_pending_orders_observation(),
        trigger={"path": "pending_count", "gte": 2},
    )

    try:
        await agent.create_monitor(active)
        await agent.create_monitor(paused)
        await agent.create_monitor(cooling)
        await agent.create_monitor(backing_off)
        await agent.pause_monitor(
            paused.id, paused_until=(NOW + timedelta(hours=1)).isoformat()
        )
        await agent.runtime.monitor_store.save_monitor_state(
            await _state_with_updates(
                agent.runtime,
                cooling.id,
                cooldown_until=(NOW + timedelta(minutes=10)).isoformat(),
            )
        )
        await agent.runtime.monitor_store.save_monitor_state(
            await _state_with_updates(
                agent.runtime,
                backing_off.id,
                error={"backoff_until": (NOW + timedelta(minutes=10)).isoformat()},
            )
        )
        runs = await agent.runtime.tick_monitors(now=NOW)
        active_snapshot = await agent.runtime.store.inspect_operation(
            next(run for run in runs if run.monitor_id == active.id).operation_id
        )
    finally:
        await agent.stop()

    by_monitor = {run.monitor_id: run for run in runs}
    assert by_monitor[active.id].status == "succeeded"
    assert by_monitor[active.id].summary["reason"] == "no_match"
    assert by_monitor[paused.id].status == "skipped"
    assert by_monitor[paused.id].summary["reason"] == "paused"
    assert by_monitor[cooling.id].status == "skipped"
    assert by_monitor[cooling.id].summary["reason"] == "cooldown"
    assert by_monitor[backing_off.id].status == "skipped"
    assert by_monitor[backing_off.id].summary["reason"] == "backoff"
    assert [task.capability_id for task in active_snapshot.tasks] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]


class MonitorDeliveryProbeExecutor:
    id = "monitor_delivery_probe.slack"
    capability_ids = frozenset({"slack.summary.send"})

    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[dict[str, Any]] = []

    async def execute(self, task: Task, operation, context: Mapping[str, Any]):
        self.calls += 1
        self.inputs.append(dict(task.input))
        return [
            Evidence(
                kind="slack.operation.result",
                owner="monitor_delivery_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "ok": True,
                    "target": task.input.get("target"),
                    "payload_source": task.input.get("payload_source"),
                    "idempotency_key": task.input.get("idempotency_key")
                    or task.metadata.get("idempotency_key"),
                },
            )
        ]


class MonitorDeliveryProbePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="monitor_delivery_probe",
        display_name="Monitor Delivery Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"monitor", "slack"}),
    )

    def __init__(self, *, requires_approval: bool = False) -> None:
        self.executor = MonitorDeliveryProbeExecutor()
        self.requires_approval = requires_approval

    def declare_capabilities(self):
        return (
            Capability(
                id="slack.summary.send",
                owner=self.manifest.id,
                description="Test monitor report delivery.",
                domains=frozenset({"monitor", "slack"}),
                operation_types=frozenset({"monitor.delivery"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.HIGH if self.requires_approval else RiskLevel.LOW,
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "object"},
                        "request": {"type": "object"},
                        "payload_source": {"type": "object"},
                    },
                },
                output_evidence=frozenset({"slack.operation.result"}),
                executor=self.executor.id,
                runtime_only=True,
                side_effecting=True,
                idempotent=False,
                replay_safe=False,
                metadata={
                    "monitor_roles": ["delivery"],
                    "delivery_kind": "slack",
                    "accepted_payload_kinds": ["monitor.report", "analysis.synthesis"],
                    "accepted_formats": ["markdown"],
                    "supports_idempotency_key": True,
                    "supports_dry_run": True,
                },
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_evidence_schemas(self):
        return (
            EvidenceSchema(
                kind="slack.operation.result",
                owner=self.manifest.id,
                json_schema={"type": "object"},
            ),
        )


def _monitor(
    monitor_id: str,
    *,
    observation_plan: dict[str, Any],
    trigger: dict[str, Any],
    action_plan: dict[str, Any] | None = None,
    source_scope: tuple[str, ...] = (),
    budgets: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> DbMonitor:
    return DbMonitor(
        id=monitor_id,
        name=monitor_id.replace("_", " ").title(),
        status="active",
        source_scope=source_scope,
        schedule={"interval_seconds": 0},
        trigger=trigger,
        observation_plan=observation_plan,
        action_plan=action_plan or {},
        policy=dict(policy or {}),
        budgets=dict(budgets or {}),
        metadata={"suite": "from_db_live_monitors"},
    )


def _pending_orders_observation() -> dict[str, Any]:
    return {
        "kind": "metric_sql",
        "metric": "pending_count",
        "sql": "select count(*)::int as pending_count from orders where status = 'pending'",
        "value_path": "rows.0.pending_count",
        "source_scope": ["orders"],
        "capability_owner": "postgresql",
    }


def _investigation_action() -> dict[str, Any]:
    return {
        "kind": "investigation",
        "goal": "Explain why the pending order monitor triggered.",
        "steps": [
            {
                "id": "final_synthesis",
                "kind": "synthesis",
                "purpose": "Summarize the monitor trigger from the observed evidence.",
                "expected_evidence": ["analysis.synthesis"],
            },
        ],
    }


def _scheduled_report_action(
    *, delivery_intent: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "kind": "scheduled_report",
        "title": "Daily order monitor report",
        "steps": [
            {
                "id": "pending_orders",
                "kind": "metric_sql",
                "metric": "pending_count",
                "sql": (
                    "select count(*)::int as pending_count "
                    "from orders where status = 'pending'"
                ),
                "value_path": "rows.0.pending_count",
                "source_scope": ["orders"],
                "capability_owner": "postgresql",
            },
            {
                "id": "revenue",
                "kind": "metric_sql",
                "metric": "total_revenue",
                "sql": "select sum(total)::float as total_revenue from orders",
                "value_path": "rows.0.total_revenue",
                "source_scope": ["orders"],
                "capability_owner": "postgresql",
            },
            {
                "id": "report_summary",
                "kind": "synthesis",
                "purpose": "Generate the monitor report narrative.",
                "expected_evidence": ["analysis.synthesis"],
            },
        ],
        "output": {"kind": "report", "format": "markdown"},
        "delivery_intent": dict(delivery_intent or {}),
    }


def _write_proposal_action(sql: str) -> dict[str, Any]:
    return {
        "kind": "write_proposal",
        "sql": sql,
        "capability_owner": "postgresql",
        "source_scope": ["monitor_actions"],
    }


async def _seed_postgres(url: str) -> None:
    deadline = time.time() + 30
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            connection = await asyncpg.connect(url, ssl=False)
            await connection.execute(SEED_SQL)
            await connection.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Could not seed Postgres monitor test database: {last_error}")


async def _fetchval(url: str, sql: str) -> Any:
    connection = await asyncpg.connect(url, ssl=False)
    try:
        return await connection.fetchval(sql)
    finally:
        await connection.close()


async def _execute(url: str, sql: str) -> None:
    connection = await asyncpg.connect(url, ssl=False)
    try:
        await connection.execute(sql)
    finally:
        await connection.close()


async def _runtime_agent_with_sqlite_store(
    url: str,
    store_path: Path,
    live_openai_kwargs: dict[str, Any],
    *,
    read_only: bool = True,
    plugins: tuple[Any, ...] = (),
) -> DbAgent:
    source_plugin = PostgreSQLPlugin(
        connection_string=url,
        read_only=read_only,
        query_default_limit=50,
        query_max_rows=200,
        query_max_chars=50000,
    )
    runtime = DbRuntime(
        source=url,
        config=DbRuntimeConfig(
            profile="analyst",
            plugins=(
                CatalogPlugin(auto_persist=False),
                source_plugin,
                *plugins,
            ),
            metadata={
                "from_db_options": {
                    "llm_provider": live_openai_kwargs["llm_provider"],
                    "model": live_openai_kwargs["model"],
                    "temperature": live_openai_kwargs.get("temperature"),
                }
            },
        ),
        store=SQLiteRuntimeStore(store_path),
        db_llm_service=db_llm_service_from_config(
            llm_provider=str(live_openai_kwargs["llm_provider"]),
            model=str(live_openai_kwargs["model"]),
            api_key=str(live_openai_kwargs["api_key"]),
            temperature=live_openai_kwargs.get("temperature"),
            agent_id="LiveFromDbPersistentMonitor",
        ),
    )
    await runtime.setup(agent_id="LiveFromDbPersistentMonitor")
    return DbAgent(runtime=runtime, name="LiveFromDbPersistentMonitor")


async def _state_with_updates(runtime, monitor_id: str, **updates):
    state = await runtime.monitor_store.load_monitor_state(monitor_id)
    assert state is not None
    return type(state).from_dict({**state.to_dict(), **updates})


def _evidence_kinds(evidence: tuple[Evidence, ...] | list[Evidence]) -> set[str]:
    return {item.kind for item in evidence}


def _latest(evidence: tuple[Evidence, ...] | list[Evidence], kind: str) -> Evidence:
    matches = [item for item in evidence if item.kind == kind]
    assert matches, f"missing evidence kind {kind}"
    return matches[-1]


def _started_before(events, first: str, second: str) -> bool:
    started = [
        event.capability_id
        for event in events
        if event.type is RuntimeEventType.EXECUTOR_STARTED
    ]
    return (
        first in started
        and second in started
        and started.index(first) < started.index(second)
    )
