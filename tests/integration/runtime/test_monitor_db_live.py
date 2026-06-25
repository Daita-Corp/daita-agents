"""Live PostgreSQL monitor-to-DB integration tests."""

from __future__ import annotations

import time

import pytest

from daita.runtime import MonitorRuntime, MonitorSpec, RuntimeEventType, TaskStatus

from tests.integration.runtime.live_postgres_runtime_helpers import (
    build_live_kernel,
    event_types,
    require_live_postgres_runtime,
    start_seeded_postgres,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.requires_db,
]


@pytest.fixture(scope="module")
def seeded_postgres_url():
    require_live_postgres_runtime()
    container, url = start_seeded_postgres("daita-monitor-db-live")
    try:
        yield url
    finally:
        container.remove()


async def test_live_monitor_to_db_action_runs_through_declared_capabilities(
    seeded_postgres_url,
):
    harness = await build_live_kernel(seeded_postgres_url)
    monitor = MonitorRuntime(kernel=harness.kernel)
    spec = _open_high_ticket_monitor(cooldown_seconds=60)

    try:
        result = await monitor.tick(spec)
        duplicate = await monitor.tick(spec)
        snapshot = await harness.store.inspect_operation(result.operation_id)
        events = await harness.store.list_events()
    finally:
        await harness.stop()

    assert result.triggered is True
    assert duplicate.triggered is False
    assert RuntimeEventType.MONITOR_TICKED in event_types(result.events)
    assert RuntimeEventType.MONITOR_TRIGGERED in event_types(result.events)
    assert RuntimeEventType.MONITOR_SKIPPED in event_types(duplicate.events)
    assert snapshot.operation.metadata["monitor_id"] == "open_high_tickets"
    assert {
        "live_runtime.monitor.action",
        "db.sql.validate",
        "db.sql.execute_read",
    } <= {task.capability_id for task in snapshot.tasks}
    assert all(task.status is TaskStatus.SUCCEEDED for task in snapshot.tasks)
    assert _capability_started_before(
        snapshot.events,
        "db.sql.validate",
        "db.sql.execute_read",
    )
    assert (
        sum(event.type is RuntimeEventType.MONITOR_TRIGGERED for event in events) == 1
    )


async def test_live_monitor_cooldown_non_trigger_and_retrigger(
    seeded_postgres_url,
):
    harness = await build_live_kernel(seeded_postgres_url)
    monitor = MonitorRuntime(kernel=harness.kernel)
    non_matching = _open_high_ticket_monitor(
        trigger={"path": "0.open_high_count", "gt": 5}
    )
    cooling = _open_high_ticket_monitor(cooldown_seconds=60)
    latencies: list[float] = []

    try:
        start = time.perf_counter()
        skipped = await monitor.tick(non_matching)
        latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        triggered = await monitor.tick(cooling)
        latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        cooldown = await monitor.tick(cooling)
        latencies.append((time.perf_counter() - start) * 1000)

        state = monitor.state_for(cooling.id)
        assert state is not None
        state.cooldown_until = 0

        start = time.perf_counter()
        retriggered = await monitor.tick(cooling)
        latencies.append((time.perf_counter() - start) * 1000)
    finally:
        await harness.stop()

    assert skipped.triggered is False
    assert RuntimeEventType.MONITOR_SKIPPED in event_types(skipped.events)
    assert triggered.triggered is True
    assert cooldown.triggered is False
    assert RuntimeEventType.MONITOR_SKIPPED in event_types(cooldown.events)
    assert retriggered.triggered is True
    state = monitor.state_for(cooling.id)
    assert state is not None
    assert state.last_tick_at is not None
    assert state.last_triggered_at is not None
    assert state.last_value_summary == {"type": "array", "count": 1}
    assert sorted(latencies)[int((len(latencies) - 1) * 0.95)] < 5000


async def test_live_monitor_source_capability_uses_runtime_kernel(
    seeded_postgres_url,
):
    harness = await build_live_kernel(seeded_postgres_url)
    monitor = MonitorRuntime(kernel=harness.kernel)

    try:
        result = await monitor.tick(_open_high_ticket_monitor())
        all_tasks = await harness.store.list_tasks()
        operations = await harness.store.list_operations()
        all_evidence = [
            evidence
            for operation in operations
            for evidence in await harness.store.list_evidence(operation.id)
        ]
        action_snapshot = await harness.store.inspect_operation(result.operation_id)
    finally:
        await harness.stop()

    assert result.triggered is True
    assert any(
        task.capability_id == "live_runtime.monitor.source"
        and task.status is TaskStatus.SUCCEEDED
        for task in all_tasks
    )
    assert any(
        evidence.kind == "live_runtime.monitor.source"
        and evidence.payload["open_high_count"] == 2
        for evidence in all_evidence
    )
    action_task = next(
        task
        for task in action_snapshot.tasks
        if task.capability_id == "live_runtime.monitor.action"
    )
    assert action_task.input["value"][0]["open_high_count"] == 2
    assert {
        "live_runtime.monitor.source",
        "live_runtime.monitor.action",
        "db.sql.validate",
        "db.sql.execute_read",
    } <= {task.capability_id for task in all_tasks}


def _open_high_ticket_monitor(
    *,
    trigger: dict | None = None,
    cooldown_seconds: float | None = None,
) -> MonitorSpec:
    return MonitorSpec(
        id="open_high_tickets",
        name="Open High Tickets",
        source_capability_id="live_runtime.monitor.source",
        trigger=trigger or {"path": "0.open_high_count", "gte": 2},
        action_capability_id="live_runtime.monitor.action",
        action_input={
            "sql": (
                "SELECT c.name "
                "FROM customers c "
                "JOIN support_tickets st ON st.customer_id = c.id "
                "WHERE st.status = 'open' AND st.severity = 'high' "
                "ORDER BY c.name"
            )
        },
        cooldown_seconds=cooldown_seconds,
        metadata={"suite": "monitor_db_live"},
    )


def _capability_started_before(events, first: str, second: str) -> bool:
    started = [
        event.capability_id
        for event in events
        if event.type is RuntimeEventType.EXECUTOR_STARTED
    ]
    return started.index(first) < started.index(second)
