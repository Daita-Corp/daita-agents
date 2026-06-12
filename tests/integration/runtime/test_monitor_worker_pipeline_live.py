"""Live PostgreSQL monitor-to-worker pipeline integration test."""

from __future__ import annotations

import pytest

from daita.runtime import (
    MonitorRuntime,
    MonitorSpec,
    OperationStatus,
    RuntimeEventType,
    WorkerRuntime,
    WorkerRuntimeOptions,
)

from tests.integration.runtime.live_postgres_runtime_helpers import (
    build_live_kernel,
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
    container, url = start_seeded_postgres("daita-monitor-worker-live")
    try:
        yield url
    finally:
        container.remove()


async def test_live_monitor_to_worker_pipeline_runs_without_direct_bypass(
    seeded_postgres_url,
):
    harness = await build_live_kernel(seeded_postgres_url)
    monitor = MonitorRuntime(kernel=harness.kernel)
    worker = WorkerRuntime(
        kernel=harness.kernel,
        options=WorkerRuntimeOptions(
            worker_id="live_runtime.db_worker",
            owner="live_runtime",
            poll_interval_seconds=0,
            max_concurrency=5,
        ),
    )

    try:
        monitor_result = await monitor.tick(
            MonitorSpec(
                id="open_high_ticket_worker_pipeline",
                name="Open High Ticket Worker Pipeline",
                source_capability_id="live_runtime.monitor.source",
                trigger={"path": "0.open_high_count", "gte": 2},
                action_capability_id="live_runtime.worker.query",
                action_input={
                    "sql": (
                        "SELECT c.name "
                        "FROM customers c "
                        "JOIN support_tickets st ON st.customer_id = c.id "
                        "WHERE st.status = 'open' AND st.severity = 'high' "
                        "ORDER BY c.name"
                    )
                },
                metadata={"suite": "monitor_worker_pipeline_live"},
            ),
            execute_actions=False,
        )
        worker_result = await worker.run_once()
        snapshot = await harness.store.inspect_operation(monitor_result.operation_id)
    finally:
        await harness.stop()

    assert monitor_result.triggered is True
    assert worker_result is not None
    assert worker_result.execution is not None
    event_types = {event.type for event in snapshot.events}
    assert {
        RuntimeEventType.MONITOR_TRIGGERED,
        RuntimeEventType.WORKER_HANDOFF,
        RuntimeEventType.WORKER_LEASE_CLAIMED,
        RuntimeEventType.WORKER_COMPLETED,
        RuntimeEventType.EXECUTOR_STARTED,
        RuntimeEventType.EXECUTOR_COMPLETED,
    } <= event_types
    assert snapshot.operation.status is OperationStatus.SUCCEEDED
    started = [
        event.capability_id
        for event in snapshot.events
        if event.type is RuntimeEventType.EXECUTOR_STARTED
    ]
    assert "live_runtime.worker.query" in started
    assert "db.sql.validate" in started
    assert "db.sql.execute_read" in started
    assert started.index("db.sql.validate") < started.index("db.sql.execute_read")
