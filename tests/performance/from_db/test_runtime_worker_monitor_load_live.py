"""Live PostgreSQL runtime worker and monitor load benchmarks."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import time

import pytest

from daita.runtime import (
    MonitorRuntime,
    MonitorSpec,
    RuntimeEventType,
    TaskStatus,
    WorkerRuntime,
    WorkerRuntimeOptions,
)

from tests.integration.runtime.live_postgres_runtime_helpers import build_live_kernel

from .postgres_scale_fixtures import (
    postgres_version,
    rich_schema_sql,
    scale_int_env,
    seed_worker_task_backlog,
    start_seeded_postgres,
)
from .scale_runner import (
    ScaleBenchmarkParameters,
    artifact_output_dir,
    default_environment_metadata,
    operation_record_from_snapshot,
    postgres_live_required,
    summarize_operations,
    write_artifact,
)

pytestmark = [
    pytest.mark.performance,
    pytest.mark.requires_db,
]


async def test_worker_concurrent_polling_no_duplicate_execution(tmp_path):
    postgres_live_required()
    task_count = scale_int_env("DAITA_PERF_WORKER_TASKS", 100)
    harness = await start_seeded_postgres(
        rich_schema_sql(),
        tag_prefix="daita-perf-worker-poll-pg",
    )
    live = await build_live_kernel(harness.url)
    try:
        tasks = await seed_worker_task_backlog(live.kernel, count=task_count)
        started = time.perf_counter()
        await _drain_with_concurrent_workers(live.kernel, worker_count=10)
        elapsed = max(time.perf_counter() - started, 0.000001)
        records = []
        for index, task in enumerate(tasks):
            snapshot = await live.store.inspect_operation(task.operation_id)
            records.append(
                operation_record_from_snapshot(
                    index=index,
                    latency_ms=elapsed * 1000,
                    started_at=_iso_now(),
                    snapshot=snapshot,
                )
            )
        artifact = _write_runtime_artifact(
            tmp_path,
            suite="runtime-worker-monitor-load",
            scenario="concurrent-polling",
            records=records,
            elapsed_seconds=elapsed,
            environment={
                "postgres_version": await postgres_version(harness.url),
                "dataset": "small_rich_schema",
            },
            parameters={"worker_count": 10, "tasks": task_count},
        )
        assert artifact["summary"]["success_rate"] == 1.0
        assert (
            await _evidence_kind_count(live.store, "live_runtime.worker.result")
            == task_count
        )
        assert await _duplicate_terminal_executions(live.store) == 0
    finally:
        await live.stop()
        await harness.stop()


async def test_worker_expired_lease_recovery_under_load(tmp_path):
    postgres_live_required()
    task_count = scale_int_env("DAITA_PERF_WORKER_TASKS", 100)
    harness = await start_seeded_postgres(
        rich_schema_sql(),
        tag_prefix="daita-perf-worker-recovery-pg",
    )
    live = await build_live_kernel(harness.url)
    try:
        tasks = await seed_worker_task_backlog(live.kernel, count=task_count)
        for task in tasks[: max(1, task_count // 2)]:
            await live.store.save_task(
                replace(
                    task,
                    status=TaskStatus.RUNNING,
                    metadata={
                        **task.metadata,
                        "lease_id": "expired-lease",
                        "lease_owner": "stale-worker",
                        "lease_expires_at": 0.0,
                    },
                )
            )
        started = time.perf_counter()
        await _drain_with_concurrent_workers(live.kernel, worker_count=10)
        elapsed = max(time.perf_counter() - started, 0.000001)
        records = [
            operation_record_from_snapshot(
                index=index,
                latency_ms=elapsed * 1000,
                started_at=_iso_now(),
                snapshot=await live.store.inspect_operation(task.operation_id),
            )
            for index, task in enumerate(tasks)
        ]
        artifact = _write_runtime_artifact(
            tmp_path,
            suite="runtime-worker-monitor-load",
            scenario="expired-lease-recovery",
            records=records,
            elapsed_seconds=elapsed,
            environment={
                "postgres_version": await postgres_version(harness.url),
                "dataset": "small_rich_schema",
            },
            parameters={"worker_count": 10, "tasks": task_count},
        )
        assert artifact["summary"]["success_rate"] == 1.0
        assert (
            await _evidence_kind_count(live.store, "live_runtime.worker.result")
            == task_count
        )
        assert await _duplicate_terminal_executions(live.store) == 0
        recovered = [
            task
            for task in await live.store.list_tasks()
            if task.metadata.get("expired_lease_recovered")
        ]
        assert len(recovered) >= max(1, task_count // 2)
    finally:
        await live.stop()
        await harness.stop()


async def test_monitor_to_db_action_load(tmp_path):
    postgres_live_required()
    ticks = scale_int_env("DAITA_PERF_MONITOR_TICKS", 100)
    harness = await start_seeded_postgres(
        rich_schema_sql(),
        tag_prefix="daita-perf-monitor-db-pg",
    )
    live = await build_live_kernel(harness.url)
    monitor = MonitorRuntime(kernel=live.kernel)
    spec = MonitorSpec(
        id="open_high_ticket_monitor",
        name="Open High Ticket Monitor",
        source_capability_id="live_runtime.monitor.source",
        trigger={"path": "0.open_high_count", "gt": 0},
        action_capability_id="live_runtime.monitor.action",
    )
    try:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(monitor.tick(spec, context={"tick": index}) for index in range(ticks))
        )
        elapsed = max(time.perf_counter() - started, 0.000001)
        records = []
        for index, result in enumerate(results):
            snapshot = await live.store.inspect_operation(result.operation_id)
            records.append(
                operation_record_from_snapshot(
                    index=index,
                    latency_ms=elapsed * 1000 / ticks,
                    started_at=_iso_now(),
                    snapshot=snapshot,
                    metadata={"triggered": result.triggered},
                )
            )
        artifact = _write_runtime_artifact(
            tmp_path,
            suite="runtime-worker-monitor-load",
            scenario="monitor-to-db",
            records=records,
            elapsed_seconds=elapsed,
            environment={
                "postgres_version": await postgres_version(harness.url),
                "dataset": "small_rich_schema",
            },
            parameters={"ticks": ticks},
        )
        assert artifact["summary"]["success_rate"] == 1.0
        assert await _monitor_direct_executor_bypasses(live.store) == 0
    finally:
        await live.stop()
        await harness.stop()


async def test_monitor_to_worker_pipeline_load(tmp_path):
    postgres_live_required()
    ticks = scale_int_env("DAITA_PERF_MONITOR_TICKS", 100)
    harness = await start_seeded_postgres(
        rich_schema_sql(),
        tag_prefix="daita-perf-monitor-worker-pg",
    )
    live = await build_live_kernel(harness.url)
    monitor = MonitorRuntime(kernel=live.kernel)
    spec = MonitorSpec(
        id="worker_pipeline_monitor",
        name="Worker Pipeline Monitor",
        trigger={"truthy": True},
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
    )
    try:
        tick_results = await asyncio.gather(
            *(
                monitor.tick(
                    spec,
                    value={"trigger": True, "index": index},
                    execute_actions=False,
                )
                for index in range(ticks)
            )
        )
        started = time.perf_counter()
        await _drain_with_concurrent_workers(live.kernel, worker_count=10)
        elapsed = max(time.perf_counter() - started, 0.000001)
        records = []
        for index, result in enumerate(tick_results):
            snapshot = await live.store.inspect_operation(result.operation_id)
            records.append(
                operation_record_from_snapshot(
                    index=index,
                    latency_ms=elapsed * 1000 / ticks,
                    started_at=_iso_now(),
                    snapshot=snapshot,
                    metadata={"triggered": result.triggered},
                )
            )
        artifact = _write_runtime_artifact(
            tmp_path,
            suite="runtime-worker-monitor-load",
            scenario="monitor-to-worker",
            records=records,
            elapsed_seconds=elapsed,
            environment={
                "postgres_version": await postgres_version(harness.url),
                "dataset": "small_rich_schema",
            },
            parameters={"ticks": ticks, "worker_count": 10},
        )
        assert artifact["summary"]["success_rate"] == 1.0
        assert (
            await _evidence_kind_count(live.store, "live_runtime.worker.result")
            == ticks
        )
        assert await _task_evidence_dependency_violations(live.store) == 0
    finally:
        await live.stop()
        await harness.stop()


async def _drain_with_concurrent_workers(kernel, *, worker_count: int) -> None:
    workers = [
        WorkerRuntime(
            kernel=kernel,
            options=WorkerRuntimeOptions(
                worker_id="live_runtime.db_worker",
                owner="live_runtime",
                lease_seconds=2,
                poll_interval_seconds=0,
                max_concurrency=5,
            ),
        )
        for _ in range(worker_count)
    ]
    while True:
        results = await asyncio.gather(*(worker.run_once() for worker in workers))
        if not any(result is not None and not result.skipped for result in results):
            pending = [
                task
                for task in await kernel.store.list_tasks()
                if task.status
                in {TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.RUNNING}
                and task.capability_id.startswith("live_runtime.worker")
            ]
            if not pending:
                return


def _write_runtime_artifact(
    tmp_path,
    *,
    suite: str,
    scenario: str,
    records: list[dict],
    elapsed_seconds: float,
    environment: dict,
    parameters: dict,
) -> dict:
    artifact = {
        "suite": suite,
        "started_at": _iso_now(),
        "finished_at": _iso_now(),
        "environment": {
            **default_environment_metadata(model=None),
            **environment,
        },
        "parameters": ScaleBenchmarkParameters(
            concurrency=int(parameters.get("worker_count") or 1),
            operations=len(records),
            scenario=scenario,
            extra=parameters,
        ).to_dict(),
        "summary": summarize_operations(records, elapsed_seconds),
        "operations": records,
    }
    write_artifact(
        artifact,
        artifact_output_dir(tmp_path, suite) / f"{suite}-{scenario}.json",
    )
    return artifact


async def _duplicate_terminal_executions(store) -> int:
    duplicates = 0
    for task in await store.list_tasks():
        events = [
            event
            for event in await store.list_events(task.operation_id)
            if event.task_id == task.id
            and event.type is RuntimeEventType.EXECUTOR_COMPLETED
        ]
        duplicates += max(0, len(events) - 1)
    return duplicates


async def _monitor_direct_executor_bypasses(store) -> int:
    bypasses = 0
    for operation in await store.list_operations():
        events = await store.list_events(operation.id)
        types = {event.type for event in events}
        if (
            RuntimeEventType.EXECUTOR_STARTED in types
            and RuntimeEventType.TASK_STARTED not in types
        ):
            bypasses += 1
    return bypasses


async def _task_evidence_dependency_violations(store) -> int:
    violations = 0
    for operation in await store.list_operations():
        tasks = await store.list_tasks(operation.id)
        evidence = await store.list_evidence(operation.id)
        task_ids = {task.id for task in tasks}
        for item in evidence:
            if item.task_id and item.task_id not in task_ids:
                violations += 1
    return violations


async def _evidence_kind_count(store, kind: str) -> int:
    count = 0
    for operation in await store.list_operations():
        count += sum(
            1 for item in await store.list_evidence(operation.id) if item.kind == kind
        )
    return count


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
