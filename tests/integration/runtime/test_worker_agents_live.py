"""Live PostgreSQL worker-runtime integration tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from daita.runtime import (
    OperationStatus,
    RuntimeEventType,
    RuntimeKernelLeaseLost,
    TaskStatus,
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
    container, url = start_seeded_postgres("daita-worker-agents-live")
    try:
        yield url
    finally:
        container.remove()


async def test_live_worker_polling_executes_task_set_once(seeded_postgres_url):
    harness = await build_live_kernel(seeded_postgres_url)
    worker = _worker(harness.kernel)

    try:
        operation, worker_tasks = await _plan_worker_tasks(harness.kernel, count=3)
        ignored = await harness.kernel.plan_task(
            operation_id=operation.id,
            capability_id="db.sql.validate",
            owner="postgresql",
            input={"sql": "SELECT 1", "operation": "query"},
        )

        results = []
        while True:
            result = await worker.run_once()
            if result is None:
                break
            results.append(result)

        snapshot = await harness.store.inspect_operation(operation.id)
        evidence = [
            item
            for item in snapshot.evidence
            if item.kind == "live_runtime.worker.result"
        ]
    finally:
        await harness.stop()

    assert len([result for result in results if result.execution is not None]) == 3
    assert ignored.id in {task.id for task in snapshot.tasks}
    assert ignored.status is TaskStatus.PENDING
    assert {
        task.status for task in snapshot.tasks if task.id in _ids(worker_tasks)
    } == {TaskStatus.SUCCEEDED}
    assert len(evidence) == 3
    assert {item.payload["worker_id"] for item in evidence} == {
        "live_runtime.db_worker"
    }
    assert {item.owner for item in evidence} == {"live_runtime"}
    assert await worker.run_once() is None
    assert snapshot.operation.status is OperationStatus.RUNNING


async def test_live_concurrent_workers_claim_each_task_once(seeded_postgres_url):
    harness = await build_live_kernel(seeded_postgres_url)
    workers = [_worker(harness.kernel) for _ in range(3)]

    try:
        operation, worker_tasks = await _plan_worker_tasks(harness.kernel, count=8)
        while True:
            batch = await asyncio.gather(
                *(worker.run_once() for worker in workers),
                return_exceptions=True,
            )
            if all(result is None for result in batch):
                break

        snapshot = await harness.store.inspect_operation(operation.id)
        worker_evidence = [
            item
            for item in snapshot.evidence
            if item.kind == "live_runtime.worker.result"
        ]
    finally:
        await harness.stop()

    assert len(worker_evidence) == len(worker_tasks)
    assert len({item.task_id for item in worker_evidence}) == len(worker_tasks)
    assert {
        task.status for task in snapshot.tasks if task.id in _ids(worker_tasks)
    } == {TaskStatus.SUCCEEDED}
    assert snapshot.operation.status is OperationStatus.SUCCEEDED


async def test_live_worker_heartbeat_and_lease_expiry_paths(seeded_postgres_url):
    harness = await build_live_kernel(seeded_postgres_url)
    worker = _worker(harness.kernel, lease_seconds=30)

    try:
        heartbeat_operation, heartbeat_task = await _plan_one_worker_task(
            harness.kernel,
            sql="SELECT 1::int AS count",
        )
        lease = await harness.kernel.claim_task(
            heartbeat_task.id,
            lease_owner="worker:live_runtime:live_runtime.db_worker",
            lease_seconds=30,
            worker_id="live_runtime.db_worker",
            worker_owner="live_runtime",
        )
        extended = await worker.heartbeat(lease)
        heartbeat_snapshot = await harness.store.inspect_operation(
            heartbeat_operation.id
        )

        stale_operation, stale_task = await _plan_one_worker_task(harness.kernel)
        stale = await harness.kernel.claim_task(
            stale_task.id,
            lease_owner="worker-1",
            lease_seconds=0.01,
        )
        claimed = await harness.store.load_task(stale_task.id)
        await harness.store.save_task(
            replace(claimed, metadata={**claimed.metadata, "lease_expires_at": 0.0})
        )
        fresh = await harness.kernel.claim_task(
            stale_task.id,
            lease_owner="worker-2",
            lease_seconds=30,
        )
        with pytest.raises(RuntimeKernelLeaseLost):
            await harness.kernel.execute_claimed_task(stale)
        await harness.kernel.execute_claimed_task(fresh)

        recovered_operation, recovered_task = await _plan_one_worker_task(
            harness.kernel
        )
        await harness.store.save_task(
            replace(
                recovered_task,
                status=TaskStatus.RUNNING,
                metadata={
                    **recovered_task.metadata,
                    "lease_id": "expired-safe",
                    "lease_owner": "worker-1",
                    "lease_expires_at": 0.0,
                },
            )
        )
        recovered = await worker.run_once()
        recovered_snapshot = await harness.store.inspect_operation(
            recovered_operation.id
        )

        side_effect_operation, side_effect_task = await _plan_one_worker_task(
            harness.kernel,
            capability_id="live_runtime.worker.side_effect",
        )
        await harness.store.save_task(
            replace(
                side_effect_task,
                status=TaskStatus.RUNNING,
                metadata={
                    **side_effect_task.metadata,
                    "lease_id": "expired-unsafe",
                    "lease_owner": "worker-1",
                    "lease_expires_at": 0.0,
                },
            )
        )
        blocked = await worker.run_once()
        blocked_snapshot = await harness.store.inspect_operation(
            side_effect_operation.id
        )
    finally:
        await harness.stop()

    assert extended.lease_expires_at > lease.lease_expires_at
    assert RuntimeEventType.WORKER_HEARTBEAT in {
        event.type for event in heartbeat_snapshot.events
    }
    assert recovered.execution is not None
    recovered_stored = next(
        task for task in recovered_snapshot.tasks if task.id == recovered_task.id
    )
    assert recovered_stored.status is TaskStatus.SUCCEEDED
    assert recovered_stored.metadata["expired_lease_recovered"] is True
    assert blocked is None
    blocked_stored = next(
        task for task in blocked_snapshot.tasks if task.id == side_effect_task.id
    )
    assert blocked_stored.status is TaskStatus.BLOCKED
    assert blocked_stored.metadata["manual_recovery_required"] is True
    assert (
        blocked_stored.metadata["manual_recovery_reason"]
        == "expired_side_effecting_lease"
    )


def _worker(kernel, *, lease_seconds: float = 10) -> WorkerRuntime:
    return WorkerRuntime(
        kernel=kernel,
        options=WorkerRuntimeOptions(
            worker_id="live_runtime.db_worker",
            owner="live_runtime",
            lease_seconds=lease_seconds,
            poll_interval_seconds=0,
            max_concurrency=5,
        ),
    )


async def _plan_worker_tasks(kernel, *, count: int):
    operation = await kernel.create_operation(
        operation_type="worker.query",
        request={"kind": "worker_live"},
    )
    tasks = []
    for index in range(count):
        tasks.append(
            await kernel.plan_task(
                operation_id=operation.id,
                capability_id="live_runtime.worker.query",
                owner="live_runtime",
                input={
                    "sql": (
                        "SELECT COUNT(*)::int AS count "
                        "FROM customers "
                        f"WHERE id >= {index + 1}"
                    )
                },
            )
        )
    return operation, tasks


async def _plan_one_worker_task(
    kernel,
    *,
    sql: str | None = None,
    capability_id: str = "live_runtime.worker.query",
):
    operation = await kernel.create_operation(
        operation_type="worker.query",
        request={"kind": "worker_live_single"},
    )
    task = await kernel.plan_task(
        operation_id=operation.id,
        capability_id=capability_id,
        owner="live_runtime",
        input={
            "sql": sql
            or ("SELECT c.name " "FROM customers c " "ORDER BY c.name " "LIMIT 1")
        },
    )
    return operation, task


def _ids(tasks) -> set[str]:
    return {task.id for task in tasks}
