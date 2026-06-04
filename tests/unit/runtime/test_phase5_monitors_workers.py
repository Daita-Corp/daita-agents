import asyncio
import importlib

import pytest

from daita.plugins import ExtensionRegistry, PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    InMemoryRuntimeStore,
    MonitorRuntime,
    MonitorSpec,
    OperationScheduleResult,
    OperationTaskScheduler,
    RiskLevel,
    RuntimeEventType,
    RuntimeKernel,
    TaskStatus,
    Worker,
    WorkerRuntime,
    WorkerRuntimeOptions,
)


class Phase5Executor:
    id = "phase5.executor"
    capability_ids = frozenset(
        {
            "phase5.monitor.action",
            "phase5.worker.task",
        }
    )

    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay

    async def execute(self, task, operation, context):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return [
            Evidence(
                kind="phase5.result",
                owner="phase5",
                payload={
                    "task_id": task.id,
                    "input": task.input,
                    "worker_id": context.get("worker_id"),
                    "monitor_id": context.get("monitor_id"),
                },
            )
        ]


class Phase5Plugin:
    manifest = PluginManifest(
        id="phase5",
        display_name="Phase 5",
        version="1.0.0",
        kind=PluginKind.WORKER_PROVIDER,
    )

    def __init__(self, *, delay: float = 0.0) -> None:
        self.executor = Phase5Executor(delay=delay)

    def declare_capabilities(self):
        common = {
            "owner": "phase5",
            "description": "Phase 5 test capability.",
            "domains": frozenset({"phase5"}),
            "operation_types": frozenset({"phase5.execute"}),
            "access": AccessMode.READ,
            "risk": RiskLevel.LOW,
            "input_schema": {"type": "object"},
            "output_evidence": frozenset({"phase5.result"}),
            "executor": self.executor.id,
            "runtime_only": True,
            "side_effecting": False,
            "replay_safe": True,
        }
        return (
            Capability(id="phase5.monitor.action", **common),
            Capability(id="phase5.worker.task", specialist_only=True, **common),
        )

    def get_executors(self):
        return (self.executor,)

    def get_workers(self):
        return (
            Worker(
                id="phase5.worker",
                owner="phase5",
                role="phase5_worker",
                capability_ids=frozenset({"phase5.worker.task"}),
                input_schema={"type": "object"},
                output_evidence=frozenset({"phase5.result"}),
                max_concurrency=2,
            ),
        )


def _kernel(*, delay: float = 0.0):
    plugin = Phase5Plugin(delay=delay)
    registry = ExtensionRegistry()
    registry.register(plugin)
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="phase5-runtime",
        runtime_kind="phase5",
        extension_registry=registry,
        runtime_store=store,
    )
    return kernel, store, plugin


async def test_monitor_creates_runtime_operation_and_task_without_direct_handler():
    kernel, store, plugin = _kernel()
    monitor = MonitorRuntime(kernel=kernel)

    result = await monitor.tick(
        MonitorSpec(
            id="orders_backlog",
            name="Orders Backlog",
            trigger={"path": "count", "gt": 5},
            action_capability_id="phase5.monitor.action",
            action_input={"severity": "high"},
        ),
        value={"count": 8},
        execute_actions=False,
    )
    snapshot = await store.inspect_operation(result.operation_id)

    assert result.triggered is True
    assert plugin.executor.calls == 0
    assert snapshot.operation.metadata["monitor_id"] == "orders_backlog"
    assert snapshot.tasks[0].capability_id == "phase5.monitor.action"
    assert snapshot.tasks[0].status is TaskStatus.PENDING
    assert {
        RuntimeEventType.MONITOR_TICKED,
        RuntimeEventType.MONITOR_TRIGGERED,
        RuntimeEventType.TASK_CREATED,
    }.issubset({event.type for event in await store.list_events()})


async def test_worker_handoff_claims_and_executes_through_kernel_events():
    kernel, store, plugin = _kernel()
    operation = await kernel.create_operation(
        operation_type="phase5.execute",
        request={"kind": "worker"},
    )
    task = await kernel.plan_task(
        operation_id=operation.id,
        capability_id="phase5.worker.task",
        owner="phase5",
        input={"value": "alpha"},
    )
    worker = WorkerRuntime(
        kernel=kernel,
        options=WorkerRuntimeOptions(worker_id="phase5.worker", owner="phase5"),
    )

    handoff = await worker.handoff_task(task.id, reason="test")
    result = await worker.execute_handoff(handoff)
    snapshot = await store.inspect_operation(operation.id)

    assert plugin.executor.calls == 1
    assert result.execution.task.status is TaskStatus.SUCCEEDED
    assert result.execution.evidence[0].payload["worker_id"] == "phase5.worker"
    event_types = {event.type for event in snapshot.events}
    assert RuntimeEventType.WORKER_HANDOFF in event_types
    assert RuntimeEventType.WORKER_LEASE_CLAIMED in event_types
    assert RuntimeEventType.WORKER_COMPLETED in event_types
    assert RuntimeEventType.EXECUTOR_STARTED in event_types
    worker_event = next(
        event
        for event in snapshot.events
        if event.type is RuntimeEventType.WORKER_LEASE_CLAIMED
    )
    assert worker_event.runtime_id == "phase5-runtime"
    assert worker_event.task_id == task.id
    assert worker_event.capability_id == "phase5.worker.task"


async def test_worker_lease_fencing_prevents_duplicate_execution():
    kernel, store, plugin = _kernel(delay=0.01)
    operation = await kernel.create_operation(
        operation_type="phase5.execute",
        request={"kind": "worker"},
    )
    task = await kernel.plan_task(
        operation_id=operation.id,
        capability_id="phase5.worker.task",
        owner="phase5",
        input={"value": "alpha"},
    )
    worker = WorkerRuntime(
        kernel=kernel,
        options=WorkerRuntimeOptions(worker_id="phase5.worker", owner="phase5"),
    )
    handoff = await worker.handoff_task(task.id, reason="race")

    results = await asyncio.gather(
        worker.execute_handoff(handoff),
        worker.execute_handoff(handoff),
    )
    snapshot = await store.inspect_operation(operation.id)

    assert plugin.executor.calls == 1
    assert (
        sum(result.execution is not None and not result.error for result in results)
        == 1
    )
    assert sum(result.error is not None for result in results) == 1
    assert snapshot.tasks[0].status is TaskStatus.SUCCEEDED


async def test_operation_scheduler_skips_completed_and_resumes_pending_tasks():
    kernel, store, plugin = _kernel()
    operation = await kernel.create_operation(
        operation_type="phase5.execute",
        request={"kind": "schedule"},
    )
    completed = await kernel.plan_task(
        operation_id=operation.id,
        capability_id="phase5.monitor.action",
        owner="phase5",
        input={"value": "done"},
    )
    pending = await kernel.plan_task(
        operation_id=operation.id,
        capability_id="phase5.monitor.action",
        owner="phase5",
        input={"value": "pending"},
    )
    await kernel.execute_task(completed.id)

    result = await OperationTaskScheduler(kernel=kernel).run_operation(operation.id)

    assert isinstance(result, OperationScheduleResult)
    assert plugin.executor.calls == 2
    assert completed.id in result.skipped_task_ids
    assert pending.id in result.executed_task_ids
    assert result.complete is True
    assert result.snapshot.operation.status.value == "succeeded"


def test_legacy_workflow_relay_scaling_modules_are_removed():
    for module_name in (
        "daita.core.workflow",
        "daita.core.relay",
        "daita.core.scaling",
        "daita.plugins.orchestrator",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
