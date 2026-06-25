import asyncio
from dataclasses import replace

import pytest

from daita.plugins import ExtensionRegistry, PluginKind, PluginManifest
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    Evidence,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    RuntimeKernel,
    RuntimeKernelGovernanceBlocked,
    RuntimeKernelLeaseLost,
    RuntimeKernelTaskNotRunnable,
    Task,
    TaskStatus,
    RuntimeEventType,
)


class CountingExecutor:
    id = "kernel_test.echo"
    capability_ids = frozenset({"kernel.echo"})

    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay

    async def execute(self, task, operation, context):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return [
            Evidence(
                kind="kernel.echo.result",
                owner="kernel_test",
                payload={"value": task.input["value"]},
            )
        ]


class TaskApprovalPolicy:
    id = "kernel_test.require_task_approval"
    owner = "kernel_test"

    def applies_to(self, request, operation_type):
        return request.get("governance_stage") == "task"

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="Task requires approval.",
            severity=RiskLevel.MEDIUM,
            operation_id=operation.id,
            required_approvals=("human",),
        )


class KernelPlugin:
    manifest = PluginManifest(
        id="kernel_test",
        display_name="Kernel Test",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
    )

    def __init__(self, *, policy=None, delay: float = 0.0) -> None:
        self.executor = CountingExecutor(delay=delay)
        self.policy = policy

    def declare_capabilities(self):
        return (
            Capability(
                id="kernel.echo",
                owner="kernel_test",
                description="Echo input.",
                domains=frozenset({"kernel"}),
                operation_types=frozenset({"kernel.echo"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"kernel.echo.result"}),
                executor=self.executor.id,
                side_effecting=False,
                replay_safe=True,
            ),
        )

    def get_executors(self):
        return (self.executor,)

    def declare_policies(self):
        return (self.policy,) if self.policy is not None else ()


def _kernel(*, policy=None, delay: float = 0.0):
    plugin = KernelPlugin(policy=policy, delay=delay)
    registry = ExtensionRegistry()
    registry.register(plugin)
    store = InMemoryRuntimeStore()
    kernel = RuntimeKernel(
        runtime_id="runtime-1",
        runtime_kind="test",
        extension_registry=registry,
        runtime_store=store,
    )
    return kernel, store, plugin


async def _persist_echo_task(store):
    operation = Operation(
        id="op-1",
        operation_type="kernel.echo",
        status=OperationStatus.RUNNING,
        request={"value": "alpha"},
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="kernel.echo",
        executor_id="kernel_test.echo",
        input={"value": "alpha"},
        metadata={"owner": "kernel_test"},
    )
    await store.save_operation(operation)
    await store.save_task(task)
    return operation, task


async def test_execute_task_requires_persisted_task():
    kernel, _, plugin = _kernel()

    with pytest.raises(RuntimeKernelTaskNotRunnable, match="not persisted"):
        await kernel.execute_task("missing-task")

    assert plugin.executor.calls == 0


async def test_execute_task_persists_evidence_and_correlated_events():
    kernel, store, plugin = _kernel()
    operation, task = await _persist_echo_task(store)

    result = await kernel.execute_task(task.id)
    snapshot = await store.inspect_operation(operation.id)

    assert plugin.executor.calls == 1
    assert result.evidence[0].operation_id == operation.id
    assert result.evidence[0].task_id == task.id
    assert snapshot.evidence == result.evidence
    success = snapshot.events[-1]
    assert success.runtime_id == "runtime-1"
    assert success.runtime_kind == "test"
    assert success.operation_id == operation.id
    assert success.task_id == task.id
    assert success.capability_id == "kernel.echo"
    assert success.executor_id == "kernel_test.echo"
    event_types = [event.type for event in snapshot.events]
    assert RuntimeEventType.EXECUTOR_STARTED in event_types
    assert RuntimeEventType.EVIDENCE_ACCEPTED in event_types
    assert RuntimeEventType.EXECUTOR_COMPLETED in event_types
    evidence_event = next(
        event
        for event in snapshot.events
        if event.type is RuntimeEventType.EVIDENCE_ACCEPTED
    )
    assert evidence_event.evidence_id == result.evidence[0].id
    assert evidence_event.plugin_id == "kernel_test"


async def test_operation_helpers_complete_and_emit_consistent_event():
    kernel, store, _ = _kernel()
    operation = await kernel.create_operation(
        operation_type="kernel.echo",
        request={"value": "alpha"},
    )

    updated = await kernel.complete_operation(
        operation.id,
        payload={"answer": "done"},
    )
    snapshot = await store.inspect_operation(operation.id)

    assert updated.status is OperationStatus.SUCCEEDED
    assert snapshot.operation.status is OperationStatus.SUCCEEDED
    event = snapshot.events[-1]
    assert event.type is RuntimeEventType.OPERATION_UPDATED
    assert event.runtime_id == "runtime-1"
    assert event.runtime_kind == "test"
    assert event.operation_id == operation.id
    assert event.payload["status"] == "succeeded"
    assert event.payload["answer"] == "done"


async def test_fail_operation_if_active_emits_error_and_terminal_noops():
    kernel, store, _ = _kernel()
    operation = await kernel.create_operation(
        operation_type="kernel.echo",
        request={"value": "alpha"},
    )

    failed = await kernel.fail_operation_if_active(
        operation.id,
        ValueError("boom"),
    )
    second = await kernel.fail_operation_if_active(
        operation.id,
        RuntimeError("ignored"),
    )
    snapshot = await store.inspect_operation(operation.id)

    assert failed.status is OperationStatus.FAILED
    assert second is None
    assert snapshot.operation.status is OperationStatus.FAILED
    error_events = [
        event for event in snapshot.events if event.type is RuntimeEventType.ERROR
    ]
    assert len(error_events) == 1
    assert error_events[0].payload["status"] == "failed"
    assert error_events[0].payload["error"]["type"] == "ValueError"


async def test_append_event_correlates_runtime_task_and_capability():
    kernel, store, _ = _kernel()
    operation, task = await _persist_echo_task(store)
    capability = kernel.extension_registry.get_capability(
        "kernel.echo",
        owner="kernel_test",
    )

    event = await kernel.append_event(
        RuntimeEventType.DIAGNOSTIC,
        operation_id=operation.id,
        task=task,
        capability=capability,
        message="diagnostic",
        payload={"ok": True},
    )

    assert event.runtime_id == "runtime-1"
    assert event.runtime_kind == "test"
    assert event.task_id == task.id
    assert event.capability_id == capability.id
    assert event.executor_id == capability.executor
    assert event.plugin_id == capability.owner
    assert (await store.inspect_operation(operation.id)).events[-1] == event


async def test_apply_terminal_approval_state_updates_operation():
    kernel, store, _ = _kernel()
    operation = await kernel.create_operation(
        operation_type="kernel.echo",
        request={"value": "alpha"},
    )
    await store.save_approval_request(
        ApprovalRequest(
            approval_id="approval-1",
            operation_id=operation.id,
            reason="Rejected.",
            proposed_action={"approval": "human"},
            risk=RiskLevel.MEDIUM,
            status=ApprovalStatus.REJECTED,
        )
    )

    updated = await kernel.apply_terminal_approval_state(operation.id)
    snapshot = await store.inspect_operation(operation.id)

    assert updated.status is OperationStatus.FAILED
    assert snapshot.operation.status is OperationStatus.FAILED
    assert snapshot.events[-1].type is RuntimeEventType.OPERATION_UPDATED


async def test_recover_expired_task_claims_requeues_safe_task():
    kernel, store, _ = _kernel()
    operation, task = await _persist_echo_task(store)
    await store.save_task(
        replace(
            task,
            status=TaskStatus.RUNNING,
            metadata={
                **task.metadata,
                "lease_id": "lease-1",
                "lease_owner": "worker",
                "lease_expires_at": 0.0,
            },
        )
    )

    recovered = await kernel.recover_expired_task_claims(operation.id)
    stored = await store.load_task(task.id)

    assert recovered[0].status is TaskStatus.PENDING
    assert stored.status is TaskStatus.PENDING
    assert stored.metadata["expired_lease_recovered"] is True
    assert "lease_id" not in stored.metadata


async def test_policy_block_prevents_executor_invocation():
    kernel, store, plugin = _kernel(policy=TaskApprovalPolicy())
    operation, task = await _persist_echo_task(store)

    with pytest.raises(RuntimeKernelGovernanceBlocked):
        await kernel.execute_task(task.id)
    snapshot = await store.inspect_operation(operation.id)

    assert plugin.executor.calls == 0
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert snapshot.tasks[0].status is TaskStatus.BLOCKED
    assert snapshot.approval_requests[0].status.value == "pending"
    event_types = [event.type for event in snapshot.events]
    assert RuntimeEventType.APPROVAL_REQUESTED in event_types
    assert RuntimeEventType.EXECUTOR_STARTED not in event_types
    approval_event = next(
        event
        for event in snapshot.events
        if event.type is RuntimeEventType.APPROVAL_REQUESTED
    )
    assert approval_event.approval_id == snapshot.approval_requests[0].approval_id
    assert approval_event.policy_id == "kernel_test.require_task_approval"
    assert approval_event.task_id == task.id


async def test_stale_lease_cannot_execute_or_commit():
    kernel, store, plugin = _kernel()
    operation, task = await _persist_echo_task(store)
    stale = await kernel.claim_task(
        task.id,
        lease_owner="worker-1",
        lease_seconds=0.01,
    )
    claimed = await store.load_task(task.id)
    await store.save_task(
        replace(
            claimed,
            metadata={**claimed.metadata, "lease_expires_at": 0.0},
        )
    )
    fresh = await kernel.claim_task(
        task.id,
        lease_owner="worker-2",
        lease_seconds=30,
    )

    with pytest.raises(RuntimeKernelLeaseLost):
        await kernel.execute_claimed_task(stale)
    result = await kernel.execute_claimed_task(fresh)
    snapshot = await store.inspect_operation(operation.id)

    assert plugin.executor.calls == 1
    assert result.task.status is TaskStatus.SUCCEEDED
    assert snapshot.evidence == result.evidence


async def test_racing_workers_invoke_executor_at_most_once():
    kernel, store, plugin = _kernel(delay=0.01)
    _, task = await _persist_echo_task(store)

    results = await asyncio.gather(
        kernel.execute_task(task.id),
        kernel.execute_task(task.id),
        return_exceptions=True,
    )

    assert plugin.executor.calls == 1
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert any(isinstance(result, RuntimeKernelTaskNotRunnable) for result in results)
