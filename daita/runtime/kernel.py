"""Shared runtime execution kernel.

The kernel coordinates persisted operation/task execution through the existing
runtime store, extension registry, governance, and approval contracts. Concrete
runtimes remain responsible for planning and domain facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

from .governance import PolicyEvaluator
from .primitives import (
    Capability,
    Evidence,
    GovernanceAuditRecord,
    GovernanceResult,
    Operation,
    OperationStatus,
    ApprovalStatus,
    PolicyDecisionTrace,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStore,
    Task,
    TaskDependency,
    TaskDependencyKind,
    TaskStatus,
)
from .store import OperationSnapshot

if TYPE_CHECKING:
    from daita.plugins import ExtensionRegistry

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }
)
_TERMINAL_OPERATION_STATUSES = frozenset(
    {
        OperationStatus.SUCCEEDED,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
    }
)
_DEFAULT_LEASE_SECONDS = 300.0


@dataclass(frozen=True)
class _KernelGovernancePersistence:
    result: GovernanceResult
    audit_record: GovernanceAuditRecord
    approvals_to_request: tuple[Any, ...]
    events: tuple[RuntimeEvent, ...]


class RuntimeFactProvider(Protocol):
    """Runtime-owned facts and hooks consumed by the shared kernel."""

    async def build_policy_facts(
        self,
        *,
        operation: Operation,
        stage: Literal["operation", "task"],
        task: Task | None = None,
        capability: Capability | None = None,
    ) -> Mapping[str, Any]:
        """Return domain facts for policy evaluation."""


@dataclass(frozen=True)
class TaskLease:
    """A fenced lease held by a runtime or worker for one task."""

    lease_id: str
    task_id: str
    operation_id: str
    lease_owner: str
    lease_expires_at: float
    attempt_count: int
    worker_id: str | None = None
    worker_owner: str | None = None


@dataclass(frozen=True)
class TaskExecutionResult:
    """Result of one kernel-managed task execution attempt."""

    operation: Operation
    task: Task
    capability: Capability
    evidence: tuple[Evidence, ...] = ()
    events: tuple[RuntimeEvent, ...] = ()
    governance: GovernanceResult | None = None
    blocked: bool = False
    error: BaseException | None = None


class RuntimeKernelExecutionError(Exception):
    """Base class for kernel execution failures with optional persisted state."""

    def __init__(
        self,
        message: str,
        *,
        result: TaskExecutionResult | None = None,
        operation: Operation | None = None,
        governance: GovernanceResult | None = None,
    ) -> None:
        self.result = result
        self.operation = operation
        self.governance = governance
        super().__init__(message)


class RuntimeKernelGovernanceBlocked(RuntimeKernelExecutionError):
    """Raised when governance denies execution or requires approval."""


class RuntimeKernelTaskNotRunnable(RuntimeKernelExecutionError):
    """Raised when a persisted task cannot currently run."""


class RuntimeKernelTaskAlreadyTerminal(RuntimeKernelTaskNotRunnable):
    """Raised when callers try to replay a terminal task."""


class RuntimeKernelLeaseLost(RuntimeKernelTaskNotRunnable):
    """Raised when a stale lease tries to commit task output."""


class RuntimeKernelExecutorFailed(RuntimeKernelExecutionError):
    """Raised when an executor fails after the kernel claimed a task."""


class RuntimeKernel:
    """Shared operation/task/governance/executor boundary for runtimes."""

    def __init__(
        self,
        *,
        runtime_id: str,
        runtime_kind: str,
        extension_registry: "ExtensionRegistry",
        runtime_store: RuntimeStore,
        approval_channel: Any | None = None,
        governance: Any | None = None,
        fact_provider: RuntimeFactProvider | Any | None = None,
    ) -> None:
        self.runtime_id = runtime_id
        self.runtime_kind = runtime_kind
        self.extension_registry = extension_registry
        self.store = runtime_store
        self.approval_channel = approval_channel
        self.governance = governance
        self.fact_provider = fact_provider

    async def create_operation(
        self,
        *,
        operation_type: str,
        request: Mapping[str, Any],
        required_evidence: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        evaluate_governance: bool = True,
    ) -> Operation:
        """Create and persist one operation plus an operation-created event."""
        operation = Operation(
            id=operation_id or f"{self.runtime_kind}-op-{uuid4()}",
            operation_type=operation_type,
            status=OperationStatus.RUNNING,
            request=dict(request),
            required_evidence=frozenset(required_evidence),
            metadata={
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                **dict(metadata or {}),
            },
        )
        await self.store.save_operation(operation)
        await self.store.append_event(
            self._event(
                RuntimeEventType.OPERATION_CREATED,
                operation=operation,
                message=f"Operation {operation.id} created.",
                payload={"operation_type": operation.operation_type},
            )
        )
        if not evaluate_governance:
            return operation
        await self.evaluate_operation_governance(operation.id)
        return operation

    async def evaluate_operation_governance(
        self,
        operation_id: str,
        *,
        capability: Capability | None = None,
    ) -> GovernanceResult:
        """Evaluate and persist operation-stage governance through the kernel."""
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        governance_persistence = await self._evaluate_governance_persistence(
            operation,
            stage="operation",
            capability=capability,
        )
        governance = governance_persistence.result
        if governance.blocked or governance.pending_approval:
            blocked = replace(operation, status=OperationStatus.BLOCKED)
            await self.store.commit_governance_blocked(
                operation=blocked,
                task=None,
                decisions=governance.decisions,
                audit_record=governance_persistence.audit_record,
                approval_requests=governance_persistence.approvals_to_request,
                events=(
                    *governance_persistence.events,
                    self._event(
                        RuntimeEventType.OPERATION_UPDATED,
                        operation=operation,
                        message=f"Operation {operation.id} blocked by governance policy.",
                        payload={"governance": governance.to_dict()},
                    ),
                ),
            )
            raise RuntimeKernelGovernanceBlocked(
                "Operation blocked by governance policy.",
                result=None,
                operation=blocked,
                governance=governance,
            )
        await self._commit_allowed_governance(governance_persistence)
        return governance

    async def plan_task(
        self,
        *,
        operation_id: str,
        capability_id: str,
        owner: str | None = None,
        input: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        dependencies: Iterable[TaskDependency] = (),
        task_id: str | None = None,
    ) -> Task:
        """Persist one planned task for an existing operation."""
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        capability = self.extension_registry.get_capability(capability_id, owner=owner)
        task = Task(
            id=task_id or f"{self.runtime_kind}-task-{uuid4()}",
            operation_id=operation_id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input=dict(input),
            status=TaskStatus.PENDING,
            required_evidence=capability.output_evidence,
            dependencies=tuple(dependencies),
            metadata={
                "owner": capability.owner,
                **dict(metadata or {}),
            },
        )
        await self.store.save_task(task)
        await self.store.append_event(
            self._event(
                RuntimeEventType.TASK_CREATED,
                operation=operation,
                task=task,
                capability=capability,
                message=f"Task {task.id} planned.",
                payload={
                    "capability_id": task.capability_id,
                    "executor_id": task.executor_id,
                    "input": task.input,
                    "required_evidence": sorted(task.required_evidence),
                    "metadata": task.metadata,
                },
            )
        )
        return task

    async def append_event(
        self,
        type: RuntimeEventType,
        *,
        operation_id: str,
        message: str,
        task: Task | None = None,
        capability: Capability | None = None,
        task_id: str | None = None,
        capability_id: str | None = None,
        executor_id: str | None = None,
        plugin_id: str | None = None,
        policy_id: str | None = None,
        approval_id: str | None = None,
        evidence_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> RuntimeEvent:
        """Append one runtime-correlated event and return the persisted shape."""
        current_trace_id, current_span_id = _current_trace_ids()
        event = RuntimeEvent(
            type=type,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation_id,
            task_id=task.id if task is not None else task_id,
            capability_id=(capability.id if capability is not None else capability_id),
            executor_id=(
                capability.executor
                if capability is not None
                else task.executor_id if task is not None else executor_id
            ),
            plugin_id=capability.owner if capability is not None else plugin_id,
            policy_id=policy_id,
            approval_id=approval_id,
            evidence_id=evidence_id,
            trace_id=trace_id or current_trace_id,
            span_id=span_id or current_span_id,
            message=message,
            payload=dict(payload or {}),
        )
        await self.store.append_event(event)
        return event

    async def update_operation(
        self,
        operation_id: str,
        status: OperationStatus,
        *,
        event_type: RuntimeEventType = RuntimeEventType.OPERATION_UPDATED,
        message: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Operation:
        """Persist an operation status transition and matching runtime event."""
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        updated = replace(operation, status=OperationStatus(status))
        await self.store.save_operation(updated)
        await self.append_event(
            event_type,
            operation_id=operation_id,
            message=message
            or f"Operation {operation_id} updated to {updated.status.value}.",
            payload={"status": updated.status.value, **dict(payload or {})},
        )
        return updated

    async def complete_operation(
        self,
        operation_id: str,
        *,
        status: OperationStatus = OperationStatus.SUCCEEDED,
        payload: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> Operation:
        """Mark an operation terminal and emit a generic update event."""
        status = OperationStatus(status)
        if status not in _TERMINAL_OPERATION_STATUSES:
            raise ValueError("complete_operation requires a terminal status")
        return await self.update_operation(
            operation_id,
            status,
            event_type=RuntimeEventType.OPERATION_UPDATED,
            message=message
            or f"Operation {operation_id} finished with {status.value}.",
            payload=payload,
        )

    async def block_operation(
        self,
        operation_id: str,
        *,
        payload: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> Operation:
        """Mark an operation blocked and emit a generic update event."""
        return await self.update_operation(
            operation_id,
            OperationStatus.BLOCKED,
            event_type=RuntimeEventType.OPERATION_UPDATED,
            message=message or f"Operation {operation_id} blocked.",
            payload=payload,
        )

    async def block_task(
        self,
        task_id: str,
        *,
        message: str | None = None,
        payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Task:
        """Mark one persisted task blocked and emit a generic update event."""
        task = await self.store.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        capability = self._capability_for_task(task)
        blocked = replace(
            task,
            status=TaskStatus.BLOCKED,
            metadata={
                **_metadata_without_active_lease(task.metadata),
                **dict(metadata or {}),
            },
        )
        operation = await self.store.load_operation(task.operation_id)
        if operation is None:
            raise KeyError(task.operation_id)
        event = self._event(
            RuntimeEventType.TASK_UPDATED,
            operation=operation,
            task=blocked,
            capability=capability,
            message=message or f"Task {task.id} blocked.",
            payload=payload,
        )
        commit = getattr(self.store, "commit_task_blocked", None)
        if commit is not None:
            await commit(operation=None, task=blocked, events=(event,))
        else:
            await self.store.save_task(blocked)
            await self.store.append_event(event)
        return blocked

    async def apply_terminal_approval_state(
        self,
        operation_id: str,
    ) -> Operation | None:
        """Apply terminal approval status to an operation when present."""
        approvals = await self.store.list_approval_requests(operation_id)
        statuses = {request.status for request in approvals}
        if ApprovalStatus.REJECTED in statuses:
            return await self.complete_operation(
                operation_id,
                status=OperationStatus.FAILED,
                message=f"Operation {operation_id} failed because approval was rejected.",
            )
        if ApprovalStatus.CANCELLED in statuses:
            return await self.complete_operation(
                operation_id,
                status=OperationStatus.CANCELLED,
                message=f"Operation {operation_id} cancelled by approval state.",
            )
        if ApprovalStatus.EXPIRED in statuses:
            return await self.block_operation(
                operation_id,
                message=(
                    f"Operation {operation_id} remains blocked because approval expired."
                ),
            )
        return None

    async def recover_expired_task_claims(self, operation_id: str) -> tuple[Task, ...]:
        """Recover expired running task leases or block unsafe replays."""
        recovered: list[Task] = []
        now = time.time()
        for task in await self.store.list_tasks(operation_id):
            if task.status is not TaskStatus.RUNNING:
                continue
            lease_expires_at = task.metadata.get("lease_expires_at")
            if lease_expires_at is None or float(lease_expires_at) > now:
                continue
            capability = self._capability_for_task(task)
            if (
                capability.idempotent
                or capability.replay_safe
                or not capability.side_effecting
            ):
                updated = replace(
                    task,
                    status=TaskStatus.PENDING,
                    metadata={
                        **_metadata_without_active_lease(task.metadata),
                        "expired_lease_recovered": True,
                    },
                )
                await self.store.save_task(updated)
                recovered.append(updated)
                continue
            await self.block_operation(operation_id)
            recovered.append(
                await self.block_task(
                    task.id,
                    message=(
                        f"Task {task.id} lease expired; manual recovery required "
                        "before replaying side-effecting work."
                    ),
                    metadata={
                        "manual_recovery_required": True,
                        "manual_recovery_reason": "expired_side_effecting_lease",
                    },
                )
            )
        return tuple(recovered)

    async def fail_operation_if_active(
        self,
        operation_id: str,
        error: BaseException,
    ) -> Operation | None:
        """Fail a non-terminal operation; terminal operations are left unchanged."""
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        if operation.status in _TERMINAL_OPERATION_STATUSES:
            return None
        return await self.update_operation(
            operation_id,
            OperationStatus.FAILED,
            event_type=RuntimeEventType.ERROR,
            message=f"Operation {operation_id} failed.",
            payload={"error": {"type": type(error).__name__, "message": str(error)}},
        )

    async def execute_task(
        self,
        task_id: str,
        *,
        context: Mapping[str, Any] | None = None,
        lease_owner: str | None = None,
        lease_seconds: float | None = None,
    ) -> TaskExecutionResult:
        """Execute a persisted task through a claimed lease."""
        stored = await self.store.load_task(task_id)
        if stored is None:
            raise RuntimeKernelTaskNotRunnable(
                f"Task {task_id} is not persisted and cannot execute."
            )
        lease = await self.claim_task(
            task_id,
            lease_owner=lease_owner or self.runtime_id,
            lease_seconds=lease_seconds or _DEFAULT_LEASE_SECONDS,
        )
        return await self.execute_claimed_task(lease, context=context)

    async def claim_task(
        self,
        task_id: str,
        *,
        lease_owner: str,
        lease_seconds: float,
        worker_id: str | None = None,
        worker_owner: str | None = None,
    ) -> TaskLease:
        """Claim one persisted task and return its fencing token."""
        await self._recover_expired_claim_if_needed(task_id)
        lease_id = f"lease-{uuid4()}"
        lease_expires_at = time.time() + lease_seconds
        claim = getattr(self.store, "claim_task", None)
        if claim is None:
            raise RuntimeKernelTaskNotRunnable("Runtime store cannot claim tasks.")
        claimed = await claim(
            task_id,
            lease_id=lease_id,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
            worker_id=worker_id,
            worker_owner=worker_owner,
        )
        if claimed is None:
            task = await self.store.load_task(task_id)
            result = await self._result_for_task(task) if task is not None else None
            if task is not None and task.status in _TERMINAL_TASK_STATUSES:
                raise RuntimeKernelTaskAlreadyTerminal(
                    f"Task {task.id} is already {task.status.value}.",
                    result=result,
                )
            raise RuntimeKernelTaskNotRunnable(
                f"Task {task_id} could not be claimed.",
                result=result,
            )
        return TaskLease(
            lease_id=lease_id,
            task_id=claimed.id,
            operation_id=claimed.operation_id,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
            attempt_count=int(claimed.metadata.get("attempt_count") or 0),
            worker_id=worker_id,
            worker_owner=worker_owner,
        )

    async def heartbeat_task(
        self,
        lease: TaskLease,
        *,
        extend_by_seconds: float,
    ) -> TaskLease:
        """Extend the current lease using its fencing token."""
        heartbeat = getattr(self.store, "heartbeat_task", None)
        if heartbeat is None:
            raise RuntimeKernelTaskNotRunnable("Runtime store cannot heartbeat tasks.")
        lease_expires_at = time.time() + extend_by_seconds
        task = await heartbeat(
            lease.task_id,
            lease_id=lease.lease_id,
            lease_expires_at=lease_expires_at,
        )
        if task is None:
            raise RuntimeKernelLeaseLost(
                f"Task {lease.task_id} lease is no longer current."
            )
        return replace(lease, lease_expires_at=lease_expires_at)

    async def execute_claimed_task(
        self,
        lease: TaskLease,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> TaskExecutionResult:
        """Execute a task using an already-owned lease."""
        task = await self.store.load_task(lease.task_id)
        if task is None:
            raise RuntimeKernelTaskNotRunnable(f"Task {lease.task_id} is missing.")
        if task.metadata.get("lease_id") != lease.lease_id:
            raise RuntimeKernelLeaseLost(
                f"Task {lease.task_id} lease is no longer current."
            )
        operation = await self.store.load_operation(task.operation_id)
        if operation is None:
            raise RuntimeKernelTaskNotRunnable(
                f"Operation {task.operation_id} is missing for task {task.id}."
            )
        capability = self._capability_for_task(task)
        if capability.executor != task.executor_id:
            raise ValueError(
                f"task executor {task.executor_id!r} does not match capability "
                f"{capability.id!r} executor {capability.executor!r}"
            )
        if task.metadata.get("manual_recovery_required"):
            raise RuntimeKernelTaskNotRunnable(
                f"Task {task.id} requires manual recovery before replay.",
                result=TaskExecutionResult(operation, task, capability),
            )
        if task.status in _TERMINAL_TASK_STATUSES:
            raise RuntimeKernelTaskAlreadyTerminal(
                f"Task {task.id} is already {task.status.value}.",
                result=TaskExecutionResult(operation, task, capability),
            )
        governance_persistence = await self._evaluate_governance_persistence(
            operation,
            stage="task",
            task=task,
            capability=capability,
        )
        governance = governance_persistence.result
        if governance.blocked or governance.pending_approval:
            blocked_task = replace(
                task,
                status=TaskStatus.BLOCKED,
                metadata=_metadata_without_active_lease(task.metadata),
            )
            blocked_operation = replace(operation, status=OperationStatus.BLOCKED)
            events = (
                *governance_persistence.events,
                self._event(
                    RuntimeEventType.OPERATION_UPDATED,
                    operation=operation,
                    task=task,
                    capability=capability,
                    message=f"Operation {operation.id} blocked by governance policy.",
                    payload={"governance": governance.to_dict()},
                ),
                self._event(
                    RuntimeEventType.TASK_UPDATED,
                    operation=operation,
                    task=task,
                    capability=capability,
                    message=f"Task {task.id} blocked by governance policy.",
                    payload={"governance": governance.to_dict()},
                ),
            )
            await self.store.commit_governance_blocked(
                operation=blocked_operation,
                task=blocked_task,
                decisions=governance.decisions,
                audit_record=governance_persistence.audit_record,
                approval_requests=governance_persistence.approvals_to_request,
                events=events,
            )
            result = TaskExecutionResult(
                blocked_operation,
                blocked_task,
                capability,
                governance=governance,
                events=events,
                blocked=True,
            )
            raise RuntimeKernelGovernanceBlocked(
                "Task blocked by governance policy.",
                result=result,
                operation=blocked_operation,
                governance=governance,
            )
        await self._commit_allowed_governance(governance_persistence)
        task = await self._executable_task(task, operation)
        readiness = await self._task_readiness(task, operation)
        if not readiness["ready"]:
            blocked_task = replace(
                task,
                status=TaskStatus.BLOCKED,
                metadata=_metadata_without_active_lease(task.metadata),
            )
            event = self._event(
                RuntimeEventType.TASK_UPDATED,
                operation=operation,
                task=task,
                capability=capability,
                message=f"Task {task.id} blocked by unsatisfied dependencies.",
                payload={"readiness": readiness},
            )
            committed = await self.store.commit_task_blocked(
                operation=None,
                task=blocked_task,
                events=(event,),
                lease_id=lease.lease_id,
            )
            result = TaskExecutionResult(
                operation,
                blocked_task,
                capability,
                events=(event,),
                governance=governance,
                blocked=True,
            )
            if not committed:
                raise RuntimeKernelLeaseLost(
                    f"Task {task.id} lease was lost before blocking.",
                    result=result,
                )
            raise RuntimeKernelTaskNotRunnable(
                f"Task {task.id} dependencies are not satisfied.",
                result=result,
            )

        started_event = self._event(
            RuntimeEventType.TASK_UPDATED,
            operation=operation,
            task=task,
            capability=capability,
            message=f"Task {task.id} started.",
        )
        await self.store.commit_task_started(task, started_event)
        executor = self.extension_registry.get_executor(task.executor_id)
        from daita.core.tracing import TraceType, get_trace_manager

        async with get_trace_manager().span(
            "runtime_executor",
            TraceType.TOOL_EXECUTION,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation.id,
            execution_id=operation.id,
            operation_type=operation.operation_type,
            task_id=task.id,
            capability_id=capability.id,
            executor_id=task.executor_id,
            plugin_id=capability.owner,
            lease_owner=lease.lease_owner,
            attempt_count=lease.attempt_count,
        ):
            trace_id, span_id = _current_trace_ids()
            executor_started_event = self._event(
                RuntimeEventType.EXECUTOR_STARTED,
                operation=operation,
                task=task,
                capability=capability,
                message=f"Executor {task.executor_id} started.",
                payload={
                    "executor_id": task.executor_id,
                    "attempt_count": lease.attempt_count,
                },
                trace_id=trace_id,
                span_id=span_id,
            )
            await self.store.append_event(executor_started_event)
            try:
                evidence = tuple(
                    await executor.execute(task, operation, dict(context or {}))
                )
            except Exception as exc:
                failed_task = replace(
                    task,
                    status=TaskStatus.FAILED,
                    metadata=_metadata_without_active_lease(task.metadata),
                )
                event = self._event(
                    RuntimeEventType.ERROR,
                    operation=operation,
                    task=task,
                    capability=capability,
                    message=f"Task {task.id} failed.",
                    payload={
                        "error": {"type": type(exc).__name__, "message": str(exc)}
                    },
                    trace_id=trace_id,
                    span_id=span_id,
                )
                committed = await self.store.commit_task_failed(
                    failed_task,
                    event,
                    lease_id=lease.lease_id,
                )
                executor_failed_event = self._event(
                    RuntimeEventType.EXECUTOR_FAILED,
                    operation=operation,
                    task=failed_task,
                    capability=capability,
                    message=f"Executor {task.executor_id} failed.",
                    payload={
                        "executor_id": task.executor_id,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    },
                    trace_id=trace_id,
                    span_id=span_id,
                )
                result = TaskExecutionResult(
                    operation,
                    failed_task,
                    capability,
                    events=(started_event, executor_started_event, event),
                    governance=governance,
                    error=exc,
                )
                if not committed:
                    raise RuntimeKernelLeaseLost(
                        f"Task {task.id} lease was lost before failure commit.",
                        result=result,
                    ) from exc
                await self.store.append_event(executor_failed_event)
                result = replace(
                    result,
                    events=(*result.events, executor_failed_event),
                )
                raise RuntimeKernelExecutorFailed(
                    f"Executor {task.executor_id} failed for task {task.id}.",
                    result=result,
                ) from exc
            accepted_evidence = tuple(
                self._accepted_task_evidence(item, operation=operation, task=task)
                for item in evidence
            )
            succeeded_task = replace(
                task,
                status=TaskStatus.SUCCEEDED,
                metadata={
                    **_metadata_without_active_lease(task.metadata),
                    "output_evidence_refs": [
                        item.id for item in accepted_evidence if item.id is not None
                    ],
                },
            )
            event = self._event(
                RuntimeEventType.TASK_UPDATED,
                operation=operation,
                task=task,
                capability=capability,
                message=f"Task {task.id} succeeded.",
                payload={
                    "evidence_ids": [
                        item.id for item in accepted_evidence if item.id is not None
                    ]
                },
                trace_id=trace_id,
                span_id=span_id,
            )
            committed = await self.store.commit_task_succeeded(
                succeeded_task,
                accepted_evidence,
                event,
                lease_id=lease.lease_id,
            )
            result = TaskExecutionResult(
                operation,
                succeeded_task,
                capability,
                evidence=accepted_evidence,
                events=(started_event, executor_started_event, event),
                governance=governance,
            )
            if not committed:
                raise RuntimeKernelLeaseLost(
                    f"Task {task.id} lease was lost before success commit.",
                    result=result,
                )
            evidence_events = tuple(
                self._event(
                    RuntimeEventType.EVIDENCE_ACCEPTED,
                    operation=operation,
                    task=succeeded_task,
                    capability=capability,
                    message=f"Evidence {item.id} accepted.",
                    payload=_evidence_event_payload(item),
                    evidence_id=item.id,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                for item in accepted_evidence
            )
            executor_completed_event = self._event(
                RuntimeEventType.EXECUTOR_COMPLETED,
                operation=operation,
                task=succeeded_task,
                capability=capability,
                message=f"Executor {task.executor_id} completed.",
                payload={
                    "executor_id": task.executor_id,
                    "evidence_ids": [
                        item.id for item in accepted_evidence if item.id is not None
                    ],
                },
                trace_id=trace_id,
                span_id=span_id,
            )
            for emitted in (*evidence_events, executor_completed_event):
                await self.store.append_event(emitted)
            result = replace(
                result,
                events=(*result.events, *evidence_events, executor_completed_event),
            )
            return result

    async def execute_capability(
        self,
        capability_id: str,
        *,
        input: Mapping[str, Any],
        owner: str | None = None,
        operation_id: str | None = None,
        operation_type: str | None = None,
        task_metadata: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TaskExecutionResult:
        """Persist operation/task state for a capability, then execute the task."""
        capability = self.extension_registry.get_capability(capability_id, owner=owner)
        if operation_id is None:
            selected_operation_type = (
                operation_type
                or (
                    next(iter(capability.operation_types))
                    if len(capability.operation_types) == 1
                    else None
                )
                or "capability.execute"
            )
            operation = await self.create_operation(
                operation_type=selected_operation_type,
                request={
                    "input": dict(input),
                    "capability_id": capability.id,
                    "capability_owner": capability.owner,
                },
                required_evidence=capability.output_evidence,
                metadata={
                    "direct_capability_id": capability.id,
                    "direct_capability_owner": capability.owner,
                },
            )
        else:
            operation = await self.store.load_operation(operation_id)
            if operation is None:
                raise KeyError(operation_id)
        task = await self.plan_task(
            operation_id=operation.id,
            capability_id=capability.id,
            owner=capability.owner,
            input=input,
            metadata=task_metadata,
        )
        return await self.execute_task(task.id, context=context)

    async def resume_operation(self, operation_id: str) -> OperationSnapshot:
        """Return persisted operation state for runtime-specific resume flows."""
        inspect = getattr(self.store, "inspect_operation", None)
        if inspect is None:
            raise RuntimeKernelTaskNotRunnable(
                "Runtime store cannot inspect operations."
            )
        snapshot = await inspect(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        return snapshot

    async def _evaluate_governance(
        self,
        operation: Operation,
        *,
        stage: Literal["operation", "task"],
        task: Task | None = None,
        capability: Capability | None = None,
    ) -> GovernanceResult:
        provider_evaluator = getattr(self.fact_provider, "evaluate_governance", None)
        if provider_evaluator is not None:
            return await provider_evaluator(
                operation,
                task=task,
                capability=capability,
                stage=stage,
            )
        policies = tuple(self.extension_registry.policies)
        facts = await self._policy_facts(
            operation=operation,
            stage=stage,
            task=task,
            capability=capability,
        )
        governance_operation = replace(
            operation,
            request={
                **operation.request,
                "governance_stage": stage,
                "runtime_facts": facts,
                **({"task": task.to_dict()} if task is not None else {}),
                **(
                    {"capability": capability.to_dict()}
                    if capability is not None
                    else {}
                ),
            },
        )
        return PolicyEvaluator(policies).evaluate_operation(
            governance_operation,
            contract={"runtime_facts": facts},
        )

    async def _evaluate_governance_persistence(
        self,
        operation: Operation,
        *,
        stage: Literal["operation", "task"],
        task: Task | None = None,
        capability: Capability | None = None,
    ) -> _KernelGovernancePersistence:
        provider_persistence = getattr(
            self.fact_provider,
            "evaluate_governance_persistence",
            None,
        )
        if provider_persistence is not None:
            persistence = await provider_persistence(
                operation,
                task=task,
                capability=capability,
                stage=stage,
            )
            return _KernelGovernancePersistence(
                result=persistence.result,
                audit_record=persistence.audit_record,
                approvals_to_request=tuple(persistence.approvals_to_request),
                events=tuple(
                    self._correlated_event(event, task=task, capability=capability)
                    for event in persistence.events
                ),
            )
        facts = await self._policy_facts(
            operation=operation,
            stage=stage,
            task=task,
            capability=capability,
        )
        result = await self._evaluate_governance(
            operation,
            stage=stage,
            task=task,
            capability=capability,
        )
        return _KernelGovernancePersistence(
            result=result,
            audit_record=self._governance_audit_record(
                operation,
                result,
                stage=stage,
                facts=facts,
                task=task,
                capability=capability,
            ),
            approvals_to_request=result.approval_requests,
            events=self._governance_events(operation, result, task, capability),
        )

    async def _policy_facts(
        self,
        *,
        operation: Operation,
        stage: Literal["operation", "task"],
        task: Task | None,
        capability: Capability | None,
    ) -> Mapping[str, Any]:
        builder = getattr(self.fact_provider, "build_policy_facts", None)
        if builder is None:
            return {
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "operation_id": operation.id,
                "operation_type": operation.operation_type,
                "stage": stage,
                "task_id": task.id if task is not None else None,
                "capability_id": capability.id if capability is not None else None,
            }
        return await builder(
            operation=operation,
            stage=stage,
            task=task,
            capability=capability,
        )

    async def _commit_allowed_governance(
        self,
        governance_persistence: _KernelGovernancePersistence,
    ) -> None:
        await self.store.commit_governance_evaluation(
            decisions=governance_persistence.result.decisions,
            audit_record=governance_persistence.audit_record,
            approval_requests=governance_persistence.approvals_to_request,
            events=governance_persistence.events,
        )

    async def _task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        provider_readiness = getattr(self.fact_provider, "task_readiness", None)
        if provider_readiness is not None:
            return await provider_readiness(task, operation)
        unsatisfied: list[dict[str, Any]] = []
        for dependency in task.dependencies:
            if dependency.kind is TaskDependencyKind.EVIDENCE:
                if not await self._evidence_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
            elif dependency.kind is TaskDependencyKind.APPROVAL:
                if not await self._approval_dependency_satisfied(dependency, operation):
                    unsatisfied.append(dependency.to_dict())
        return {
            "ready": not unsatisfied,
            "unsatisfied_dependencies": unsatisfied,
            "dependency_count": len(task.dependencies),
        }

    async def _executable_task(self, task: Task, operation: Operation) -> Task:
        provider_input = getattr(self.fact_provider, "executable_input_for_task", None)
        if provider_input is None:
            return task
        executable_input = await provider_input(task, operation)
        if executable_input == task.input:
            return task
        updated = replace(task, input=dict(executable_input))
        await self.store.save_task(updated)
        return updated

    async def _evidence_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for evidence in await self.store.list_evidence(operation_id):
            if evidence.kind != dependency.evidence_kind:
                continue
            if (
                dependency.evidence_id is not None
                and evidence.id != dependency.evidence_id
            ):
                continue
            if (
                dependency.evidence_owner is not None
                and evidence.owner != dependency.evidence_owner
            ):
                continue
            if (
                dependency.producer_task_id is not None
                and evidence.task_id != dependency.producer_task_id
            ):
                continue
            if evidence.accepted is not dependency.evidence_accepted:
                continue
            if (
                dependency.input_hash is not None
                and evidence.metadata.get("task_input_hash") != dependency.input_hash
            ):
                continue
            if _payload_contains(evidence.payload, dependency.evidence_payload):
                if (
                    dependency.payload_fingerprint is not None
                    and dependency.payload_fingerprint
                    != _payload_fingerprint(evidence.payload)
                ):
                    continue
                return True
        return False

    async def _approval_dependency_satisfied(
        self,
        dependency: TaskDependency,
        operation: Operation,
    ) -> bool:
        operation_id = dependency.operation_id or operation.id
        for approval in await self.store.list_approval_requests(operation_id):
            if (
                dependency.approval_id is not None
                and approval.approval_id != dependency.approval_id
            ):
                continue
            if (
                dependency.approval_policy_id is not None
                and approval.requested_by_policy_id != dependency.approval_policy_id
            ):
                continue
            if (
                dependency.approval_name is not None
                and approval.proposed_action.get("approval") != dependency.approval_name
            ):
                continue
            if (
                dependency.approval_version is not None
                and approval.metadata.get("version") != dependency.approval_version
            ):
                continue
            if approval.status is dependency.approval_status:
                return True
        return False

    async def _recover_expired_claim_if_needed(self, task_id: str) -> None:
        task = await self.store.load_task(task_id)
        if task is None or task.status is not TaskStatus.RUNNING:
            return
        lease_expires_at = task.metadata.get("lease_expires_at")
        if lease_expires_at is None or float(lease_expires_at) > time.time():
            return
        capability = self._capability_for_task(task)
        if (
            capability.idempotent
            or capability.replay_safe
            or not capability.side_effecting
        ):
            await self.store.save_task(
                replace(
                    task,
                    status=TaskStatus.PENDING,
                    metadata={
                        **_metadata_without_active_lease(task.metadata),
                        "expired_lease_recovered": True,
                    },
                )
            )
            return
        await self.store.save_task(
            replace(
                task,
                status=TaskStatus.BLOCKED,
                metadata={
                    **_metadata_without_active_lease(task.metadata),
                    "manual_recovery_required": True,
                    "manual_recovery_reason": "expired_side_effecting_lease",
                },
            )
        )

    async def _result_for_task(self, task: Task) -> TaskExecutionResult | None:
        operation = await self.store.load_operation(task.operation_id)
        if operation is None:
            return None
        capability = self._capability_for_task(task)
        return TaskExecutionResult(operation, task, capability)

    def _capability_for_task(self, task: Task) -> Capability:
        owner = task.metadata.get("owner") if task.metadata else None
        if owner:
            return self.extension_registry.get_capability(
                task.capability_id, owner=str(owner)
            )
        try:
            return self.extension_registry.get_capability(task.capability_id)
        except ValueError:
            for capability in self.extension_registry.capabilities:
                if (
                    capability.id == task.capability_id
                    and capability.executor == task.executor_id
                ):
                    return capability
            raise

    def _event(
        self,
        type: RuntimeEventType,
        *,
        operation: Operation,
        message: str,
        task: Task | None = None,
        capability: Capability | None = None,
        payload: Mapping[str, Any] | None = None,
        policy_id: str | None = None,
        approval_id: str | None = None,
        evidence_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> RuntimeEvent:
        current_trace_id, current_span_id = _current_trace_ids()
        return RuntimeEvent(
            type=type,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            operation_id=operation.id,
            task_id=task.id if task is not None else None,
            capability_id=capability.id if capability is not None else None,
            executor_id=(
                capability.executor
                if capability is not None
                else task.executor_id if task is not None else None
            ),
            plugin_id=capability.owner if capability is not None else None,
            policy_id=policy_id,
            approval_id=approval_id,
            evidence_id=evidence_id,
            trace_id=trace_id or current_trace_id,
            span_id=span_id or current_span_id,
            message=message,
            payload=dict(payload or {}),
        )

    def _correlated_event(
        self,
        event: RuntimeEvent,
        *,
        task: Task | None,
        capability: Capability | None,
    ) -> RuntimeEvent:
        return replace(
            event,
            runtime_id=event.runtime_id or self.runtime_id,
            runtime_kind=event.runtime_kind or self.runtime_kind,
            task_id=event.task_id or (task.id if task is not None else None),
            capability_id=event.capability_id
            or (capability.id if capability is not None else None),
            executor_id=event.executor_id
            or (
                capability.executor
                if capability is not None
                else task.executor_id if task is not None else None
            ),
            plugin_id=event.plugin_id
            or (capability.owner if capability is not None else None),
            trace_id=event.trace_id or _current_trace_ids()[0],
            span_id=event.span_id or _current_trace_ids()[1],
        )

    def _governance_events(
        self,
        operation: Operation,
        governance: GovernanceResult,
        task: Task | None = None,
        capability: Capability | None = None,
    ) -> tuple[RuntimeEvent, ...]:
        events: list[RuntimeEvent] = []
        for decision in governance.decisions:
            events.append(
                self._event(
                    RuntimeEventType.POLICY_DECISION,
                    operation=operation,
                    task=task,
                    capability=capability,
                    message=(
                        f"Policy {decision.owner}:{decision.policy_id} returned "
                        f"{decision.effect.value}."
                    ),
                    payload={"decision": decision.to_dict()},
                    policy_id=decision.policy_id,
                )
            )
        for approval in governance.approval_requests:
            events.append(
                self._event(
                    RuntimeEventType.APPROVAL_REQUESTED,
                    operation=operation,
                    task=task,
                    capability=capability,
                    message=f"Approval {approval.approval_id} requested.",
                    payload={"approval": approval.to_dict()},
                    policy_id=approval.requested_by_policy_id,
                    approval_id=approval.approval_id,
                )
            )
        return tuple(events)

    def _governance_audit_record(
        self,
        operation: Operation,
        governance: GovernanceResult,
        *,
        stage: Literal["operation", "task"],
        facts: Mapping[str, Any],
        task: Task | None,
        capability: Capability | None,
    ) -> GovernanceAuditRecord:
        audit_id = f"governance-audit-{uuid4()}"
        traces = tuple(
            PolicyDecisionTrace(
                trace_id=f"{audit_id}:decision:{index}",
                operation_id=operation.id,
                policy_id=decision.policy_id,
                owner=decision.owner,
                policy_version=decision.policy_version,
                policy_identity=str(decision.policy_identity),
                effect=decision.effect,
                reason=decision.reason,
                stage=stage,
                task_id=task.id if task is not None else None,
                capability_id=capability.id if capability is not None else None,
                runtime_facts=dict(facts),
            )
            for index, decision in enumerate(governance.decisions, start=1)
        )
        return GovernanceAuditRecord(
            audit_id=audit_id,
            operation_id=operation.id,
            stage=stage,
            allowed=governance.allowed,
            blocked=governance.blocked,
            pending_approval=governance.pending_approval,
            policy_decisions=governance.decisions,
            traces=traces,
            task_id=task.id if task is not None else None,
            capability_id=capability.id if capability is not None else None,
            runtime_facts={
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "stage": stage,
                "operation_type": operation.operation_type,
                "facts": dict(facts),
            },
            metadata={"governance_metadata": governance.metadata},
        )

    def _accepted_task_evidence(
        self,
        evidence: Evidence,
        *,
        operation: Operation,
        task: Task,
    ) -> Evidence:
        evidence_identity = {
            "operation_id": operation.id,
            "task_id": task.id,
            "kind": evidence.kind,
            "payload": evidence.payload,
        }
        evidence_id = evidence.id or f"evidence-{_stable_hash(evidence_identity)}"
        return replace(
            evidence,
            id=evidence_id,
            operation_id=evidence.operation_id or operation.id,
            task_id=evidence.task_id or task.id,
            metadata={
                **evidence.metadata,
                **_evidence_trace_metadata(task.metadata),
                "payload_fingerprint": _payload_fingerprint(evidence.payload),
                "task_input_hash": task.metadata.get("input_hash"),
            },
        )


def _evidence_trace_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    trace_metadata = metadata.get("evidence_trace_metadata")
    if isinstance(trace_metadata, Mapping):
        return {str(key): value for key, value in trace_metadata.items()}
    trace_keys = metadata.get("evidence_trace_keys")
    if not isinstance(trace_keys, (list, tuple, set)):
        return {}
    return {
        str(key): metadata[key]
        for key in trace_keys
        if key in metadata and metadata[key] is not None
    }


def _metadata_without_active_lease(metadata: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(metadata)
    for key in (
        "lease_id",
        "lease_owner",
        "lease_expires_at",
        "worker_id",
        "worker_owner",
    ):
        copied.pop(key, None)
    return copied


def _payload_contains(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return True


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    return _stable_hash(dict(payload))


def _current_trace_ids() -> tuple[str | None, str | None]:
    try:
        from daita.core.tracing import get_trace_manager

        context = get_trace_manager().trace_context
        return context.current_trace_id, context.current_span_id
    except Exception:
        return None, None


def _evidence_event_payload(evidence: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.id,
        "kind": evidence.kind,
        "owner": evidence.owner,
        "operation_id": evidence.operation_id,
        "task_id": evidence.task_id,
        "accepted": evidence.accepted,
        "payload_fingerprint": evidence.metadata.get("payload_fingerprint"),
        "payload_size": len(json.dumps(evidence.payload, default=str)),
    }
