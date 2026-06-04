"""Default runtime operation store implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import json
import time
from typing import Any

from .primitives import (
    ApprovalRequest,
    Evidence,
    GovernanceAuditRecord,
    Operation,
    OperationStatus,
    PolicyDecision,
    RuntimeEvent,
    RuntimeStore,
    Task,
    TaskStatus,
)


@dataclass(frozen=True)
class OperationSnapshot:
    """Inspectable persisted state for one runtime operation."""

    operation: Operation
    tasks: tuple[Task, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    events: tuple[RuntimeEvent, ...] = ()
    policy_decisions: tuple[PolicyDecision, ...] = ()
    governance_audit_records: tuple[GovernanceAuditRecord, ...] = ()
    approval_requests: tuple[ApprovalRequest, ...] = ()
    resumable_task_ids: tuple[str, ...] = ()
    completed_task_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "policy_decisions", tuple(self.policy_decisions))
        object.__setattr__(
            self, "governance_audit_records", tuple(self.governance_audit_records)
        )
        object.__setattr__(self, "approval_requests", tuple(self.approval_requests))
        object.__setattr__(self, "resumable_task_ids", tuple(self.resumable_task_ids))
        object.__setattr__(self, "completed_task_ids", tuple(self.completed_task_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "evidence": [item.to_dict() for item in self.evidence],
            "events": [event.to_dict() for event in self.events],
            "policy_decisions": [
                decision.to_dict() for decision in self.policy_decisions
            ],
            "governance_audit_records": [
                record.to_dict() for record in self.governance_audit_records
            ],
            "approval_requests": [
                request.to_dict() for request in self.approval_requests
            ],
            "resumable_task_ids": list(self.resumable_task_ids),
            "completed_task_ids": list(self.completed_task_ids),
            "metadata": dict(self.metadata),
        }


class InMemoryRuntimeStore(RuntimeStore):
    """Transient default store for operations, tasks, evidence, and approvals."""

    def __init__(self) -> None:
        self._operations: dict[str, Operation] = {}
        self._operation_order: list[str] = []
        self._tasks: dict[str, Task] = {}
        self._task_order: list[str] = []
        self._evidence: list[Evidence] = []
        self._events: list[RuntimeEvent] = []
        self._policy_decisions: list[PolicyDecision] = []
        self._governance_audit_records: list[GovernanceAuditRecord] = []
        self._approval_requests: dict[str, ApprovalRequest] = {}
        self._approval_order: list[str] = []
        self._task_claim_lock = asyncio.Lock()
        self._commit_lock = asyncio.Lock()

    async def save_operation(self, operation: Operation) -> None:
        if operation.id not in self._operations:
            self._operation_order.append(operation.id)
        self._operations[operation.id] = operation

    async def load_operation(self, operation_id: str) -> Operation | None:
        return self._operations.get(operation_id)

    async def list_operations(self) -> list[Operation]:
        return [
            self._operations[operation_id]
            for operation_id in self._operation_order
            if operation_id in self._operations
        ]

    async def save_task(self, task: Task) -> None:
        if task.id not in self._tasks:
            self._task_order.append(task.id)
        self._tasks[task.id] = task

    async def load_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def list_tasks(self, operation_id: str | None = None) -> list[Task]:
        tasks = [
            self._tasks[task_id]
            for task_id in self._task_order
            if task_id in self._tasks
        ]
        if operation_id is None:
            return tasks
        return [task for task in tasks if task.operation_id == operation_id]

    async def claim_task(
        self,
        task_id: str,
        *,
        lease_id: str | None = None,
        lease_owner: str,
        lease_expires_at: float | None = None,
        worker_id: str | None = None,
        worker_owner: str | None = None,
    ) -> Task | None:
        """Atomically claim pending or blocked work before it runs."""
        async with self._task_claim_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                return None
            attempt_count = int(task.metadata.get("attempt_count") or 0) + 1
            lease_metadata = {
                "attempt_count": attempt_count,
                "lease_id": lease_id,
                "lease_owner": lease_owner,
                "lease_expires_at": lease_expires_at,
                "claimed_at": time.time(),
            }
            if worker_id is not None:
                lease_metadata["worker_id"] = worker_id
            if worker_owner is not None:
                lease_metadata["worker_owner"] = worker_owner
            claimed = replace(
                task,
                status=TaskStatus.RUNNING,
                metadata={
                    **task.metadata,
                    **lease_metadata,
                },
            )
            self._tasks[task_id] = claimed
            return claimed

    async def heartbeat_task(
        self,
        task_id: str,
        *,
        lease_id: str,
        lease_expires_at: float,
    ) -> Task | None:
        """Extend a running task lease only when the lease fencing token matches."""
        async with self._task_claim_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status is not TaskStatus.RUNNING:
                return None
            if task.metadata.get("lease_id") != lease_id:
                return None
            updated = replace(
                task,
                metadata={
                    **task.metadata,
                    "lease_expires_at": lease_expires_at,
                    "heartbeat_at": time.time(),
                },
            )
            self._tasks[task_id] = updated
            return updated

    async def commit_task_blocked(
        self,
        *,
        operation: Operation | None,
        task: Task,
        events: tuple[RuntimeEvent, ...],
        lease_id: str | None = None,
    ) -> bool:
        async with self._commit_lock:
            if not self._lease_matches(task.id, lease_id):
                return False
            if operation is not None:
                if operation.id not in self._operations:
                    self._operation_order.append(operation.id)
                self._operations[operation.id] = operation
            if task.id not in self._tasks:
                self._task_order.append(task.id)
            self._tasks[task.id] = task
            self._events.extend(events)
            return True

    async def commit_task_started(self, task: Task, event: RuntimeEvent) -> None:
        async with self._commit_lock:
            if task.id not in self._tasks:
                self._task_order.append(task.id)
            self._tasks[task.id] = task
            self._events.append(event)

    async def commit_task_succeeded(
        self,
        task: Task,
        evidence: tuple[Evidence, ...],
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        async with self._commit_lock:
            if not self._lease_matches(task.id, lease_id):
                return False
            if task.id not in self._tasks:
                self._task_order.append(task.id)
            self._tasks[task.id] = task
            self._evidence.extend(evidence)
            self._events.append(event)
            return True

    async def commit_task_failed(
        self,
        task: Task,
        event: RuntimeEvent,
        *,
        lease_id: str | None = None,
    ) -> bool:
        async with self._commit_lock:
            if not self._lease_matches(task.id, lease_id):
                return False
            if task.id not in self._tasks:
                self._task_order.append(task.id)
            self._tasks[task.id] = task
            self._events.append(event)
            return True

    async def commit_approval_update(
        self,
        request: ApprovalRequest,
        event: RuntimeEvent,
    ) -> None:
        async with self._commit_lock:
            if request.approval_id not in self._approval_requests:
                self._approval_order.append(request.approval_id)
            self._approval_requests[request.approval_id] = request
            self._events.append(event)

    async def save_evidence(self, evidence: Evidence) -> None:
        self._evidence.append(evidence)

    async def append_event(self, event: RuntimeEvent) -> None:
        self._events.append(event)

    async def save_policy_decision(self, decision: PolicyDecision) -> None:
        self._policy_decisions.append(decision)

    async def save_governance_audit_record(self, record: GovernanceAuditRecord) -> None:
        self._governance_audit_records.append(_audit_record_copy(record))

    async def commit_governance_evaluation(
        self,
        *,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        async with self._commit_lock:
            self._policy_decisions.extend(decisions)
            self._governance_audit_records.append(_audit_record_copy(audit_record))
            for request in approval_requests:
                if request.approval_id not in self._approval_requests:
                    self._approval_order.append(request.approval_id)
                self._approval_requests[request.approval_id] = request
            self._events.extend(events)

    async def commit_governance_blocked(
        self,
        *,
        operation: Operation,
        task: Task | None,
        decisions: tuple[PolicyDecision, ...],
        audit_record: GovernanceAuditRecord,
        approval_requests: tuple[ApprovalRequest, ...],
        events: tuple[RuntimeEvent, ...],
    ) -> None:
        async with self._commit_lock:
            if operation.id not in self._operations:
                self._operation_order.append(operation.id)
            self._operations[operation.id] = operation
            if task is not None:
                if task.id not in self._tasks:
                    self._task_order.append(task.id)
                self._tasks[task.id] = task
            self._policy_decisions.extend(decisions)
            self._governance_audit_records.append(_audit_record_copy(audit_record))
            for request in approval_requests:
                if request.approval_id not in self._approval_requests:
                    self._approval_order.append(request.approval_id)
                self._approval_requests[request.approval_id] = request
            self._events.extend(events)

    async def save_approval_request(self, request: ApprovalRequest) -> None:
        if request.approval_id not in self._approval_requests:
            self._approval_order.append(request.approval_id)
        self._approval_requests[request.approval_id] = request

    async def list_evidence(self, operation_id: str) -> list[Evidence]:
        return [item for item in self._evidence if item.operation_id == operation_id]

    async def list_events(self, operation_id: str | None = None) -> list[RuntimeEvent]:
        if operation_id is None:
            return list(self._events)
        return [event for event in self._events if event.operation_id == operation_id]

    async def list_policy_decisions(
        self, operation_id: str | None = None
    ) -> list[PolicyDecision]:
        if operation_id is None:
            return list(self._policy_decisions)
        return [
            decision
            for decision in self._policy_decisions
            if decision.operation_id == operation_id
        ]

    async def list_governance_audit_records(
        self, operation_id: str | None = None
    ) -> list[GovernanceAuditRecord]:
        if operation_id is None:
            return [
                _audit_record_copy(record) for record in self._governance_audit_records
            ]
        return [
            _audit_record_copy(record)
            for record in self._governance_audit_records
            if record.operation_id == operation_id
        ]

    async def list_approval_requests(
        self, operation_id: str | None = None
    ) -> list[ApprovalRequest]:
        requests = [
            self._approval_requests[approval_id]
            for approval_id in self._approval_order
            if approval_id in self._approval_requests
        ]
        if operation_id is None:
            return requests
        return [request for request in requests if request.operation_id == operation_id]

    async def inspect_operation(self, operation_id: str) -> OperationSnapshot | None:
        """Return an inspectable snapshot assembled from persisted state."""
        operation = await self.load_operation(operation_id)
        if operation is None:
            return None
        tasks = tuple(await self.list_tasks(operation_id))
        evidence = tuple(await self.list_evidence(operation_id))
        completed_task_ids = tuple(
            task.id
            for task in tasks
            if task.status
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.SKIPPED,
            }
        )

        resumable_task_ids = tuple(
            task.id
            for task in tasks
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
        )
        return OperationSnapshot(
            operation=operation,
            tasks=tasks,
            evidence=evidence,
            events=tuple(await self.list_events(operation_id)),
            policy_decisions=tuple(await self.list_policy_decisions(operation_id)),
            governance_audit_records=tuple(
                await self.list_governance_audit_records(operation_id)
            ),
            approval_requests=tuple(await self.list_approval_requests(operation_id)),
            resumable_task_ids=resumable_task_ids,
            completed_task_ids=completed_task_ids,
            metadata={
                "operation_status": operation.status.value,
                "resumable": operation.status
                in {OperationStatus.BLOCKED, OperationStatus.RUNNING},
            },
        )

    def _lease_matches(self, task_id: str, lease_id: str | None) -> bool:
        if lease_id is None:
            return True
        current = self._tasks.get(task_id)
        if current is None:
            return False
        return (
            current.status is TaskStatus.RUNNING
            and current.metadata.get("lease_id") == lease_id
        )


def _audit_record_copy(record: GovernanceAuditRecord) -> GovernanceAuditRecord:
    return GovernanceAuditRecord.from_dict(json.loads(json.dumps(record.to_dict())))
