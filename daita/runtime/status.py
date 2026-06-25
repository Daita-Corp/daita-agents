"""Runtime-owned operation status reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .primitives import OperationStatus, TaskStatus
from .store import OperationSnapshot

TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }
)


@dataclass(frozen=True)
class OperationStatusReconciliation:
    """Result of reconciling operation status from persisted task state."""

    snapshot: OperationSnapshot
    complete: bool = False
    blocked: bool = False
    failed: bool = False
    cancelled: bool = False
    pending: bool = False


async def reconcile_operation_status(
    kernel: Any,
    operation_id: str,
) -> OperationStatusReconciliation | None:
    """Refresh an operation's status from its persisted tasks."""
    store = kernel.store
    inspect = getattr(store, "inspect_operation", None)
    if inspect is None:
        return None
    snapshot = await inspect(operation_id)
    if snapshot is None or not snapshot.tasks:
        return None

    tasks = snapshot.tasks
    failed = any(task.status is TaskStatus.FAILED for task in tasks)
    cancelled = any(task.status is TaskStatus.CANCELLED for task in tasks)
    pending = any(
        task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} for task in tasks
    )
    blocked = any(task.status is TaskStatus.BLOCKED for task in tasks)
    complete = all(task.status in TERMINAL_TASK_STATUSES for task in tasks)

    status = snapshot.operation.status
    if failed:
        status = OperationStatus.FAILED
    elif cancelled:
        status = OperationStatus.CANCELLED
    elif blocked:
        status = OperationStatus.BLOCKED
    elif complete:
        status = OperationStatus.SUCCEEDED
    elif pending:
        status = OperationStatus.RUNNING

    if status is not snapshot.operation.status:
        await store.save_operation(replace(snapshot.operation, status=status))
        snapshot = await inspect(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)

    return OperationStatusReconciliation(
        snapshot=snapshot,
        complete=complete and not failed and not cancelled,
        blocked=blocked,
        failed=failed,
        cancelled=cancelled,
        pending=pending,
    )
