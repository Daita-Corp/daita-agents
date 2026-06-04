"""Small persisted-task scheduler for runtime operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .kernel import (
    RuntimeKernel,
    RuntimeKernelExecutorFailed,
    RuntimeKernelGovernanceBlocked,
    RuntimeKernelTaskAlreadyTerminal,
    RuntimeKernelTaskNotRunnable,
    TaskExecutionResult,
)
from .primitives import OperationStatus, RuntimeEventType, TaskStatus
from .store import OperationSnapshot

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }
)


@dataclass(frozen=True)
class OperationScheduleResult:
    """Result from one persisted operation scheduling pass."""

    snapshot: OperationSnapshot
    executed_task_ids: tuple[str, ...] = ()
    skipped_task_ids: tuple[str, ...] = ()
    blocked_task_ids: tuple[str, ...] = ()
    complete: bool = False
    blocked: bool = False
    failed: bool = False
    cancelled: bool = False
    pending: bool = False


class OperationTaskScheduler:
    """Run ready persisted tasks through the runtime kernel."""

    def __init__(self, *, kernel: RuntimeKernel) -> None:
        self.kernel = kernel
        self.store = kernel.store

    async def run_operation(
        self,
        operation_id: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> OperationScheduleResult:
        """Execute ready non-terminal tasks for an operation."""
        inspect = getattr(self.store, "inspect_operation", None)
        if inspect is None:
            raise RuntimeKernelTaskNotRunnable(
                "Runtime store cannot inspect operations."
            )
        snapshot = await inspect(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        executed: list[str] = []
        skipped: list[str] = []
        blocked: list[str] = []
        made_progress = True

        while made_progress:
            made_progress = False
            snapshot = await inspect(operation_id)
            if snapshot is None:
                raise KeyError(operation_id)
            for task in snapshot.tasks:
                if task.status in _TERMINAL_TASK_STATUSES:
                    if task.id not in skipped:
                        skipped.append(task.id)
                        await self.store.append_event(
                            self.kernel._event(
                                RuntimeEventType.TASK_SKIPPED,
                                operation=snapshot.operation,
                                task=task,
                                message=(
                                    f"Task {task.id} already terminal; not re-running."
                                ),
                            )
                        )
                    continue
                if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                    continue
                try:
                    await self.kernel.execute_task(task.id, context=context)
                except RuntimeKernelTaskAlreadyTerminal:
                    continue
                except RuntimeKernelGovernanceBlocked:
                    blocked.append(task.id)
                    return await self._finish(
                        operation_id,
                        executed=executed,
                        skipped=skipped,
                        blocked=blocked,
                    )
                except RuntimeKernelTaskNotRunnable:
                    blocked.append(task.id)
                    continue
                except RuntimeKernelExecutorFailed:
                    executed.append(task.id)
                    return await self._finish(
                        operation_id,
                        executed=executed,
                        skipped=skipped,
                        blocked=blocked,
                    )
                else:
                    executed.append(task.id)
                    made_progress = True

        return await self._finish(
            operation_id,
            executed=executed,
            skipped=skipped,
            blocked=blocked,
        )

    async def _finish(
        self,
        operation_id: str,
        *,
        executed: list[str],
        skipped: list[str],
        blocked: list[str],
    ) -> OperationScheduleResult:
        inspect = getattr(self.store, "inspect_operation")
        snapshot = await inspect(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        tasks = snapshot.tasks
        failed = any(task.status is TaskStatus.FAILED for task in tasks)
        cancelled = any(task.status is TaskStatus.CANCELLED for task in tasks)
        pending = any(
            task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} for task in tasks
        )
        blocked_now = any(task.status is TaskStatus.BLOCKED for task in tasks)
        complete = bool(tasks) and all(
            task.status in _TERMINAL_TASK_STATUSES for task in tasks
        )
        status = snapshot.operation.status
        if failed:
            status = OperationStatus.FAILED
        elif cancelled:
            status = OperationStatus.CANCELLED
        elif blocked_now:
            status = OperationStatus.BLOCKED
        elif complete:
            status = OperationStatus.SUCCEEDED
        elif pending:
            status = OperationStatus.RUNNING
        if status is not snapshot.operation.status:
            await self.store.save_operation(replace(snapshot.operation, status=status))
            snapshot = await inspect(operation_id)
            if snapshot is None:
                raise KeyError(operation_id)
        return OperationScheduleResult(
            snapshot=snapshot,
            executed_task_ids=tuple(dict.fromkeys(executed)),
            skipped_task_ids=tuple(dict.fromkeys(skipped)),
            blocked_task_ids=tuple(dict.fromkeys(blocked)),
            complete=complete and not failed and not cancelled,
            blocked=blocked_now,
            failed=failed,
            cancelled=cancelled,
            pending=pending,
        )
