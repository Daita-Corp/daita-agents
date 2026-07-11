"""Bounded task-batch orchestration for the model-planned DB loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    Operation,
    Task,
    TaskDependencyKind,
    TaskStatus,
)

from ..runtime.types import DbRuntimeGovernanceBlocked, DbRuntimeTaskNotRunnable

_CONCURRENT_READ_ACCESS = frozenset(
    {AccessMode.NONE, AccessMode.METADATA_READ, AccessMode.READ}
)
_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }
)


@dataclass(frozen=True)
class DbLoopTaskOutcome:
    """One loop task result associated with its original plan position."""

    task_index: int
    task: Task
    evidence: tuple[Evidence, ...] = ()
    governance_error: DbRuntimeGovernanceBlocked | None = None
    readiness_error: DbRuntimeTaskNotRunnable | None = None
    error: Exception | None = None


class DbLoopTaskBatchExecutor:
    """Execute one task-plan batch through the existing DB task facade."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def execute(
        self,
        tasks: tuple[Task, ...],
        operation: Operation,
    ) -> tuple[DbLoopTaskOutcome, ...]:
        indexed = tuple(
            (index, task)
            for index, task in enumerate(tasks)
            if task.status not in _TERMINAL_TASK_STATUSES
        )
        if not indexed:
            return ()
        max_concurrency = self.runtime.config.execution.max_read_concurrency
        if max_concurrency == 1 or len(indexed) < 2:
            return await self._execute_serial(indexed, operation)
        if self._contains_serial_only_work(indexed):
            return await self._execute_serial(indexed, operation)
        outcomes: list[DbLoopTaskOutcome] = []
        remaining = list(indexed)
        while remaining:
            try:
                ready_prefix: list[tuple[int, Task]] = []
                for index, task in remaining:
                    readiness = await self.runtime.task_readiness(task, operation)
                    if not readiness.get("ready"):
                        break
                    ready_prefix.append((index, task))
            except Exception:
                ready_prefix = []
            ready = tuple(ready_prefix)
            if len(ready) > 1 and await self._concurrent_batch_eligible(
                ready,
                operation,
            ):
                wave = await self._execute_concurrent(
                    ready,
                    operation,
                    max_concurrency=max_concurrency,
                )
            else:
                wave = await self._execute_serial((remaining[0],), operation)
            outcomes.extend(wave)
            executed_ids = {outcome.task.id for outcome in wave}
            remaining = [item for item in remaining if item[1].id not in executed_ids]
            if any(outcome.governance_error is not None for outcome in wave):
                break
        return tuple(sorted(outcomes, key=lambda item: item.task_index))

    async def _execute_concurrent(
        self,
        indexed: tuple[tuple[int, Task], ...],
        operation: Operation,
        *,
        max_concurrency: int,
    ) -> tuple[DbLoopTaskOutcome, ...]:
        outcomes: list[DbLoopTaskOutcome] = []
        for offset in range(0, len(indexed), max_concurrency):
            chunk = indexed[offset : offset + max_concurrency]
            async with asyncio.TaskGroup() as group:
                children = [
                    group.create_task(self._execute_one(index, task, operation))
                    for index, task in chunk
                ]
            outcomes.extend(child.result() for child in children)
        return tuple(outcomes)

    async def _execute_serial(
        self,
        indexed: tuple[tuple[int, Task], ...],
        operation: Operation,
    ) -> tuple[DbLoopTaskOutcome, ...]:
        outcomes: list[DbLoopTaskOutcome] = []
        for index, task in indexed:
            outcome = await self._execute_one(index, task, operation)
            outcomes.append(outcome)
            if outcome.governance_error is not None:
                break
        return tuple(outcomes)

    async def _concurrent_batch_eligible(
        self,
        indexed: tuple[tuple[int, Task], ...],
        operation: Operation,
    ) -> bool:
        try:
            capabilities: list[Capability] = []
            selected_task_ids = {task.id for _, task in indexed}
            for _, task in indexed:
                if task.status is not TaskStatus.PENDING:
                    return False
                capability = self._capability_for_task(task)
                if (
                    not capability.concurrent_safe
                    or capability.side_effecting
                    or capability.access not in _CONCURRENT_READ_ACCESS
                    or any(
                        dependency.kind is TaskDependencyKind.APPROVAL
                        or dependency.producer_task_id in selected_task_ids
                        for dependency in task.dependencies
                    )
                ):
                    return False
                capabilities.append(capability)
            for (_, task), capability in zip(indexed, capabilities):
                governance = await self.runtime.evaluate_governance_persistence(
                    operation,
                    task=task,
                    capability=capability,
                    stage="task",
                )
                if governance.result.blocked or governance.result.pending_approval:
                    return False
        except Exception:
            return False
        return True

    def _contains_serial_only_work(
        self,
        indexed: tuple[tuple[int, Task], ...],
    ) -> bool:
        try:
            return any(
                capability.side_effecting
                or capability.access not in _CONCURRENT_READ_ACCESS
                for _, task in indexed
                for capability in (self._capability_for_task(task),)
            )
        except Exception:
            return True

    def _capability_for_task(self, task: Task) -> Capability:
        owner = task.metadata.get("owner") if task.metadata else None
        if owner:
            return self.runtime.registry.get_capability(
                task.capability_id, owner=str(owner)
            )
        try:
            return self.runtime.registry.get_capability(task.capability_id)
        except ValueError:
            for capability in self.runtime.registry.capabilities:
                if (
                    capability.id == task.capability_id
                    and capability.executor == task.executor_id
                ):
                    return capability
            raise

    async def _execute_one(
        self,
        index: int,
        task: Task,
        operation: Operation,
    ) -> DbLoopTaskOutcome:
        try:
            evidence = await self.runtime.execute_task(task, operation)
        except DbRuntimeGovernanceBlocked as exc:
            return DbLoopTaskOutcome(index, task, governance_error=exc)
        except DbRuntimeTaskNotRunnable as exc:
            return DbLoopTaskOutcome(index, task, readiness_error=exc)
        except Exception as exc:
            return DbLoopTaskOutcome(index, task, error=exc)
        return DbLoopTaskOutcome(index, task, evidence=tuple(evidence))
