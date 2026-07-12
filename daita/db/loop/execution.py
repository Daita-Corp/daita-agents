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
        concurrency_disabled = False
        while remaining:
            cohort: tuple[tuple[int, Task], ...]
            preparations: tuple[tuple[int, Task], ...]
            if concurrency_disabled:
                cohort, preparations = (), ()
            else:
                cohort, preparations = self._concurrent_read_cohort(tuple(remaining))
            preparation = next(iter(preparations), None)
            if preparation is not None and await self._task_ready(
                preparation[1], operation
            ):
                preparing = True
                wave = await self._execute_serial((preparation,), operation)
            elif (
                len(cohort) > 1
                and not preparations
                and await self._tasks_ready(cohort, operation)
                and await self._concurrent_batch_eligible(cohort, operation)
            ):
                preparing = False
                wave = await self._execute_concurrent(
                    cohort,
                    operation,
                    max_concurrency=max_concurrency,
                )
            else:
                preparing = False
                wave = await self._execute_serial((remaining[0],), operation)
            outcomes.extend(wave)
            if preparing and any(
                outcome.error is not None or outcome.readiness_error is not None
                for outcome in wave
            ):
                concurrency_disabled = True
            executed_ids = {outcome.task.id for outcome in wave}
            remaining = [item for item in remaining if item[1].id not in executed_ids]
            if any(outcome.governance_error is not None for outcome in wave):
                break
        return tuple(sorted(outcomes, key=lambda item: item.task_index))

    def _concurrent_read_cohort(
        self,
        indexed: tuple[tuple[int, Task], ...],
    ) -> tuple[
        tuple[tuple[int, Task], ...],
        tuple[tuple[int, Task], ...],
    ]:
        """Return a leading read cohort and its safe pending prerequisites."""
        try:
            task_by_id = {task.id: task for _, task in indexed}
            candidates = tuple(
                item for item in indexed if self._concurrent_candidate(item[1])
            )
            if len(candidates) < 2:
                return (), ()

            candidate_ids = {task.id for _, task in candidates}
            ancestors = {
                task.id: self._task_ancestor_ids(task, task_by_id)
                for _, task in candidates
            }
            if any(ancestor_ids & candidate_ids for ancestor_ids in ancestors.values()):
                return (), ()

            prerequisite_ids = set().union(*ancestors.values())
            if any(
                not self._safe_preparation_task(task_by_id[task_id])
                for task_id in prerequisite_ids
            ):
                return (), ()

            allowed_ids = candidate_ids | prerequisite_ids
            leading: list[tuple[int, Task]] = []
            for item in indexed:
                if item[1].id not in allowed_ids:
                    break
                leading.append(item)

            leading_candidate_ids = {
                task.id for _, task in leading if task.id in candidate_ids
            }
            if len(leading_candidate_ids) < 2:
                return (), ()
            if any(
                ancestors[task_id] & leading_candidate_ids
                for task_id in leading_candidate_ids
            ):
                return (), ()

            leading_prerequisite_ids = set().union(
                *(ancestors[task_id] for task_id in leading_candidate_ids)
            )
            selected_ids = leading_candidate_ids | leading_prerequisite_ids
            last_candidate_position = max(
                position
                for position, (_, task) in enumerate(leading)
                if task.id in leading_candidate_ids
            )
            if any(
                task.id not in selected_ids
                for _, task in leading[: last_candidate_position + 1]
            ):
                return (), ()
            cohort = tuple(
                item for item in leading if item[1].id in leading_candidate_ids
            )
            preparations = tuple(
                item for item in leading if item[1].id in leading_prerequisite_ids
            )
            return cohort, preparations
        except Exception:
            return (), ()

    def _concurrent_candidate(self, task: Task) -> bool:
        capability = self._capability_for_task(task)
        return (
            task.status is TaskStatus.PENDING
            and capability.concurrent_safe
            and not capability.side_effecting
            and capability.access in _CONCURRENT_READ_ACCESS
            and not any(
                dependency.kind is TaskDependencyKind.APPROVAL
                for dependency in task.dependencies
            )
        )

    def _safe_preparation_task(self, task: Task) -> bool:
        capability = self._capability_for_task(task)
        return (
            task.status is TaskStatus.PENDING
            and not capability.side_effecting
            and capability.access in _CONCURRENT_READ_ACCESS
            and not any(
                dependency.kind is TaskDependencyKind.APPROVAL
                for dependency in task.dependencies
            )
        )

    @staticmethod
    def _task_ancestor_ids(
        task: Task,
        task_by_id: dict[str, Task],
    ) -> set[str]:
        ancestors: set[str] = set()
        pending = [
            dependency.producer_task_id
            for dependency in task.dependencies
            if dependency.kind is TaskDependencyKind.EVIDENCE
            and dependency.producer_task_id in task_by_id
        ]
        while pending:
            task_id = pending.pop()
            if task_id is None or task_id in ancestors:
                continue
            ancestors.add(task_id)
            pending.extend(
                dependency.producer_task_id
                for dependency in task_by_id[task_id].dependencies
                if dependency.kind is TaskDependencyKind.EVIDENCE
                and dependency.producer_task_id in task_by_id
            )
        return ancestors

    async def _task_ready(self, task: Task, operation: Operation) -> bool:
        try:
            readiness = await self.runtime.task_readiness(task, operation)
        except Exception:
            return False
        return bool(readiness.get("ready"))

    async def _tasks_ready(
        self,
        indexed: tuple[tuple[int, Task], ...],
        operation: Operation,
    ) -> bool:
        for _, task in indexed:
            if not await self._task_ready(task, operation):
                return False
        return True

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
            chunk_outcomes = await asyncio.gather(
                *(self._execute_one(index, task, operation) for index, task in chunk)
            )
            outcomes.extend(chunk_outcomes)
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
