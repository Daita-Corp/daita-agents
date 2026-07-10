"""Composed task execution for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from daita.runtime import (
    Capability,
    Evidence,
    GovernanceResult,
    Operation,
    RuntimeKernelExecutorFailed,
    RuntimeKernelGovernanceBlocked,
    RuntimeKernelLeaseLost,
    RuntimeKernelTaskAlreadyTerminal,
    RuntimeKernelTaskNotRunnable,
    Task,
    TaskStatus,
)

from .dependencies import _task_dependencies_for_capability
from .context import DbTaskContext
from .planning import (
    _direct_capability_tasks,
    _prompt_from_direct_input,
    _validation_capability_for_sql_execute,
)
from ..types import (
    _DEFAULT_TASK_LEASE_SECONDS,
    DbRuntimeGovernanceBlocked,
    DbRuntimeTaskNotRunnable,
)


class DbTaskExecutor:
    """Prepare DB tasks and delegate their execution to ``RuntimeKernel``."""

    def __init__(self, context: DbTaskContext) -> None:
        self.context = context

    async def execute_task(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        """Prepare one interactive DB task and delegate it to the kernel."""
        capability = self.capability_for_task(task)
        if capability.executor != task.executor_id:
            raise ValueError(
                f"task executor {task.executor_id!r} does not match capability "
                f"{task.capability_id!r} executor {capability.executor!r}"
            )
        stored_task = await self.context.store.load_task(task.id)
        if stored_task is None:
            task = replace(
                task,
                dependencies=task.dependencies
                or _task_dependencies_for_capability(operation, capability),
            )
            task = await self.plan_kernel_task(task)
        elif (
            stored_task.status is TaskStatus.PENDING and stored_task.input != task.input
        ):
            task = replace(
                stored_task,
                input=task.input,
                dependencies=task.dependencies or stored_task.dependencies,
                metadata={
                    **stored_task.metadata,
                    **{
                        key: value
                        for key, value in task.metadata.items()
                        if key in {"owner", "reason"}
                    },
                },
            )
            await self.context.store.save_task(task)
        else:
            task = stored_task
        default_dependencies = _task_dependencies_for_capability(operation, capability)
        if not task.dependencies and default_dependencies:
            task = replace(task, dependencies=default_dependencies)
            await self.context.store.save_task(task)
        try:
            result = await self.context.kernel.execute_task(
                task.id,
                context={
                    "capability_owner": capability.owner,
                    **(context or {}),
                },
                lease_owner=self.context.runtime_id,
                lease_seconds=_DEFAULT_TASK_LEASE_SECONDS,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            result = exc.result
            raise DbRuntimeGovernanceBlocked(
                operation=result.operation if result is not None else operation,
                task=result.task if result is not None else task,
                governance=(
                    result.governance
                    if result is not None and result.governance is not None
                    else GovernanceResult(False, True, False)
                ),
            ) from exc
        except RuntimeKernelTaskAlreadyTerminal as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                f"Task {blocked_task.id} is already {blocked_task.status.value}; "
                "completed tasks are not replayed without explicit invalidation.",
            ) from exc
        except RuntimeKernelLeaseLost as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                f"Task {blocked_task.id} lease was lost before commit.",
            ) from exc
        except RuntimeKernelTaskNotRunnable as exc:
            result = exc.result
            blocked_task = result.task if result is not None else task
            readiness = (
                result.events[-1].payload.get("readiness", {})
                if result is not None and result.events
                else {}
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                str(exc),
                readiness=readiness,
            ) from exc
        except RuntimeKernelExecutorFailed as exc:
            raise (exc.__cause__ or exc) from exc
        return tuple(result.evidence)

    async def execute_capability(
        self,
        capability_id: str,
        *,
        owner: str | None = None,
        operation_type: str,
        input: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> tuple[Evidence, ...]:
        """Create and execute a single task for one registered capability."""
        capability = self.context.registry.get_capability(capability_id, owner=owner)
        output_evidence = capability.output_evidence
        validation_capability = _validation_capability_for_sql_execute(
            self.context,
            capability,
        )
        if (
            capability.id
            in {
                "db.sql.execute_read",
                "db.sql.execute_write",
            }
            and validation_capability is not None
        ):
            output_evidence = frozenset(
                (
                    *sorted(validation_capability.output_evidence),
                    *sorted(output_evidence),
                )
            )
        try:
            operation = await self.context.kernel.create_operation(
                operation_id=operation_id,
                operation_type=operation_type,
                request={
                    "prompt": _prompt_from_direct_input(input or {}),
                    "input": input or {},
                    "capability_id": capability.id,
                    "capability_owner": capability.owner,
                },
                required_evidence=output_evidence,
                metadata={
                    "direct_capability_id": capability.id,
                    "direct_capability_owner": capability.owner,
                    "access": capability.access.value,
                },
                evaluate_governance=False,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            raise DbRuntimeGovernanceBlocked(
                operation=exc.operation
                or await self.context.store.load_operation(operation_id or ""),
                task=None,
                governance=exc.governance or GovernanceResult(False, True, False),
            ) from exc
        task_plans = _direct_capability_tasks(
            operation,
            capability,
            input or {},
            validation_capability=validation_capability,
        )
        tasks = []
        for task in task_plans:
            tasks.append(await self.plan_kernel_task(task))
        primary_task = tasks[-1]
        if (
            capability.id == "db.sql.execute_write"
            and validation_capability is None
            and not (input or {}).get("validated_evidence_id")
        ):
            blocked_task = await self.context.kernel.block_task(
                primary_task.id,
                message=(
                    "Direct write execution requires db.sql.validate "
                    "or a validated_evidence_id."
                ),
            )
            await self.context.kernel.block_operation(
                operation.id,
                message=(
                    "Direct write execution requires db.sql.validate "
                    "or a validated_evidence_id."
                ),
            )
            raise DbRuntimeTaskNotRunnable(
                blocked_task,
                "Direct write execution requires db.sql.validate or validated_evidence_id.",
            )
        try:
            await self.context.kernel.evaluate_operation_governance(
                operation.id,
                capability=capability,
            )
        except RuntimeKernelGovernanceBlocked as exc:
            await self.context.kernel.block_task(
                primary_task.id,
                message=f"Task {primary_task.id} blocked by operation governance.",
                payload={
                    "governance": (
                        exc.governance.to_dict() if exc.governance is not None else {}
                    )
                },
            )
            raise DbRuntimeGovernanceBlocked(
                operation=exc.operation or operation,
                task=replace(primary_task, status=TaskStatus.BLOCKED),
                governance=exc.governance or GovernanceResult(False, True, False),
            ) from exc
        try:
            collected: list[Evidence] = []
            for task in tasks:
                collected.extend(await self.execute_task(task, operation))
            evidence = tuple(collected)
        except DbRuntimeGovernanceBlocked:
            raise
        except Exception as exc:
            await self.context.kernel.fail_operation_if_active(operation.id, exc)
            raise
        await self.context.kernel.complete_operation(operation.id)
        return evidence

    async def plan_kernel_task(self, task: Task) -> Task:
        """Persist a DB-planned task through the shared kernel planner."""
        return await self.context.kernel.plan_task(
            task_id=task.id,
            operation_id=task.operation_id,
            capability_id=task.capability_id,
            owner=str(task.metadata["owner"]) if task.metadata.get("owner") else None,
            input=task.input,
            metadata=task.metadata,
            dependencies=task.dependencies,
        )

    def capability_for_task(self, task: Task) -> Capability:
        owner = task.metadata.get("owner") if task.metadata else None
        if owner:
            return self.context.registry.get_capability(
                task.capability_id,
                owner=str(owner),
            )
        try:
            return self.context.registry.get_capability(task.capability_id)
        except ValueError:
            for capability in self.context.registry.capabilities:
                if (
                    capability.id == task.capability_id
                    and capability.executor == task.executor_id
                ):
                    return capability
            raise


class DbRuntimeTaskExecutionMixin:
    """Temporary ``DbRuntime`` delegates for the representative conversion."""

    if TYPE_CHECKING:
        _is_setup: bool
        _tasks: DbTaskExecutor

        async def setup(self, *, agent_id: str | None = None) -> None: ...

    async def execute_task(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        return await self._tasks.execute_task(task, operation, context)

    async def execute_capability(
        self,
        capability_id: str,
        *,
        owner: str | None = None,
        operation_type: str,
        input: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> tuple[Evidence, ...]:
        if not self._is_setup:
            await self.setup()
        return await self._tasks.execute_capability(
            capability_id,
            owner=owner,
            operation_type=operation_type,
            input=input,
            operation_id=operation_id,
        )

    async def _plan_kernel_task(self, task: Task) -> Task:
        return await self._tasks.plan_kernel_task(task)

    def _capability_for_task(self, task: Task) -> Capability:
        return self._tasks.capability_for_task(task)
