"""Composed owner for DB task-runtime behavior."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from daita.runtime import Capability, Evidence, Operation, Task, TaskDependency

from ...models import DbIntent, DbOperationContract
from .context import DbTaskContext
from .evidence import (
    accepted_evidence_for_dependency,
    authoritative_validation_evidence,
    latest_accepted_evidence,
    latest_evidence,
    persist_verification_result_evidence,
)
from .execution import DbTaskExecutor
from .inputs import executable_input_for_task
from .models import DbTaskPlan, DbTaskSpec
from .planning import _validation_capability_for_sql_execute, plan_task_specs
from .readiness import task_readiness
from .synthesis import execute_answer_synthesis


class DbTaskRuntime:
    """Compose active task behavior over one explicit dependency context."""

    def __init__(self, context: DbTaskContext) -> None:
        self.context = context
        self.executor = DbTaskExecutor(context)

    async def execute_task(
        self,
        task: Task,
        operation: Operation,
        context: dict[str, Any] | None = None,
    ) -> tuple[Evidence, ...]:
        return await self.executor.execute_task(task, operation, context)

    async def execute_capability(
        self,
        capability_id: str,
        *,
        owner: str | None = None,
        operation_type: str,
        input: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> tuple[Evidence, ...]:
        return await self.executor.execute_capability(
            capability_id,
            owner=owner,
            operation_type=operation_type,
            input=input,
            operation_id=operation_id,
        )

    async def plan_task_specs(
        self,
        operation: Operation,
        specs: Iterable[DbTaskSpec],
        *,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
    ) -> DbTaskPlan:
        return await plan_task_specs(
            self.context,
            operation,
            specs,
            contract=contract,
        )

    async def task_readiness(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        return await task_readiness(self.context, task, operation)

    async def executable_input_for_task(
        self,
        task: Task,
        operation: Operation,
    ) -> dict[str, Any]:
        return await executable_input_for_task(self.context, task, operation)

    async def persist_verification_result_evidence(
        self,
        operation: Operation,
        verification: Any,
        evidence: tuple[Evidence, ...],
    ) -> Evidence:
        return await persist_verification_result_evidence(
            self.context,
            operation,
            verification,
            evidence,
        )

    async def accepted_evidence_for_dependency(
        self,
        operation_id: str,
        dependency: TaskDependency,
    ) -> Evidence | None:
        return await accepted_evidence_for_dependency(
            self.context,
            operation_id,
            dependency,
        )

    async def authoritative_validation_evidence(
        self,
        operation: Operation,
        task: Task | None,
    ) -> tuple[Evidence, ...]:
        return await authoritative_validation_evidence(self.context, operation, task)

    async def latest_accepted_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Evidence | None:
        return await latest_accepted_evidence(
            self.context,
            operation_id,
            kind,
            payload=payload,
        )

    async def latest_evidence(
        self,
        operation_id: str,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        accepted: bool | None = None,
    ) -> Evidence | None:
        return await latest_evidence(
            self.context,
            operation_id,
            kind,
            payload=payload,
            accepted=accepted,
        )

    async def execute_answer_synthesis(
        self,
        *,
        operation: Operation,
        intent: DbIntent,
        outcome_evidence: tuple[Evidence, ...],
    ) -> tuple[Evidence, Task]:
        return await execute_answer_synthesis(
            self.context,
            operation=operation,
            intent=intent,
            outcome_evidence=outcome_evidence,
        )

    def validation_capability_for_sql_execute(
        self,
        capability: Capability,
    ) -> Capability | None:
        return _validation_capability_for_sql_execute(self.context, capability)

    def capability_for_task(self, task: Task) -> Capability:
        return self.executor.capability_for_task(task)
