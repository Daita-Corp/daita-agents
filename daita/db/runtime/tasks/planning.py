"""Task planning helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.runtime import Capability, Operation, Task, TaskStatus

from ...models import DbOperationContract
from .common import _json_dict, _stable_hash
from .dependencies import (
    _combine_dependencies,
    _has_sql_validation_dependency,
    _task_dependencies_for_capability,
)
from .models import DbTaskPlan, DbTaskSpec


class DbRuntimeTaskPlanningMixin:
    async def plan_task_specs(
        self,
        operation: Operation,
        specs: Iterable[DbTaskSpec],
        *,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
    ) -> DbTaskPlan:
        """Materialize DB task specs through the shared runtime kernel."""
        planned: list[Task] = []
        diagnostics: dict[str, Any] = {
            "spec_count": 0,
            "reused_task_count": 0,
            "planned_task_count": 0,
        }
        prior_by_capability_owner: dict[tuple[str, str], Task] = {}
        for spec in specs:
            diagnostics["spec_count"] += 1
            capability = self.registry.get_capability(
                spec.capability_id,
                owner=spec.owner,
            )
            validation_task = prior_by_capability_owner.get(
                ("db.sql.validate", capability.owner)
            )
            task = self._task_for_spec(
                operation,
                capability,
                spec,
                contract=contract,
                validation_task=validation_task,
            )
            existing = await self.store.load_task(task.id)
            if existing is not None:
                planned.append(existing)
                diagnostics["reused_task_count"] += 1
            else:
                planned.append(await self._plan_kernel_task(task))
                diagnostics["planned_task_count"] += 1
            prior_by_capability_owner[(capability.id, capability.owner)] = planned[-1]
        return DbTaskPlan(tasks=tuple(planned), diagnostics=diagnostics)

    async def _persist_contract_tasks(
        self,
        operation: Operation,
        contract: DbOperationContract,
    ) -> None:
        existing = {
            (task.capability_id, task.executor_id, task.metadata.get("owner"))
            for task in await self.store.list_tasks(operation.id)
        }
        planned_by_capability: dict[tuple[str, str], Task] = {
            (task.capability_id, str(task.metadata.get("owner") or "")): task
            for task in await self.store.list_tasks(operation.id)
        }
        for sequence, selected in enumerate(
            contract.metadata.get("selected_capabilities", ()), start=1
        ):
            capability = self.registry.get_capability(
                str(selected["id"]),
                owner=str(selected["owner"]),
            )
            key = (capability.id, capability.executor, capability.owner)
            if key in existing:
                continue
            task = Task(
                id=f"db-task-{uuid4()}",
                operation_id=operation.id,
                capability_id=capability.id,
                executor_id=capability.executor,
                input=self._planned_task_input(operation, capability),
                required_evidence=capability.output_evidence,
                metadata={
                    "owner": capability.owner,
                    "reason": str(selected.get("reason") or "contract"),
                    "sequence": sequence,
                },
            )
            validation_task = planned_by_capability.get(
                ("db.sql.validate", capability.owner)
            )
            task = replace(
                task,
                input={
                    **task.input,
                    "input_hash": _stable_hash(task.input),
                },
                dependencies=_task_dependencies_for_capability(
                    operation,
                    capability,
                    validation_task=validation_task,
                ),
                metadata={
                    **task.metadata,
                    "input_hash": _stable_hash(task.input),
                    "idempotency_key": _stable_hash(
                        {
                            "operation_id": operation.id,
                            "task_id": task.id,
                            "capability_id": task.capability_id,
                            "input": task.input,
                        }
                    ),
                    "idempotent": capability.idempotent,
                    "replay_safe": capability.replay_safe,
                    "side_effecting": capability.side_effecting,
                },
            )
            await self._plan_kernel_task(task)
            existing.add(key)
            planned_by_capability[(capability.id, capability.owner)] = task

    async def _planned_task_for_capability(
        self,
        operation_id: str,
        capability: Capability,
        *,
        metadata_match: dict[str, Any] | None = None,
    ) -> Task | None:
        for task in await self.store.list_tasks(operation_id):
            if task.status is not TaskStatus.PENDING:
                continue
            if task.capability_id != capability.id:
                continue
            if task.executor_id != capability.executor:
                continue
            if task.metadata.get("owner") != capability.owner:
                continue
            if metadata_match and any(
                task.metadata.get(key) != value for key, value in metadata_match.items()
            ):
                continue
            return task
        return None

    def _validation_capability_for_sql_execute(
        self,
        capability: Capability,
    ) -> Capability | None:
        if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
            return None
        try:
            return self.registry.get_capability(
                "db.sql.validate", owner=capability.owner
            )
        except KeyError:
            return None

    def _direct_capability_tasks(
        self,
        operation: Operation,
        capability: Capability,
        input: dict[str, Any],
        *,
        validation_capability: Capability | None,
    ) -> tuple[Task, ...]:
        if (
            capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}
            or validation_capability is None
        ):
            task = self._task_for_capability(
                operation,
                capability,
                input=input,
                reason="direct",
                sequence=1,
            )
            return (task,)
        validation_task = self._task_for_capability(
            operation,
            validation_capability,
            input={
                "sql": str(input.get("sql") or ""),
                "operation": (
                    "query"
                    if capability.id == "db.sql.execute_read"
                    else operation.operation_type
                ),
            },
            reason="direct_validation",
            sequence=1,
        )
        execute_input = (
            {
                "sql_ref": "sql.validation",
                "params": list(input.get("params") or []),
                **(
                    {"param_specs": list(input.get("param_specs") or [])}
                    if input.get("param_specs")
                    else {}
                ),
            }
            if capability.id == "db.sql.execute_read"
            else {
                "sql_ref": "sql.validation",
                "params": list(input.get("params") or []),
                **(
                    {"param_specs": list(input.get("param_specs") or [])}
                    if input.get("param_specs")
                    else {}
                ),
            }
        )
        execute_task = self._task_for_capability(
            operation,
            capability,
            input=execute_input,
            reason="direct",
            sequence=2,
            validation_task=validation_task,
        )
        return (validation_task, execute_task)

    def _task_for_capability(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input: dict[str, Any],
        reason: str,
        sequence: int,
        validation_task: Task | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        input_hash = _stable_hash(input)
        task = Task(
            id=f"db-task-{uuid4()}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            metadata={
                **dict(metadata or {}),
                "owner": capability.owner,
                "reason": reason,
                "sequence": sequence,
                "input_hash": input_hash,
                "idempotency_key": _stable_hash(
                    {
                        "operation_id": operation.id,
                        "capability_id": capability.id,
                        "executor_id": capability.executor,
                        "input": input,
                    }
                ),
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        return replace(
            task,
            dependencies=_task_dependencies_for_capability(
                operation,
                capability,
                validation_task=validation_task,
            ),
        )

    def _task_for_spec(
        self,
        operation: Operation,
        capability: Capability,
        spec: DbTaskSpec,
        *,
        contract: DbOperationContract | Mapping[str, Any] | None = None,
        validation_task: Task | None = None,
    ) -> Task:
        input_hash = _stable_hash(spec.input)
        idempotency_key = spec.idempotency_key or _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "owner": capability.owner,
                "input_hash": input_hash,
                "sequence": spec.sequence,
                "deterministic_key": spec.deterministic_key,
            }
        )
        task_fingerprint = _stable_hash(
            {
                "operation_id": operation.id,
                "idempotency_key": idempotency_key,
            }
        )
        task_id = spec.task_id or f"db-task-{task_fingerprint[:32]}"
        default_dependencies = _task_dependencies_for_capability(
            operation,
            capability,
            validation_task=validation_task,
        )
        if _has_sql_validation_dependency(spec.dependencies):
            default_dependencies = tuple(
                dependency
                for dependency in default_dependencies
                if not (
                    dependency.kind.value == "evidence"
                    and dependency.evidence_kind == "sql.validation"
                )
            )
        dependencies = _combine_dependencies(default_dependencies, spec.dependencies)
        metadata = {
            **spec.metadata,
            "owner": capability.owner,
            "reason": spec.reason,
            "sequence": spec.sequence,
            "input_hash": input_hash,
            "idempotency_key": idempotency_key,
            "deterministic_key": spec.deterministic_key,
            "idempotent": capability.idempotent,
            "replay_safe": capability.replay_safe,
            "side_effecting": capability.side_effecting,
        }
        contract_snapshot = self._contract_snapshot(contract)
        if contract_snapshot is not None:
            metadata["contract"] = contract_snapshot
        return Task(
            id=task_id,
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**spec.input, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            dependencies=dependencies,
            metadata=metadata,
        )

    def _planned_task_input(
        self,
        operation: Operation,
        capability: Capability,
    ) -> dict[str, Any]:
        prompt = str(operation.request.get("prompt") or "")
        if capability.id in {"db.sql.execute_read", "db.sql.execute_write"}:
            return {"sql_ref": "sql.validation"}
        if capability.id == "db.sql.validate":
            return {"sql": prompt, "operation": operation.operation_type}
        return {"prompt": prompt}

    def _prompt_from_direct_input(self, input: dict[str, Any]) -> str:
        for key in ("prompt", "sql", "query", "content"):
            value = input.get(key)
            if value:
                return str(value)
        return ""

    def _contract_snapshot(
        self,
        contract: DbOperationContract | Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if contract is None:
            return None
        if isinstance(contract, Mapping):
            return _json_dict(contract)
        return {
            "operation_type": contract.operation_type,
            "required_capabilities": list(contract.required_capabilities),
            "required_evidence": list(contract.required_evidence),
            "access": contract.access.value,
            "limits": contract.limits.to_dict(),
            "policy_ids": list(contract.policy_ids),
            "metadata": contract.metadata,
        }
