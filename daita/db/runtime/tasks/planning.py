"""Task planning helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.runtime import Capability, Operation, Task

from ...fingerprints import persisted_fingerprint
from ...models import DbOperationContract
from .common import _json_dict
from .context import DbTaskContext
from .dependencies import (
    _combine_dependencies,
    _has_sql_validation_dependency,
    _task_dependencies_for_capability,
)
from .models import DbTaskPlan, DbTaskSpec


def _validation_capability_for_sql_execute(
    context: DbTaskContext,
    capability: Capability,
) -> Capability | None:
    if capability.id not in {"db.sql.execute_read", "db.sql.execute_write"}:
        return None
    try:
        return context.registry.get_capability(
            "db.sql.validate", owner=capability.owner
        )
    except KeyError:
        return None


def _direct_capability_tasks(
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
        task = _task_for_capability(
            operation,
            capability,
            input=input,
            reason="direct",
            sequence=1,
        )
        return (task,)
    validation_task = _task_for_capability(
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
    execute_input = {
        "sql_ref": "sql.validation",
        "params": list(input.get("params") or []),
        **(
            {"param_specs": list(input.get("param_specs") or [])}
            if input.get("param_specs")
            else {}
        ),
    }
    execute_task = _task_for_capability(
        operation,
        capability,
        input=execute_input,
        reason="direct",
        sequence=2,
        validation_task=validation_task,
    )
    return (validation_task, execute_task)


def _task_for_capability(
    operation: Operation,
    capability: Capability,
    *,
    input: dict[str, Any],
    reason: str,
    sequence: int,
    validation_task: Task | None = None,
    metadata: dict[str, Any] | None = None,
) -> Task:
    input_hash = persisted_fingerprint(input)
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
            "idempotency_key": persisted_fingerprint(
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


def _prompt_from_direct_input(input: dict[str, Any]) -> str:
    for key in ("prompt", "sql", "query", "content"):
        value = input.get(key)
        if value:
            return str(value)
    return ""


async def plan_task_specs(
    context: DbTaskContext,
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
        capability = context.registry.get_capability(
            spec.capability_id,
            owner=spec.owner,
        )
        validation_task = prior_by_capability_owner.get(
            ("db.sql.validate", capability.owner)
        )
        task = _task_for_spec(
            operation,
            capability,
            spec,
            contract=contract,
            validation_task=validation_task,
        )
        existing = await context.store.load_task(task.id)
        if existing is not None:
            planned.append(existing)
            diagnostics["reused_task_count"] += 1
        else:
            planned.append(await plan_kernel_task(context, task))
            diagnostics["planned_task_count"] += 1
        prior_by_capability_owner[(capability.id, capability.owner)] = planned[-1]
    return DbTaskPlan(tasks=tuple(planned), diagnostics=diagnostics)


async def plan_kernel_task(context: DbTaskContext, task: Task) -> Task:
    """Persist a DB-planned task through the shared kernel planner."""
    return await context.kernel.plan_task(
        task_id=task.id,
        operation_id=task.operation_id,
        capability_id=task.capability_id,
        owner=str(task.metadata["owner"]) if task.metadata.get("owner") else None,
        input=task.input,
        metadata=task.metadata,
        dependencies=task.dependencies,
    )


def _task_for_spec(
    operation: Operation,
    capability: Capability,
    spec: DbTaskSpec,
    *,
    contract: DbOperationContract | Mapping[str, Any] | None = None,
    validation_task: Task | None = None,
) -> Task:
    input_hash = persisted_fingerprint(spec.input)
    idempotency_key = spec.idempotency_key or persisted_fingerprint(
        {
            "operation_id": operation.id,
            "capability_id": capability.id,
            "owner": capability.owner,
            "input_hash": input_hash,
            "sequence": spec.sequence,
            "deterministic_key": spec.deterministic_key,
        }
    )
    task_fingerprint = persisted_fingerprint(
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
                dependency.kind_value == "evidence"
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
    contract_snapshot = _contract_snapshot(contract)
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


def _contract_snapshot(
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
