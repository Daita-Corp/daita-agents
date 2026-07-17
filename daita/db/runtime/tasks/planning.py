"""Task planning helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping
from uuid import uuid4

from daita.core.exceptions import ValidationError
from daita.runtime import (
    Capability,
    Evidence,
    Operation,
    Task,
    TaskDependency,
    ToolView,
)

from ...fingerprints import persisted_fingerprint
from ...models import DbOperationContract
from ...query_sql_validation import sql_fingerprint
from .common import _json_dict
from .context import DbTaskContext
from .dependencies import (
    _combine_dependencies,
    _has_sql_validation_dependency,
    _task_dependencies_for_capability,
)
from .models import DbTaskPlan, DbTaskSpec

SLIM_READ_OPERATION_NAMES = (
    "search_schema",
    "inspect_asset",
    "find_relationships",
    "search_column_values",
    "query",
)

_SLIM_RECIPE_KINDS = {
    "search_schema": "catalog",
    "inspect_asset": "catalog",
    "find_relationships": "catalog",
    "search_column_values": "catalog",
    "query": "query",
}


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


async def plan_slim_operation(
    context: DbTaskContext,
    operation: Operation,
    *,
    operation_name: str,
    arguments: Mapping[str, Any],
    source_owner: str,
    attempt: int = 1,
    groundings: Iterable[Mapping[str, Any]] = (),
) -> DbTaskPlan:
    """Plan the first stage of one closed Phase 2 operation recipe.

    Registry declarations own the model schema and capability identity. This
    owner binds catalog/source identity, policy scope, validation schema,
    dependencies, expected evidence, and deterministic task identity.
    """

    name = str(operation_name)
    recipe_kind = _SLIM_RECIPE_KINDS.get(name)
    if recipe_kind is None:
        raise ValidationError(
            "Unsupported Phase 2 database operation.",
            field="operation",
            context={"allowed_operations": list(SLIM_READ_OPERATION_NAMES)},
        )
    view, view_owner = _slim_tool_view(context, name)
    model_input = _validated_model_arguments(view.parameters, arguments)
    owner = str(source_owner)
    recipe_attempt = max(1, int(attempt))
    policy_scope = _runtime_policy_scope(context, operation, owner)

    if recipe_kind == "catalog":
        if view_owner != "catalog":
            raise RuntimeError(f"Phase 2 operation {name!r} must be catalog-owned")
        capability = context.registry.get_capability(
            view.capability_id,
            owner=view_owner,
        )
        store_id = _ready_catalog_store_id(context, source_owner=owner)
        task_input = {
            **model_input,
            "store_id": store_id,
            **policy_scope,
        }
        spec = DbTaskSpec(
            capability_id=capability.id,
            owner=capability.owner,
            input=task_input,
            reason=f"slim_read:{name}",
            sequence=1,
            metadata={
                "slim_operation": name,
                "selection_attempt": recipe_attempt,
                "source_owner": owner,
            },
            deterministic_key=(
                f"slim:{owner}:{name}:{recipe_attempt}:"
                f"{persisted_fingerprint(model_input)}"
            ),
        )
        plan = await plan_task_specs(context, operation, (spec,))
        return DbTaskPlan(
            tasks=plan.tasks,
            diagnostics={
                **plan.diagnostics,
                "operation": name,
                "recipe": "catalog_single_task",
                "expected_evidence": sorted(capability.output_evidence),
                "selection_attempt": recipe_attempt,
            },
        )

    if view_owner != owner or view.capability_id != "db.sql.execute_read":
        raise RuntimeError("Phase 2 query declaration does not match the source owner")
    sql = str(model_input["sql"]).strip()
    params = list(model_input["params"])
    param_specs = [dict(item) for item in model_input.get("param_specs") or ()]
    grounding_facts = [
        dict(item) for item in list(groundings or ())[:12] if isinstance(item, Mapping)
    ]
    fingerprint = sql_fingerprint(sql)
    validation = context.registry.get_capability("db.sql.validate", owner=owner)
    schema = _runtime_validation_schema(
        context,
        source_owner=owner,
        policy_scope=policy_scope,
    )
    validation_input: dict[str, Any] = {
        "sql": sql,
        "operation": "query",
        "schema": schema,
        "params": params,
        "groundings": grounding_facts,
        "source_owner": owner,
    }
    recipe_identity = persisted_fingerprint(
        {
            "sql_fingerprint": fingerprint,
            "params": params,
            "param_specs": param_specs,
            "groundings": grounding_facts,
            "attempt": recipe_attempt,
        }
    )
    spec = DbTaskSpec(
        capability_id=validation.id,
        owner=validation.owner,
        input=validation_input,
        reason="slim_read:validation",
        sequence=(recipe_attempt * 2) - 1,
        metadata={
            "slim_operation": name,
            "query_attempt": recipe_attempt,
            "sql_fingerprint": fingerprint,
            "source_owner": owner,
            "source_scope": list(policy_scope.get("source_scope") or ()),
            "query_params": params,
            "query_param_specs": param_specs,
        },
        deterministic_key=f"slim:{owner}:validate:{recipe_identity}",
    )
    plan = await plan_task_specs(context, operation, (spec,))
    return DbTaskPlan(
        tasks=plan.tasks,
        diagnostics={
            **plan.diagnostics,
            "operation": name,
            "recipe": "db.sql.validate",
            "sql_fingerprint": fingerprint,
            "query_attempt": recipe_attempt,
            "expected_evidence": sorted(validation.output_evidence),
        },
    )


async def plan_validated_slim_read(
    context: DbTaskContext,
    operation: Operation,
    *,
    source_owner: str,
    validation: Evidence,
) -> DbTaskPlan:
    """Persist the read stage only after applicable validation was accepted."""

    owner = str(source_owner)
    validation_task = await context.store.load_task(str(validation.task_id or ""))
    if (
        not validation.accepted
        or validation.kind != "sql.validation"
        or validation.owner != owner
        or validation.operation_id != operation.id
        or validation_task is None
        or validation_task.capability_id != "db.sql.validate"
        or validation.payload.get("valid") is not True
    ):
        raise ValidationError(
            "Accepted current SQL validation evidence is required before read planning.",
            field="validation",
        )
    execute = context.registry.get_capability("db.sql.execute_read", owner=owner)
    attempt = max(1, int(validation_task.metadata.get("query_attempt") or 1))
    fingerprint = str(
        validation.payload.get("sql_fingerprint")
        or validation_task.metadata.get("sql_fingerprint")
        or sql_fingerprint(str(validation.payload.get("sql") or ""))
    )
    params = list(validation_task.metadata.get("query_params") or ())
    param_specs = [
        dict(item)
        for item in validation_task.metadata.get("query_param_specs") or ()
        if isinstance(item, Mapping)
    ]
    spec = DbTaskSpec(
        capability_id=execute.id,
        owner=execute.owner,
        input={
            "sql_ref": "sql.validation",
            "sql_fingerprint": fingerprint,
            "params": params,
            **({"param_specs": param_specs} if param_specs else {}),
        },
        reason="slim_read:execute",
        sequence=attempt * 2,
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                evidence_id=validation.id,
                evidence_owner=owner,
                producer_task_id=validation_task.id,
                producer_capability_id=validation_task.capability_id,
                producer_executor_id=validation_task.executor_id,
                evidence_payload={"valid": True},
                evidence_accepted=True,
                operation_id=operation.id,
            ),
        ),
        metadata={
            "slim_operation": "query",
            "query_attempt": attempt,
            "sql_fingerprint": fingerprint,
            "source_owner": owner,
            "validation_evidence_id": validation.id,
        },
        deterministic_key=f"slim:{owner}:execute:{attempt}:{fingerprint}",
    )
    plan = await plan_task_specs(context, operation, (spec,))
    return DbTaskPlan(
        tasks=plan.tasks,
        diagnostics={
            **plan.diagnostics,
            "operation": "query",
            "recipe": "db.sql.execute_read",
            "sql_fingerprint": fingerprint,
            "query_attempt": attempt,
            "expected_evidence": sorted(execute.output_evidence),
        },
    )


def _slim_tool_view(
    context: DbTaskContext,
    operation_name: str,
) -> tuple[ToolView, str]:
    matches = [
        view
        for view in context.registry.tool_views
        if view.name == operation_name
        and view.model_visible
        and view.metadata.get("db_slim_phase") == 2
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Phase 2 operation {operation_name!r} must have one declared ToolView"
        )
    view = matches[0]
    return view, context.registry.get_tool_view_owner(view.name)


def _validated_model_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = {str(item) for item in schema.get("required") or ()}
    unknown = sorted(str(key) for key in arguments if key not in properties)
    missing = sorted(key for key in required if key not in arguments)
    invalid = sorted(
        str(key)
        for key, value in arguments.items()
        if key in properties
        and isinstance(properties[key], Mapping)
        and not _matches_declared_schema(value, properties[key])
    )
    if unknown or missing or invalid:
        raise ValidationError(
            "Selected operation arguments do not match the registered ToolView.",
            field="arguments",
            context={
                "unknown_arguments": unknown,
                "missing_arguments": missing,
                "invalid_arguments": invalid,
            },
        )
    return {str(key): value for key, value in arguments.items()}


def _matches_declared_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    declared = schema.get("type")
    types = tuple(declared) if isinstance(declared, list) else (declared,)
    if declared is not None and not any(
        (
            kind == "null"
            and value is None
            or kind == "string"
            and isinstance(value, str)
            or kind == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
            or kind == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            or kind == "boolean"
            and isinstance(value, bool)
            or kind == "array"
            and isinstance(value, list)
            or kind == "object"
            and isinstance(value, Mapping)
        )
        for kind in types
    ):
        return False
    if isinstance(value, str) and int(schema.get("minLength") or 0) > len(value):
        return False
    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < int(minimum):
            return False
        if maximum is not None and value > int(maximum):
            return False
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        if int(schema.get("minItems") or 0) > len(value):
            return False
        maximum_items = schema.get("maxItems")
        if maximum_items is not None and len(value) > int(maximum_items):
            return False
        if not all(_matches_declared_schema(item, schema["items"]) for item in value):
            return False
    if isinstance(value, Mapping) and schema.get("type") == "object":
        try:
            _validated_model_arguments(schema, value)
        except ValidationError:
            return False
    return True


def _runtime_policy_scope(
    context: DbTaskContext,
    operation: Operation,
    source_owner: str,
) -> dict[str, Any]:
    source = context.registry.get_plugin(source_owner)
    allowed_tables = sorted(str(item) for item in source.allowed_tables)
    blocked_tables = sorted(str(item) for item in source.blocked_tables)
    blocked_columns = sorted(str(item) for item in source.blocked_columns)
    raw_scope = operation.request.get("source_scope") or operation.metadata.get(
        "source_scope"
    )
    source_scope = (
        [str(raw_scope)]
        if isinstance(raw_scope, str)
        else [str(item) for item in raw_scope or ()]
    )
    requested_tables = [
        item
        for item in source_scope
        if item not in {source_owner, f"{source_owner}:default"}
    ]
    restricted = bool(getattr(source, "_allowed_tables_restricted", False))
    if requested_tables:
        if allowed_tables:
            wanted = {item.lower() for item in requested_tables}
            allowed_tables = [
                item
                for item in allowed_tables
                if item.lower() in wanted or item.split(".")[-1].lower() in wanted
            ]
        else:
            allowed_tables = requested_tables
        restricted = True
    return {
        "allowed_tables": allowed_tables,
        "allowed_tables_restricted": restricted,
        "blocked_tables": blocked_tables,
        "blocked_columns": blocked_columns,
        "source_scope": source_scope,
    }


def _ready_catalog_store_id(
    context: DbTaskContext,
    *,
    source_owner: str,
) -> str:
    catalog = context.registry.get_plugin("catalog")
    binding = catalog.runtime_binding_facts()
    state = catalog.runtime_source_state()
    if (
        state.get("status") != "ready"
        or binding.get("source_owner") != source_owner
        or not binding.get("store_id")
    ):
        raise RuntimeError("Phase 2 catalog source is not bound and ready")
    return str(binding["store_id"])


def _runtime_validation_schema(
    context: DbTaskContext,
    *,
    source_owner: str,
    policy_scope: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = context.registry.get_plugin("catalog")
    binding = catalog.runtime_binding_facts()
    if binding.get("source_owner") != source_owner:
        raise RuntimeError("Phase 2 catalog/source binding mismatch")
    schema = catalog.runtime_validation_schema(policy_scope=policy_scope)
    if not schema:
        raise RuntimeError("Phase 2 catalog validation schema is unavailable")
    return dict(schema)


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
