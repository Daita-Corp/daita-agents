"""Stateless preparation rules for graph task attempts.

The helpers in this module calculate immutable contracts, bindings, context, and
call policy.  They never claim work, dispatch a capability, acquire a lock or
permit, or write durable state; those responsibilities remain with the existing
supervisor, runtime, and SQLite store boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256

from ..._json import FrozenJsonObject
from ...capabilities import (
    AccessMode,
    CapabilityGrant,
    CapabilityInputError,
    ExecutionContractBindings,
    ExecutionScope,
    ExecutionScopeKind,
    GraphTaskBinding,
    OperationalEffect,
)
from ...loop.models import RunOrigin
from .capabilities import REVIEW_CAPABILITY_IDS
from .context import TaskContextBundle
from .models import (
    BudgetAmount,
    GraphInspection,
    GraphTask,
    TaskAttempt,
    TaskExecutionKind,
    TaskRole,
    TaskState,
    canonical_digest,
)

PROFILE_WORK_KIND = "data_profile_work"
PROFILE_FINALIZER_KIND = "data_profile_finalizer"
GRAPH_RESULT_FINALIZER_KIND = "graph_result_finalizer"

PLANNER_CATALOG_CAPABILITY_IDS = frozenset(
    {"catalog.search", "catalog.schema", "catalog.inspect", "catalog.traverse"}
)
PLANNER_CAPABILITY_PREFIX = "jobs.graph.planner_"
TASK_LIFECYCLE_CAPABILITY_PREFIX = "jobs.graph.task_"


@dataclass(frozen=True, slots=True)
class GraphAttemptPreparation:
    """The immutable, I/O-free inputs needed before a durable claim."""

    contract: Mapping[str, object]
    executor_id: str
    absolute_deadline_at: datetime
    budget_reservations: tuple[BudgetAmount, ...]


def inspection_task(inspection: GraphInspection, task_id: str) -> GraphTask:
    task = next((item for item in inspection.tasks if item.task_id == task_id), None)
    if task is None:
        raise ValueError("graph task is absent from its inspection")
    return task


def internal_task_contract(task: GraphTask) -> Mapping[str, object]:
    if task.execution_kind is not TaskExecutionKind.INTERNAL_CAPABILITY:
        raise ValueError("the graph task is not an internal capability task")
    contract = task.specification.expected_result_contract
    kind = contract.get("kind")
    capability_id = contract.get("capability_id")
    contract_digest = contract.get("contract_digest")
    output_kind = contract.get("output_kind")
    if kind not in {
        PROFILE_WORK_KIND,
        PROFILE_FINALIZER_KIND,
        GRAPH_RESULT_FINALIZER_KIND,
    }:
        raise ValueError("the internal graph task kind is outside the admitted slice")
    if any(
        not isinstance(value, str) or not value
        for value in (capability_id, contract_digest, output_kind)
    ):
        raise ValueError("the internal graph task contract is malformed")
    assert isinstance(capability_id, str)
    assert isinstance(contract_digest, str)
    if (
        kind in {PROFILE_FINALIZER_KIND, GRAPH_RESULT_FINALIZER_KIND}
        and task.role is not TaskRole.FINALIZER
    ):
        raise ValueError("the graph finalizer contract is not reserved")
    if kind == PROFILE_WORK_KIND and task.role is TaskRole.FINALIZER:
        raise ValueError("the reserved finalizer cannot execute work")
    if capability_id not in task.specification.authority.capability_ids:
        raise ValueError("the task capability exceeds its immutable authority")
    binding = execution_contract_bindings(
        task.specification.authority.contract_bindings,
        capability_ids=task.specification.authority.capability_ids,
        resource_ids=task.specification.authority.resource_ids,
        model_route_ids=task.specification.authority.model_route_ids,
    ).capability_contracts.get(capability_id)
    if binding != contract_digest:
        raise ValueError("the task capability digest differs from its authority")
    return contract


def model_task_contract(task: GraphTask) -> Mapping[str, object]:
    if task.execution_kind is not TaskExecutionKind.MODEL:
        raise ValueError("the graph task is not model-driven")
    contract = task.specification.expected_result_contract
    if contract.get("kind") != "model_task":
        raise ValueError("the model graph task contract is malformed")
    route_id = contract.get("model_route_id")
    max_tokens = contract.get("per_run_max_tokens")
    max_cost = contract.get("per_run_max_cost_usd")
    if (
        not isinstance(route_id, str)
        or route_id not in task.specification.authority.model_route_ids
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
        or not isinstance(max_cost, str)
    ):
        raise ValueError("the model graph task route or budget is malformed")
    try:
        cost = Decimal(max_cost)
    except Exception as error:
        raise ValueError("the model graph task cost budget is malformed") from error
    if not cost.is_finite() or cost < 0:
        raise ValueError("the model graph task cost budget is malformed")
    return {**contract, "executor_id": route_id}


def graph_task_contract(task: GraphTask) -> Mapping[str, object]:
    if task.execution_kind is TaskExecutionKind.MODEL:
        return model_task_contract(task)
    contract = internal_task_contract(task)
    return {**contract, "executor_id": str(contract["capability_id"])}


def attempt_budget_reservations(task: GraphTask) -> tuple[BudgetAmount, ...]:
    raw = task.specification.expected_result_contract.get("attempt_budgets")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("the graph task attempt budget is malformed")
    task_limits = {item.dimension: item.amount for item in task.specification.budgets}
    reservations: list[BudgetAmount] = []
    for dimension, amount in raw.items():
        if (
            not isinstance(dimension, str)
            or not isinstance(amount, int)
            or isinstance(amount, bool)
        ):
            raise TypeError("the graph task attempt budget is malformed")
        if dimension not in task_limits or amount > task_limits[dimension]:
            raise ValueError("the attempt budget exceeds the immutable task budget")
        reservations.append(BudgetAmount(dimension, amount))
    return tuple(sorted(reservations))


def prepare_graph_attempt(
    inspection: GraphInspection,
    task: GraphTask,
    *,
    claimed_at: datetime,
) -> GraphAttemptPreparation:
    contract = graph_task_contract(task)
    return GraphAttemptPreparation(
        contract=contract,
        executor_id=str(contract["executor_id"]),
        absolute_deadline_at=min(
            inspection.job.deadline_at,
            claimed_at + timedelta(seconds=task.specification.max_wall_time_seconds),
        ),
        budget_reservations=attempt_budget_reservations(task),
    )


def graph_task_binding(
    inspection: GraphInspection,
    task: GraphTask,
    attempt: TaskAttempt,
) -> GraphTaskBinding:
    return GraphTaskBinding(
        agent_id=attempt.agent_id,
        job_id=attempt.job_id,
        root_authority_digest=inspection.job.specification.authority.digest,
        task_id=attempt.task_id,
        task_revision=task.task_revision,
        task_spec_digest=task.task_spec_digest,
        task_scope_digest=task.task_scope_digest,
        attempt_id=attempt.attempt_id,
        claim_token_digest="sha256:"
        + sha256(attempt.claim_token.encode("utf-8")).hexdigest(),
        fencing_epoch=attempt.fencing_epoch,
        graph_revision_at_claim=inspection.graph.revision,
        task_role=task.role.value,
        task_deadline_at=attempt.absolute_deadline_at,
        job_deadline_at=inspection.job.deadline_at,
        budget_reservation_identity=canonical_digest(
            {
                "attempt_id": attempt.attempt_id,
                "reservations": tuple(
                    {"dimension": item.dimension, "amount": item.amount}
                    for item in attempt.reserved_budgets
                ),
            }
        ),
    )


def execution_contract_bindings(
    material: Mapping[str, object],
    *,
    capability_ids: tuple[str, ...],
    resource_ids: tuple[str, ...],
    model_route_ids: tuple[str, ...],
) -> ExecutionContractBindings:
    nested = material.get("capability_contracts")
    if isinstance(nested, Mapping):
        raw_resources = material.get("resource_revisions", {})
        raw_routes = material.get("model_routes", {})
        raw_origins = material.get("tool_origins", {})
        if not isinstance(raw_resources, Mapping):
            raise TypeError("graph execution contract bindings are malformed")
        if not isinstance(raw_routes, Mapping):
            raise TypeError("graph execution contract bindings are malformed")
        if not isinstance(raw_origins, Mapping):
            raise TypeError("graph execution contract bindings are malformed")
        return ExecutionContractBindings(
            capability_contracts=dict(nested),
            resource_revisions=dict(raw_resources),
            model_routes=dict(raw_routes),
            tool_origins=dict(raw_origins),
        )
    capabilities: dict[str, str] = {}
    for capability_id in capability_ids:
        value = material.get(capability_id)
        if not isinstance(value, str):
            raise TypeError("graph capability contract binding is unavailable")
        capabilities[capability_id] = value
    resources: dict[str, str] = {}
    for resource_id in resource_ids:
        value = material.get(resource_id)
        if not isinstance(value, Mapping):
            raise TypeError("graph resource contract binding is unavailable")
        revision = value.get("resource_revision")
        if not isinstance(revision, str):
            raise TypeError("graph resource revision binding is unavailable")
        resources[resource_id] = revision
    routes: dict[str, str] = {}
    for route_id in model_route_ids:
        value = material.get(route_id)
        if not isinstance(value, str):
            raise TypeError("graph model route contract binding is unavailable")
        routes[route_id] = value
    return ExecutionContractBindings(
        capability_contracts=capabilities,
        resource_revisions=resources,
        model_routes=routes,
    )


def graph_capability_grants(
    material: Mapping[str, object],
) -> tuple[CapabilityGrant, ...]:
    raw = material.get("capability_grants", {})
    if not isinstance(raw, Mapping):
        raise TypeError("graph capability grants are malformed")
    grants: list[CapabilityGrant] = []
    for capability_id, entry in sorted(raw.items()):
        if not isinstance(capability_id, str) or not isinstance(entry, Mapping):
            raise TypeError("graph capability grant entry is malformed")
        grant_material = entry.get("grant")
        expected_digest = entry.get("grant_digest")
        if (
            set(entry) != {"grant", "grant_digest"}
            or not isinstance(grant_material, Mapping)
            or not isinstance(expected_digest, str)
            or set(grant_material)
            != {
                "grant_id",
                "domain_owner_id",
                "capability_id",
                "capability_contract_digest",
                "constraints_kind",
                "constraints",
                "max_calls_per_occurrence",
            }
            or not isinstance(grant_material.get("constraints"), Mapping)
        ):
            raise TypeError("graph capability grant entry is malformed")
        grant = CapabilityGrant(
            grant_id=str(grant_material["grant_id"]),
            domain_owner_id=str(grant_material["domain_owner_id"]),
            capability_id=str(grant_material["capability_id"]),
            capability_contract_digest=str(
                grant_material["capability_contract_digest"]
            ),
            constraints_kind=str(grant_material["constraints_kind"]),
            constraints=FrozenJsonObject.from_mapping(grant_material["constraints"]),
            max_calls_per_occurrence=grant_material["max_calls_per_occurrence"],
        )
        if (
            grant.capability_id != capability_id
            or grant.grant_digest != expected_digest
        ):
            raise ValueError("graph capability grant digest is invalid")
        grants.append(grant)
    return tuple(grants)


def model_task_execution_scope(
    inspection: GraphInspection,
    task: GraphTask,
    attempt: TaskAttempt,
    binding: GraphTaskBinding,
    contract: Mapping[str, object],
) -> ExecutionScope:
    authority = task.specification.authority
    bindings = execution_contract_bindings(
        authority.contract_bindings,
        capability_ids=authority.capability_ids,
        resource_ids=authority.resource_ids,
        model_route_ids=authority.model_route_ids,
    )
    max_tokens = contract["per_run_max_tokens"]
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise TypeError("the model graph task token budget is malformed")
    access_modes = frozenset(
        AccessMode(item) for item in authority.access_modes
    ) | frozenset({AccessMode.NONE})
    grants = graph_capability_grants(authority.contract_bindings)
    return ExecutionScope(
        scope_id=f"graph-scope:{attempt.attempt_id}",
        revision=1,
        agent_id=attempt.agent_id,
        principal_id=inspection.job.specification.principal_id,
        grant_id=f"graph-attempt:{attempt.attempt_id}",
        job_id=attempt.job_id,
        job_revision=inspection.graph.revision + 1,
        allowed_source_ids=authority.source_ids,
        allowed_resource_ids=authority.resource_ids,
        allowed_capability_ids=authority.capability_ids,
        allowed_access_modes=access_modes,
        allowed_operational_effects=frozenset(
            OperationalEffect(item) for item in authority.operational_effects
        ),
        sensitivity_ceiling=authority.sensitivity,
        eligible_model_routes=authority.model_route_ids,
        per_run_max_cost_usd=Decimal(str(contract["per_run_max_cost_usd"])),
        per_run_max_tokens=max_tokens,
        distribution_plan_digest=inspection.job.specification.distribution_plan_digest,
        contract_bindings=bindings,
        allowed_connector_binding_ids=authority.connector_ids,
        scope_kind=ExecutionScopeKind.GRAPH_TASK,
        capability_grants=grants,
        graph_task_binding=binding,
    )


def task_context_bundle(
    inspection: GraphInspection,
    task: GraphTask,
    attempt: TaskAttempt,
    binding: GraphTaskBinding,
    *,
    created_at: datetime,
) -> TaskContextBundle:
    direct_parent_ids = {
        edge.upstream_task_id
        for edge in inspection.dependencies
        if edge.downstream_task_id == task.task_id
    }
    parent_ids = effective_task_ids(inspection, direct_parent_ids)
    parents = tuple(
        {
            "task_id": result.task_id,
            "result_id": result.result_id,
            "result_kind": result.result_kind,
            "summary": result.summary,
            "payload": result.payload,
            "result_digest": result.result_digest,
            "sensitivity": result.sensitivity.value,
            "artifact_ids": result.artifact_ids,
            "provenance": result.provenance,
        }
        for result in inspection.results
        if result.task_id in parent_ids
    )
    prior = tuple(
        {
            "attempt_id": item.attempt_id,
            "ordinal": item.ordinal,
            "state": item.state.value,
            "error_code": item.error_code,
            "diagnostic": item.diagnostic,
            "checkpoint_ids": item.checkpoint_ids,
            "measured_usage": tuple(
                {"dimension": usage.dimension, "amount": usage.amount}
                for usage in item.measured_usage
            ),
        }
        for item in sorted(
            (
                value
                for value in inspection.attempts
                if value.task_id == task.task_id
                and value.attempt_id != attempt.attempt_id
            ),
            key=lambda value: value.ordinal,
        )[-2:]
    )
    checkpoints = tuple(
        {
            "attempt_id": item.attempt_id,
            "checkpoint_id": item.checkpoint_id,
            "ordinal": item.ordinal,
            "milestone": item.milestone,
            "payload": item.payload,
            "payload_digest": item.payload_digest,
        }
        for item in inspection.checkpoints
        if item.task_id == task.task_id
    )[-8:]
    comments = tuple(
        {
            "comment_id": item.comment_id,
            "author_kind": item.author_kind,
            "body": item.body,
            "body_digest": item.body_digest,
            "sensitivity": item.sensitivity.value,
        }
        for item in inspection.comments
        if item.task_id == task.task_id
    )[-16:]
    review_control_id = (
        task.specification.expected_result_contract.get("review_control_id")
        if task.role is TaskRole.REVIEWER
        else None
    )
    controls = tuple(
        {
            "control_id": item.control_id,
            "kind": item.kind.value,
            "state": item.state.value,
            "payload": item.payload,
            "resolution": item.resolution,
            "resolved_by_kind": item.resolved_by_kind,
            "resolved_by_id": item.resolved_by_id,
        }
        for item in inspection.controls
        if item.task_id == task.task_id
        or (isinstance(review_control_id, str) and item.control_id == review_control_id)
    )[-16:]
    try:
        return TaskContextBundle(
            binding=binding,
            root_objective=inspection.job.specification.objective,
            outcome_contract=inspection.job.specification.outcome_contract,
            task_specification=task.specification.digest_material(),
            parent_results=parents,
            prior_attempts=prior,
            checkpoints=checkpoints,
            comments=comments,
            controls=controls,
            created_at=created_at,
        )
    except ValueError as error:
        if "aggregate byte bound" not in str(error):
            raise
        bounded_parents = tuple(
            {
                key: value
                for key, value in parent.items()
                if key not in {"payload", "provenance"}
            }
            | {"omitted_payload_digest": parent["result_digest"]}
            for parent in parents
        )
        return TaskContextBundle(
            binding=binding,
            root_objective=inspection.job.specification.objective,
            outcome_contract=inspection.job.specification.outcome_contract,
            task_specification=task.specification.digest_material(),
            parent_results=bounded_parents,
            prior_attempts=prior,
            checkpoints=checkpoints,
            comments=comments,
            controls=controls,
            created_at=created_at,
        )


def task_conversation_id(job_id: str, attempt_id: str) -> str:
    digest = sha256(f"{job_id}:{attempt_id}".encode()).hexdigest()[:32]
    return f"graph-task-{digest}"


def effective_task_ids(inspection: GraphInspection, task_ids: set[str]) -> set[str]:
    """Resolve immutable supersession chains without treating supersession as success."""

    by_id = {task.task_id: task for task in inspection.tasks}
    resolved: set[str] = set()
    for original in task_ids:
        current_id = original
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            current = by_id.get(current_id)
            if current is None:
                raise ValueError("graph dependency task is unavailable")
            if (
                current.state is TaskState.SUPERSEDED
                and current.superseded_by_task_id is not None
            ):
                current_id = current.superseded_by_task_id
                continue
            resolved.add(current_id)
            break
        else:
            raise ValueError("graph task supersession contains a cycle")
    return resolved


def task_role_allows_capability(role: str, capability_id: str) -> bool:
    """Enforce the narrow role surface independently of domain projection."""

    if role == TaskRole.PLANNER.value:
        return (
            capability_id.startswith(
                (TASK_LIFECYCLE_CAPABILITY_PREFIX, PLANNER_CAPABILITY_PREFIX)
            )
            or capability_id in PLANNER_CATALOG_CAPABILITY_IDS
        )
    if role == TaskRole.FINALIZER.value:
        return (
            capability_id == "jobs.graph.result_finalize"
            or capability_id.startswith(TASK_LIFECYCLE_CAPABILITY_PREFIX)
        )
    if role == TaskRole.REVIEWER.value:
        return capability_id in REVIEW_CAPABILITY_IDS
    if role == TaskRole.WORKER.value:
        return not capability_id.startswith(PLANNER_CAPABILITY_PREFIX)
    return False


def validate_graph_task_arguments(
    *,
    origin: RunOrigin,
    context: object,
    capability_id: str,
    arguments: Mapping[str, object],
) -> None:
    """Validate a graph call against its immutable task role and proposal."""

    if origin is not RunOrigin.JOB_TASK:
        return
    binding = getattr(context, "binding", None)
    if not isinstance(binding, GraphTaskBinding):
        raise CapabilityInputError(
            "task_context_invalid", "The code-owned task binding is unavailable."
        )
    if not task_role_allows_capability(binding.task_role, capability_id):
        raise CapabilityInputError(
            "task_role_capability_forbidden",
            "This capability is outside the exact graph-task role projection.",
        )
    specification = getattr(context, "task_specification", None)
    if not isinstance(specification, Mapping):
        raise CapabilityInputError(
            "task_context_invalid", "The code-owned task specification is unavailable."
        )
    expected = specification.get("expected_result_contract")
    if not isinstance(expected, Mapping):
        raise CapabilityInputError(
            "task_context_invalid",
            "The code-owned task result contract is unavailable.",
        )
    initial_call = expected.get("initial_call")
    if initial_call is None:
        return
    if not isinstance(initial_call, Mapping):
        raise CapabilityInputError(
            "task_context_invalid", "The frozen initial call is malformed."
        )
    if capability_id.startswith("jobs.graph."):
        return
    effect_call = initial_call.get("effect_call")
    if effect_call is not None:
        if not isinstance(effect_call, Mapping):
            raise CapabilityInputError(
                "task_context_invalid", "The frozen effect call is malformed."
            )
        effect_capability_id = effect_call.get("capability_id")
        preview_capability_id = effect_call.get("preview_capability_id")
        intent = effect_call.get("intent_arguments")
        if not isinstance(effect_capability_id, str) or not isinstance(intent, Mapping):
            raise CapabilityInputError(
                "task_context_invalid", "The frozen effect intent is malformed."
            )
        if preview_capability_id is None:
            if capability_id == effect_capability_id and arguments == intent:
                return
        elif isinstance(preview_capability_id, str):
            if capability_id == preview_capability_id and arguments == intent:
                return
            if capability_id == effect_capability_id and all(
                name in arguments and arguments[name] == value
                for name, value in intent.items()
            ):
                return
        raise CapabilityInputError(
            "task_call_differs_from_proposal",
            "The graph effect call differs from its exact admitted intent.",
        )
    if (
        initial_call.get("capability_id") != capability_id
        or initial_call.get("arguments") != arguments
    ):
        raise CapabilityInputError(
            "task_call_differs_from_proposal",
            "The graph work call differs from its exact admitted proposal.",
        )


__all__ = [
    "GRAPH_RESULT_FINALIZER_KIND",
    "PLANNER_CATALOG_CAPABILITY_IDS",
    "PROFILE_FINALIZER_KIND",
    "PROFILE_WORK_KIND",
    "GraphAttemptPreparation",
    "attempt_budget_reservations",
    "effective_task_ids",
    "execution_contract_bindings",
    "graph_capability_grants",
    "graph_task_binding",
    "graph_task_contract",
    "inspection_task",
    "internal_task_contract",
    "model_task_contract",
    "model_task_execution_scope",
    "prepare_graph_attempt",
    "task_context_bundle",
    "task_conversation_id",
    "task_role_allows_capability",
    "validate_graph_task_arguments",
]
