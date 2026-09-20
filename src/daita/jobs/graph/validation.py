"""Pure invariant validation for the jobs-owned current graph model."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping

from ...llm.models import ModelSensitivity
from .models import (
    ACTIVE_ATTEMPT_STATES,
    CONTROL_BUDGET_ROLES,
    AttemptState,
    GraphAdmission,
    GraphAuthority,
    GraphMutationRequest,
    GraphState,
    GraphTask,
    JobGraph,
    TaskAttempt,
    TaskControl,
    TaskDependency,
    TaskExecutionKind,
    TaskResult,
    TaskRole,
    TaskState,
    topology_digest,
)


class GraphValidationError(ValueError):
    """A bounded graph proposal or transition violates the frozen contract."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)


GRAPH_TRANSITIONS = {
    GraphState.QUEUED: frozenset(
        {
            GraphState.ACTIVE,
            GraphState.CANCEL_REQUESTED,
            GraphState.NEEDS_ATTENTION,
            GraphState.FAILED,
        }
    ),
    GraphState.ACTIVE: frozenset(
        {
            GraphState.BLOCKED,
            GraphState.NEEDS_ATTENTION,
            GraphState.CANCEL_REQUESTED,
            GraphState.FAILED,
            GraphState.SUCCEEDED,
        }
    ),
    GraphState.BLOCKED: frozenset(
        {
            GraphState.ACTIVE,
            GraphState.NEEDS_ATTENTION,
            GraphState.CANCEL_REQUESTED,
            GraphState.FAILED,
        }
    ),
    GraphState.NEEDS_ATTENTION: frozenset(
        {GraphState.ACTIVE, GraphState.CANCEL_REQUESTED, GraphState.FAILED}
    ),
    GraphState.CANCEL_REQUESTED: frozenset(
        {GraphState.CANCELLED, GraphState.NEEDS_ATTENTION}
    ),
    GraphState.SUCCEEDED: frozenset(),
    GraphState.FAILED: frozenset(),
    GraphState.CANCELLED: frozenset(),
}

TASK_TRANSITIONS = {
    TaskState.PENDING: frozenset(
        {
            TaskState.READY,
            TaskState.BLOCKED,
            TaskState.CANCELLED,
            TaskState.SUPERSEDED,
            TaskState.SKIPPED,
        }
    ),
    TaskState.READY: frozenset(
        {
            TaskState.RUNNING,
            TaskState.CANCELLED,
            TaskState.SUPERSEDED,
            TaskState.SKIPPED,
        }
    ),
    TaskState.RUNNING: frozenset(
        {
            TaskState.SUCCEEDED,
            TaskState.READY,
            TaskState.BLOCKED,
            TaskState.REVIEW,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }
    ),
    TaskState.BLOCKED: frozenset(
        {
            TaskState.READY,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.SUPERSEDED,
        }
    ),
    TaskState.REVIEW: frozenset(
        {
            TaskState.SUCCEEDED,
            TaskState.BLOCKED,
            TaskState.CANCELLED,
            TaskState.SUPERSEDED,
        }
    ),
    TaskState.SUCCEEDED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
    TaskState.SKIPPED: frozenset(),
    TaskState.SUPERSEDED: frozenset(),
}

ATTEMPT_TRANSITIONS = {
    AttemptState.CLAIMED: frozenset(
        {
            AttemptState.RUNNING,
            AttemptState.FAILED,
            AttemptState.CANCELLED,
            AttemptState.TIMED_OUT,
            AttemptState.FENCED,
        }
    ),
    AttemptState.RUNNING: frozenset(
        {
            AttemptState.SUCCEEDED,
            AttemptState.FAILED,
            AttemptState.CANCELLED,
            AttemptState.BLOCKED,
            AttemptState.REVIEW_REQUESTED,
            AttemptState.TIMED_OUT,
            AttemptState.PROTOCOL_VIOLATION,
            AttemptState.FENCED,
        }
    ),
    AttemptState.SUCCEEDED: frozenset(),
    AttemptState.FAILED: frozenset(),
    AttemptState.CANCELLED: frozenset(),
    AttemptState.BLOCKED: frozenset(),
    AttemptState.REVIEW_REQUESTED: frozenset(),
    AttemptState.TIMED_OUT: frozenset(),
    AttemptState.PROTOCOL_VIOLATION: frozenset(),
    AttemptState.FENCED: frozenset(),
}


def require_graph_transition(before: GraphState, after: GraphState) -> None:
    if after not in GRAPH_TRANSITIONS[before]:
        raise GraphValidationError(
            "illegal_graph_transition",
            f"graph transition is not allowed: {before.value} -> {after.value}",
        )


def require_task_transition(before: TaskState, after: TaskState) -> None:
    if after not in TASK_TRANSITIONS[before]:
        raise GraphValidationError(
            "illegal_task_transition",
            f"task transition is not allowed: {before.value} -> {after.value}",
        )


def require_attempt_transition(before: AttemptState, after: AttemptState) -> None:
    if after not in ATTEMPT_TRANSITIONS[before]:
        raise GraphValidationError(
            "illegal_attempt_transition",
            f"attempt transition is not allowed: {before.value} -> {after.value}",
        )


def _sensitivity_rank(value: ModelSensitivity) -> int:
    return {
        ModelSensitivity.PUBLIC: 0,
        ModelSensitivity.INTERNAL: 1,
        ModelSensitivity.CONFIDENTIAL: 2,
        ModelSensitivity.RESTRICTED: 3,
    }[value]


def require_authority_subset(child: GraphAuthority, parent: GraphAuthority) -> None:
    for field_name in (
        "source_ids",
        "resource_ids",
        "connector_ids",
        "capability_ids",
        "access_modes",
        "operational_effects",
        "model_route_ids",
    ):
        if not set(getattr(child, field_name)).issubset(getattr(parent, field_name)):
            raise GraphValidationError(
                "authority_expansion",
                f"task {field_name} expands root authority",
            )
    if _sensitivity_rank(child.sensitivity) > _sensitivity_rank(parent.sensitivity):
        raise GraphValidationError(
            "authority_expansion", "task sensitivity exceeds root authority"
        )
    parent_bindings = dict(parent.contract_bindings)
    for key, value in child.contract_bindings.items():
        parent_value = parent_bindings.get(key)
        if isinstance(value, Mapping) and isinstance(parent_value, Mapping):
            differs = any(
                nested_key not in parent_value
                or parent_value[nested_key] != nested_value
                for nested_key, nested_value in value.items()
            )
        else:
            differs = key not in parent_bindings or parent_value != value
        if differs:
            raise GraphValidationError(
                "contract_binding_expansion",
                "task contract binding is not an exact root subset",
            )


def _topology(
    tasks: tuple[GraphTask, ...],
    dependencies: tuple[TaskDependency, ...],
    *,
    graph: JobGraph,
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    task_ids = {task.task_id for task in tasks}
    if len(task_ids) != len(tasks):
        raise GraphValidationError("duplicate_task", "graph task IDs must be unique")
    pairs = {(edge.upstream_task_id, edge.downstream_task_id) for edge in dependencies}
    if len(pairs) != len(dependencies):
        raise GraphValidationError("duplicate_edge", "graph edges must be unique")
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    for edge in dependencies:
        if (
            edge.upstream_task_id not in task_ids
            or edge.downstream_task_id not in task_ids
        ):
            raise GraphValidationError(
                "foreign_edge_endpoint", "dependency endpoint is not in this graph"
            )
        parents[edge.downstream_task_id].append(edge.upstream_task_id)
        children[edge.upstream_task_id].append(edge.downstream_task_id)
    indegree = {task_id: len(parents[task_id]) for task_id in task_ids}
    queue = deque(
        sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    )
    depths = {task_id: 1 for task_id in queue}
    visited: list[str] = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for child in sorted(children[current]):
            depths[child] = max(depths.get(child, 1), depths[current] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(task_ids):
        raise GraphValidationError("graph_cycle", "graph dependencies contain a cycle")
    return (
        {key: tuple(sorted(value)) for key, value in parents.items()},
        depths,
    )


def validate_graph_topology(
    tasks: tuple[GraphTask, ...],
    dependencies: tuple[TaskDependency, ...],
    *,
    graph: JobGraph,
    max_tasks: int,
    max_edges: int,
    max_depth: int,
    max_direct_parents: int,
) -> None:
    if len(tasks) > max_tasks:
        raise GraphValidationError("task_limit", "graph task limit exceeded")
    if len(dependencies) > max_edges:
        raise GraphValidationError("edge_limit", "graph edge limit exceeded")
    parents, depths = _topology(tasks, dependencies, graph=graph)
    if any(len(value) > max_direct_parents for value in parents.values()):
        raise GraphValidationError("parent_limit", "graph direct-parent limit exceeded")
    if depths and max(depths.values()) > max_depth:
        raise GraphValidationError("depth_limit", "graph dependency depth exceeded")


def _validate_budget_envelope(
    tasks: tuple[GraphTask, ...], admission: GraphAdmission
) -> None:
    root = {item.dimension: item for item in admission.job.specification.budgets}
    ordinary: dict[str, int] = defaultdict(int)
    control: dict[str, int] = defaultdict(int)
    for task in tasks:
        for budget in task.specification.budgets:
            if budget.dimension not in root:
                raise GraphValidationError(
                    "unknown_budget_dimension",
                    "task budget dimension is absent from the root ledger",
                )
            target = control if task.role in CONTROL_BUDGET_ROLES else ordinary
            target[budget.dimension] += budget.amount
    for dimension, limit in root.items():
        if ordinary[dimension] > limit.ceiling - limit.control_reserved:
            raise GraphValidationError(
                "worker_budget_expansion",
                "ordinary task ceilings consume the root control reserve",
            )
        if control[dimension] > limit.control_reserved:
            raise GraphValidationError(
                "control_budget_expansion",
                "control task ceilings exceed the root control reserve",
            )


def validate_graph_admission(admission: GraphAdmission) -> None:
    job = admission.job
    graph = admission.graph
    tasks = admission.tasks
    edges = admission.dependencies
    if job.agent_id != graph.agent_id or job.job_id != graph.job_id:
        raise GraphValidationError("ownership", "job and graph ownership differ")
    if job.state not in {
        GraphState.QUEUED,
        GraphState.SUCCEEDED,
        GraphState.FAILED,
        GraphState.CANCELLED,
        GraphState.NEEDS_ATTENTION,
    }:
        raise GraphValidationError(
            "admission_state", "graph admission state is invalid"
        )
    if (
        graph.revision != 0
        or graph.mutation_count != 0
        or graph.active_attempt_count != 0
    ):
        raise GraphValidationError(
            "admission_projection", "new graph projection must be pristine"
        )
    if graph.task_count != len(tasks) or graph.edge_count != len(edges):
        raise GraphValidationError(
            "counter_mismatch", "graph counters do not match rows"
        )
    if graph.topology_digest != topology_digest(tasks, edges):
        raise GraphValidationError(
            "topology_digest", "graph topology digest is invalid"
        )
    finalizers = tuple(task for task in tasks if task.role is TaskRole.FINALIZER)
    if len(finalizers) != 1 or finalizers[0].task_id != job.finalizer_task_id:
        raise GraphValidationError(
            "finalizer_uniqueness", "graph requires exactly one reserved finalizer"
        )
    for task in tasks:
        if task.agent_id != job.agent_id or task.job_id != job.job_id:
            raise GraphValidationError("ownership", "task belongs to another graph")
        require_authority_subset(
            task.specification.authority, job.specification.authority
        )
    for edge in edges:
        if edge.agent_id != job.agent_id or edge.job_id != job.job_id:
            raise GraphValidationError("ownership", "edge belongs to another graph")
    limits = job.specification.limits
    validate_graph_topology(
        tasks,
        edges,
        graph=graph,
        max_tasks=limits.max_tasks,
        max_edges=limits.max_edges,
        max_depth=limits.max_depth,
        max_direct_parents=limits.max_direct_parents,
    )
    parents = defaultdict(set)
    for edge in edges:
        parents[edge.downstream_task_id].add(edge.upstream_task_id)
    for task in tasks:
        if task.state is TaskState.READY and parents[task.task_id]:
            raise GraphValidationError(
                "ready_with_dependencies",
                "new ready task cannot have unsatisfied dependencies",
            )
        if (
            task.state is TaskState.PENDING
            and not parents[task.task_id]
            and not job.terminal
        ):
            raise GraphValidationError(
                "pending_without_dependencies",
                "new root task must be ready rather than pending",
            )
    _validate_budget_envelope(tasks, admission)


def validate_mutation(
    *,
    job_authority: GraphAuthority,
    graph: JobGraph,
    existing_tasks: tuple[GraphTask, ...],
    existing_dependencies: tuple[TaskDependency, ...],
    existing_results: tuple[TaskResult, ...] = (),
    existing_controls: tuple[TaskControl, ...] = (),
    request: GraphMutationRequest,
    creator_authority: GraphAuthority | None = None,
    max_tasks: int,
    max_edges: int,
    max_depth: int,
    max_direct_parents: int,
    max_fan_out: int,
) -> tuple[GraphTask, ...]:
    if not (request.tasks or request.dependencies or request.supersessions):
        raise GraphValidationError(
            "empty_mutation", "graph mutation must make one bounded topology change"
        )
    if graph.finalization_attempt_id is not None:
        raise GraphValidationError(
            "finalization_sealed", "topology mutation is sealed by finalization"
        )
    if request.expected_revision != graph.revision:
        raise GraphValidationError(
            "stale_graph_revision", "mutation expected graph revision is stale"
        )
    if len(request.tasks) > max_fan_out:
        raise GraphValidationError("fan_out_limit", "mutation fan-out exceeded")
    if request.creator_task_id is not None and request.actor_kind != "planner_attempt":
        raise GraphValidationError(
            "mutation_actor", "attempt-bound topology mutation requires planner actor"
        )
    if request.creator_task_id is None and request.actor_kind not in {
        "job_owner",
        "human_policy",
        "owner",
        "supervisor",
    }:
        raise GraphValidationError(
            "mutation_actor", "unbound graph mutation actor is not admitted"
        )
    if request.actor_kind == "supervisor" and (
        len(request.tasks) != 1
        or request.tasks[0].role is not TaskRole.PLANNER
        or request.dependencies
        or request.supersessions
        or request.actor_key != "supervisor_replan"
    ):
        raise GraphValidationError(
            "supervisor_mutation_not_allowed",
            "supervisor mutation is limited to one policy-owned replan task",
        )
    existing_ids = {task.task_id for task in existing_tasks}
    if any(task.task_id in existing_ids for task in request.tasks):
        raise GraphValidationError("duplicate_task", "mutation task already exists")
    if any(task.role is TaskRole.FINALIZER for task in request.tasks):
        raise GraphValidationError(
            "finalizer_uniqueness", "mutation cannot create another finalizer"
        )
    for task in request.tasks:
        if task.agent_id != graph.agent_id or task.job_id != graph.job_id:
            raise GraphValidationError("ownership", "mutation task is foreign")
        if task.state not in {TaskState.PENDING, TaskState.READY}:
            raise GraphValidationError(
                "mutation_task_state", "new mutation task must be pending or ready"
            )
        require_authority_subset(task.specification.authority, job_authority)
        if creator_authority is not None:
            require_authority_subset(task.specification.authority, creator_authority)
        if (
            request.actor_kind != "owner"
            and task.specification.created_by != request.actor_key
        ):
            raise GraphValidationError(
                "mutation_actor", "new task creator differs from mutation actor"
            )
        _validate_contract_binding_coverage(task)
    all_tasks = (*existing_tasks, *request.tasks)
    all_ids = {task.task_id for task in all_tasks}
    existing_pairs = {
        (edge.upstream_task_id, edge.downstream_task_id)
        for edge in existing_dependencies
    }
    for edge in request.dependencies:
        if edge.agent_id != graph.agent_id or edge.job_id != graph.job_id:
            raise GraphValidationError("ownership", "mutation edge is foreign")
        if (
            request.actor_kind != "owner" and edge.creator_key != request.actor_key
        ) or edge.mutation_id not in {None, request.mutation_id}:
            raise GraphValidationError(
                "mutation_actor", "dependency creator differs from mutation actor"
            )
        pair = (edge.upstream_task_id, edge.downstream_task_id)
        if pair in existing_pairs:
            raise GraphValidationError("duplicate_edge", "mutation edge already exists")
        if not set(pair).issubset(all_ids):
            raise GraphValidationError(
                "foreign_edge_endpoint", "mutation edge endpoint is unavailable"
            )
        downstream = next(task for task in all_tasks if task.task_id == pair[1])
        new_ids = {item.task_id for item in request.tasks}
        allowed_states = (
            {TaskState.PENDING, TaskState.READY}
            if downstream.task_id in new_ids
            else {TaskState.PENDING}
        )
        if downstream.attempt_count or downstream.state not in allowed_states:
            raise GraphValidationError(
                "dependency_after_claim",
                "dependencies target only unclaimed pending tasks",
            )
    by_id = {task.task_id: task for task in all_tasks}
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in (*existing_dependencies, *request.dependencies):
        incoming[edge.downstream_task_id].add(edge.upstream_task_id)
    controls_by_id = {control.control_id: control for control in existing_controls}
    if request.actor_kind == "human_policy" and (
        len(request.tasks) != 1
        or len(request.supersessions) != 1
        or request.supersessions[0][1] != request.tasks[0].task_id
    ):
        raise GraphValidationError(
            "human_mutation_not_allowed",
            "human policy may create only one derived replacement task",
        )
    for replaced_id, replacement_id in request.supersessions:
        if replaced_id not in by_id or replacement_id not in by_id:
            raise GraphValidationError(
                "supersession_target", "supersession task is unavailable"
            )
        replaced = by_id[replaced_id]
        replacement = by_id[replacement_id]
        if replaced.role is TaskRole.FINALIZER:
            raise GraphValidationError(
                "finalizer_immutable", "the reserved finalizer cannot be superseded"
            )
        if replaced.state not in {
            TaskState.PENDING,
            TaskState.READY,
            TaskState.BLOCKED,
            TaskState.REVIEW,
        }:
            raise GraphValidationError(
                "supersession_after_claim",
                "running or terminal task cannot be superseded",
            )
        if replacement_id not in {item.task_id for item in request.tasks} or (
            replacement.attempt_count
            or replacement.state not in {TaskState.PENDING, TaskState.READY}
        ):
            raise GraphValidationError(
                "supersession_replacement", "replacement task is not unstarted"
            )
        if replacement.supersedes_task_id not in {None, replaced_id}:
            raise GraphValidationError(
                "supersession_replacement", "replacement link differs from request"
            )
        if not incoming[replaced_id] <= incoming[replacement_id]:
            raise GraphValidationError(
                "supersession_dependency_loss",
                "replacement task must preserve every incoming dependency",
            )
        if (
            request.actor_kind == "human_policy"
            and replaced.state is not TaskState.BLOCKED
        ):
            raise GraphValidationError(
                "human_replacement_not_allowed",
                "human policy replaces only blocked work; review requires a typed "
                "decision",
            )
        if request.actor_kind == "human_policy":
            expected_prefix = (
                replaced.specification.description + "\n\nUntrusted human advisory: "
            )
            replacement_spec = replacement.specification
            if (
                replacement.role is not replaced.role
                or replacement.execution_kind is not replaced.execution_kind
                or replacement.priority != replaced.priority
                or replacement_spec.title != replaced.specification.title
                or not replacement_spec.description.startswith(expected_prefix)
                or not replacement_spec.description.removeprefix(expected_prefix)
                or replacement_spec.expected_result_contract
                != replaced.specification.expected_result_contract
                or replacement_spec.authority != replaced.specification.authority
                or replacement_spec.budgets != replaced.specification.budgets
                or replacement_spec.max_steps != replaced.specification.max_steps
                or replacement_spec.max_wall_time_seconds
                != replaced.specification.max_wall_time_seconds
                or replacement_spec.created_by != request.actor_key
            ):
                raise GraphValidationError(
                    "human_mutation_not_allowed",
                    "human replacement must be derived from the immutable task policy",
                )
        if replaced.state in {TaskState.BLOCKED, TaskState.REVIEW}:
            control = (
                None
                if replaced.latest_control_id is None
                else controls_by_id.get(replaced.latest_control_id)
            )
            if control is None or control.state.value != "open":
                raise GraphValidationError(
                    "supersession_control_missing",
                    "blocked replacement requires its exact open control",
                )
            if (
                request.actor_kind == "planner_attempt"
                and control.kind.value != "needs_replan"
            ):
                raise GraphValidationError(
                    "control_requires_principal",
                    "planner replacement cannot bypass a human-owned control",
                )
    provisional_graph = JobGraph(
        agent_id=graph.agent_id,
        job_id=graph.job_id,
        revision=graph.revision,
        task_count=len(all_tasks),
        edge_count=len(existing_dependencies) + len(request.dependencies),
        mutation_count=graph.mutation_count,
        active_attempt_count=graph.active_attempt_count,
        next_ready_at=graph.next_ready_at,
        finalization_attempt_id=graph.finalization_attempt_id,
        finalization_started_revision=graph.finalization_started_revision,
        created_at=graph.created_at,
        updated_at=graph.updated_at,
        topology_digest=topology_digest(
            tuple(all_tasks), (*existing_dependencies, *request.dependencies)
        ),
    )
    validate_graph_topology(
        tuple(all_tasks),
        (*existing_dependencies, *request.dependencies),
        graph=provisional_graph,
        max_tasks=max_tasks,
        max_edges=max_edges,
        max_depth=max_depth,
        max_direct_parents=max_direct_parents,
    )
    _validate_model_finalizer_join(
        tuple(all_tasks), (*existing_dependencies, *request.dependencies)
    )
    _validate_task_input_references(
        tuple(request.tasks),
        (*existing_dependencies, *request.dependencies),
        existing_results,
    )
    return tuple(all_tasks)


def _validate_contract_binding_coverage(task: GraphTask) -> None:
    authority = task.specification.authority
    material = authority.contract_bindings
    nested_capabilities = material.get("capability_contracts")
    nested_resources = material.get("resource_revisions")
    nested_routes = material.get("model_routes")
    if isinstance(nested_capabilities, Mapping):
        capabilities = nested_capabilities
        resources = nested_resources if isinstance(nested_resources, Mapping) else {}
        routes = nested_routes if isinstance(nested_routes, Mapping) else {}
    else:
        capabilities = material
        resources = material
        routes = material
    if any(
        not isinstance(capabilities.get(item), str) for item in authority.capability_ids
    ):
        raise GraphValidationError(
            "contract_binding_missing",
            "task capability contract coverage is incomplete",
        )
    for resource_id in authority.resource_ids:
        value = resources.get(resource_id)
        if isinstance(value, Mapping):
            value = value.get("resource_revision")
        if not isinstance(value, str):
            raise GraphValidationError(
                "contract_binding_missing",
                "task resource contract coverage is incomplete",
            )
    if any(not isinstance(routes.get(item), str) for item in authority.model_route_ids):
        raise GraphValidationError(
            "contract_binding_missing", "task model-route coverage is incomplete"
        )


def _validate_model_finalizer_join(
    tasks: tuple[GraphTask, ...], dependencies: tuple[TaskDependency, ...]
) -> None:
    finalizer = next(task for task in tasks if task.role is TaskRole.FINALIZER)
    if finalizer.execution_kind is not TaskExecutionKind.MODEL:
        return
    direct = {
        edge.upstream_task_id
        for edge in dependencies
        if edge.downstream_task_id == finalizer.task_id
    }
    reduction_roots = {
        task.task_id
        for task in tasks
        if task.task_id in direct
        and task.specification.expected_result_contract.get("reduction") is True
    }
    parents: dict[str, set[str]] = defaultdict(set)
    for edge in dependencies:
        parents[edge.downstream_task_id].add(edge.upstream_task_id)

    covered = set(direct)
    frontier = list(reduction_roots)
    while frontier:
        current = frontier.pop()
        for parent in parents[current]:
            if parent not in covered:
                covered.add(parent)
                frontier.append(parent)
    required = {
        task.task_id
        for task in tasks
        if task.role is not TaskRole.FINALIZER
        and task.state not in {TaskState.SUPERSEDED, TaskState.SKIPPED}
    }
    if not required <= covered:
        raise GraphValidationError(
            "finalizer_join_unbounded",
            "model finalizer requires explicit bounded reduction ancestry",
        )


def _validate_task_input_references(
    tasks: tuple[GraphTask, ...],
    dependencies: tuple[TaskDependency, ...],
    results: tuple[TaskResult, ...],
) -> None:
    accepted = {result.result_id: result.task_id for result in results}
    parents: dict[str, set[str]] = defaultdict(set)
    for edge in dependencies:
        parents[edge.downstream_task_id].add(edge.upstream_task_id)
    for task in tasks:
        raw = task.specification.expected_result_contract.get("input_result_ids", ())
        if not isinstance(raw, tuple) or any(not isinstance(item, str) for item in raw):
            raise GraphValidationError(
                "input_reference_invalid", "task input result references are malformed"
            )
        if any(
            result_id not in accepted
            or accepted[result_id] not in parents[task.task_id]
            for result_id in raw
        ):
            raise GraphValidationError(
                "input_reference_invalid",
                "task input reference is not an accepted direct upstream result",
            )


def require_current_attempt(
    *,
    task: GraphTask,
    attempt: TaskAttempt,
    claim_token: str,
    fencing_epoch: int,
) -> None:
    if (
        task.state is not TaskState.RUNNING
        or task.current_attempt_id != attempt.attempt_id
        or attempt.state not in ACTIVE_ATTEMPT_STATES
        or attempt.claim_token != claim_token
        or attempt.fencing_epoch != fencing_epoch
        or task.fencing_epoch != fencing_epoch
    ):
        raise GraphValidationError(
            "stale_attempt", "attempt claim token or fencing epoch is stale"
        )


__all__ = [
    "ATTEMPT_TRANSITIONS",
    "GRAPH_TRANSITIONS",
    "TASK_TRANSITIONS",
    "GraphValidationError",
    "require_attempt_transition",
    "require_authority_subset",
    "require_current_attempt",
    "require_graph_transition",
    "require_task_transition",
    "validate_graph_admission",
    "validate_graph_topology",
    "validate_mutation",
]
