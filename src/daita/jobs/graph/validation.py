"""Pure invariant validation for the jobs-owned draft graph model."""

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
    TaskDependency,
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
    request: GraphMutationRequest,
    max_tasks: int,
    max_edges: int,
    max_depth: int,
    max_direct_parents: int,
    max_fan_out: int,
) -> tuple[GraphTask, ...]:
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
    all_tasks = (*existing_tasks, *request.tasks)
    all_ids = {task.task_id for task in all_tasks}
    existing_pairs = {
        (edge.upstream_task_id, edge.downstream_task_id)
        for edge in existing_dependencies
    }
    for edge in request.dependencies:
        if edge.agent_id != graph.agent_id or edge.job_id != graph.job_id:
            raise GraphValidationError("ownership", "mutation edge is foreign")
        pair = (edge.upstream_task_id, edge.downstream_task_id)
        if pair in existing_pairs:
            raise GraphValidationError("duplicate_edge", "mutation edge already exists")
        if not set(pair).issubset(all_ids):
            raise GraphValidationError(
                "foreign_edge_endpoint", "mutation edge endpoint is unavailable"
            )
        downstream = next(task for task in all_tasks if task.task_id == pair[1])
        if downstream.attempt_count or downstream.state not in {
            TaskState.PENDING,
            TaskState.READY,
        }:
            raise GraphValidationError(
                "dependency_after_claim",
                "dependencies target only unclaimed pending/ready tasks",
            )
    by_id = {task.task_id: task for task in all_tasks}
    for replaced_id, replacement_id in request.supersessions:
        if replaced_id not in by_id or replacement_id not in by_id:
            raise GraphValidationError(
                "supersession_target", "supersession task is unavailable"
            )
        replaced = by_id[replaced_id]
        replacement = by_id[replacement_id]
        if replaced.attempt_count or replaced.state not in {
            TaskState.PENDING,
            TaskState.READY,
            TaskState.BLOCKED,
        }:
            raise GraphValidationError(
                "supersession_after_claim", "claimed task cannot be superseded"
            )
        if replacement.state not in {TaskState.PENDING, TaskState.READY}:
            raise GraphValidationError(
                "supersession_replacement", "replacement task is not unstarted"
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
    return tuple(all_tasks)


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
