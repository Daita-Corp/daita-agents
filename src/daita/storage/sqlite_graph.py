"""Private SQL implementation for draft graph methods on ``SQLiteStateStore``."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from ..artifacts.models import ArtifactRef, artifact_ref_from_mapping
from ..distribution.models import GraphJobDelivery
from ..jobs.graph.models import (
    ACTIVE_ATTEMPT_STATES,
    CONTROL_BUDGET_ROLES,
    MAX_GRAPH_EVENTS,
    MAX_GRAPH_INSPECTION_EVENTS,
    AttemptBudgetReservation,
    AttemptState,
    BudgetAmount,
    BudgetLedger,
    ControlKind,
    ControlState,
    GraphAdmission,
    GraphDesiredState,
    GraphEventPage,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphMutationRequest,
    GraphState,
    GraphTask,
    JobGraph,
    MutationDecision,
    TaskAttempt,
    TaskCheckpoint,
    TaskComment,
    TaskControl,
    TaskDependency,
    TaskResult,
    TaskRole,
    TaskState,
    reserved_artifact_id,
    topology_digest,
)
from ..jobs.graph.validation import (
    GraphValidationError,
    require_attempt_transition,
    require_current_attempt,
    require_graph_transition,
    require_task_transition,
    validate_graph_admission,
    validate_mutation,
)
from .draft_graph_codecs import (
    decode_draft_delivery,
    decode_graph_event,
    decode_graph_job,
    decode_graph_mutation,
    decode_graph_task,
    decode_job_graph,
    decode_task_attempt,
    decode_task_control,
    decode_task_dependency,
    decode_task_result,
    encode_graph_event_payload,
    encode_graph_job,
    encode_graph_job_delivery,
    encode_graph_mutation,
    encode_graph_task,
    encode_job_graph,
    encode_task_attempt,
    encode_task_checkpoint,
    encode_task_comment,
    encode_task_control,
    encode_task_dependency,
    encode_task_result,
)


class GraphStoreConflictError(RuntimeError):
    """A graph CAS or idempotency precondition did not match."""


class GraphBudgetError(ValueError):
    """A graph budget reservation or settlement would violate conservation."""


def datetime_to_us(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("graph timestamp must be timezone-aware UTC")
    delta = value - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def datetime_from_us(value: object, label: str) -> datetime:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"stored {label} timestamp is invalid")
    return datetime.fromtimestamp(value / 1_000_000, tz=UTC)


def optional_datetime_from_us(value: object, label: str) -> datetime | None:
    return None if value is None else datetime_from_us(value, label)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"stored {label} is invalid")
    return value


def _required_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"stored {label} is invalid")
    return value


def _require_projection(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(f"stored graph projection is inconsistent: {label}")


def _load_job(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> tuple[GraphJob, str] | None:
    row = connection.execute(
        """SELECT conversation_id, origin_run_id, origin_call_id, state,
                  desired_state, created_at_us, updated_at_us, deadline_at_us,
                  terminal_at_us, finalizer_task_id, data
           FROM job_runs WHERE agent_id = ? AND job_id = ?""",
        (agent_id, job_id),
    ).fetchone()
    if row is None:
        return None
    data = _required_text(row[10], "graph job payload")
    job = decode_graph_job(data)
    _require_projection(job.agent_id == agent_id and job.job_id == job_id, "job owner")
    _require_projection(job.conversation_id == row[0], "job conversation")
    _require_projection(job.origin_run_id == row[1], "job origin run")
    _require_projection(job.origin_call_id == row[2], "job origin call")
    _require_projection(job.state.value == row[3], "job state")
    _require_projection(job.desired_state.value == row[4], "job desired state")
    _require_projection(datetime_to_us(job.created_at) == row[5], "job created time")
    _require_projection(datetime_to_us(job.updated_at) == row[6], "job updated time")
    _require_projection(datetime_to_us(job.deadline_at) == row[7], "job deadline")
    _require_projection(datetime_to_us(job.terminal_at) == row[8], "job terminal time")
    _require_projection(job.finalizer_task_id == row[9], "job finalizer")
    return job, data


def _load_graph(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> tuple[JobGraph, str] | None:
    row = connection.execute(
        """SELECT revision, task_count, edge_count, mutation_count,
                  active_attempt_count, next_ready_at_us,
                  finalization_attempt_id, finalization_started_revision,
                  created_at_us, updated_at_us, data
           FROM job_graphs WHERE agent_id = ? AND job_id = ?""",
        (agent_id, job_id),
    ).fetchone()
    if row is None:
        return None
    data = _required_text(row[10], "job graph payload")
    graph = decode_job_graph(data)
    _require_projection(
        graph.agent_id == agent_id and graph.job_id == job_id, "graph owner"
    )
    _require_projection(graph.revision == row[0], "graph revision")
    _require_projection(graph.task_count == row[1], "graph task count")
    _require_projection(graph.edge_count == row[2], "graph edge count")
    _require_projection(graph.mutation_count == row[3], "graph mutation count")
    _require_projection(graph.active_attempt_count == row[4], "graph active count")
    _require_projection(
        datetime_to_us(graph.next_ready_at) == row[5], "graph next ready"
    )
    _require_projection(
        graph.finalization_attempt_id == row[6], "graph finalization attempt"
    )
    _require_projection(
        graph.finalization_started_revision == row[7],
        "graph finalization revision",
    )
    _require_projection(datetime_to_us(graph.created_at) == row[8], "graph created")
    _require_projection(datetime_to_us(graph.updated_at) == row[9], "graph updated")
    return graph, data


def _load_task(
    connection: sqlite3.Connection, agent_id: str, job_id: str, task_id: str
) -> tuple[GraphTask, str] | None:
    row = connection.execute(
        """SELECT state, role, task_kind, priority, not_before_us,
                  current_attempt_id, task_revision, task_spec_digest,
                  task_scope_digest, supersedes_task_id, superseded_by_task_id,
                  latest_result_id, latest_control_id, latest_checkpoint_id,
                  created_at_us, updated_at_us, terminal_at_us, data
           FROM job_tasks
           WHERE agent_id = ? AND job_id = ? AND task_id = ?""",
        (agent_id, job_id, task_id),
    ).fetchone()
    if row is None:
        return None
    data = _required_text(row[17], "graph task payload")
    task = decode_graph_task(data)
    _require_projection(
        (task.agent_id, task.job_id, task.task_id) == (agent_id, job_id, task_id),
        "task owner",
    )
    projected = (
        task.state.value,
        task.role.value,
        task.execution_kind.value,
        task.priority,
        datetime_to_us(task.not_before),
        task.current_attempt_id,
        task.task_revision,
        task.task_spec_digest,
        task.task_scope_digest,
        task.supersedes_task_id,
        task.superseded_by_task_id,
        task.latest_result_id,
        task.latest_control_id,
        task.latest_checkpoint_id,
        datetime_to_us(task.created_at),
        datetime_to_us(task.updated_at),
        datetime_to_us(task.terminal_at),
    )
    _require_projection(projected == row[:17], "task columns")
    return task, data


def _load_attempt(
    connection: sqlite3.Connection,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
) -> tuple[TaskAttempt, str] | None:
    row = connection.execute(
        """SELECT ordinal, fencing_epoch, state, claim_token, run_id,
                  lease_expires_at_us, absolute_deadline_at_us, started_at_us,
                  heartbeat_at_us, ended_at_us, active_slot, data
           FROM job_task_attempts
           WHERE agent_id = ? AND job_id = ? AND task_id = ? AND attempt_id = ?""",
        (agent_id, job_id, task_id, attempt_id),
    ).fetchone()
    if row is None:
        return None
    data = _required_text(row[11], "task attempt payload")
    attempt = decode_task_attempt(data)
    _require_projection(
        (attempt.agent_id, attempt.job_id, attempt.task_id, attempt.attempt_id)
        == (agent_id, job_id, task_id, attempt_id),
        "attempt owner",
    )
    projected = (
        attempt.ordinal,
        attempt.fencing_epoch,
        attempt.state.value,
        attempt.claim_token,
        attempt.run_id,
        datetime_to_us(attempt.lease_expires_at),
        datetime_to_us(attempt.absolute_deadline_at),
        datetime_to_us(attempt.started_at),
        datetime_to_us(attempt.heartbeat_at),
        datetime_to_us(attempt.ended_at),
        1 if attempt.state in ACTIVE_ATTEMPT_STATES else None,
    )
    _require_projection(projected == row[:11], "attempt columns")
    return attempt, data


def _load_tasks(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> tuple[GraphTask, ...]:
    ids = tuple(
        _required_text(row[0], "task ID")
        for row in connection.execute(
            "SELECT task_id FROM job_tasks WHERE agent_id = ? AND job_id = ? "
            "ORDER BY task_id",
            (agent_id, job_id),
        )
    )
    tasks = []
    for task_id in ids:
        loaded = _load_task(connection, agent_id, job_id, task_id)
        if loaded is None:
            raise ValueError("stored graph task disappeared during inspection")
        tasks.append(loaded[0])
    return tuple(tasks)


def _load_dependencies(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> tuple[TaskDependency, ...]:
    rows = tuple(
        connection.execute(
            """SELECT upstream_task_id, downstream_task_id, edge_kind,
                      created_at_us, data
               FROM job_task_dependencies
               WHERE agent_id = ? AND job_id = ?
               ORDER BY upstream_task_id, downstream_task_id""",
            (agent_id, job_id),
        )
    )
    decoded: list[TaskDependency] = []
    for upstream, downstream, kind, created_at_us, data in rows:
        edge = decode_task_dependency(_required_text(data, "dependency payload"))
        _require_projection(
            (
                edge.agent_id,
                edge.job_id,
                edge.upstream_task_id,
                edge.downstream_task_id,
                edge.edge_kind.value,
                datetime_to_us(edge.created_at),
            )
            == (agent_id, job_id, upstream, downstream, kind, created_at_us),
            "dependency columns",
        )
        decoded.append(edge)
    return tuple(decoded)


def _load_all_attempts(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> tuple[TaskAttempt, ...]:
    keys = tuple(
        (str(row[0]), str(row[1]))
        for row in connection.execute(
            """SELECT task_id, attempt_id FROM job_task_attempts
               WHERE agent_id = ? AND job_id = ?
               ORDER BY task_id, ordinal""",
            (agent_id, job_id),
        )
    )
    values = []
    for task_id, attempt_id in keys:
        loaded = _load_attempt(connection, agent_id, job_id, task_id, attempt_id)
        if loaded is None:
            raise ValueError("stored task attempt disappeared during inspection")
        values.append(loaded[0])
    return tuple(values)


def _replace_job(
    connection: sqlite3.Connection, current_data: str, job: GraphJob
) -> None:
    result = connection.execute(
        """UPDATE job_runs
           SET state = ?, desired_state = ?, updated_at_us = ?, terminal_at_us = ?,
               finalizer_task_id = ?, data = ?
           WHERE agent_id = ? AND job_id = ? AND data = ?""",
        (
            job.state.value,
            job.desired_state.value,
            datetime_to_us(job.updated_at),
            datetime_to_us(job.terminal_at),
            job.finalizer_task_id,
            encode_graph_job(job),
            job.agent_id,
            job.job_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise GraphStoreConflictError("graph job changed during CAS")


def _replace_graph(
    connection: sqlite3.Connection, current_data: str, graph: JobGraph
) -> None:
    result = connection.execute(
        """UPDATE job_graphs
           SET revision = ?, task_count = ?, edge_count = ?, mutation_count = ?,
               active_attempt_count = ?, next_ready_at_us = ?,
               finalization_attempt_id = ?, finalization_started_revision = ?,
               updated_at_us = ?, data = ?
           WHERE agent_id = ? AND job_id = ? AND data = ?""",
        (
            graph.revision,
            graph.task_count,
            graph.edge_count,
            graph.mutation_count,
            graph.active_attempt_count,
            datetime_to_us(graph.next_ready_at),
            graph.finalization_attempt_id,
            graph.finalization_started_revision,
            datetime_to_us(graph.updated_at),
            encode_job_graph(graph),
            graph.agent_id,
            graph.job_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise GraphStoreConflictError("job graph changed during CAS")


def _replace_task(
    connection: sqlite3.Connection, current_data: str, task: GraphTask
) -> None:
    result = connection.execute(
        """UPDATE job_tasks
           SET state = ?, role = ?, task_kind = ?, priority = ?, not_before_us = ?,
               current_attempt_id = ?, task_revision = ?, task_spec_digest = ?,
               task_scope_digest = ?, supersedes_task_id = ?,
               superseded_by_task_id = ?, latest_result_id = ?,
               latest_control_id = ?, latest_checkpoint_id = ?,
               updated_at_us = ?, terminal_at_us = ?, data = ?
           WHERE agent_id = ? AND job_id = ? AND task_id = ? AND data = ?""",
        (
            task.state.value,
            task.role.value,
            task.execution_kind.value,
            task.priority,
            datetime_to_us(task.not_before),
            task.current_attempt_id,
            task.task_revision,
            task.task_spec_digest,
            task.task_scope_digest,
            task.supersedes_task_id,
            task.superseded_by_task_id,
            task.latest_result_id,
            task.latest_control_id,
            task.latest_checkpoint_id,
            datetime_to_us(task.updated_at),
            datetime_to_us(task.terminal_at),
            encode_graph_task(task),
            task.agent_id,
            task.job_id,
            task.task_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise GraphStoreConflictError("graph task changed during CAS")


def _replace_attempt(
    connection: sqlite3.Connection, current_data: str, attempt: TaskAttempt
) -> None:
    active_slot = 1 if attempt.state in ACTIVE_ATTEMPT_STATES else None
    result = connection.execute(
        """UPDATE job_task_attempts
           SET state = ?, lease_expires_at_us = ?, started_at_us = ?,
               heartbeat_at_us = ?, ended_at_us = ?, active_slot = ?, data = ?
           WHERE agent_id = ? AND job_id = ? AND task_id = ? AND attempt_id = ?
             AND data = ?""",
        (
            attempt.state.value,
            datetime_to_us(attempt.lease_expires_at),
            datetime_to_us(attempt.started_at),
            datetime_to_us(attempt.heartbeat_at),
            datetime_to_us(attempt.ended_at),
            active_slot,
            encode_task_attempt(attempt),
            attempt.agent_id,
            attempt.job_id,
            attempt.task_id,
            attempt.attempt_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise GraphStoreConflictError("task attempt changed during CAS")


def _insert_task(connection: sqlite3.Connection, task: GraphTask) -> None:
    connection.execute(
        """INSERT INTO job_tasks(
               agent_id, job_id, task_id, state, role, task_kind, priority,
               not_before_us, current_attempt_id, task_revision,
               task_spec_digest, task_scope_digest, supersedes_task_id,
               superseded_by_task_id, latest_result_id, latest_control_id,
               latest_checkpoint_id, created_at_us, updated_at_us,
               terminal_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            task.agent_id,
            task.job_id,
            task.task_id,
            task.state.value,
            task.role.value,
            task.execution_kind.value,
            task.priority,
            datetime_to_us(task.not_before),
            task.current_attempt_id,
            task.task_revision,
            task.task_spec_digest,
            task.task_scope_digest,
            task.supersedes_task_id,
            task.superseded_by_task_id,
            task.latest_result_id,
            task.latest_control_id,
            task.latest_checkpoint_id,
            datetime_to_us(task.created_at),
            datetime_to_us(task.updated_at),
            datetime_to_us(task.terminal_at),
            encode_graph_task(task),
        ),
    )


def _insert_dependency(connection: sqlite3.Connection, edge: TaskDependency) -> None:
    connection.execute(
        """INSERT INTO job_task_dependencies(
               agent_id, job_id, upstream_task_id, downstream_task_id,
               edge_kind, created_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            edge.agent_id,
            edge.job_id,
            edge.upstream_task_id,
            edge.downstream_task_id,
            edge.edge_kind.value,
            datetime_to_us(edge.created_at),
            encode_task_dependency(edge),
        ),
    )


def _insert_event(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    kind: str,
    created_at: datetime,
    payload: dict[str, object],
    task_id: str | None = None,
    attempt_id: str | None = None,
    maximum: int = MAX_GRAPH_EVENTS,
) -> int:
    count = connection.execute(
        "SELECT COUNT(*) FROM job_graph_events WHERE agent_id = ? AND job_id = ?",
        (agent_id, job_id),
    ).fetchone()
    if count is None or _required_int(count[0], "graph event count") >= maximum:
        raise GraphValidationError("event_limit", "graph event limit exceeded")
    cursor = connection.execute(
        """INSERT INTO job_graph_events(
               agent_id, job_id, task_id, attempt_id, kind, created_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            agent_id,
            job_id,
            task_id,
            attempt_id,
            kind,
            datetime_to_us(created_at),
            encode_graph_event_payload(kind=kind, payload=payload),
        ),
    ).lastrowid
    if cursor is None:
        raise RuntimeError("graph event insert did not return a cursor")
    return int(cursor)


def _next_ready_at(tasks: tuple[GraphTask, ...]) -> datetime | None:
    ready = tuple(
        task.not_before or task.updated_at
        for task in tasks
        if task.state is TaskState.READY
    )
    return min(ready) if ready else None


def _validate_budget_envelope(job: GraphJob, tasks: tuple[GraphTask, ...]) -> None:
    root = {item.dimension: item for item in job.specification.budgets}
    ordinary: dict[str, int] = defaultdict(int)
    control: dict[str, int] = defaultdict(int)
    for task in tasks:
        for budget in task.specification.budgets:
            limit = root.get(budget.dimension)
            if limit is None:
                raise GraphValidationError(
                    "unknown_budget_dimension",
                    "task budget dimension is absent from the root ledger",
                )
            bucket = control if task.role in CONTROL_BUDGET_ROLES else ordinary
            bucket[budget.dimension] += budget.amount
    for dimension, limit in root.items():
        if ordinary[dimension] > limit.ceiling - limit.control_reserved:
            raise GraphValidationError(
                "worker_budget_expansion", "ordinary task ceilings exceed root budget"
            )
        if control[dimension] > limit.control_reserved:
            raise GraphValidationError(
                "control_budget_expansion", "control task ceilings exceed root reserve"
            )


def admit_graph(connection: sqlite3.Connection, admission: GraphAdmission) -> GraphJob:
    validate_graph_admission(admission)
    job = admission.job
    graph = admission.graph
    if _load_job(connection, job.agent_id, job.job_id) is not None:
        raise ValueError("graph job identity already exists")
    origin = connection.execute(
        """SELECT job_id FROM job_runs
           WHERE agent_id = ? AND origin_run_id = ? AND origin_call_id = ?""",
        (job.agent_id, job.origin_run_id, job.origin_call_id),
    ).fetchone()
    if origin is not None:
        raise ValueError("graph origin identity already exists")
    connection.execute(
        """INSERT INTO job_runs(
               agent_id, job_id, conversation_id, origin_run_id, origin_call_id,
               state, desired_state, created_at_us, updated_at_us, deadline_at_us,
               terminal_at_us, finalizer_task_id, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job.agent_id,
            job.job_id,
            job.conversation_id,
            job.origin_run_id,
            job.origin_call_id,
            job.state.value,
            job.desired_state.value,
            datetime_to_us(job.created_at),
            datetime_to_us(job.updated_at),
            datetime_to_us(job.deadline_at),
            datetime_to_us(job.terminal_at),
            job.finalizer_task_id,
            encode_graph_job(job),
        ),
    )
    connection.execute(
        """INSERT INTO job_graphs(
               agent_id, job_id, revision, task_count, edge_count, mutation_count,
               active_attempt_count, next_ready_at_us, finalization_attempt_id,
               finalization_started_revision, created_at_us, updated_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            graph.agent_id,
            graph.job_id,
            graph.revision,
            graph.task_count,
            graph.edge_count,
            graph.mutation_count,
            graph.active_attempt_count,
            datetime_to_us(graph.next_ready_at),
            graph.finalization_attempt_id,
            graph.finalization_started_revision,
            datetime_to_us(graph.created_at),
            datetime_to_us(graph.updated_at),
            encode_job_graph(graph),
        ),
    )
    for task in admission.tasks:
        _insert_task(connection, task)
    for edge in admission.dependencies:
        _insert_dependency(connection, edge)
    for root_budget in job.specification.budgets:
        connection.execute(
            """INSERT INTO job_budget_ledger(
                   agent_id, job_id, dimension, ceiling, settled, reserved,
                   control_reserved, updated_at_us
               ) VALUES (?, ?, ?, ?, 0, 0, ?, ?)""",
            (
                job.agent_id,
                job.job_id,
                root_budget.dimension,
                root_budget.ceiling,
                root_budget.control_reserved,
                datetime_to_us(job.created_at),
            ),
        )
    for task in admission.tasks:
        for task_budget in task.specification.budgets:
            connection.execute(
                """INSERT INTO job_task_budget_ledger(
                       agent_id, job_id, task_id, dimension, ceiling,
                       settled, reserved, updated_at_us
                   ) VALUES (?, ?, ?, ?, ?, 0, 0, ?)""",
                (
                    task.agent_id,
                    task.job_id,
                    task.task_id,
                    task_budget.dimension,
                    task_budget.amount,
                    datetime_to_us(task.created_at),
                ),
            )
    _insert_event(
        connection,
        agent_id=job.agent_id,
        job_id=job.job_id,
        kind="graph_admitted",
        created_at=job.created_at,
        payload={
            "specification_digest": job.specification_digest,
            "task_count": graph.task_count,
            "edge_count": graph.edge_count,
        },
        maximum=job.specification.limits.max_events,
    )
    return job


def _insert_graph_delivery(
    connection: sqlite3.Connection,
    delivery: GraphJobDelivery,
) -> None:
    connection.execute(
        """INSERT INTO deliveries(
               agent_id, delivery_id, conversation_id, subject_kind, subject_id,
               logical_key, target_kind, target_fingerprint, state,
               created_at_us, data
           ) VALUES (?, ?, ?, 'graph_job', ?, ?, 'conversation_inbox', ?, ?, ?, ?)""",
        (
            delivery.agent_id,
            delivery.delivery_id,
            delivery.conversation_id,
            delivery.job_id,
            delivery.logical_key,
            delivery.target.target_fingerprint,
            delivery.visibility_state.value,
            datetime_to_us(delivery.created_at),
            encode_graph_job_delivery(delivery),
        ),
    )


def inspect_graph(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> GraphInspection | None:
    loaded_job = _load_job(connection, agent_id, job_id)
    if loaded_job is None:
        return None
    loaded_graph = _load_graph(connection, agent_id, job_id)
    if loaded_graph is None:
        raise ValueError("stored graph job has no topology projection")
    tasks = _load_tasks(connection, agent_id, job_id)
    dependencies = _load_dependencies(connection, agent_id, job_id)
    attempts = _load_all_attempts(connection, agent_id, job_id)
    results = tuple(
        decode_task_result(_required_text(row[0], "task result payload"))
        for row in connection.execute(
            """SELECT data FROM job_task_results
               WHERE agent_id = ? AND job_id = ? ORDER BY task_id""",
            (agent_id, job_id),
        )
    )
    controls = tuple(
        decode_task_control(_required_text(row[0], "task control payload"))
        for row in connection.execute(
            """SELECT data FROM job_task_controls
               WHERE agent_id = ? AND job_id = ? ORDER BY created_at_us, control_id""",
            (agent_id, job_id),
        )
    )
    graph = loaded_graph[0]
    _require_projection(graph.task_count == len(tasks), "inspection task count")
    _require_projection(graph.edge_count == len(dependencies), "inspection edge count")
    _require_projection(
        graph.topology_digest == topology_digest(tasks, dependencies),
        "inspection topology digest",
    )
    return GraphInspection(
        job=loaded_job[0],
        graph=graph,
        tasks=tasks,
        dependencies=dependencies,
        attempts=attempts,
        results=results,
        controls=controls,
        budget_ledgers=list_budget_ledgers(connection, agent_id, job_id),
        events=list_graph_events(
            connection,
            agent_id,
            job_id,
            limit=MAX_GRAPH_INSPECTION_EVENTS,
        ).events,
        delivery_ids=tuple(
            item.delivery_id
            for item in list_graph_deliveries(
                connection,
                agent_id,
                job_id=job_id,
                limit=2,
            )
        ),
    )


def list_graph_events(
    connection: sqlite3.Connection,
    agent_id: str,
    job_id: str,
    *,
    after_event_id: int = 0,
    limit: int = MAX_GRAPH_INSPECTION_EVENTS,
) -> GraphEventPage:
    if not 1 <= limit <= MAX_GRAPH_INSPECTION_EVENTS:
        raise ValueError("graph event page limit is outside its bound")
    if after_event_id < 0:
        raise ValueError("graph event cursor must be non-negative")
    if _load_job(connection, agent_id, job_id) is None:
        return GraphEventPage(events=(), next_cursor=None)
    rows = tuple(
        connection.execute(
            """SELECT event_id, task_id, attempt_id, kind, created_at_us, data
               FROM job_graph_events
               WHERE agent_id = ? AND job_id = ? AND event_id > ?
               ORDER BY event_id LIMIT ?""",
            (agent_id, job_id, after_event_id, limit + 1),
        )
    )
    page_rows = rows[:limit]
    events = tuple(
        decode_graph_event(
            _required_text(data, "graph event payload"),
            event_id=_required_int(event_id, "event ID"),
            agent_id=agent_id,
            job_id=job_id,
            task_id=None if task_id is None else str(task_id),
            attempt_id=None if attempt_id is None else str(attempt_id),
            kind=_required_text(kind, "event kind"),
            created_at=datetime_from_us(created_at_us, "event"),
        )
        for event_id, task_id, attempt_id, kind, created_at_us, data in page_rows
    )
    return GraphEventPage(
        events=events,
        next_cursor=events[-1].event_id if len(rows) > limit and events else None,
    )


def list_ready_tasks(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    now: datetime,
    limit: int,
) -> tuple[GraphTask, ...]:
    if not 1 <= limit <= 64:
        raise ValueError("ready-task list limit is outside its bound")
    now_us = datetime_to_us(now)
    rows = tuple(
        connection.execute(
            """SELECT t.job_id, t.task_id
               FROM job_tasks AS t
               JOIN job_runs AS j
                 ON j.agent_id = t.agent_id AND j.job_id = t.job_id
               WHERE t.agent_id = ? AND t.state = 'ready'
                 AND (t.not_before_us IS NULL OR t.not_before_us <= ?)
                 AND j.state IN ('queued','active') AND j.desired_state = 'run'
                 AND j.deadline_at_us > ?
                 AND NOT EXISTS (
                     SELECT 1 FROM job_task_dependencies AS d
                     JOIN job_tasks AS p
                       ON p.agent_id = d.agent_id AND p.job_id = d.job_id
                      AND p.task_id = d.upstream_task_id
                     LEFT JOIN job_task_results AS r
                       ON r.agent_id = p.agent_id AND r.job_id = p.job_id
                      AND r.task_id = p.task_id
                     WHERE d.agent_id = t.agent_id AND d.job_id = t.job_id
                       AND d.downstream_task_id = t.task_id
                       AND (p.state <> 'succeeded' OR r.task_id IS NULL)
                 )
               ORDER BY t.priority DESC, COALESCE(t.not_before_us, t.updated_at_us),
                        t.task_id
               LIMIT ?""",
            (agent_id, now_us, now_us, limit),
        )
    )
    tasks = []
    for job_id, task_id in rows:
        loaded = _load_task(connection, agent_id, str(job_id), str(task_id))
        if loaded is not None:
            tasks.append(loaded[0])
    return tuple(tasks)


def expire_due_graphs(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    expired_at: datetime,
    limit: int = 64,
) -> tuple[GraphJob, ...]:
    """Fail deadline-expired graphs once every active attempt has been fenced."""

    if not 1 <= limit <= 64:
        raise ValueError("expired graph limit is outside its bound")
    rows = tuple(
        connection.execute(
            """SELECT job_id FROM job_runs
               WHERE agent_id = ? AND state IN ('queued','active')
                 AND desired_state = 'run' AND deadline_at_us <= ?
               ORDER BY deadline_at_us, job_id LIMIT ?""",
            (agent_id, datetime_to_us(expired_at), limit),
        )
    )
    expired: list[GraphJob] = []
    for (job_id_raw,) in rows:
        job_id = str(job_id_raw)
        loaded_job = _load_job(connection, agent_id, job_id)
        loaded_graph = _load_graph(connection, agent_id, job_id)
        if loaded_job is None or loaded_graph is None:
            continue
        job, job_data = loaded_job
        graph, graph_data = loaded_graph
        if graph.active_attempt_count != 0:
            continue
        require_graph_transition(job.state, GraphState.FAILED)
        terminal_job = replace(
            job,
            state=GraphState.FAILED,
            updated_at=expired_at,
            terminal_at=expired_at,
            failure_code="deadline_exceeded",
        )
        _replace_job(connection, job_data, terminal_job)
        tasks: list[GraphTask] = []
        for task in _load_tasks(connection, agent_id, job_id):
            if task.state in {TaskState.PENDING, TaskState.READY}:
                require_task_transition(task.state, TaskState.SKIPPED)
                loaded_task = _load_task(connection, agent_id, job_id, task.task_id)
                if loaded_task is None:
                    raise GraphStoreConflictError(
                        "graph task disappeared during expiry"
                    )
                skipped = replace(
                    task,
                    state=TaskState.SKIPPED,
                    task_revision=task.task_revision + 1,
                    updated_at=expired_at,
                    terminal_at=expired_at,
                )
                _replace_task(connection, loaded_task[1], skipped)
                tasks.append(skipped)
            else:
                tasks.append(task)
        terminal_graph = replace(
            graph,
            next_ready_at=_next_ready_at(tuple(tasks)),
            updated_at=expired_at,
        )
        _replace_graph(connection, graph_data, terminal_graph)
        _insert_event(
            connection,
            agent_id=agent_id,
            job_id=job_id,
            kind="graph_deadline_exceeded",
            created_at=expired_at,
            payload={"deadline_at": job.deadline_at.isoformat()},
            maximum=job.specification.limits.max_events,
        )
        expired.append(terminal_job)
    return tuple(expired)


def list_stale_attempts(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    now: datetime,
    limit: int,
) -> tuple[TaskAttempt, ...]:
    if not 1 <= limit <= 64:
        raise ValueError("stale-attempt list limit is outside its bound")
    rows = tuple(
        connection.execute(
            """SELECT job_id, task_id, attempt_id
               FROM job_task_attempts
               WHERE agent_id = ? AND state IN ('claimed','running')
                 AND lease_expires_at_us <= ?
               ORDER BY lease_expires_at_us, attempt_id LIMIT ?""",
            (agent_id, datetime_to_us(now), limit),
        )
    )
    attempts = []
    for job_id, task_id, attempt_id in rows:
        loaded = _load_attempt(
            connection, agent_id, str(job_id), str(task_id), str(attempt_id)
        )
        if loaded is not None:
            attempts.append(loaded[0])
    return tuple(attempts)


def list_active_attempts(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    limit: int,
) -> tuple[TaskAttempt, ...]:
    if not 1 <= limit <= 64:
        raise ValueError("active-attempt list limit is outside its bound")
    rows = tuple(
        connection.execute(
            """SELECT job_id, task_id, attempt_id
               FROM job_task_attempts
               WHERE agent_id = ? AND state IN ('claimed','running')
               ORDER BY absolute_deadline_at_us, attempt_id LIMIT ?""",
            (agent_id, limit),
        )
    )
    attempts = []
    for job_id, task_id, attempt_id in rows:
        loaded = _load_attempt(
            connection, agent_id, str(job_id), str(task_id), str(attempt_id)
        )
        if loaded is not None:
            attempts.append(loaded[0])
    return tuple(attempts)


def _load_graph_delivery(
    connection: sqlite3.Connection,
    agent_id: str,
    job_id: str,
) -> GraphJobDelivery | None:
    row = connection.execute(
        """SELECT delivery_id, conversation_id, subject_kind, subject_id,
                  logical_key, state, data
           FROM deliveries
           WHERE agent_id = ? AND subject_kind = 'graph_job' AND subject_id = ?""",
        (agent_id, job_id),
    ).fetchone()
    if row is None:
        return None
    decoded = decode_draft_delivery(
        _required_text(row[6], "graph delivery payload"),
        agent_id=agent_id,
        delivery_id=_required_text(row[0], "graph delivery ID"),
        conversation_id=_required_text(row[1], "graph delivery conversation"),
        subject_kind=_required_text(row[2], "graph delivery subject kind"),
        subject_id=_required_text(row[3], "graph delivery subject ID"),
        logical_key=_required_text(row[4], "graph delivery logical key"),
        state=_required_text(row[5], "graph delivery state"),
    )
    if not isinstance(decoded, GraphJobDelivery):
        raise ValueError("migrated delivery cannot be a live graph finalization")
    return decoded


def list_graph_deliveries(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    job_id: str | None = None,
    limit: int = 100,
) -> tuple[GraphJobDelivery, ...]:
    if not 1 <= limit <= 100:
        raise ValueError("graph delivery list limit is outside its bound")
    clauses = ["agent_id = ?", "subject_kind = 'graph_job'"]
    parameters: list[object] = [agent_id]
    if job_id is not None:
        clauses.append("subject_id = ?")
        parameters.append(job_id)
    parameters.append(limit)
    rows = tuple(
        connection.execute(
            """SELECT delivery_id, conversation_id, subject_kind, subject_id,
                      logical_key, state, data
               FROM deliveries WHERE """
            + " AND ".join(clauses)
            + " ORDER BY created_at_us, delivery_id LIMIT ?",
            tuple(parameters),
        )
    )
    deliveries: list[GraphJobDelivery] = []
    for row in rows:
        decoded = decode_draft_delivery(
            _required_text(row[6], "graph delivery payload"),
            agent_id=agent_id,
            delivery_id=_required_text(row[0], "graph delivery ID"),
            conversation_id=_required_text(row[1], "graph delivery conversation"),
            subject_kind=_required_text(row[2], "graph delivery subject kind"),
            subject_id=_required_text(row[3], "graph delivery subject ID"),
            logical_key=_required_text(row[4], "graph delivery logical key"),
            state=_required_text(row[5], "graph delivery state"),
        )
        if isinstance(decoded, GraphJobDelivery):
            deliveries.append(decoded)
    return tuple(deliveries)


def list_graph_artifact_refs(
    connection: sqlite3.Connection,
    agent_id: str,
    *,
    run_id: str | None = None,
    conversation_id: str | None = None,
) -> tuple[ArtifactRef, ...]:
    clauses = ["r.agent_id = ?"]
    parameters: list[object] = [agent_id]
    if conversation_id is not None:
        clauses.append("j.conversation_id = ?")
        parameters.append(conversation_id)
    rows = tuple(
        connection.execute(
            """SELECT r.data, j.conversation_id
               FROM job_task_results AS r
               JOIN job_runs AS j
                 ON j.agent_id = r.agent_id AND j.job_id = r.job_id
               WHERE """
            + " AND ".join(clauses)
            + " ORDER BY r.completed_at_us, r.result_id",
            tuple(parameters),
        )
    )
    refs: dict[str, ArtifactRef] = {}
    for data, stored_conversation_id in rows:
        result = decode_task_result(_required_text(data, "task result payload"))
        raw_refs = result.provenance.get("artifact_refs", ())
        if not isinstance(raw_refs, tuple):
            raise ValueError("graph task result artifact references are malformed")
        decoded: list[ArtifactRef] = []
        for raw in raw_refs:
            if not isinstance(raw, Mapping):
                raise ValueError("graph task result artifact reference is malformed")
            decoded.append(artifact_ref_from_mapping(raw))
        if tuple(sorted(item.artifact_id for item in decoded)) != result.artifact_ids:
            raise ValueError("graph task result artifact identities differ")
        for ref in decoded:
            if ref.run_id != result.run_id or ref.conversation_id != str(
                stored_conversation_id
            ):
                raise ValueError("graph task result artifact ownership differs")
            if run_id is not None and ref.run_id != run_id:
                continue
            current = refs.get(ref.artifact_id)
            if current is not None and current != ref:
                raise ValueError("graph task result artifact identity is ambiguous")
            refs[ref.artifact_id] = ref
    return tuple(
        sorted(refs.values(), key=lambda item: (item.created_at, item.artifact_id))
    )


def list_graph_reserved_artifact_ids(
    connection: sqlite3.Connection,
    agent_id: str,
) -> frozenset[tuple[str, str]]:
    attempts = list_active_attempts(connection, agent_id, limit=64)
    return frozenset(
        (attempt.run_id, reserved_artifact_id(attempt.attempt_id))
        for attempt in attempts
    )


def _attempt_binding_is_current(
    connection: sqlite3.Connection, request: GraphMutationRequest
) -> None:
    if request.creator_task_id is None:
        return
    assert request.creator_attempt_id is not None
    assert request.claim_token is not None
    assert request.fencing_epoch is not None
    loaded_task = _load_task(
        connection, request.agent_id, request.job_id, request.creator_task_id
    )
    loaded_attempt = _load_attempt(
        connection,
        request.agent_id,
        request.job_id,
        request.creator_task_id,
        request.creator_attempt_id,
    )
    if loaded_task is None or loaded_attempt is None:
        raise GraphValidationError("stale_attempt", "mutation attempt is unavailable")
    task = loaded_task[0]
    if task.role is not TaskRole.PLANNER:
        raise GraphValidationError(
            "mutation_role", "only a planner attempt may mutate graph topology"
        )
    require_current_attempt(
        task=task,
        attempt=loaded_attempt[0],
        claim_token=request.claim_token,
        fencing_epoch=request.fencing_epoch,
    )


def _insert_mutation(connection: sqlite3.Connection, mutation: GraphMutation) -> None:
    connection.execute(
        """INSERT INTO job_graph_mutations(
               agent_id, job_id, mutation_id, actor_kind, actor_key,
               creator_task_id, creator_attempt_id, idempotency_key,
               payload_digest, expected_revision, committed_revision,
               decision, created_at_us, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            mutation.agent_id,
            mutation.job_id,
            mutation.mutation_id,
            mutation.actor_kind,
            mutation.actor_key,
            mutation.creator_task_id,
            mutation.creator_attempt_id,
            mutation.idempotency_key,
            mutation.payload_digest,
            mutation.expected_revision,
            mutation.committed_revision,
            mutation.decision.value,
            datetime_to_us(mutation.created_at),
            encode_graph_mutation(mutation),
        ),
    )


def apply_mutation(
    connection: sqlite3.Connection, request: GraphMutationRequest
) -> GraphMutation:
    existing = connection.execute(
        """SELECT payload_digest, data FROM job_graph_mutations
           WHERE agent_id = ? AND job_id = ? AND actor_key = ?
             AND idempotency_key = ?""",
        (
            request.agent_id,
            request.job_id,
            request.actor_key,
            request.idempotency_key,
        ),
    ).fetchone()
    if existing is not None:
        if existing[0] != request.payload_digest:
            raise GraphStoreConflictError(
                "graph mutation idempotency key was reused with different content"
            )
        return decode_graph_mutation(_required_text(existing[1], "mutation payload"))
    loaded_job = _load_job(connection, request.agent_id, request.job_id)
    loaded_graph = _load_graph(connection, request.agent_id, request.job_id)
    if loaded_job is None or loaded_graph is None:
        raise GraphValidationError("unknown_graph", "graph job is unavailable")
    job, _ = loaded_job
    graph, graph_data = loaded_graph
    if job.terminal or job.desired_state is GraphDesiredState.CANCEL:
        raise GraphValidationError("terminal_graph", "terminal graph cannot mutate")
    _attempt_binding_is_current(connection, request)
    tasks = _load_tasks(connection, request.agent_id, request.job_id)
    edges = _load_dependencies(connection, request.agent_id, request.job_id)
    limits = job.specification.limits
    if graph.mutation_count >= limits.max_mutations:
        raise GraphValidationError("mutation_limit", "graph mutation limit exceeded")
    try:
        all_tasks = validate_mutation(
            job_authority=job.specification.authority,
            graph=graph,
            existing_tasks=tasks,
            existing_dependencies=edges,
            request=request,
            max_tasks=limits.max_tasks,
            max_edges=limits.max_edges,
            max_depth=limits.max_depth,
            max_direct_parents=limits.max_direct_parents,
            max_fan_out=limits.max_fan_out,
        )
        _validate_budget_envelope(job, all_tasks)
    except GraphValidationError as error:
        rejected = GraphMutation(
            agent_id=request.agent_id,
            job_id=request.job_id,
            mutation_id=request.mutation_id,
            actor_kind=request.actor_kind,
            actor_key=request.actor_key,
            idempotency_key=request.idempotency_key,
            payload_digest=request.payload_digest,
            expected_revision=request.expected_revision,
            committed_revision=graph.revision,
            decision=MutationDecision.REJECTED,
            created_at=request.created_at,
            failure_code=error.code,
            creator_task_id=request.creator_task_id,
            creator_attempt_id=request.creator_attempt_id,
        )
        _insert_mutation(connection, rejected)
        updated_graph = replace(
            graph,
            mutation_count=graph.mutation_count + 1,
            updated_at=request.created_at,
        )
        _replace_graph(connection, graph_data, updated_graph)
        _insert_event(
            connection,
            agent_id=request.agent_id,
            job_id=request.job_id,
            task_id=request.creator_task_id,
            attempt_id=request.creator_attempt_id,
            kind="graph_mutation_rejected",
            created_at=request.created_at,
            payload={"mutation_id": request.mutation_id, "failure_code": error.code},
            maximum=limits.max_events,
        )
        return rejected

    parents = defaultdict(set)
    for edge in (*edges, *request.dependencies):
        parents[edge.downstream_task_id].add(edge.upstream_task_id)
    inserted_tasks: list[GraphTask] = []
    for task in request.tasks:
        desired_state = TaskState.PENDING if parents[task.task_id] else TaskState.READY
        material = (
            task
            if task.state is desired_state
            else replace(
                task, state=desired_state, task_revision=task.task_revision + 1
            )
        )
        _insert_task(connection, material)
        for budget in material.specification.budgets:
            connection.execute(
                """INSERT INTO job_task_budget_ledger(
                       agent_id, job_id, task_id, dimension, ceiling,
                       settled, reserved, updated_at_us
                   ) VALUES (?, ?, ?, ?, ?, 0, 0, ?)""",
                (
                    material.agent_id,
                    material.job_id,
                    material.task_id,
                    budget.dimension,
                    budget.amount,
                    datetime_to_us(request.created_at),
                ),
            )
        inserted_tasks.append(material)
    for edge in request.dependencies:
        _insert_dependency(connection, edge)
    current_tasks = {
        task.task_id: task
        for task in _load_tasks(connection, request.agent_id, request.job_id)
    }
    for replaced_id, replacement_id in request.supersessions:
        replaced_loaded = _load_task(
            connection, request.agent_id, request.job_id, replaced_id
        )
        replacement_loaded = _load_task(
            connection, request.agent_id, request.job_id, replacement_id
        )
        if replaced_loaded is None or replacement_loaded is None:
            raise GraphStoreConflictError("supersession task disappeared")
        replaced_task, replaced_data = replaced_loaded
        replacement_task, replacement_data = replacement_loaded
        _replace_task(
            connection,
            replaced_data,
            replace(
                replaced_task,
                state=TaskState.SUPERSEDED,
                superseded_by_task_id=replacement_id,
                task_revision=replaced_task.task_revision + 1,
                updated_at=request.created_at,
                terminal_at=request.created_at,
            ),
        )
        _replace_task(
            connection,
            replacement_data,
            replace(
                replacement_task,
                supersedes_task_id=replaced_id,
                task_revision=replacement_task.task_revision + 1,
                updated_at=request.created_at,
            ),
        )
        current_tasks[replaced_id] = replace(
            replaced_task,
            state=TaskState.SUPERSEDED,
            superseded_by_task_id=replacement_id,
            task_revision=replaced_task.task_revision + 1,
            updated_at=request.created_at,
            terminal_at=request.created_at,
        )
    final_tasks = _load_tasks(connection, request.agent_id, request.job_id)
    final_edges = _load_dependencies(connection, request.agent_id, request.job_id)
    committed_revision = graph.revision + 1
    mutation = GraphMutation(
        agent_id=request.agent_id,
        job_id=request.job_id,
        mutation_id=request.mutation_id,
        actor_kind=request.actor_kind,
        actor_key=request.actor_key,
        idempotency_key=request.idempotency_key,
        payload_digest=request.payload_digest,
        expected_revision=request.expected_revision,
        committed_revision=committed_revision,
        decision=MutationDecision.COMMITTED,
        created_at=request.created_at,
        resulting_task_ids=tuple(sorted(task.task_id for task in inserted_tasks)),
        resulting_edges=tuple(
            sorted(
                (edge.upstream_task_id, edge.downstream_task_id)
                for edge in request.dependencies
            )
        ),
        creator_task_id=request.creator_task_id,
        creator_attempt_id=request.creator_attempt_id,
    )
    _insert_mutation(connection, mutation)
    updated_graph = replace(
        graph,
        revision=committed_revision,
        task_count=len(final_tasks),
        edge_count=len(final_edges),
        mutation_count=graph.mutation_count + 1,
        next_ready_at=_next_ready_at(final_tasks),
        updated_at=request.created_at,
        topology_digest=topology_digest(final_tasks, final_edges),
    )
    _replace_graph(connection, graph_data, updated_graph)
    _insert_event(
        connection,
        agent_id=request.agent_id,
        job_id=request.job_id,
        task_id=request.creator_task_id,
        attempt_id=request.creator_attempt_id,
        kind="graph_mutation_committed",
        created_at=request.created_at,
        payload={
            "mutation_id": request.mutation_id,
            "committed_revision": committed_revision,
            "task_ids": mutation.resulting_task_ids,
            "edges": mutation.resulting_edges,
        },
        maximum=limits.max_events,
    )
    return mutation


def _dependencies_satisfied(
    connection: sqlite3.Connection, agent_id: str, job_id: str, task_id: str
) -> bool:
    missing = connection.execute(
        """SELECT 1 FROM job_task_dependencies AS d
           JOIN job_tasks AS p
             ON p.agent_id = d.agent_id AND p.job_id = d.job_id
            AND p.task_id = d.upstream_task_id
           LEFT JOIN job_task_results AS r
             ON r.agent_id = p.agent_id AND r.job_id = p.job_id
            AND r.task_id = p.task_id
           WHERE d.agent_id = ? AND d.job_id = ? AND d.downstream_task_id = ?
             AND (p.state <> 'succeeded' OR r.task_id IS NULL)
           LIMIT 1""",
        (agent_id, job_id, task_id),
    ).fetchone()
    return missing is None


def _finalizer_barrier_satisfied(
    connection: sqlite3.Connection, job: GraphJob, task: GraphTask
) -> bool:
    if task.role is not TaskRole.FINALIZER:
        return True
    blocked = connection.execute(
        """SELECT 1 FROM job_tasks AS t
           LEFT JOIN job_task_results AS r
             ON r.agent_id = t.agent_id AND r.job_id = t.job_id
            AND r.task_id = t.task_id
           WHERE t.agent_id = ? AND t.job_id = ? AND t.task_id <> ?
             AND (
                 t.state NOT IN ('succeeded','superseded','skipped')
                 OR (t.state = 'succeeded' AND r.task_id IS NULL)
             )
           LIMIT 1""",
        (job.agent_id, job.job_id, task.task_id),
    ).fetchone()
    controls = connection.execute(
        """SELECT 1 FROM job_task_controls
           WHERE agent_id = ? AND job_id = ? AND state = 'open' LIMIT 1""",
        (job.agent_id, job.job_id),
    ).fetchone()
    return blocked is None and controls is None


def _reserve_budgets(
    connection: sqlite3.Connection,
    *,
    job: GraphJob,
    task: GraphTask,
    attempt_id: str,
    reservations: tuple[BudgetAmount, ...],
    updated_at: datetime,
) -> None:
    material = tuple(reservations)
    if material != tuple(sorted(material)) or len(
        {item.dimension for item in material}
    ) != len(material):
        raise GraphBudgetError("attempt budget reservations must be unique and sorted")
    task_limits = {item.dimension: item.amount for item in task.specification.budgets}
    for reservation in material:
        if reservation.dimension not in task_limits:
            raise GraphBudgetError(
                "attempt reservation dimension is not task-authorized"
            )
        root = connection.execute(
            """SELECT ceiling, settled, reserved, control_reserved
               FROM job_budget_ledger
               WHERE agent_id = ? AND job_id = ? AND dimension = ?""",
            (job.agent_id, job.job_id, reservation.dimension),
        ).fetchone()
        task_row = connection.execute(
            """SELECT ceiling, settled, reserved FROM job_task_budget_ledger
               WHERE agent_id = ? AND job_id = ? AND task_id = ? AND dimension = ?""",
            (job.agent_id, job.job_id, task.task_id, reservation.dimension),
        ).fetchone()
        if root is None or task_row is None:
            raise GraphBudgetError("budget ledger dimension is missing")
        root_ceiling, root_settled, root_reserved, control_reserved = map(int, root)
        task_ceiling, task_settled, task_reserved = map(int, task_row)
        if root_settled + root_reserved + reservation.amount > root_ceiling:
            raise GraphBudgetError("root budget reservation would exceed its ceiling")
        if task_settled + task_reserved + reservation.amount > task_ceiling:
            raise GraphBudgetError("task budget reservation would exceed its ceiling")
        if task.role not in CONTROL_BUDGET_ROLES:
            ordinary = connection.execute(
                """SELECT COALESCE(SUM(l.settled + l.reserved), 0)
                   FROM job_task_budget_ledger AS l
                   JOIN job_tasks AS t
                     ON t.agent_id = l.agent_id AND t.job_id = l.job_id
                    AND t.task_id = l.task_id
                   WHERE l.agent_id = ? AND l.job_id = ? AND l.dimension = ?
                     AND t.role NOT IN ('planner','reviewer','finalizer')""",
                (job.agent_id, job.job_id, reservation.dimension),
            ).fetchone()
            if int(ordinary[0]) + reservation.amount > root_ceiling - control_reserved:
                raise GraphBudgetError("ordinary work cannot consume control reserve")
        else:
            control = connection.execute(
                """SELECT COALESCE(SUM(l.settled + l.reserved), 0)
                   FROM job_task_budget_ledger AS l
                   JOIN job_tasks AS t
                     ON t.agent_id = l.agent_id AND t.job_id = l.job_id
                    AND t.task_id = l.task_id
                   WHERE l.agent_id = ? AND l.job_id = ? AND l.dimension = ?
                     AND t.role IN ('planner','reviewer','finalizer')""",
                (job.agent_id, job.job_id, reservation.dimension),
            ).fetchone()
            if int(control[0]) + reservation.amount > control_reserved:
                raise GraphBudgetError("control work exceeds reserved root budget")
        connection.execute(
            """UPDATE job_budget_ledger SET reserved = reserved + ?, updated_at_us = ?
               WHERE agent_id = ? AND job_id = ? AND dimension = ?""",
            (
                reservation.amount,
                datetime_to_us(updated_at),
                job.agent_id,
                job.job_id,
                reservation.dimension,
            ),
        )
        connection.execute(
            """UPDATE job_task_budget_ledger
               SET reserved = reserved + ?, updated_at_us = ?
               WHERE agent_id = ? AND job_id = ? AND task_id = ? AND dimension = ?""",
            (
                reservation.amount,
                datetime_to_us(updated_at),
                job.agent_id,
                job.job_id,
                task.task_id,
                reservation.dimension,
            ),
        )
        connection.execute(
            """INSERT INTO job_attempt_budget_reservations(
                   agent_id, job_id, task_id, attempt_id, dimension,
                   reserved, settled, updated_at_us
               ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)""",
            (
                job.agent_id,
                job.job_id,
                task.task_id,
                attempt_id,
                reservation.dimension,
                reservation.amount,
                datetime_to_us(updated_at),
            ),
        )


def _settle_budgets(
    connection: sqlite3.Connection,
    *,
    attempt: TaskAttempt,
    usage: tuple[BudgetAmount, ...] | None,
    settled_at: datetime,
) -> tuple[BudgetAmount, ...]:
    rows = tuple(
        connection.execute(
            """SELECT dimension, reserved, settled
               FROM job_attempt_budget_reservations
               WHERE agent_id = ? AND job_id = ? AND task_id = ? AND attempt_id = ?
               ORDER BY dimension""",
            (attempt.agent_id, attempt.job_id, attempt.task_id, attempt.attempt_id),
        )
    )
    if any(row[2] is not None for row in rows):
        raise GraphStoreConflictError("attempt budgets are already settled")
    known: dict[str, int] | None = None
    if usage is not None:
        known = {item.dimension: item.amount for item in usage}
        if tuple(usage) != tuple(sorted(usage)) or len(known) != len(usage):
            raise GraphBudgetError("measured usage must be unique and sorted")
    reserved_dimensions = {str(row[0]) for row in rows}
    if known is not None and not set(known).issubset(reserved_dimensions):
        raise GraphBudgetError("measured usage contains an unreserved dimension")
    measured: list[BudgetAmount] = []
    for dimension_raw, reserved_raw, _ in rows:
        dimension = str(dimension_raw)
        reserved = int(reserved_raw)
        settled = reserved if known is None else known.get(dimension, 0)
        measured.append(BudgetAmount(dimension, settled))
        root = connection.execute(
            """SELECT ceiling, settled, reserved FROM job_budget_ledger
               WHERE agent_id = ? AND job_id = ? AND dimension = ?""",
            (attempt.agent_id, attempt.job_id, dimension),
        ).fetchone()
        task = connection.execute(
            """SELECT ceiling, settled, reserved FROM job_task_budget_ledger
               WHERE agent_id = ? AND job_id = ? AND task_id = ? AND dimension = ?""",
            (attempt.agent_id, attempt.job_id, attempt.task_id, dimension),
        ).fetchone()
        if root is None or task is None:
            raise GraphBudgetError("budget ledger disappeared during settlement")
        if int(root[2]) < reserved or int(task[2]) < reserved:
            raise GraphBudgetError("budget reservation aggregates are inconsistent")
        if int(root[1]) + settled + int(root[2]) - reserved > int(root[0]):
            raise GraphBudgetError("measured root usage exceeds its ceiling")
        if int(task[1]) + settled + int(task[2]) - reserved > int(task[0]):
            raise GraphBudgetError("measured task usage exceeds its ceiling")
        connection.execute(
            """UPDATE job_budget_ledger
               SET settled = settled + ?, reserved = reserved - ?, updated_at_us = ?
               WHERE agent_id = ? AND job_id = ? AND dimension = ?""",
            (
                settled,
                reserved,
                datetime_to_us(settled_at),
                attempt.agent_id,
                attempt.job_id,
                dimension,
            ),
        )
        connection.execute(
            """UPDATE job_task_budget_ledger
               SET settled = settled + ?, reserved = reserved - ?, updated_at_us = ?
               WHERE agent_id = ? AND job_id = ? AND task_id = ? AND dimension = ?""",
            (
                settled,
                reserved,
                datetime_to_us(settled_at),
                attempt.agent_id,
                attempt.job_id,
                attempt.task_id,
                dimension,
            ),
        )
        connection.execute(
            """UPDATE job_attempt_budget_reservations
               SET settled = ?, updated_at_us = ?
               WHERE agent_id = ? AND job_id = ? AND task_id = ?
                 AND attempt_id = ? AND dimension = ? AND settled IS NULL""",
            (
                settled,
                datetime_to_us(settled_at),
                attempt.agent_id,
                attempt.job_id,
                attempt.task_id,
                attempt.attempt_id,
                dimension,
            ),
        )
    return tuple(measured)


def claim_task(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
    claim_token: str,
    run_id: str,
    executor_id: str,
    claimed_at: datetime,
    lease_seconds: int,
    absolute_deadline_at: datetime,
    budget_reservations: tuple[BudgetAmount, ...],
) -> TaskAttempt | None:
    existing = _load_attempt(connection, agent_id, job_id, task_id, attempt_id)
    if existing is not None:
        attempt = existing[0]
        if (
            attempt.claim_token != claim_token
            or attempt.run_id != run_id
            or attempt.executor_id != executor_id
            or attempt.reserved_budgets != budget_reservations
        ):
            raise GraphStoreConflictError("attempt identity was reused")
        return attempt
    loaded_job = _load_job(connection, agent_id, job_id)
    loaded_graph = _load_graph(connection, agent_id, job_id)
    loaded_task = _load_task(connection, agent_id, job_id, task_id)
    if loaded_job is None or loaded_graph is None or loaded_task is None:
        return None
    job, job_data = loaded_job
    graph, graph_data = loaded_graph
    task, task_data = loaded_task
    if (
        job.state not in {GraphState.QUEUED, GraphState.ACTIVE}
        or job.desired_state is not GraphDesiredState.RUN
        or job.deadline_at <= claimed_at
        or task.state is not TaskState.READY
        or (task.not_before is not None and task.not_before > claimed_at)
        or task.attempt_count >= job.specification.limits.max_attempts_per_task
        or graph.active_attempt_count >= job.specification.limits.max_parallelism
        or not _dependencies_satisfied(connection, agent_id, job_id, task_id)
        or not _finalizer_barrier_satisfied(connection, job, task)
    ):
        return None
    if not 1 <= lease_seconds <= 300:
        raise ValueError("task claim lease is outside its bound")
    deadline = min(absolute_deadline_at, job.deadline_at)
    if deadline <= claimed_at:
        return None
    fence = task.fencing_epoch + 1
    attempt = TaskAttempt(
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        attempt_id=attempt_id,
        ordinal=task.attempt_count + 1,
        fencing_epoch=fence,
        state=AttemptState.CLAIMED,
        claim_token=claim_token,
        run_id=run_id,
        lease_expires_at=min(claimed_at + timedelta(seconds=lease_seconds), deadline),
        absolute_deadline_at=deadline,
        started_at=None,
        heartbeat_at=None,
        ended_at=None,
        execution_scope_digest=task.task_scope_digest,
        executor_id=executor_id,
        reserved_budgets=budget_reservations,
    )
    connection.execute(
        """INSERT INTO job_task_attempts(
               agent_id, job_id, task_id, attempt_id, ordinal, fencing_epoch,
               state, claim_token, run_id, lease_expires_at_us,
               absolute_deadline_at_us, started_at_us, heartbeat_at_us,
               ended_at_us, active_slot, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 1, ?)""",
        (
            agent_id,
            job_id,
            task_id,
            attempt_id,
            attempt.ordinal,
            fence,
            attempt.state.value,
            claim_token,
            run_id,
            datetime_to_us(attempt.lease_expires_at),
            datetime_to_us(attempt.absolute_deadline_at),
            encode_task_attempt(attempt),
        ),
    )
    _reserve_budgets(
        connection,
        job=job,
        task=task,
        attempt_id=attempt_id,
        reservations=budget_reservations,
        updated_at=claimed_at,
    )
    claimed_task = replace(
        task,
        state=TaskState.RUNNING,
        current_attempt_id=attempt_id,
        task_revision=task.task_revision + 1,
        attempt_count=task.attempt_count + 1,
        fencing_epoch=fence,
        updated_at=claimed_at,
    )
    _replace_task(connection, task_data, claimed_task)
    active_job = (
        job
        if job.state is GraphState.ACTIVE
        else replace(job, state=GraphState.ACTIVE, updated_at=claimed_at)
    )
    if active_job is not job:
        require_graph_transition(job.state, active_job.state)
        _replace_job(connection, job_data, active_job)
    tasks = tuple(
        claimed_task if item.task_id == task_id else item
        for item in _load_tasks(connection, agent_id, job_id)
    )
    claimed_graph = replace(
        graph,
        active_attempt_count=graph.active_attempt_count + 1,
        next_ready_at=_next_ready_at(tasks),
        finalization_attempt_id=(
            attempt_id
            if task.role is TaskRole.FINALIZER
            else graph.finalization_attempt_id
        ),
        finalization_started_revision=(
            graph.revision
            if task.role is TaskRole.FINALIZER
            else graph.finalization_started_revision
        ),
        updated_at=claimed_at,
    )
    _replace_graph(connection, graph_data, claimed_graph)
    _insert_event(
        connection,
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        attempt_id=attempt_id,
        kind="task_claimed",
        created_at=claimed_at,
        payload={"fencing_epoch": fence, "ordinal": attempt.ordinal},
        maximum=job.specification.limits.max_events,
    )
    return attempt


def start_attempt(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
    claim_token: str,
    fencing_epoch: int,
    started_at: datetime,
) -> TaskAttempt | None:
    loaded_task = _load_task(connection, agent_id, job_id, task_id)
    loaded_attempt = _load_attempt(connection, agent_id, job_id, task_id, attempt_id)
    if loaded_task is None or loaded_attempt is None:
        return None
    task = loaded_task[0]
    attempt, attempt_data = loaded_attempt
    require_current_attempt(
        task=task,
        attempt=attempt,
        claim_token=claim_token,
        fencing_epoch=fencing_epoch,
    )
    if attempt.state is AttemptState.RUNNING:
        return attempt
    require_attempt_transition(attempt.state, AttemptState.RUNNING)
    if started_at >= attempt.absolute_deadline_at:
        return None
    running = replace(
        attempt,
        state=AttemptState.RUNNING,
        started_at=started_at,
        heartbeat_at=started_at,
    )
    _replace_attempt(connection, attempt_data, running)
    return running


def heartbeat_attempt(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
    claim_token: str,
    fencing_epoch: int,
    heartbeat_at: datetime,
    lease_seconds: int,
) -> TaskAttempt | None:
    loaded_task = _load_task(connection, agent_id, job_id, task_id)
    loaded_attempt = _load_attempt(connection, agent_id, job_id, task_id, attempt_id)
    if loaded_task is None or loaded_attempt is None:
        return None
    task = loaded_task[0]
    attempt, attempt_data = loaded_attempt
    require_current_attempt(
        task=task,
        attempt=attempt,
        claim_token=claim_token,
        fencing_epoch=fencing_epoch,
    )
    if attempt.state is not AttemptState.RUNNING:
        raise GraphValidationError("attempt_not_running", "only running attempt renews")
    if not 1 <= lease_seconds <= 30:
        raise ValueError("heartbeat lease is outside its bound")
    prior = attempt.heartbeat_at or attempt.started_at
    if prior is not None and heartbeat_at < prior + timedelta(seconds=10):
        raise ValueError("task heartbeat is rate limited")
    if heartbeat_at >= attempt.absolute_deadline_at:
        return None
    renewed = replace(
        attempt,
        heartbeat_at=heartbeat_at,
        lease_expires_at=min(
            heartbeat_at + timedelta(seconds=lease_seconds),
            attempt.absolute_deadline_at,
        ),
    )
    _replace_attempt(connection, attempt_data, renewed)
    return renewed


def checkpoint_attempt(
    connection: sqlite3.Connection,
    checkpoint: TaskCheckpoint,
    *,
    claim_token: str,
) -> TaskCheckpoint:
    loaded_job = _load_job(connection, checkpoint.agent_id, checkpoint.job_id)
    loaded_task = _load_task(
        connection, checkpoint.agent_id, checkpoint.job_id, checkpoint.task_id
    )
    loaded_attempt = _load_attempt(
        connection,
        checkpoint.agent_id,
        checkpoint.job_id,
        checkpoint.task_id,
        checkpoint.attempt_id,
    )
    if loaded_job is None or loaded_task is None or loaded_attempt is None:
        raise GraphValidationError("stale_attempt", "checkpoint attempt is unavailable")
    task, task_data = loaded_task
    attempt, attempt_data = loaded_attempt
    require_current_attempt(
        task=task,
        attempt=attempt,
        claim_token=claim_token,
        fencing_epoch=checkpoint.fencing_epoch,
    )
    count = connection.execute(
        """SELECT COUNT(*) FROM job_task_checkpoints
           WHERE agent_id = ? AND job_id = ? AND task_id = ? AND attempt_id = ?""",
        (
            checkpoint.agent_id,
            checkpoint.job_id,
            checkpoint.task_id,
            checkpoint.attempt_id,
        ),
    ).fetchone()
    limit = loaded_job[0].specification.limits.max_checkpoints_per_attempt
    if (
        count is None
        or int(count[0]) >= limit
        or checkpoint.ordinal != int(count[0]) + 1
    ):
        raise GraphValidationError(
            "checkpoint_limit", "checkpoint bound or order failed"
        )
    connection.execute(
        """INSERT INTO job_task_checkpoints(
               agent_id, job_id, task_id, attempt_id, checkpoint_id, ordinal,
               created_at_us, payload_digest, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            checkpoint.agent_id,
            checkpoint.job_id,
            checkpoint.task_id,
            checkpoint.attempt_id,
            checkpoint.checkpoint_id,
            checkpoint.ordinal,
            datetime_to_us(checkpoint.created_at),
            checkpoint.payload_digest,
            encode_task_checkpoint(checkpoint),
        ),
    )
    _replace_attempt(
        connection,
        attempt_data,
        replace(
            attempt,
            checkpoint_ids=tuple(
                sorted((*attempt.checkpoint_ids, checkpoint.checkpoint_id))
            ),
        ),
    )
    _replace_task(
        connection,
        task_data,
        replace(
            task,
            latest_checkpoint_id=checkpoint.checkpoint_id,
            task_revision=task.task_revision + 1,
            updated_at=checkpoint.created_at,
        ),
    )
    _insert_event(
        connection,
        agent_id=checkpoint.agent_id,
        job_id=checkpoint.job_id,
        task_id=checkpoint.task_id,
        attempt_id=checkpoint.attempt_id,
        kind="task_checkpointed",
        created_at=checkpoint.created_at,
        payload={
            "checkpoint_id": checkpoint.checkpoint_id,
            "payload_digest": checkpoint.payload_digest,
        },
        maximum=loaded_job[0].specification.limits.max_events,
    )
    return checkpoint


def add_comment(connection: sqlite3.Connection, comment: TaskComment) -> TaskComment:
    loaded_job = _load_job(connection, comment.agent_id, comment.job_id)
    if (
        loaded_job is None
        or _load_task(connection, comment.agent_id, comment.job_id, comment.task_id)
        is None
    ):
        raise GraphValidationError("unknown_task", "comment task is unavailable")
    count = connection.execute(
        """SELECT COUNT(*) FROM job_task_comments
           WHERE agent_id = ? AND job_id = ? AND task_id = ?""",
        (comment.agent_id, comment.job_id, comment.task_id),
    ).fetchone()
    if (
        count is None
        or int(count[0]) >= loaded_job[0].specification.limits.max_comments_per_task
    ):
        raise GraphValidationError("comment_limit", "task comment limit exceeded")
    connection.execute(
        """INSERT INTO job_task_comments(
               agent_id, job_id, task_id, comment_id, author_kind, author_id,
               sensitivity, created_at_us, body_digest, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            comment.agent_id,
            comment.job_id,
            comment.task_id,
            comment.comment_id,
            comment.author_kind,
            comment.author_id,
            comment.sensitivity.value,
            datetime_to_us(comment.created_at),
            comment.body_digest,
            encode_task_comment(comment),
        ),
    )
    _insert_event(
        connection,
        agent_id=comment.agent_id,
        job_id=comment.job_id,
        task_id=comment.task_id,
        kind="task_commented",
        created_at=comment.created_at,
        payload={"comment_id": comment.comment_id},
        maximum=loaded_job[0].specification.limits.max_events,
    )
    return comment


def _promote_ready(
    connection: sqlite3.Connection,
    *,
    job: GraphJob,
    changed_at: datetime,
) -> tuple[GraphTask, ...]:
    tasks = _load_tasks(connection, job.agent_id, job.job_id)
    promoted: list[GraphTask] = []
    for task in tasks:
        if task.state is not TaskState.PENDING:
            continue
        if not _dependencies_satisfied(
            connection, job.agent_id, job.job_id, task.task_id
        ):
            continue
        if not _finalizer_barrier_satisfied(connection, job, task):
            continue
        loaded = _load_task(connection, job.agent_id, job.job_id, task.task_id)
        if loaded is None:
            raise GraphStoreConflictError("ready task disappeared")
        require_task_transition(task.state, TaskState.READY)
        ready = replace(
            task,
            state=TaskState.READY,
            task_revision=task.task_revision + 1,
            updated_at=changed_at,
        )
        _replace_task(connection, loaded[1], ready)
        promoted.append(ready)
    return tuple(promoted)


def complete_attempt(
    connection: sqlite3.Connection,
    result: TaskResult,
    *,
    claim_token: str,
    fencing_epoch: int,
    usage: tuple[BudgetAmount, ...] | None,
    delivery: GraphJobDelivery | None = None,
) -> TaskResult:
    existing_row = connection.execute(
        """SELECT data FROM job_task_results
           WHERE agent_id = ? AND job_id = ? AND task_id = ?""",
        (result.agent_id, result.job_id, result.task_id),
    ).fetchone()
    if existing_row is not None:
        existing = decode_task_result(
            _required_text(existing_row[0], "task result payload")
        )
        if existing == result:
            if (
                delivery is not None
                and _load_graph_delivery(connection, result.agent_id, result.job_id)
                != delivery
            ):
                raise GraphStoreConflictError(
                    "finalization result exists without its exact delivery"
                )
            return existing
        raise GraphStoreConflictError(
            "task completion response was retried with different content"
        )
    loaded_job = _load_job(connection, result.agent_id, result.job_id)
    loaded_graph = _load_graph(connection, result.agent_id, result.job_id)
    loaded_task = _load_task(connection, result.agent_id, result.job_id, result.task_id)
    loaded_attempt = _load_attempt(
        connection,
        result.agent_id,
        result.job_id,
        result.task_id,
        result.attempt_id,
    )
    if (
        loaded_job is None
        or loaded_graph is None
        or loaded_task is None
        or loaded_attempt is None
    ):
        raise GraphValidationError("stale_attempt", "completion attempt is unavailable")
    job, job_data = loaded_job
    graph, graph_data = loaded_graph
    task, task_data = loaded_task
    attempt, attempt_data = loaded_attempt
    require_current_attempt(
        task=task,
        attempt=attempt,
        claim_token=claim_token,
        fencing_epoch=fencing_epoch,
    )
    if attempt.state is not AttemptState.RUNNING:
        raise GraphValidationError(
            "attempt_not_running", "only running attempt completes"
        )
    if result.run_id != attempt.run_id:
        raise GraphValidationError("result_run", "result belongs to another run")
    if result.completed_at > attempt.absolute_deadline_at:
        raise GraphValidationError("attempt_deadline", "result arrived after deadline")
    if (
        result.sensitivity.routing_rank
        < task.specification.authority.sensitivity.routing_rank
    ):
        raise GraphValidationError(
            "result_sensitivity", "result lowers task sensitivity"
        )
    if task.role is TaskRole.FINALIZER and (
        graph.finalization_attempt_id != attempt.attempt_id
        or graph.finalization_started_revision != graph.revision
        or not _finalizer_barrier_satisfied(connection, job, task)
    ):
        raise GraphValidationError("finalizer_seal", "finalizer seal is stale")
    if delivery is not None:
        if task.role is not TaskRole.FINALIZER:
            raise GraphValidationError(
                "delivery_task", "only the finalizer can publish a graph delivery"
            )
        if (
            delivery.agent_id != job.agent_id
            or delivery.job_id != job.job_id
            or delivery.conversation_id != job.conversation_id
            or delivery.outcome.conclusion_id != result.result_id
            or delivery.outcome.conclusion_digest != result.result_digest
            or tuple(item.artifact_id for item in delivery.outcome.artifact_references)
            != result.artifact_ids
            or delivery.outcome.effective_sensitivity != result.sensitivity
        ):
            raise GraphValidationError(
                "delivery_result", "graph delivery differs from the finalizer result"
            )
    connection.execute(
        """INSERT INTO job_task_results(
               agent_id, job_id, task_id, result_id, attempt_id, completed_at_us,
               sensitivity, result_digest, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            result.agent_id,
            result.job_id,
            result.task_id,
            result.result_id,
            result.attempt_id,
            datetime_to_us(result.completed_at),
            result.sensitivity.value,
            result.result_digest,
            encode_task_result(result),
        ),
    )
    measured_usage = _settle_budgets(
        connection, attempt=attempt, usage=usage, settled_at=result.completed_at
    )
    require_attempt_transition(attempt.state, AttemptState.SUCCEEDED)
    settled_attempt = replace(
        attempt,
        state=AttemptState.SUCCEEDED,
        lease_expires_at=None,
        ended_at=result.completed_at,
        measured_usage=measured_usage,
        result_id=result.result_id,
        artifact_ids=result.artifact_ids,
        effect_receipt_ids=result.effect_receipt_ids,
    )
    _replace_attempt(connection, attempt_data, settled_attempt)
    require_task_transition(task.state, TaskState.SUCCEEDED)
    succeeded_task = replace(
        task,
        state=TaskState.SUCCEEDED,
        current_attempt_id=None,
        latest_result_id=result.result_id,
        failure_streak=0,
        task_revision=task.task_revision + 1,
        updated_at=result.completed_at,
        terminal_at=result.completed_at,
    )
    _replace_task(connection, task_data, succeeded_task)
    if task.role is TaskRole.FINALIZER:
        require_graph_transition(job.state, GraphState.SUCCEEDED)
        terminal_job = replace(
            job,
            state=GraphState.SUCCEEDED,
            updated_at=result.completed_at,
            terminal_at=result.completed_at,
            terminal_result_id=result.result_id,
        )
        _replace_job(connection, job_data, terminal_job)
        if delivery is not None:
            _insert_graph_delivery(connection, delivery)
    else:
        _promote_ready(connection, job=job, changed_at=result.completed_at)
    final_tasks = _load_tasks(connection, result.agent_id, result.job_id)
    updated_graph = replace(
        graph,
        active_attempt_count=graph.active_attempt_count - 1,
        next_ready_at=_next_ready_at(final_tasks),
        finalization_attempt_id=(
            None if task.role is TaskRole.FINALIZER else graph.finalization_attempt_id
        ),
        finalization_started_revision=(
            None
            if task.role is TaskRole.FINALIZER
            else graph.finalization_started_revision
        ),
        updated_at=result.completed_at,
    )
    _replace_graph(connection, graph_data, updated_graph)
    _insert_event(
        connection,
        agent_id=result.agent_id,
        job_id=result.job_id,
        task_id=result.task_id,
        attempt_id=result.attempt_id,
        kind=(
            "graph_succeeded" if task.role is TaskRole.FINALIZER else "task_succeeded"
        ),
        created_at=result.completed_at,
        payload={"result_id": result.result_id, "result_digest": result.result_digest},
        maximum=job.specification.limits.max_events,
    )
    return result


def fence_attempt(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
    fencing_epoch: int,
    fenced_at: datetime,
    requeue: bool,
    reason_code: str,
) -> TaskAttempt | None:
    loaded_job = _load_job(connection, agent_id, job_id)
    loaded_graph = _load_graph(connection, agent_id, job_id)
    loaded_task = _load_task(connection, agent_id, job_id, task_id)
    loaded_attempt = _load_attempt(connection, agent_id, job_id, task_id, attempt_id)
    if (
        loaded_job is None
        or loaded_graph is None
        or loaded_task is None
        or loaded_attempt is None
    ):
        return None
    job = loaded_job[0]
    graph, graph_data = loaded_graph
    task, task_data = loaded_task
    attempt, attempt_data = loaded_attempt
    if (
        task.current_attempt_id != attempt_id
        or task.fencing_epoch != fencing_epoch
        or attempt.fencing_epoch != fencing_epoch
        or attempt.state not in ACTIVE_ATTEMPT_STATES
    ):
        return None
    measured_usage = _settle_budgets(
        connection, attempt=attempt, usage=None, settled_at=fenced_at
    )
    require_attempt_transition(attempt.state, AttemptState.FENCED)
    fenced = replace(
        attempt,
        state=AttemptState.FENCED,
        lease_expires_at=None,
        ended_at=fenced_at,
        error_code=reason_code,
        measured_usage=measured_usage,
    )
    _replace_attempt(connection, attempt_data, fenced)
    can_retry = (
        requeue
        and job.desired_state is GraphDesiredState.RUN
        and task.attempt_count < job.specification.limits.max_attempts_per_task
        and job.deadline_at > fenced_at
    )
    next_state = TaskState.READY if can_retry else TaskState.FAILED
    require_task_transition(task.state, next_state)
    updated_task = replace(
        task,
        state=next_state,
        current_attempt_id=None,
        fencing_epoch=fencing_epoch + 1,
        failure_streak=task.failure_streak + 1,
        task_revision=task.task_revision + 1,
        not_before=(
            (fenced_at + timedelta(seconds=1 if task.attempt_count == 1 else 5))
            if can_retry
            else task.not_before
        ),
        updated_at=fenced_at,
        terminal_at=None if can_retry else fenced_at,
    )
    _replace_task(connection, task_data, updated_task)
    tasks = tuple(
        updated_task if item.task_id == task_id else item
        for item in _load_tasks(connection, agent_id, job_id)
    )
    updated_graph = replace(
        graph,
        active_attempt_count=graph.active_attempt_count - 1,
        next_ready_at=_next_ready_at(tasks),
        finalization_attempt_id=(
            None
            if graph.finalization_attempt_id == attempt_id
            else graph.finalization_attempt_id
        ),
        finalization_started_revision=(
            None
            if graph.finalization_attempt_id == attempt_id
            else graph.finalization_started_revision
        ),
        updated_at=fenced_at,
    )
    _replace_graph(connection, graph_data, updated_graph)
    _insert_event(
        connection,
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        attempt_id=attempt_id,
        kind="task_attempt_fenced",
        created_at=fenced_at,
        payload={"fencing_epoch": fencing_epoch, "requeued": can_retry},
        maximum=job.specification.limits.max_events,
    )
    return fenced


def fail_attempt(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
    claim_token: str,
    fencing_epoch: int,
    failed_at: datetime,
    retryable: bool,
    reason_code: str,
) -> TaskAttempt | None:
    loaded_job = _load_job(connection, agent_id, job_id)
    loaded_graph = _load_graph(connection, agent_id, job_id)
    loaded_task = _load_task(connection, agent_id, job_id, task_id)
    loaded_attempt = _load_attempt(connection, agent_id, job_id, task_id, attempt_id)
    if (
        loaded_job is None
        or loaded_graph is None
        or loaded_task is None
        or loaded_attempt is None
    ):
        return None
    job, job_data = loaded_job
    graph, graph_data = loaded_graph
    task, task_data = loaded_task
    attempt, attempt_data = loaded_attempt
    require_current_attempt(
        task=task,
        attempt=attempt,
        claim_token=claim_token,
        fencing_epoch=fencing_epoch,
    )
    if attempt.state not in ACTIVE_ATTEMPT_STATES:
        return attempt
    measured_usage = _settle_budgets(
        connection, attempt=attempt, usage=None, settled_at=failed_at
    )
    require_attempt_transition(attempt.state, AttemptState.FAILED)
    failed_attempt = replace(
        attempt,
        state=AttemptState.FAILED,
        lease_expires_at=None,
        ended_at=failed_at,
        error_code=reason_code,
        diagnostic="The internal graph attempt failed within its bounded contract.",
        measured_usage=measured_usage,
    )
    _replace_attempt(connection, attempt_data, failed_attempt)
    can_retry = (
        retryable
        and job.desired_state is GraphDesiredState.RUN
        and task.attempt_count < job.specification.limits.max_attempts_per_task
        and job.deadline_at > failed_at
    )
    next_state = TaskState.READY if can_retry else TaskState.FAILED
    require_task_transition(task.state, next_state)
    updated_task = replace(
        task,
        state=next_state,
        current_attempt_id=None,
        failure_streak=task.failure_streak + 1,
        task_revision=task.task_revision + 1,
        not_before=(
            failed_at + timedelta(seconds=1 if task.attempt_count == 1 else 5)
            if can_retry
            else task.not_before
        ),
        updated_at=failed_at,
        terminal_at=None if can_retry else failed_at,
    )
    _replace_task(connection, task_data, updated_task)
    updated_job = job
    if not can_retry:
        require_graph_transition(job.state, GraphState.FAILED)
        updated_job = replace(
            job,
            state=GraphState.FAILED,
            updated_at=failed_at,
            terminal_at=failed_at,
            failure_code=reason_code,
        )
        _replace_job(connection, job_data, updated_job)
    tasks = tuple(
        updated_task if item.task_id == task_id else item
        for item in _load_tasks(connection, agent_id, job_id)
    )
    updated_graph = replace(
        graph,
        active_attempt_count=graph.active_attempt_count - 1,
        next_ready_at=_next_ready_at(tasks),
        finalization_attempt_id=(
            None
            if graph.finalization_attempt_id == attempt_id
            else graph.finalization_attempt_id
        ),
        finalization_started_revision=(
            None
            if graph.finalization_attempt_id == attempt_id
            else graph.finalization_started_revision
        ),
        updated_at=failed_at,
    )
    _replace_graph(connection, graph_data, updated_graph)
    _insert_event(
        connection,
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        attempt_id=attempt_id,
        kind="task_attempt_failed",
        created_at=failed_at,
        payload={"reason_code": reason_code, "requeued": can_retry},
        maximum=updated_job.specification.limits.max_events,
    )
    return failed_attempt


def open_control(
    connection: sqlite3.Connection,
    control: TaskControl,
    *,
    claim_token: str,
    fencing_epoch: int,
) -> TaskControl:
    if control.state is not ControlState.OPEN or control.requesting_attempt_id is None:
        raise ValueError("new task control must be open and attempt-bound")
    loaded_job = _load_job(connection, control.agent_id, control.job_id)
    loaded_graph = _load_graph(connection, control.agent_id, control.job_id)
    loaded_task = _load_task(
        connection, control.agent_id, control.job_id, control.task_id
    )
    loaded_attempt = _load_attempt(
        connection,
        control.agent_id,
        control.job_id,
        control.task_id,
        control.requesting_attempt_id,
    )
    if (
        loaded_job is None
        or loaded_graph is None
        or loaded_task is None
        or loaded_attempt is None
    ):
        raise GraphValidationError("stale_attempt", "control attempt is unavailable")
    job, job_data = loaded_job
    graph, graph_data = loaded_graph
    task, task_data = loaded_task
    attempt, attempt_data = loaded_attempt
    require_current_attempt(
        task=task,
        attempt=attempt,
        claim_token=claim_token,
        fencing_epoch=fencing_epoch,
    )
    if attempt.state is not AttemptState.RUNNING:
        raise GraphValidationError(
            "attempt_not_running", "control requires running attempt"
        )
    count = connection.execute(
        """SELECT COUNT(*) FROM job_task_controls
           WHERE agent_id = ? AND job_id = ? AND task_id = ?""",
        (control.agent_id, control.job_id, control.task_id),
    ).fetchone()
    if count is None or int(count[0]) >= job.specification.limits.max_controls_per_task:
        raise GraphValidationError("control_limit", "task control limit exceeded")
    connection.execute(
        """INSERT INTO job_task_controls(
               agent_id, job_id, task_id, control_id, kind, state,
               requesting_attempt_id, created_at_us, resolved_at_us,
               resolved_by_kind, resolved_by_id, payload_digest, data
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)""",
        (
            control.agent_id,
            control.job_id,
            control.task_id,
            control.control_id,
            control.kind.value,
            control.state.value,
            control.requesting_attempt_id,
            datetime_to_us(control.created_at),
            control.payload_digest,
            encode_task_control(control),
        ),
    )
    attempt_state = (
        AttemptState.REVIEW_REQUESTED
        if control.kind is ControlKind.REVIEW_REQUESTED
        else AttemptState.BLOCKED
    )
    require_attempt_transition(attempt.state, attempt_state)
    measured_usage = _settle_budgets(
        connection, attempt=attempt, usage=None, settled_at=control.created_at
    )
    _replace_attempt(
        connection,
        attempt_data,
        replace(
            attempt,
            state=attempt_state,
            lease_expires_at=None,
            ended_at=control.created_at,
            measured_usage=measured_usage,
            control_ids=tuple(sorted((*attempt.control_ids, control.control_id))),
        ),
    )
    task_state = (
        TaskState.REVIEW
        if control.kind is ControlKind.REVIEW_REQUESTED
        else TaskState.BLOCKED
    )
    require_task_transition(task.state, task_state)
    _replace_task(
        connection,
        task_data,
        replace(
            task,
            state=task_state,
            current_attempt_id=None,
            latest_control_id=control.control_id,
            task_revision=task.task_revision + 1,
            updated_at=control.created_at,
        ),
    )
    target_graph_state = (
        GraphState.NEEDS_ATTENTION
        if control.kind
        in {
            ControlKind.NEEDS_AUTHORIZATION,
            ControlKind.EFFECT_UNCERTAIN,
            ControlKind.SOURCE_OR_CONTRACT_DRIFT,
            ControlKind.RETRY_CIRCUIT_OPEN,
        }
        else GraphState.BLOCKED
    )
    if job.state is not target_graph_state:
        require_graph_transition(job.state, target_graph_state)
        _replace_job(
            connection,
            job_data,
            replace(job, state=target_graph_state, updated_at=control.created_at),
        )
    updated_graph = replace(
        graph,
        active_attempt_count=graph.active_attempt_count - 1,
        finalization_attempt_id=(
            None
            if graph.finalization_attempt_id == attempt.attempt_id
            else graph.finalization_attempt_id
        ),
        finalization_started_revision=(
            None
            if graph.finalization_attempt_id == attempt.attempt_id
            else graph.finalization_started_revision
        ),
        next_ready_at=None,
        updated_at=control.created_at,
    )
    _replace_graph(connection, graph_data, updated_graph)
    _insert_event(
        connection,
        agent_id=control.agent_id,
        job_id=control.job_id,
        task_id=control.task_id,
        attempt_id=control.requesting_attempt_id,
        kind="task_control_opened",
        created_at=control.created_at,
        payload={"control_id": control.control_id, "kind": control.kind.value},
        maximum=job.specification.limits.max_events,
    )
    return control


def resolve_control(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    job_id: str,
    task_id: str,
    control_id: str,
    state: ControlState,
    resolved_at: datetime,
    resolved_by_kind: str,
    resolved_by_id: str,
    resolution: dict[str, object],
    make_ready: bool,
) -> TaskControl | None:
    row = connection.execute(
        """SELECT data FROM job_task_controls
           WHERE agent_id = ? AND job_id = ? AND task_id = ? AND control_id = ?""",
        (agent_id, job_id, task_id, control_id),
    ).fetchone()
    if row is None:
        return None
    current_data = _required_text(row[0], "task control payload")
    current = decode_task_control(current_data)
    if current.state is not ControlState.OPEN:
        return current
    if state is ControlState.OPEN:
        raise ValueError("control resolution must be terminal")
    resolved = replace(
        current,
        state=state,
        resolved_at=resolved_at,
        resolved_by_kind=resolved_by_kind,
        resolved_by_id=resolved_by_id,
        resolution=resolution,
    )
    result = connection.execute(
        """UPDATE job_task_controls
           SET state = ?, resolved_at_us = ?, resolved_by_kind = ?,
               resolved_by_id = ?, data = ?
           WHERE agent_id = ? AND job_id = ? AND task_id = ? AND control_id = ?
             AND data = ?""",
        (
            resolved.state.value,
            datetime_to_us(resolved.resolved_at),
            resolved.resolved_by_kind,
            resolved.resolved_by_id,
            encode_task_control(resolved),
            agent_id,
            job_id,
            task_id,
            control_id,
            current_data,
        ),
    )
    if result.rowcount != 1:
        raise GraphStoreConflictError("task control changed during resolution")
    loaded_job = _load_job(connection, agent_id, job_id)
    loaded_graph = _load_graph(connection, agent_id, job_id)
    loaded_task = _load_task(connection, agent_id, job_id, task_id)
    if loaded_job is None or loaded_graph is None or loaded_task is None:
        raise GraphStoreConflictError("control owner disappeared")
    job, job_data = loaded_job
    graph, graph_data = loaded_graph
    task, task_data = loaded_task
    next_state = (
        TaskState.READY
        if make_ready and state is ControlState.RESOLVED
        else TaskState.FAILED
    )
    require_task_transition(task.state, next_state)
    updated_task = replace(
        task,
        state=next_state,
        latest_control_id=control_id,
        task_revision=task.task_revision + 1,
        updated_at=resolved_at,
        terminal_at=None if next_state is TaskState.READY else resolved_at,
    )
    _replace_task(connection, task_data, updated_task)
    target_job_state = (
        GraphState.ACTIVE
        if next_state is TaskState.READY
        else GraphState.NEEDS_ATTENTION
    )
    if job.state is not target_job_state:
        require_graph_transition(job.state, target_job_state)
        _replace_job(
            connection,
            job_data,
            replace(job, state=target_job_state, updated_at=resolved_at),
        )
    tasks = tuple(
        updated_task if item.task_id == task_id else item
        for item in _load_tasks(connection, agent_id, job_id)
    )
    _replace_graph(
        connection,
        graph_data,
        replace(graph, next_ready_at=_next_ready_at(tasks), updated_at=resolved_at),
    )
    _insert_event(
        connection,
        agent_id=agent_id,
        job_id=job_id,
        task_id=task_id,
        kind="task_control_resolved",
        created_at=resolved_at,
        payload={"control_id": control_id, "state": state.value},
        maximum=job.specification.limits.max_events,
    )
    return resolved


def list_budget_ledgers(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> tuple[BudgetLedger, ...]:
    rows = tuple(
        connection.execute(
            """SELECT dimension, ceiling, settled, reserved, control_reserved,
                      updated_at_us
               FROM job_budget_ledger WHERE agent_id = ? AND job_id = ?
               ORDER BY dimension""",
            (agent_id, job_id),
        )
    )
    return tuple(
        BudgetLedger(
            agent_id=agent_id,
            job_id=job_id,
            dimension=str(dimension),
            ceiling=int(ceiling),
            settled=int(settled),
            reserved=int(reserved),
            control_reserved=int(control_reserved),
            updated_at=datetime_from_us(updated_at_us, "root budget"),
        )
        for dimension, ceiling, settled, reserved, control_reserved, updated_at_us in rows
    )


def list_attempt_reservations(
    connection: sqlite3.Connection,
    agent_id: str,
    job_id: str,
    task_id: str,
    attempt_id: str,
) -> tuple[AttemptBudgetReservation, ...]:
    rows = tuple(
        connection.execute(
            """SELECT dimension, reserved, settled, updated_at_us
               FROM job_attempt_budget_reservations
               WHERE agent_id = ? AND job_id = ? AND task_id = ? AND attempt_id = ?
               ORDER BY dimension""",
            (agent_id, job_id, task_id, attempt_id),
        )
    )
    return tuple(
        AttemptBudgetReservation(
            agent_id=agent_id,
            job_id=job_id,
            task_id=task_id,
            attempt_id=attempt_id,
            dimension=str(dimension),
            reserved=int(reserved),
            settled=None if settled is None else int(settled),
            updated_at=datetime_from_us(updated_at_us, "attempt budget"),
        )
        for dimension, reserved, settled, updated_at_us in rows
    )


__all__ = [
    "GraphBudgetError",
    "GraphStoreConflictError",
    "add_comment",
    "admit_graph",
    "apply_mutation",
    "checkpoint_attempt",
    "claim_task",
    "complete_attempt",
    "datetime_from_us",
    "datetime_to_us",
    "expire_due_graphs",
    "fail_attempt",
    "fence_attempt",
    "heartbeat_attempt",
    "inspect_graph",
    "list_active_attempts",
    "list_attempt_reservations",
    "list_budget_ledgers",
    "list_graph_artifact_refs",
    "list_graph_deliveries",
    "list_graph_events",
    "list_graph_reserved_artifact_ids",
    "list_ready_tasks",
    "list_stale_attempts",
    "open_control",
    "resolve_control",
    "start_attempt",
]
