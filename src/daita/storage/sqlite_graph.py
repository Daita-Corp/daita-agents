"""Frozen graph store that the released revision-2 migration runs.

Revision 2 converts revision-1 jobs into the task graph that revision 4 then
turns into checklist jobs. Its checksummed conversion imports this module, the
graph records and their codec by path; nothing at runtime uses them.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from ..jobs.graph.models import (
    ACTIVE_ATTEMPT_STATES,
    MAX_GRAPH_EVENTS,
    MAX_GRAPH_INSPECTION_EVENTS,
    BudgetLedger,
    GraphAdmission,
    GraphEventPage,
    GraphInspection,
    GraphJob,
    GraphTask,
    JobGraph,
    TaskAttempt,
    TaskDependency,
    topology_digest,
)
from ..jobs.graph.validation import GraphValidationError, validate_graph_admission
from .errors import (
    GraphBudgetError as GraphBudgetError,
    GraphStoreConflictError as GraphStoreConflictError,
)
from .sql_connection import SQLConnection
from .sqlite_codecs.graph import (
    decode_graph_event,
    decode_graph_job,
    decode_graph_task,
    decode_job_graph,
    decode_task_attempt,
    decode_task_checkpoint,
    decode_task_comment,
    decode_task_control,
    decode_task_dependency,
    decode_task_result,
    encode_graph_event_payload,
    encode_graph_job,
    encode_graph_task,
    encode_job_graph,
    encode_task_dependency,
)


def admit_graph(connection: sqlite3.Connection, admission: GraphAdmission) -> GraphJob:
    from .sqlite import _SQLiteConnection

    return _admit_graph(_SQLiteConnection(connection), admission)


def inspect_graph(
    connection: sqlite3.Connection, agent_id: str, job_id: str
) -> GraphInspection | None:
    from .sqlite import _SQLiteConnection

    return _inspect_graph(_SQLiteConnection(connection), agent_id, job_id)


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
    connection: SQLConnection, agent_id: str, job_id: str
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
    connection: SQLConnection, agent_id: str, job_id: str
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
    connection: SQLConnection, agent_id: str, job_id: str, task_id: str
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
    connection: SQLConnection,
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
    connection: SQLConnection, agent_id: str, job_id: str
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
    connection: SQLConnection, agent_id: str, job_id: str
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
    connection: SQLConnection, agent_id: str, job_id: str
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


def _insert_task(connection: SQLConnection, task: GraphTask) -> None:
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


def _insert_dependency(connection: SQLConnection, edge: TaskDependency) -> None:
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
    connection: SQLConnection,
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
           ) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING event_id""",
        (
            agent_id,
            job_id,
            task_id,
            attempt_id,
            kind,
            datetime_to_us(created_at),
            encode_graph_event_payload(kind=kind, payload=payload),
        ),
    ).fetchone()
    if cursor is None:
        raise RuntimeError("graph event insert did not return a cursor")
    return int(cursor[0])


def _admit_graph(connection: SQLConnection, admission: GraphAdmission) -> GraphJob:
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


def _inspect_graph(
    connection: SQLConnection, agent_id: str, job_id: str
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
    checkpoints = tuple(
        decode_task_checkpoint(_required_text(row[0], "task checkpoint payload"))
        for row in connection.execute(
            """SELECT data FROM job_task_checkpoints
               WHERE agent_id = ? AND job_id = ?
               ORDER BY created_at_us, checkpoint_id""",
            (agent_id, job_id),
        )
    )
    comments = tuple(
        decode_task_comment(_required_text(row[0], "task comment payload"))
        for row in connection.execute(
            """SELECT data FROM job_task_comments
               WHERE agent_id = ? AND job_id = ?
               ORDER BY created_at_us, comment_id""",
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
        checkpoints=checkpoints,
        comments=comments,
        budget_ledgers=(
            *list_budget_ledgers(connection, agent_id, job_id),
            *_list_task_budget_ledgers(connection, agent_id, job_id),
        ),
        events=list_graph_events(
            connection,
            agent_id,
            job_id,
            limit=MAX_GRAPH_INSPECTION_EVENTS,
        ).events,
        delivery_ids=tuple(
            str(row[0])
            for row in connection.execute(
                """SELECT delivery_id FROM deliveries
                   WHERE agent_id = ? AND (
                       (subject_kind = 'graph_job' AND subject_id = ?)
                       OR (subject_kind = 'graph_attention'
                           AND substr(subject_id, 1, length(?) + 1) = ? || '/')
                   )
                   ORDER BY created_at_us, delivery_id LIMIT 64""",
                (agent_id, job_id, job_id, job_id),
            )
        ),
    )


def list_graph_events(
    connection: SQLConnection,
    agent_id: str,
    job_id: str,
    *,
    after_event_id: int = 0,
    limit: int = MAX_GRAPH_INSPECTION_EVENTS,
    task_id: str | None = None,
) -> GraphEventPage:
    if not 1 <= limit <= MAX_GRAPH_INSPECTION_EVENTS:
        raise ValueError("graph event page limit is outside its bound")
    if after_event_id < 0:
        raise ValueError("graph event cursor must be non-negative")
    if _load_job(connection, agent_id, job_id) is None:
        return GraphEventPage(events=(), next_cursor=None)
    task_clause = "" if task_id is None else " AND task_id = ?"
    parameters: tuple[object, ...] = (agent_id, job_id, after_event_id)
    if task_id is not None:
        parameters = (*parameters, task_id)
    rows = tuple(
        connection.execute(
            """SELECT event_id, task_id, attempt_id, kind, created_at_us, data
               FROM job_graph_events
               WHERE agent_id = ? AND job_id = ? AND event_id > ?"""
            + task_clause
            + " ORDER BY event_id LIMIT ?",
            (*parameters, limit + 1),
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


def list_budget_ledgers(
    connection: SQLConnection, agent_id: str, job_id: str
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


def _list_task_budget_ledgers(
    connection: SQLConnection, agent_id: str, job_id: str
) -> tuple[BudgetLedger, ...]:
    rows = tuple(
        connection.execute(
            """SELECT task_id, dimension, ceiling, settled, reserved, updated_at_us
               FROM job_task_budget_ledger
               WHERE agent_id = ? AND job_id = ?
               ORDER BY task_id, dimension""",
            (agent_id, job_id),
        )
    )
    return tuple(
        BudgetLedger(
            agent_id=agent_id,
            job_id=job_id,
            task_id=_required_text(task_id, "task budget task ID"),
            dimension=_required_text(dimension, "task budget dimension"),
            ceiling=_required_int(ceiling, "task budget ceiling"),
            settled=_required_int(settled, "task budget settled"),
            reserved=_required_int(reserved, "task budget reserved"),
            control_reserved=0,
            updated_at=datetime_from_us(updated_at_us, "task budget"),
        )
        for task_id, dimension, ceiling, settled, reserved, updated_at_us in rows
    )
