"""Pure bounded operational projections over authoritative graph records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .graph.models import (
    CONTROL_BUDGET_ROLES,
    BudgetLedger,
    ControlState,
    GraphEvent,
    GraphInspection,
    GraphState,
    GraphTask,
    TaskDependency,
    TaskState,
)

MAX_OPERATIONAL_ITEMS = 100


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    job_id: str
    graph_state: GraphState
    graph_revision: int
    blockers: tuple[Mapping[str, object], ...]
    exhausted_budgets: tuple[Mapping[str, object], ...]
    deadlocked: bool
    deadlock_reason: str | None
    latest_event_id: int | None


@dataclass(frozen=True, slots=True)
class GraphTimelinePage:
    job_id: str
    graph_state: GraphState
    graph_revision: int
    events: tuple[GraphEvent, ...]
    next_cursor: int | None
    diagnostics: GraphDiagnostics


@dataclass(frozen=True, slots=True)
class KanbanColumn:
    name: str
    task_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DependencyProjection:
    upstream_task_id: str
    downstream_task_id: str
    edge_kind: str
    satisfied: bool


@dataclass(frozen=True, slots=True)
class GraphBoardProjection:
    job_id: str
    graph_state: GraphState
    graph_revision: int
    columns: tuple[KanbanColumn, ...]
    dependencies: tuple[DependencyProjection, ...]
    diagnostics: GraphDiagnostics


def graph_diagnostics(inspection: GraphInspection) -> GraphDiagnostics:
    tasks_by_id = {task.task_id: task for task in inspection.tasks}
    open_controls: tuple[Mapping[str, object], ...] = tuple(
        {
            "kind": "open_control",
            "control_id": control.control_id,
            "task_id": control.task_id,
            "control_kind": control.kind.value,
            "created_at": control.created_at.isoformat(),
            "payload_digest": control.payload_digest,
            "payload": control.payload,
        }
        for control in inspection.controls
        if control.state is ControlState.OPEN
    )
    failed_tasks: tuple[Mapping[str, object], ...] = tuple(
        {
            "kind": "failed_task",
            "task_id": task.task_id,
            "failure_streak": task.failure_streak,
            "task_revision": task.task_revision,
        }
        for task in inspection.tasks
        if task.state is TaskState.FAILED
    )
    task_ledgers = tuple(
        ledger for ledger in inspection.budget_ledgers if ledger.task_id is not None
    )
    exhausted_items: list[Mapping[str, object]] = [
        {
            "scope": "task",
            "dimension": ledger.dimension,
            "task_id": ledger.task_id,
            "ceiling": ledger.ceiling,
            "settled": ledger.settled,
            "reserved": ledger.reserved,
            "control_reserved": 0,
        }
        for ledger in task_ledgers
        if ledger.settled + ledger.reserved >= ledger.ceiling
    ]
    exhausted_task_dimensions = {
        (ledger.task_id, ledger.dimension)
        for ledger in task_ledgers
        if ledger.settled + ledger.reserved >= ledger.ceiling
    }
    exhausted_root_lanes: set[tuple[str, str]] = set()

    def control_ledger(ledger: BudgetLedger) -> bool:
        assert ledger.task_id is not None
        return tasks_by_id[ledger.task_id].role in CONTROL_BUDGET_ROLES

    for root in (
        ledger for ledger in inspection.budget_ledgers if ledger.task_id is None
    ):
        control_ledgers = tuple(
            ledger
            for ledger in task_ledgers
            if ledger.dimension == root.dimension and control_ledger(ledger)
        )
        ordinary_ledgers = tuple(
            ledger
            for ledger in task_ledgers
            if ledger.dimension == root.dimension and not control_ledger(ledger)
        )
        for lane, ledgers, ceiling in (
            ("ordinary", ordinary_ledgers, root.ceiling - root.control_reserved),
            ("control", control_ledgers, root.control_reserved),
        ):
            settled = sum(item.settled for item in ledgers)
            reserved = sum(item.reserved for item in ledgers)
            if settled + reserved < ceiling:
                continue
            exhausted_root_lanes.add((root.dimension, lane))
            exhausted_items.append(
                {
                    "scope": "root_lane",
                    "lane": lane,
                    "dimension": root.dimension,
                    "task_id": None,
                    "ceiling": ceiling,
                    "settled": settled,
                    "reserved": reserved,
                    "control_reserved": root.control_reserved,
                }
            )

    def budget_blocked(task: GraphTask) -> bool:
        lane = "control" if task.role in CONTROL_BUDGET_ROLES else "ordinary"
        return any(
            (task.task_id, budget.dimension) in exhausted_task_dimensions
            or (budget.dimension, lane) in exhausted_root_lanes
            for budget in task.specification.budgets
        )

    ready_tasks = tuple(
        task for task in inspection.tasks if task.state is TaskState.READY
    )
    budget_blocked_ready = tuple(task for task in ready_tasks if budget_blocked(task))
    runnable = any(task.state is TaskState.RUNNING for task in inspection.tasks) or any(
        not budget_blocked(task) for task in ready_tasks
    )
    unfinished = tuple(
        task
        for task in inspection.tasks
        if task.state
        not in {
            TaskState.SUCCEEDED,
            TaskState.SUPERSEDED,
            TaskState.SKIPPED,
            TaskState.CANCELLED,
            TaskState.FAILED,
        }
    )
    terminal = inspection.job.state in {
        GraphState.SUCCEEDED,
        GraphState.FAILED,
        GraphState.CANCELLED,
    }
    deadlocked = bool(
        unfinished and not runnable and not open_controls and not terminal
    )
    deadlock_reason: str | None = None
    if deadlocked:
        if budget_blocked_ready:
            deadlock_reason = "budget_exhaustion_without_runnable_task"
        else:
            parent_ids = {
                edge.upstream_task_id
                for edge in inspection.dependencies
                if any(task.task_id == edge.downstream_task_id for task in unfinished)
            }
            deadlock_reason = (
                "dependency_wait_without_runnable_parent"
                if parent_ids
                else "no_runnable_task_or_open_control"
            )
    blockers: tuple[Mapping[str, object], ...] = (*open_controls, *failed_tasks)
    if deadlocked:
        blockers = (
            *blockers,
            {
                "kind": "deadlock",
                "reason": deadlock_reason,
                "unfinished_task_ids": tuple(
                    sorted(task.task_id for task in unfinished)
                ),
            },
        )
    return GraphDiagnostics(
        job_id=inspection.job.job_id,
        graph_state=inspection.job.state,
        graph_revision=inspection.graph.revision,
        blockers=tuple(blockers[:MAX_OPERATIONAL_ITEMS]),
        exhausted_budgets=tuple(exhausted_items[:MAX_OPERATIONAL_ITEMS]),
        deadlocked=deadlocked,
        deadlock_reason=deadlock_reason,
        latest_event_id=(
            None if not inspection.events else inspection.events[-1].event_id
        ),
    )


def graph_board(inspection: GraphInspection) -> GraphBoardProjection:
    groups = (
        ("backlog", {TaskState.PENDING}),
        ("ready", {TaskState.READY}),
        ("in_progress", {TaskState.RUNNING}),
        ("blocked", {TaskState.BLOCKED, TaskState.REVIEW}),
        ("done", {TaskState.SUCCEEDED, TaskState.SUPERSEDED, TaskState.SKIPPED}),
        ("failed", {TaskState.FAILED, TaskState.CANCELLED}),
    )
    tasks_by_id = {task.task_id: task for task in inspection.tasks}
    columns = tuple(
        KanbanColumn(
            name=name,
            task_ids=tuple(
                sorted(
                    task.task_id for task in inspection.tasks if task.state in states
                )
            ),
        )
        for name, states in groups
    )
    dependencies = tuple(
        DependencyProjection(
            upstream_task_id=edge.upstream_task_id,
            downstream_task_id=edge.downstream_task_id,
            edge_kind=edge.edge_kind.value,
            satisfied=(tasks_by_id[edge.upstream_task_id].state is TaskState.SUCCEEDED),
        )
        for edge in inspection.dependencies[:MAX_OPERATIONAL_ITEMS]
    )
    return GraphBoardProjection(
        job_id=inspection.job.job_id,
        graph_state=inspection.job.state,
        graph_revision=inspection.graph.revision,
        columns=columns,
        dependencies=dependencies,
        diagnostics=graph_diagnostics(inspection),
    )


def bounded_tasks(
    inspection: GraphInspection,
    *,
    states: frozenset[TaskState] = frozenset(),
    limit: int = 64,
) -> tuple[GraphTask, ...]:
    if not 1 <= limit <= 64:
        raise ValueError("task list limit must be between one and sixty-four")
    return tuple(
        task for task in inspection.tasks if not states or task.state in states
    )[:limit]


def bounded_dependencies(
    inspection: GraphInspection,
    *,
    task_id: str | None = None,
    limit: int = 100,
) -> tuple[TaskDependency, ...]:
    if not 1 <= limit <= MAX_OPERATIONAL_ITEMS:
        raise ValueError("dependency list limit is outside its bound")
    return tuple(
        edge
        for edge in inspection.dependencies
        if task_id is None
        or task_id in {edge.upstream_task_id, edge.downstream_task_id}
    )[:limit]


__all__ = [
    "DependencyProjection",
    "GraphBoardProjection",
    "GraphDiagnostics",
    "GraphTimelinePage",
    "KanbanColumn",
    "bounded_dependencies",
    "bounded_tasks",
    "graph_board",
    "graph_diagnostics",
]
