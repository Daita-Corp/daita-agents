"""Read-only work board projection over jobs, routines and their results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType

from .distribution.models import (
    DeliveryState,
    DeliverySubjectKind,
    DeliveryView,
    OutcomeState,
)
from .jobs.graph.models import GraphJob, GraphState
from .routines.models import (
    ACTIVE_OCCURRENCE_DISPOSITIONS,
    RoutineOccurrence,
    RoutineState,
    ScheduledRoutineSummary,
)

MAX_WORK_CARDS_PER_COLUMN = 25
MAX_WORK_CARD_TITLE_CHARACTERS = 120
MAX_WORK_CARD_SUMMARY_CHARACTERS = 240
DEFAULT_DONE_WINDOW = timedelta(hours=24)
# shortcut: failed jobs have no dismissable record yet; replace when one exists.
FAILED_JOB_WINDOW = timedelta(days=7)

_ROUTINE_RUN_FALLBACK_TITLE = "Routine run"


class WorkColumn(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    NEEDS_YOU = "needs_you"
    DONE = "done"


class WorkKind(str, Enum):
    JOB = "job"
    ROUTINE = "routine"
    ROUTINE_RUN = "routine_run"


def _utc(value: datetime, name: str) -> None:
    offset = value.utcoffset() if isinstance(value, datetime) else None
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _line(value: str, name: str, *, maximum: int, allow_empty: bool = False) -> None:
    if (
        not isinstance(value, str)
        or (not value and not allow_empty)
        or value != value.strip()
        or len(value) > maximum
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{name} must be bounded single-line text")


def _bounded(value: str, maximum: int) -> str:
    text = " ".join(value.split())
    return text if len(text) <= maximum else text[: maximum - 1].rstrip() + "…"


@dataclass(frozen=True, slots=True)
class WorkCard:
    """One live or recent unit of work on the board."""

    kind: WorkKind
    id: str
    column: WorkColumn
    title: str
    status: str
    summary: str
    updated_at: datetime
    started_at: datetime | None = None
    due_at: datetime | None = None
    routine_id: str | None = None
    delivery_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WorkKind):
            raise TypeError("work card kind is invalid")
        if not isinstance(self.column, WorkColumn):
            raise TypeError("work card column is invalid")
        _line(self.id, "work card id", maximum=256)
        _line(self.title, "work card title", maximum=MAX_WORK_CARD_TITLE_CHARACTERS)
        _line(self.status, "work card status", maximum=64)
        _line(
            self.summary,
            "work card summary",
            maximum=MAX_WORK_CARD_SUMMARY_CHARACTERS,
            allow_empty=True,
        )
        _utc(self.updated_at, "work card updated_at")
        for timestamp, name in (
            (self.started_at, "work card started_at"),
            (self.due_at, "work card due_at"),
        ):
            if timestamp is not None:
                _utc(timestamp, name)
        for identifier, name in (
            (self.routine_id, "work card routine_id"),
            (self.delivery_id, "work card delivery_id"),
        ):
            if identifier is not None:
                _line(identifier, name, maximum=256)


@dataclass(frozen=True, slots=True)
class WorkBoard:
    """Bounded columns of work; truncated counts cards omitted per column."""

    columns: Mapping[WorkColumn, tuple[WorkCard, ...]]
    truncated: Mapping[WorkColumn, int]
    generated_at: datetime

    def __post_init__(self) -> None:
        if set(self.columns) != set(WorkColumn) or set(self.truncated) != set(
            WorkColumn
        ):
            raise ValueError("work board must contain every column")
        for column, cards in self.columns.items():
            if len(cards) > MAX_WORK_CARDS_PER_COLUMN or any(
                not isinstance(card, WorkCard) or card.column is not column
                for card in cards
            ):
                raise ValueError("work board column contents are invalid")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in self.truncated.values()
        ):
            raise ValueError("work board truncation counts are invalid")
        _utc(self.generated_at, "work board generated_at")


def _job_card(job: GraphJob, now: datetime, done_window: timedelta) -> WorkCard | None:
    started_at: datetime | None = None
    summary = ""
    if job.state is GraphState.QUEUED:
        column = WorkColumn.SCHEDULED
    elif job.state in {GraphState.ACTIVE, GraphState.CANCEL_REQUESTED}:
        column, started_at = WorkColumn.RUNNING, job.created_at
    elif job.state is GraphState.BLOCKED:
        column, summary = WorkColumn.NEEDS_YOU, "Waiting for you"
    elif job.state is GraphState.NEEDS_ATTENTION:
        column, summary = WorkColumn.NEEDS_YOU, "Needs your attention"
    elif job.state is GraphState.FAILED:
        if now - (job.terminal_at or job.updated_at) > FAILED_JOB_WINDOW:
            return None
        column, summary = WorkColumn.NEEDS_YOU, job.failure_code or ""
    else:
        if now - (job.terminal_at or job.updated_at) > done_window:
            return None
        column = WorkColumn.DONE
    return WorkCard(
        kind=WorkKind.JOB,
        id=job.job_id,
        column=column,
        title=_bounded(job.specification.objective, MAX_WORK_CARD_TITLE_CHARACTERS),
        status=job.state.value,
        summary=_bounded(summary, MAX_WORK_CARD_SUMMARY_CHARACTERS),
        updated_at=job.updated_at,
        started_at=started_at,
    )


def _routine_card(routine: ScheduledRoutineSummary, now: datetime) -> WorkCard | None:
    # Routine summaries carry no timestamp; the board's reading time stands in.
    if routine.state is RoutineState.ACTIVE and routine.next_due_at is not None:
        column, summary = WorkColumn.SCHEDULED, ""
    elif routine.state is RoutineState.NEEDS_ATTENTION:
        failures = routine.consecutive_failures
        column, summary = WorkColumn.NEEDS_YOU, "Needs attention"
        if failures:
            noun = "failure" if failures == 1 else "failures"
            summary = f"Needs attention after {failures} consecutive {noun}"
    else:
        return None
    return WorkCard(
        kind=WorkKind.ROUTINE,
        id=routine.routine_id,
        column=column,
        title=_bounded(routine.title, MAX_WORK_CARD_TITLE_CHARACTERS),
        status=routine.state.value,
        summary=summary,
        updated_at=now,
        due_at=routine.next_due_at if column is WorkColumn.SCHEDULED else None,
        routine_id=routine.routine_id,
    )


def _delivery_card(
    delivery: DeliveryView,
    *,
    routine_id: str | None,
    title: str,
    now: datetime,
    done_window: timedelta,
) -> WorkCard | None:
    if delivery.state is DeliveryState.BLOCKED:
        status, summary = DeliveryState.BLOCKED.value, "Result withheld (sensitivity)"
        column = WorkColumn.NEEDS_YOU
    elif delivery.conclusion_state is OutcomeState.FAILED:
        status = delivery.conclusion_state.value
        summary = delivery.failure_code or delivery.conclusion_preview
        column = WorkColumn.NEEDS_YOU
    elif now - delivery.created_at <= done_window:
        status, summary = delivery.conclusion_state.value, delivery.conclusion_preview
        column = WorkColumn.DONE
    else:
        return None
    return WorkCard(
        kind=WorkKind.ROUTINE_RUN,
        id=delivery.subject_id,
        column=column,
        title=title,
        status=status,
        summary=_bounded(summary, MAX_WORK_CARD_SUMMARY_CHARACTERS),
        updated_at=delivery.created_at,
        routine_id=routine_id,
        delivery_id=delivery.delivery_id,
    )


def _sort_key(card: WorkCard) -> tuple[object, ...]:
    tie = (card.id, card.kind.value)
    if card.column is WorkColumn.SCHEDULED:
        due = 0.0 if card.due_at is None else card.due_at.timestamp()
        return (card.due_at is None, due, -card.updated_at.timestamp(), *tie)
    if card.column is WorkColumn.RUNNING:
        return (card.started_at or card.updated_at, *tie)
    return (-card.updated_at.timestamp(), *tie)


def project_work_board(
    *,
    jobs: Iterable[GraphJob],
    routines: Iterable[ScheduledRoutineSummary],
    active_occurrences: Iterable[RoutineOccurrence],
    result_occurrences: Iterable[RoutineOccurrence],
    deliveries: Iterable[DeliveryView],
    now: datetime,
    done_window: timedelta = DEFAULT_DONE_WINDOW,
) -> WorkBoard:
    """Place each job, routine and routine run in exactly one column."""

    _utc(now, "work board now")
    if not isinstance(done_window, timedelta) or done_window <= timedelta(0):
        raise ValueError("work board done window must be positive")

    latest_jobs: dict[str, GraphJob] = {}
    for job in jobs:
        current = latest_jobs.get(job.job_id)
        if current is None or job.updated_at > current.updated_at:
            latest_jobs[job.job_id] = job
    routine_list = tuple(routines)
    titles = {item.routine_id: item.title for item in routine_list}
    occurrence_routines = {
        item.occurrence_id: item.routine_id for item in result_occurrences
    }

    def run_title(routine_id: str | None) -> str:
        title = titles.get(routine_id) if routine_id is not None else None
        return _bounded(
            title or _ROUTINE_RUN_FALLBACK_TITLE, MAX_WORK_CARD_TITLE_CHARACTERS
        )

    cards: list[WorkCard] = []
    for job in latest_jobs.values():
        job_card = _job_card(job, now, done_window)
        if job_card is not None:
            cards.append(job_card)
    for routine in routine_list:
        routine_card = _routine_card(routine, now)
        if routine_card is not None:
            cards.append(routine_card)
    carded_occurrences: set[str] = set()
    for occurrence in active_occurrences:
        if (
            occurrence.disposition not in ACTIVE_OCCURRENCE_DISPOSITIONS
            or occurrence.occurrence_id in carded_occurrences
        ):
            continue
        carded_occurrences.add(occurrence.occurrence_id)
        cards.append(
            WorkCard(
                kind=WorkKind.ROUTINE_RUN,
                id=occurrence.occurrence_id,
                column=WorkColumn.RUNNING,
                title=run_title(occurrence.routine_id),
                status=occurrence.disposition.value,
                summary="",
                updated_at=occurrence.updated_at,
                started_at=occurrence.claimed_at or occurrence.scheduled_for,
                routine_id=occurrence.routine_id,
            )
        )
    for delivery in sorted(
        deliveries, key=lambda item: (item.created_at, item.delivery_id), reverse=True
    ):
        if (
            delivery.subject_kind is not DeliverySubjectKind.ROUTINE_OCCURRENCE
            or delivery.state is DeliveryState.ACKNOWLEDGED
            or delivery.subject_id in carded_occurrences
        ):
            continue
        routine_id = occurrence_routines.get(delivery.subject_id)
        delivery_card = _delivery_card(
            delivery,
            routine_id=routine_id,
            title=run_title(routine_id),
            now=now,
            done_window=done_window,
        )
        if delivery_card is not None:
            carded_occurrences.add(delivery.subject_id)
            cards.append(delivery_card)

    columns: dict[WorkColumn, tuple[WorkCard, ...]] = {}
    truncated: dict[WorkColumn, int] = {}
    for column in WorkColumn:
        ordered = sorted(
            (card for card in cards if card.column is column), key=_sort_key
        )
        columns[column] = tuple(ordered[:MAX_WORK_CARDS_PER_COLUMN])
        truncated[column] = max(0, len(ordered) - MAX_WORK_CARDS_PER_COLUMN)
    return WorkBoard(
        columns=MappingProxyType(columns),
        truncated=MappingProxyType(truncated),
        generated_at=now,
    )


def _optional_iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def work_board_projection(board: WorkBoard) -> dict[str, object]:
    """JSON-safe projection for the CLI (and, later, hosted callers)."""

    if not isinstance(board, WorkBoard):
        raise TypeError("work board projection requires WorkBoard")
    return {
        "generated_at": board.generated_at.isoformat(),
        "columns": {
            column.value: [
                {
                    "kind": card.kind.value,
                    "id": card.id,
                    "column": card.column.value,
                    "title": card.title,
                    "status": card.status,
                    "summary": card.summary,
                    "updated_at": card.updated_at.isoformat(),
                    "started_at": _optional_iso(card.started_at),
                    "due_at": _optional_iso(card.due_at),
                    "routine_id": card.routine_id,
                    "delivery_id": card.delivery_id,
                }
                for card in board.columns[column]
            ]
            for column in WorkColumn
        },
        "truncated": {column.value: board.truncated[column] for column in WorkColumn},
    }


__all__ = [
    "DEFAULT_DONE_WINDOW",
    "FAILED_JOB_WINDOW",
    "MAX_WORK_CARDS_PER_COLUMN",
    "WorkBoard",
    "WorkCard",
    "WorkColumn",
    "WorkKind",
    "project_work_board",
    "work_board_projection",
]
