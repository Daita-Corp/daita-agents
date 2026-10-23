from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daita.distribution.models import (
    DeliveryState,
    DeliverySubjectKind,
    DeliveryView,
    OutcomeConclusionKind,
    OutcomeState,
)
from daita.jobs import (
    ChecklistItem,
    ChecklistStatus,
    Delegation,
    DelegationState,
    Job,
    JobState,
)
from daita.llm.models import ModelSensitivity
from daita.routines.models import (
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineState,
    ScheduledRoutineSummary,
    ScheduleKind,
)
from daita.work_board import (
    DEFAULT_DONE_WINDOW,
    FAILED_JOB_WINDOW,
    MAX_WORK_CARDS_PER_COLUMN,
    WorkBoard,
    WorkCard,
    WorkColumn,
    WorkKind,
    project_work_board,
    work_board_projection,
)
from tests.routines._storage_support import occurrence_record

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
STARTED = NOW - timedelta(hours=1)
OBJECTIVE = "Compare the monthly totals."
_DIGEST = "sha256:" + "0" * 64


_STATE_FIELDS: dict[JobState, dict[str, object]] = {
    JobState.WAITING_FOR_YOU: {"question": "Which month?"},
    JobState.FAILED: {"failure_code": "job_run_failed"},
    JobState.DONE: {"final_summary": "September was higher."},
}


def _job(
    job_id: str,
    state: JobState,
    *,
    updated_at: datetime = NOW - timedelta(minutes=5),
    **fields: object,
) -> Job:
    values: dict[str, Any] = {**_STATE_FIELDS.get(state, {}), **fields}
    return Job(
        job_id=job_id,
        agent_id="agent-1",
        conversation_id="conversation-1",
        objective=values.pop("objective", OBJECTIVE),
        source_ids=(),
        state=state,
        created_at=min(STARTED, updated_at),
        updated_at=updated_at,
        **values,
    )


def _routine(
    routine_id: str,
    state: RoutineState = RoutineState.ACTIVE,
    *,
    title: str = "Weekly report",
    next_due_at: datetime | None = NOW + timedelta(hours=1),
    consecutive_failures: int = 0,
) -> ScheduledRoutineSummary:
    return ScheduledRoutineSummary(
        routine_id=routine_id,
        title=title,
        state=state,
        schedule_kind=ScheduleKind.INTERVAL,
        next_due_at=next_due_at,
        revision=1,
        occurrence_count=0,
        consecutive_failures=consecutive_failures,
    )


def _occurrence(
    occurrence_id: str,
    routine_id: str,
    disposition: RoutineOccurrenceDisposition = RoutineOccurrenceDisposition.RUNNING,
    *,
    claimed_at: datetime | None = NOW - timedelta(minutes=6),
) -> RoutineOccurrence:
    return replace(
        occurrence_record(),
        occurrence_id=occurrence_id,
        routine_id=routine_id,
        disposition=disposition,
        scheduled_for=NOW - timedelta(minutes=10),
        claimed_at=claimed_at,
        claim_token=None if claimed_at is None else "claim-1",
        lease_expires_at=(
            None if claimed_at is None else claimed_at + timedelta(seconds=30)
        ),
        updated_at=NOW - timedelta(minutes=1),
    )


def _delivery(
    occurrence_id: str,
    conclusion_state: OutcomeState = OutcomeState.SUCCEEDED,
    *,
    delivery_id: str | None = None,
    state: DeliveryState = DeliveryState.AVAILABLE,
    created_at: datetime = NOW - timedelta(hours=2),
    subject_kind: DeliverySubjectKind = DeliverySubjectKind.ROUTINE_OCCURRENCE,
    preview: str = "Revenue rose 4%.",
    failure_code: str | None = None,
) -> DeliveryView:
    acknowledged = state is DeliveryState.ACKNOWLEDGED
    return DeliveryView(
        delivery_id=delivery_id or f"delivery-{occurrence_id}",
        conversation_id="conversation-1",
        subject_kind=subject_kind,
        subject_id=occurrence_id,
        conclusion_kind=OutcomeConclusionKind.TERMINAL_RUN,
        conclusion_state=conclusion_state,
        conclusion_digest=_DIGEST,
        conclusion_preview=preview,
        conclusion_preview_truncated=False,
        resulting_run_id=None,
        artifact_references=(),
        effective_sensitivity=ModelSensitivity.INTERNAL,
        provenance_digest=_DIGEST,
        destination_id="destination-1",
        state=state,
        created_at=created_at,
        updated_at=created_at,
        acknowledged_at=created_at if acknowledged else None,
        blocked_reason_code=(
            "sensitivity_ceiling" if state is DeliveryState.BLOCKED else None
        ),
        failure_code=failure_code,
    )


def _board(
    *,
    jobs: tuple[Job, ...] = (),
    routines: tuple[ScheduledRoutineSummary, ...] = (),
    active: tuple[RoutineOccurrence, ...] = (),
    results: tuple[RoutineOccurrence, ...] = (),
    deliveries: tuple[DeliveryView, ...] = (),
    done_window: timedelta = DEFAULT_DONE_WINDOW,
) -> WorkBoard:
    return project_work_board(
        jobs=jobs,
        routines=routines,
        active_occurrences=active,
        result_occurrences=results,
        deliveries=deliveries,
        now=NOW,
        done_window=done_window,
    )


def _only(board: WorkBoard) -> WorkCard | None:
    cards = [card for column in WorkColumn for card in board.columns[column]]
    assert len(cards) <= 1
    return cards[0] if cards else None


@pytest.mark.parametrize(
    "state,fields,column,summary,started",
    [
        (JobState.RUNNING, {}, WorkColumn.RUNNING, "", True),
        (
            JobState.WAITING_FOR_YOU,
            {},
            WorkColumn.NEEDS_YOU,
            "Which month?",
            False,
        ),
        (
            JobState.FAILED,
            {},
            WorkColumn.NEEDS_YOU,
            "job_run_failed",
            False,
        ),
        (
            JobState.FAILED,
            {"failure_code": "job_upgraded", "final_summary": "Start it again."},
            WorkColumn.NEEDS_YOU,
            "Start it again.",
            False,
        ),
        (
            JobState.DONE,
            {},
            WorkColumn.DONE,
            "September was higher.",
            False,
        ),
        (JobState.CANCELLED, {}, WorkColumn.DONE, "", False),
    ],
)
def test_every_job_state_lands_in_its_column(
    state: JobState,
    fields: dict[str, Any],
    column: WorkColumn,
    summary: str,
    started: bool,
) -> None:
    job = _job("job-1", state, **fields)
    card = _only(_board(jobs=(job,)))
    assert card is not None
    assert (card.kind, card.id, card.column) == (WorkKind.JOB, "job-1", column)
    assert (card.status, card.summary) == (state.value, summary)
    assert card.title == OBJECTIVE
    assert card.updated_at == job.updated_at
    assert card.started_at == (STARTED if started else None)
    assert card.delivery_id is None


def test_a_running_job_shows_its_checklist_and_sub_runs() -> None:
    job = _job(
        "job-1",
        JobState.RUNNING,
        checklist=(
            ChecklistItem("a", "Read A", ChecklistStatus.DONE),
            ChecklistItem("b", "Read B", ChecklistStatus.IN_PROGRESS),
            ChecklistItem("c", "Read C", ChecklistStatus.CANCELLED),
        ),
        delegations=(Delegation("d-1", "Read B", DelegationState.RUNNING, "run-1"),),
    )
    card = _only(_board(jobs=(job,)))
    assert card is not None
    assert card.summary == "1 of 2 done · 1 sub-run running"


@pytest.mark.parametrize(
    "state,window",
    [
        (JobState.DONE, DEFAULT_DONE_WINDOW),
        (JobState.CANCELLED, DEFAULT_DONE_WINDOW),
        (JobState.FAILED, FAILED_JOB_WINDOW),
    ],
)
def test_terminal_jobs_age_off_exactly_at_their_window(
    state: JobState, window: timedelta
) -> None:
    at_edge = _job("job-edge", state, updated_at=NOW - window)
    past_edge = _job(
        "job-past", state, updated_at=NOW - window - timedelta(microseconds=1)
    )
    card = _only(_board(jobs=(at_edge, past_edge)))
    assert card is not None and card.id == "job-edge"


def test_a_job_read_twice_keeps_its_latest_state() -> None:
    waiting = _job(
        "job-1", JobState.WAITING_FOR_YOU, updated_at=NOW - timedelta(minutes=9)
    )
    running = _job("job-1", JobState.RUNNING, updated_at=NOW - timedelta(minutes=8))
    card = _only(_board(jobs=(running, waiting)))
    assert card is not None and card.column is WorkColumn.RUNNING


@pytest.mark.parametrize(
    "state,next_due,column,summary",
    [
        (RoutineState.ACTIVE, True, WorkColumn.SCHEDULED, ""),
        (RoutineState.ACTIVE, False, None, None),
        (
            RoutineState.NEEDS_ATTENTION,
            False,
            WorkColumn.NEEDS_YOU,
            "Needs attention after 3 consecutive failures",
        ),
        (RoutineState.PAUSED, True, None, None),
        (RoutineState.COMPLETED, False, None, None),
        (RoutineState.EXPIRED, False, None, None),
        (RoutineState.DISABLED, False, None, None),
    ],
)
def test_every_routine_state_lands_in_its_column(
    state: RoutineState,
    next_due: bool,
    column: WorkColumn | None,
    summary: str | None,
) -> None:
    due = NOW + timedelta(hours=1) if next_due else None
    routine = _routine("routine-1", state, next_due_at=due, consecutive_failures=3)
    card = _only(_board(routines=(routine,)))
    if column is None:
        assert card is None
        return
    assert card is not None
    assert (card.kind, card.id, card.column, card.summary) == (
        WorkKind.ROUTINE,
        "routine-1",
        column,
        summary,
    )
    assert card.routine_id == "routine-1"
    assert card.due_at == (due if column is WorkColumn.SCHEDULED else None)
    assert card.updated_at == NOW


def test_routine_attention_summary_without_failures_and_with_one() -> None:
    board = _board(
        routines=(
            _routine("routine-0", RoutineState.NEEDS_ATTENTION, next_due_at=None),
            _routine(
                "routine-1",
                RoutineState.NEEDS_ATTENTION,
                next_due_at=None,
                consecutive_failures=1,
            ),
        )
    )
    assert {card.id: card.summary for card in board.columns[WorkColumn.NEEDS_YOU]} == {
        "routine-0": "Needs attention",
        "routine-1": "Needs attention after 1 consecutive failure",
    }


@pytest.mark.parametrize(
    "disposition,running",
    [
        (disposition, True)
        for disposition in (
            RoutineOccurrenceDisposition.CLAIMED,
            RoutineOccurrenceDisposition.RUNNING,
            RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
            RoutineOccurrenceDisposition.RETRYABLE,
        )
    ]
    + [
        (disposition, False)
        for disposition in (
            RoutineOccurrenceDisposition.COMPLETED,
            RoutineOccurrenceDisposition.TERMINAL_FAILED,
        )
    ],
)
def test_only_active_routine_runs_are_running(
    disposition: RoutineOccurrenceDisposition, running: bool
) -> None:
    occurrence = _occurrence("occurrence-1", "routine-1", disposition)
    card = _only(_board(active=(occurrence,)))
    if not running:
        assert card is None
        return
    assert card is not None
    assert (card.kind, card.id, card.column, card.status) == (
        WorkKind.ROUTINE_RUN,
        "occurrence-1",
        WorkColumn.RUNNING,
        disposition.value,
    )
    assert card.started_at == occurrence.claimed_at
    assert card.updated_at == occurrence.updated_at
    assert card.routine_id == "routine-1"


def test_unclaimed_run_starts_at_its_scheduled_time() -> None:
    occurrence = _occurrence("occurrence-1", "routine-1", claimed_at=None)
    card = _only(_board(active=(occurrence,)))
    assert card is not None and card.started_at == occurrence.scheduled_for


@pytest.mark.parametrize(
    "delivery,column,status,summary",
    [
        (
            _delivery(
                "occurrence-1",
                OutcomeState.FAILED,
                created_at=NOW - timedelta(days=90),
                failure_code="routine_source_unavailable",
            ),
            WorkColumn.NEEDS_YOU,
            "failed",
            "routine_source_unavailable",
        ),
        (
            _delivery("occurrence-1", OutcomeState.FAILED, preview="Could not run."),
            WorkColumn.NEEDS_YOU,
            "failed",
            "Could not run.",
        ),
        (
            _delivery("occurrence-1", state=DeliveryState.BLOCKED, preview=""),
            WorkColumn.NEEDS_YOU,
            "blocked",
            "Result withheld (sensitivity)",
        ),
        (
            _delivery("occurrence-1"),
            WorkColumn.DONE,
            "succeeded",
            "Revenue rose 4%.",
        ),
        (
            _delivery("occurrence-1", OutcomeState.SKIPPED_NO_CHANGE),
            WorkColumn.DONE,
            "skipped_no_change",
            "Revenue rose 4%.",
        ),
        (
            _delivery("occurrence-1", created_at=NOW - DEFAULT_DONE_WINDOW),
            WorkColumn.DONE,
            "succeeded",
            "Revenue rose 4%.",
        ),
        (
            _delivery(
                "occurrence-1",
                created_at=NOW - DEFAULT_DONE_WINDOW - timedelta(microseconds=1),
            ),
            None,
            None,
            None,
        ),
        (_delivery("occurrence-1", state=DeliveryState.ACKNOWLEDGED), None, None, None),
        (
            _delivery(
                "job-1/event/1/control_opened/control-1",
                subject_kind=DeliverySubjectKind.JOB,
            ),
            None,
            None,
            None,
        ),
    ],
)
def test_routine_results_land_in_their_column(
    delivery: DeliveryView,
    column: WorkColumn | None,
    status: str | None,
    summary: str | None,
) -> None:
    result = _occurrence(
        "occurrence-1", "routine-1", RoutineOccurrenceDisposition.COMPLETED
    )
    card = _only(_board(results=(result,), deliveries=(delivery,)))
    if column is None:
        assert card is None
        return
    assert card is not None
    assert (card.kind, card.id, card.column, card.status, card.summary) == (
        WorkKind.ROUTINE_RUN,
        "occurrence-1",
        column,
        status,
        summary,
    )
    assert (card.delivery_id, card.routine_id) == (delivery.delivery_id, "routine-1")
    assert card.updated_at == delivery.created_at


def test_run_titles_resolve_through_their_routine_or_fall_back() -> None:
    board = _board(
        routines=(_routine("routine-1", RoutineState.PAUSED, title="Daily sales"),),
        active=(_occurrence("occurrence-active", "routine-1"),),
        results=(
            _occurrence(
                "occurrence-known", "routine-1", RoutineOccurrenceDisposition.COMPLETED
            ),
        ),
        deliveries=(_delivery("occurrence-known"), _delivery("occurrence-unknown")),
    )
    titles = {
        card.id: (card.title, card.routine_id)
        for column in WorkColumn
        for card in board.columns[column]
    }
    assert titles == {
        "occurrence-active": ("Daily sales", "routine-1"),
        "occurrence-known": ("Daily sales", "routine-1"),
        "occurrence-unknown": ("Routine run", None),
    }


def test_an_active_run_hides_its_delivered_result() -> None:
    occurrence = _occurrence("occurrence-1", "routine-1")
    board = _board(
        active=(occurrence,),
        results=(occurrence,),
        deliveries=(_delivery("occurrence-1", OutcomeState.FAILED),),
    )
    card = _only(board)
    assert card is not None
    assert (card.column, card.delivery_id) == (WorkColumn.RUNNING, None)


def test_a_routine_can_be_scheduled_while_one_of_its_runs_is_running() -> None:
    board = _board(
        routines=(_routine("routine-1"),),
        active=(_occurrence("occurrence-1", "routine-1"),),
    )
    assert [card.kind for card in board.columns[WorkColumn.SCHEDULED]] == [
        WorkKind.ROUTINE
    ]
    assert [card.kind for card in board.columns[WorkColumn.RUNNING]] == [
        WorkKind.ROUTINE_RUN
    ]


def test_columns_sort_by_their_own_time_then_id() -> None:
    later = NOW - timedelta(minutes=1)
    earlier = NOW - timedelta(minutes=30)
    board = _board(
        jobs=(
            _job("job-blocked-b", JobState.WAITING_FOR_YOU, updated_at=earlier),
            _job("job-blocked-a", JobState.WAITING_FOR_YOU, updated_at=earlier),
            _job("job-blocked-new", JobState.WAITING_FOR_YOU, updated_at=later),
            _job("job-done-old", JobState.DONE, updated_at=earlier),
            _job("job-done-new", JobState.DONE, updated_at=later),
        ),
        routines=(
            _routine("routine-late", next_due_at=NOW + timedelta(hours=2)),
            _routine("routine-soon", next_due_at=NOW + timedelta(hours=1)),
        ),
        active=(
            _occurrence("occurrence-b", "routine-1", claimed_at=earlier),
            _occurrence("occurrence-a", "routine-1", claimed_at=earlier),
            _occurrence(
                "occurrence-first", "routine-1", claimed_at=earlier - timedelta(1)
            ),
        ),
    )
    assert [card.id for card in board.columns[WorkColumn.SCHEDULED]] == [
        "routine-soon",
        "routine-late",
    ]
    assert [card.id for card in board.columns[WorkColumn.RUNNING]] == [
        "occurrence-first",
        "occurrence-a",
        "occurrence-b",
    ]
    assert [card.id for card in board.columns[WorkColumn.NEEDS_YOU]] == [
        "job-blocked-new",
        "job-blocked-a",
        "job-blocked-b",
    ]
    assert [card.id for card in board.columns[WorkColumn.DONE]] == [
        "job-done-new",
        "job-done-old",
    ]


def test_columns_truncate_and_every_column_is_present() -> None:
    jobs = tuple(
        _job(f"job-{index:02d}", JobState.WAITING_FOR_YOU)
        for index in range(MAX_WORK_CARDS_PER_COLUMN + 2)
    )
    board = _board(jobs=jobs)
    assert len(board.columns[WorkColumn.NEEDS_YOU]) == MAX_WORK_CARDS_PER_COLUMN
    assert board.columns[WorkColumn.NEEDS_YOU][0].id == "job-00"
    assert dict(board.truncated) == {
        WorkColumn.SCHEDULED: 0,
        WorkColumn.RUNNING: 0,
        WorkColumn.NEEDS_YOU: 2,
        WorkColumn.DONE: 0,
    }
    empty = _board()
    assert {column: empty.columns[column] for column in WorkColumn} == {
        column: () for column in WorkColumn
    }


def test_long_multiline_text_is_bounded_to_one_line() -> None:
    job = _job("job-1", JobState.WAITING_FOR_YOU, objective="Count\n" + "x" * 500)
    result = _occurrence(
        "occurrence-1", "routine-1", RoutineOccurrenceDisposition.COMPLETED
    )
    board = _board(
        jobs=(job,),
        results=(result,),
        deliveries=(_delivery("occurrence-1", preview="Line one\nline two " * 40),),
    )
    (job_card,) = board.columns[WorkColumn.NEEDS_YOU]
    (run_card,) = board.columns[WorkColumn.DONE]
    assert job_card.title.startswith("Count x") and job_card.title.endswith("…")
    assert len(job_card.title) == 120
    assert "\n" not in run_card.summary and run_card.summary.endswith("…")
    assert len(run_card.summary) <= 240


def test_projection_is_json_safe_with_every_column() -> None:
    board = _board(
        jobs=(_job("job-1", JobState.RUNNING),),
        routines=(_routine("routine-1"),),
    )
    projected = work_board_projection(board)
    assert json.loads(json.dumps(projected)) == projected
    assert projected["generated_at"] == NOW.isoformat()
    assert projected["truncated"] == {column.value: 0 for column in WorkColumn}
    columns = projected["columns"]
    assert isinstance(columns, dict)
    assert list(columns) == [column.value for column in WorkColumn]
    assert columns["running"] == [
        {
            "kind": "job",
            "id": "job-1",
            "column": "running",
            "title": OBJECTIVE,
            "status": "running",
            "summary": "",
            "updated_at": (NOW - timedelta(minutes=5)).isoformat(),
            "started_at": STARTED.isoformat(),
            "due_at": None,
            "routine_id": None,
            "delivery_id": None,
        }
    ]
    assert columns["scheduled"][0]["due_at"] == (NOW + timedelta(hours=1)).isoformat()


@pytest.mark.parametrize(
    "now,done_window",
    [
        (NOW.replace(tzinfo=None), DEFAULT_DONE_WINDOW),
        (NOW, timedelta(0)),
        (NOW, -DEFAULT_DONE_WINDOW),
    ],
)
def test_projection_rejects_naive_time_and_non_positive_window(
    now: datetime, done_window: timedelta
) -> None:
    with pytest.raises(ValueError):
        project_work_board(
            jobs=(),
            routines=(),
            active_occurrences=(),
            result_occurrences=(),
            deliveries=(),
            now=now,
            done_window=done_window,
        )
