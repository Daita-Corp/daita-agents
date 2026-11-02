"""Strict bounded records for checklist jobs: one durable record per job."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from ..capabilities import ToolRecovery, recovery_details
from ..errors import DaitaError, ErrorRetryability
from ..llm.models import ModelSensitivity
from ..tui.sanitization import sanitize_terminal_text

MAX_CHECKLIST_ITEMS = 64
MAX_DELEGATIONS = 128
MAX_ANSWERS = 32
MAX_JOB_SOURCES = 64
MAX_JOB_LIST_PAGE_SIZE = 50
MAX_OBJECTIVE_CHARACTERS = 16_384
MAX_ITEM_TITLE_CHARACTERS = 200
MAX_ITEM_NOTE_CHARACTERS = 1_000
MAX_GOAL_CHARACTERS = 4_000
MAX_DELEGATION_CONTEXT_CHARACTERS = 8_000
MAX_SUMMARY_CHARACTERS = 4_000
MAX_QUESTION_CHARACTERS = 2_000
MAX_ANSWER_CHARACTERS = 4_000
MAX_FINAL_SUMMARY_CHARACTERS = 16_384
# Code-owned limits: a job stops as failed after this many runs or this many
# times the per-run token limit in total; raising them grants the same again.
JOB_RUN_LIMIT = 20
JOB_TOKEN_LIMIT_RUNS = 10
# A main run ends after its current step once its context passes this share of
# the model's input window; the job continues in a new run.
JOB_CONTEXT_THRESHOLD = 0.6
INTERRUPTED_BY_RESTART = "Interrupted by restart."
UPGRADED_JOB_SUMMARY = "Created by an older version of Daita; start it again."
UNCERTAIN_WRITE_QUESTION = (
    "A write's outcome is uncertain. Resolve it under Effects; the job "
    "continues once nothing is uncertain."
)
UNANSWERED_APPROVAL_QUESTION = (
    "An approval this job asked for was not answered in time. Answer to "
    "continue; the job asks again if it still needs to write."
)

_ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")


class JobError(DaitaError):
    def __init__(
        self, code: str, message: str, recovery: ToolRecovery | None = None
    ) -> None:
        self.code = code
        self.details = {} if recovery is None else recovery_details(recovery)
        super().__init__(
            message, error_code=code, retryability=ErrorRetryability.PERMANENT
        )


class JobState(str, Enum):
    RUNNING = "running"
    WAITING_FOR_YOU = "waiting_for_you"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


LIVE_JOB_STATES = frozenset({JobState.RUNNING, JobState.WAITING_FOR_YOU})
TERMINAL_JOB_STATES = frozenset(JobState) - LIVE_JOB_STATES
# Failure codes a person can clear by raising the job's limits.
JOB_LIMIT_FAILURES = frozenset({"job_run_limit_reached", "job_token_limit_reached"})


class ChecklistStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class DelegationState(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def clean_text(value: object, maximum: int, *, lines: bool = False) -> str:
    """Model-written text is stored sanitized and bounded; it grants nothing."""

    if not lines and isinstance(value, str):
        value = " ".join(value.split())
    text = sanitize_terminal_text(
        value, maximum=None, preserve_lines=lines, fallback=""
    ).strip()
    return text if len(text) <= maximum else text[: maximum - 1].rstrip() + "…"


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded identifier")


def _bounded(value: str, name: str, maximum: int, *, empty: bool = False) -> None:
    if (
        not isinstance(value, str)
        or (not value and not empty)
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be bounded text")


def _utc(value: datetime, name: str) -> None:
    offset = value.utcoffset() if isinstance(value, datetime) else None
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _count(value: int, name: str, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} is outside its bound")


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    id: str
    title: str
    status: ChecklistStatus = ChecklistStatus.PENDING
    after: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _ITEM_ID.fullmatch(self.id) is None:
            raise ValueError("checklist item id must be a short identifier")
        _bounded(self.title, "checklist item title", MAX_ITEM_TITLE_CHARACTERS)
        if not isinstance(self.status, ChecklistStatus):
            raise TypeError("checklist item status is invalid")
        after = tuple(self.after)
        if len(after) != len(set(after)) or self.id in after:
            raise ValueError("checklist item dependencies must be distinct others")
        for item in after:
            if not isinstance(item, str) or _ITEM_ID.fullmatch(item) is None:
                raise ValueError("checklist item dependency id is invalid")
        object.__setattr__(self, "after", after)
        if self.note is not None:
            _bounded(self.note, "checklist item note", MAX_ITEM_NOTE_CHARACTERS)


@dataclass(frozen=True, slots=True)
class Delegation:
    id: str
    goal: str
    state: DelegationState
    run_id: str
    summary: str | None = None
    item: str | None = None

    def __post_init__(self) -> None:
        _identity(self.id, "delegation id")
        _identity(self.run_id, "delegation run_id")
        _bounded(self.goal, "delegation goal", MAX_GOAL_CHARACTERS)
        if not isinstance(self.state, DelegationState):
            raise TypeError("delegation state is invalid")
        if self.summary is not None:
            _bounded(self.summary, "delegation summary", MAX_SUMMARY_CHARACTERS)
        if self.item is not None and _ITEM_ID.fullmatch(self.item) is None:
            raise ValueError("delegation item id is invalid")


@dataclass(frozen=True, slots=True)
class UserAnswer:
    question: str
    text: str
    answered_at: datetime

    def __post_init__(self) -> None:
        _bounded(self.question, "answered question", MAX_QUESTION_CHARACTERS)
        _bounded(self.text, "answer", MAX_ANSWER_CHARACTERS)
        _utc(self.answered_at, "answer answered_at")


@dataclass(frozen=True, slots=True)
class Job:
    """A background job: an ordinary agent run that keeps going in the host.

    The model owns the checklist and delegations; code owns state, limits and
    accounting. A job continues by starting a new run from this record.
    """

    job_id: str
    agent_id: str
    conversation_id: str
    objective: str
    source_ids: tuple[str, ...]
    state: JobState
    created_at: datetime
    updated_at: datetime
    checklist: tuple[ChecklistItem, ...] = ()
    delegations: tuple[Delegation, ...] = ()
    question: str | None = None
    answers: tuple[UserAnswer, ...] = ()
    current_run_id: str | None = None
    run_count: int = 0
    tokens_used: int = 0
    run_limit: int = JOB_RUN_LIMIT
    token_limit: int = 1
    final_summary: str | None = None
    failure_code: str | None = None
    # The floor for everything model-derived in this record: the highest
    # sensitivity any of its runs or sub-runs saw.
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL
    revision: int = 1

    def __post_init__(self) -> None:
        for value, name in (
            (self.job_id, "job_id"),
            (self.agent_id, "job agent_id"),
            (self.conversation_id, "job conversation_id"),
        ):
            _identity(value, name)
        _bounded(self.objective, "job objective", MAX_OBJECTIVE_CHARACTERS)
        sources = tuple(self.source_ids)
        if len(sources) > MAX_JOB_SOURCES or len(sources) != len(set(sources)):
            raise ValueError("job sources are duplicated or exceed their bound")
        for source in sources:
            _bounded(source, "job source id", 2_048)
        object.__setattr__(self, "source_ids", tuple(sorted(sources)))
        if not isinstance(self.state, JobState):
            raise TypeError("job state is invalid")
        _utc(self.created_at, "job created_at")
        _utc(self.updated_at, "job updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("job updated_at precedes created_at")
        checklist = tuple(self.checklist)
        if len(checklist) > MAX_CHECKLIST_ITEMS or any(
            not isinstance(item, ChecklistItem) for item in checklist
        ):
            raise ValueError("job checklist is invalid or exceeds its bound")
        validate_checklist(checklist)
        object.__setattr__(self, "checklist", checklist)
        delegations = tuple(self.delegations)
        if len(delegations) > MAX_DELEGATIONS or any(
            not isinstance(item, Delegation) for item in delegations
        ):
            raise ValueError("job delegations are invalid or exceed their bound")
        if len({item.id for item in delegations}) != len(delegations):
            raise ValueError("job delegation ids must be distinct")
        object.__setattr__(self, "delegations", delegations)
        answers = tuple(self.answers)
        if len(answers) > MAX_ANSWERS or any(
            not isinstance(item, UserAnswer) for item in answers
        ):
            raise ValueError("job answers are invalid or exceed their bound")
        object.__setattr__(self, "answers", answers)
        if self.question is not None:
            _bounded(self.question, "job question", MAX_QUESTION_CHARACTERS)
        if (self.question is not None) != (self.state is JobState.WAITING_FOR_YOU):
            raise ValueError("a job has a question exactly while it waits for you")
        if self.current_run_id is not None:
            _identity(self.current_run_id, "job current_run_id")
        _count(self.run_count, "job run_count")
        _count(self.tokens_used, "job tokens_used")
        _count(self.run_limit, "job run_limit", minimum=1)
        _count(self.token_limit, "job token_limit", minimum=1)
        _count(self.revision, "job revision", minimum=1)
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("job sensitivity is invalid")
        if self.final_summary is not None:
            _bounded(
                self.final_summary,
                "job final_summary",
                MAX_FINAL_SUMMARY_CHARACTERS,
                empty=True,
            )
        if self.state is JobState.DONE and self.final_summary is None:
            raise ValueError("a done job requires its final summary")
        if self.failure_code is not None:
            if _FAILURE_CODE.fullmatch(self.failure_code) is None:
                raise ValueError("job failure code is invalid")
            if self.state is not JobState.FAILED:
                raise ValueError("only a failed job carries a failure code")
        elif self.state is JobState.FAILED:
            raise ValueError("a failed job requires its failure code")

    @property
    def progress(self) -> tuple[int, int]:
        """Done items and items that still count (cancelled items do not)."""

        counted = [
            item
            for item in self.checklist
            if item.status is not ChecklistStatus.CANCELLED
        ]
        return (
            sum(item.status is ChecklistStatus.DONE for item in counted),
            len(counted),
        )

    @property
    def running_delegations(self) -> tuple[Delegation, ...]:
        return tuple(
            item for item in self.delegations if item.state is DelegationState.RUNNING
        )


def validate_checklist(items: tuple[ChecklistItem, ...]) -> None:
    """``after`` ids name existing items and form no cycle."""

    by_id = {item.id: item for item in items}
    if len(by_id) != len(items):
        raise JobError("checklist_invalid", "Checklist item ids must be distinct.")
    unknown = sorted({ref for item in items for ref in item.after} - set(by_id))
    if unknown:
        raise JobError(
            "checklist_invalid",
            "Unknown `after` ids: " + ", ".join(unknown) + ".",
        )
    visiting: set[str] = set()
    finished: set[str] = set()

    def visit(item_id: str, path: tuple[str, ...]) -> None:
        if item_id in finished:
            return
        if item_id in visiting:
            cycle = (*path[path.index(item_id) :], item_id)
            raise JobError(
                "checklist_invalid",
                "The `after` ids form a cycle: " + " -> ".join(cycle) + ".",
            )
        visiting.add(item_id)
        for dependency in by_id[item_id].after:
            visit(dependency, (*path, item_id))
        visiting.discard(item_id)
        finished.add(item_id)

    for item in items:
        visit(item.id, ())


def unfinished_dependencies(
    items: tuple[ChecklistItem, ...], item_id: str
) -> tuple[str, ...]:
    """The items ``item_id`` comes after that are not done (cancelled counts)."""

    by_id = {item.id: item for item in items}
    return tuple(
        dependency
        for dependency in by_id[item_id].after
        if by_id[dependency].status is not ChecklistStatus.DONE
    )


def require_startable(items: tuple[ChecklistItem, ...], item_id: str) -> None:
    waiting = unfinished_dependencies(items, item_id)
    if waiting:
        raise JobError(
            "checklist_item_blocked",
            f"Item {item_id} comes after items that are not done: "
            + ", ".join(waiting)
            + ".",
        )


def written_checklist(
    current: tuple[ChecklistItem, ...],
    items: Iterable[Mapping[str, object]],
    *,
    merge: bool,
) -> tuple[ChecklistItem, ...]:
    """Apply one ``checklist_write``: replace the list, or merge by id.

    Merged items keep omitted fields. Starting or finishing an item whose
    ``after`` items are not all done is refused, naming them.
    """

    previous = {item.id: item for item in current}
    result: dict[str, ChecklistItem] = dict(previous) if merge else {}
    for raw in items:
        item_id = raw.get("id")
        if not isinstance(item_id, str) or _ITEM_ID.fullmatch(item_id) is None:
            raise JobError(
                "checklist_invalid",
                "Each item needs an id of letters, digits, '.', '_' or '-' "
                "(at most 32).",
            )
        base = result.get(item_id) if merge else None
        title = raw.get("title")
        if title is None and base is None:
            raise JobError("checklist_invalid", f"Item {item_id} needs a title.")
        status = raw.get("status")
        after = raw.get("after")
        note = raw.get("note", base.note if base is not None else None)
        try:
            result[item_id] = ChecklistItem(
                id=item_id,
                title=(
                    clean_text(title, MAX_ITEM_TITLE_CHARACTERS)
                    if title is not None
                    else base.title  # type: ignore[union-attr]
                )
                or item_id,
                status=(
                    ChecklistStatus(status)
                    if isinstance(status, str)
                    else (base.status if base is not None else ChecklistStatus.PENDING)
                ),
                after=(
                    tuple(str(ref) for ref in after)
                    if isinstance(after, (tuple, list))
                    else (base.after if base is not None else ())
                ),
                note=(
                    None
                    if note is None
                    else clean_text(note, MAX_ITEM_NOTE_CHARACTERS, lines=True) or None
                ),
            )
        except ValueError as error:
            raise JobError(
                "checklist_invalid", f"Item {item_id} is invalid: {error}."
            ) from None
    written = tuple(result.values())
    if len(written) > MAX_CHECKLIST_ITEMS:
        raise JobError(
            "checklist_too_long",
            f"A checklist holds at most {MAX_CHECKLIST_ITEMS} items.",
        )
    validate_checklist(written)
    for item in written:
        before = previous.get(item.id)
        if item.status in {ChecklistStatus.IN_PROGRESS, ChecklistStatus.DONE} and (
            before is None or before.status is not item.status
        ):
            require_startable(written, item.id)
    return written


def with_item_status(
    items: tuple[ChecklistItem, ...], item_id: str, status: ChecklistStatus
) -> tuple[ChecklistItem, ...]:
    return tuple(
        replace(item, status=status) if item.id == item_id else item for item in items
    )


__all__ = [
    "INTERRUPTED_BY_RESTART",
    "JOB_CONTEXT_THRESHOLD",
    "JOB_LIMIT_FAILURES",
    "JOB_RUN_LIMIT",
    "JOB_TOKEN_LIMIT_RUNS",
    "LIVE_JOB_STATES",
    "MAX_CHECKLIST_ITEMS",
    "MAX_JOB_LIST_PAGE_SIZE",
    "TERMINAL_JOB_STATES",
    "UNANSWERED_APPROVAL_QUESTION",
    "UNCERTAIN_WRITE_QUESTION",
    "UPGRADED_JOB_SUMMARY",
    "ChecklistItem",
    "ChecklistStatus",
    "Delegation",
    "DelegationState",
    "Job",
    "JobError",
    "JobState",
    "UserAnswer",
    "clean_text",
    "require_startable",
    "unfinished_dependencies",
    "validate_checklist",
    "with_item_status",
    "written_checklist",
]
