"""Strict bounded values for scheduled routines and occurrences."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from typing import TypeAlias, cast

from .._json import FrozenJsonObject
from ..capabilities import ExecutionScope
from ..distribution.models import (
    MAX_DISTRIBUTION_TARGETS,
    MAX_OUTCOME_ARTIFACT_BYTES,
    MAX_OUTCOME_ARTIFACT_REFERENCES,
    MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
    ArtifactRequirement,
    OutcomeContract,
)
from ..llm.models import ModelSensitivity

SCHEDULE_INTERPRETER_REVISION = 1

MAX_SCHEDULED_ROUTINES_PER_AGENT = 128
MAX_ACTIVE_ROUTINES_PER_AGENT = 32
MAX_ROUTINE_OCCURRENCES = 10_000
MAX_ROUTINE_ATTEMPTS = 3
# Code defaults: a recurring routine pauses as Needs you after this many
# consecutive failures; the counter's stored bound admits older records.
ROUTINE_FAILURE_LIMIT = 3
MAX_ROUTINE_CONSECUTIVE_FAILURES = 10
MAX_ROUTINE_LIST_PAGE_SIZE = 50
MAX_ROUTINE_HISTORY_PAGE_SIZE = 100
MAX_ROUTINE_INSTRUCTION_BYTES = 16 * 1024
MAX_ROUTINE_TITLE_CHARACTERS = 256
MAX_ROUTINE_IDENTITY_ITEMS = 64
MAX_ROUTINE_IDENTITY_CHARACTERS = 1_024
MIN_ROUTINE_INTERVAL_SECONDS = 60
MAX_ROUTINE_INTERVAL_SECONDS = 366 * 5 * 24 * 60 * 60
MAX_SCHEDULE_LOOKAHEAD_DAYS = 366 * 8
ROUTINE_CLAIM_LEASE_SECONDS = 30.0
# Code defaults for every occurrence: delivery to the conversation's Inbox with
# no sensitivity ceiling, and any artifacts the run commits within the shared
# delivery bounds. Nothing about them is chosen per routine.
ROUTINE_INBOX_SENSITIVITY = ModelSensitivity.RESTRICTED
ROUTINE_OUTCOME = OutcomeContract(
    require_terminal_conclusion=True,
    artifact_requirements=(
        ArtifactRequirement(
            required=False,
            minimum_count=0,
            maximum_count=MAX_OUTCOME_ARTIFACT_REFERENCES,
            allowed_media_types=(),
            allowed_authorships=(),
            allowed_producer_capability_ids=(),
            maximum_artifact_bytes=MAX_OUTCOME_ARTIFACT_BYTES,
            maximum_total_bytes=MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
            maximum_sensitivity=ROUTINE_INBOX_SENSITIVITY,
        ),
    ),
    maximum_total_artifact_bytes=MAX_OUTCOME_TOTAL_ARTIFACT_BYTES,
    maximum_effective_sensitivity=ROUTINE_INBOX_SENSITIVITY,
    require_current_run_provenance=True,
    require_exact_source_bindings=False,
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")


def _text(
    value: str,
    name: str,
    *,
    maximum: int = MAX_ROUTINE_IDENTITY_CHARACTERS,
    single_line: bool = True,
) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or (single_line and any(character in "\r\n" for character in value))
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _optional_utc(value: datetime | None, name: str) -> None:
    if value is not None:
        _utc(value, name)


def _positive_revision(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _bounded_count(value: int, name: str, *, maximum: int | None) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} is outside its bound")


def _money(value: Decimal, name: str, *, maximum: Decimal | None) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a bounded non-negative Decimal")


def _identities(
    values: Iterable[str],
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    items = tuple(values)
    if (not allow_empty and not items) or len(items) > MAX_ROUTINE_IDENTITY_ITEMS:
        raise ValueError(f"{name} is empty or exceeds its bound")
    for item in items:
        _text(item, name)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(items))


def _delivery_ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) > MAX_DISTRIBUTION_TARGETS:
        raise ValueError(f"{name} exceeds its bound")
    for item in values:
        _text(item, name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} cannot contain duplicates")
    return values


def text_digest(value: str) -> str:
    """Return the canonical digest used for authorized routine instructions."""

    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


class ScheduleKind(str, Enum):
    ONCE = "once"
    INTERVAL = "interval"
    CALENDAR = "calendar"


class CalendarDaySelector(str, Enum):
    EVERY_DAY = "every_day"
    WEEKDAYS = "weekdays"
    MONTH_DAYS = "month_days"


class NonexistentTimePolicy(str, Enum):
    SKIP = "skip"
    NEXT_VALID = "next_valid"


class AmbiguousTimePolicy(str, Enum):
    FIRST = "first"
    SECOND = "second"


class RoutineState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"
    DISABLED = "disabled"
    NEEDS_ATTENTION = "needs_attention"


class RoutineSlotKind(str, Enum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class RoutineOccurrenceDisposition(str, Enum):
    CLAIMED = "claimed"
    RUNNING = "running"
    RUN_TERMINAL_PENDING_FINALIZATION = "run_terminal_pending_finalization"
    COMPLETED = "completed"
    RETRYABLE = "retryable"
    TERMINAL_FAILED = "terminal_failed"


ACTIVE_OCCURRENCE_DISPOSITIONS = frozenset(
    {
        RoutineOccurrenceDisposition.CLAIMED,
        RoutineOccurrenceDisposition.RUNNING,
        RoutineOccurrenceDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
        RoutineOccurrenceDisposition.RETRYABLE,
    }
)


class RoutineControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    RUN_NOW = "run_now"
    DISABLE = "disable"


@dataclass(frozen=True, slots=True)
class OnceSchedule:
    exact_at: datetime

    def __post_init__(self) -> None:
        _utc(self.exact_at, "once schedule exact_at")

    @property
    def kind(self) -> ScheduleKind:
        return ScheduleKind.ONCE


@dataclass(frozen=True, slots=True)
class IntervalSchedule:
    interval_seconds: int
    anchor_at: datetime

    def __post_init__(self) -> None:
        _utc(self.anchor_at, "interval schedule anchor_at")
        if (
            not isinstance(self.interval_seconds, int)
            or isinstance(self.interval_seconds, bool)
            or not MIN_ROUTINE_INTERVAL_SECONDS
            <= self.interval_seconds
            <= MAX_ROUTINE_INTERVAL_SECONDS
        ):
            raise ValueError("interval schedule duration is outside its bound")

    @property
    def kind(self) -> ScheduleKind:
        return ScheduleKind.INTERVAL


@dataclass(frozen=True, slots=True)
class CalendarSchedule:
    timezone: str
    hour: int
    minute: int
    day_selector: CalendarDaySelector
    weekdays: tuple[int, ...] = ()
    month_days: tuple[int, ...] = ()
    months: tuple[int, ...] = ()
    nonexistent_time_policy: NonexistentTimePolicy = NonexistentTimePolicy.SKIP
    ambiguous_time_policy: AmbiguousTimePolicy = AmbiguousTimePolicy.FIRST

    def __post_init__(self) -> None:
        _text(self.timezone, "calendar timezone", maximum=128)
        if "/" not in self.timezone or self.timezone.upper() in {
            "EST",
            "EDT",
            "CST",
            "CDT",
            "MST",
            "MDT",
            "PST",
            "PDT",
        }:
            raise ValueError("calendar timezone must be an exact IANA zone")
        if (
            not isinstance(self.hour, int)
            or isinstance(self.hour, bool)
            or not 0 <= self.hour <= 23
        ):
            raise ValueError("calendar hour is outside its bound")
        if (
            not isinstance(self.minute, int)
            or isinstance(self.minute, bool)
            or not 0 <= self.minute <= 59
        ):
            raise ValueError("calendar minute is outside its bound")
        if not isinstance(self.day_selector, CalendarDaySelector):
            raise TypeError("calendar day_selector is invalid")
        weekdays = tuple(self.weekdays)
        month_days = tuple(self.month_days)
        months = tuple(self.months)
        for values, name, lower, upper in (
            (weekdays, "calendar weekdays", 1, 7),
            (month_days, "calendar month_days", 1, 31),
            (months, "calendar months", 1, 12),
        ):
            if len(values) != len(set(values)) or values != tuple(sorted(values)):
                raise ValueError(f"{name} must be sorted and distinct")
            if any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or not lower <= item <= upper
                for item in values
            ):
                raise ValueError(f"{name} contains an invalid value")
        if self.day_selector is CalendarDaySelector.EVERY_DAY:
            if weekdays or month_days:
                raise ValueError("every_day calendar cannot contain day filters")
        elif self.day_selector is CalendarDaySelector.WEEKDAYS:
            if not weekdays or month_days:
                raise ValueError("weekday calendar requires only exact weekdays")
        elif not month_days or weekdays:
            raise ValueError("month-day calendar requires only exact month days")
        if not isinstance(self.nonexistent_time_policy, NonexistentTimePolicy):
            raise TypeError("calendar nonexistent-time policy is invalid")
        if not isinstance(self.ambiguous_time_policy, AmbiguousTimePolicy):
            raise TypeError("calendar ambiguous-time policy is invalid")
        object.__setattr__(self, "weekdays", weekdays)
        object.__setattr__(self, "month_days", month_days)
        object.__setattr__(self, "months", months)

    @property
    def kind(self) -> ScheduleKind:
        return ScheduleKind.CALENDAR


RoutineSchedule: TypeAlias = OnceSchedule | IntervalSchedule | CalendarSchedule


@dataclass(frozen=True, slots=True)
class RoutineIntent:
    """What a person asks a routine to do: the one shape every surface uses.

    ``schedule`` is a typed schedule or its JSON shape, which may use relative
    timing that the owner resolves when the routine is saved. On update an
    omitted (``None``) field keeps its stored value.
    """

    title: str | None = None
    prompt: str | None = None
    schedule: RoutineSchedule | FrozenJsonObject | None = None
    sources: tuple[str, ...] | None = None
    run_immediately: bool = False

    def __post_init__(self) -> None:
        if self.title is not None:
            _text(self.title, "routine title", maximum=MAX_ROUTINE_TITLE_CHARACTERS)
        if self.prompt is not None:
            _text(
                self.prompt,
                "routine prompt",
                maximum=MAX_ROUTINE_INSTRUCTION_BYTES,
                single_line=False,
            )
            if len(self.prompt.encode("utf-8")) > MAX_ROUTINE_INSTRUCTION_BYTES:
                raise ValueError("routine prompt exceeds its bound")
        schedule = self.schedule
        if isinstance(schedule, Mapping) and not isinstance(schedule, FrozenJsonObject):
            schedule = FrozenJsonObject.from_mapping(schedule)
        if schedule is not None and not isinstance(
            schedule,
            (OnceSchedule, IntervalSchedule, CalendarSchedule, FrozenJsonObject),
        ):
            raise TypeError("routine schedule must be a schedule or its JSON shape")
        sources = self.sources
        if sources is not None:
            if isinstance(sources, (str, bytes)):
                raise TypeError("routine sources must be a sequence")
            sources = tuple(sources)
            if len(sources) > MAX_ROUTINE_IDENTITY_ITEMS or len(set(sources)) != len(
                sources
            ):
                raise ValueError("routine sources are duplicated or exceed their bound")
            for item in sources:
                _text(item, "routine source")
        if type(self.run_immediately) is not bool:
            raise TypeError("run_immediately must be a boolean")
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "sources", sources)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RoutineIntent:
        """Parse a tool call, CLI spec or flags; unknown keys are refused."""

        unknown = set(value) - set(_INTENT_FIELDS)
        if unknown:
            raise ValueError(
                "unknown routine fields: " + ", ".join(sorted(map(str, unknown)))
            )
        sources = value.get("sources")
        schedule = value.get("schedule")
        if schedule is not None and not isinstance(schedule, Mapping):
            raise TypeError("routine schedule must be an object")
        for name in ("title", "prompt"):
            if value.get(name) is not None and not isinstance(value[name], str):
                raise TypeError(f"routine {name} must be text")
        if sources is not None and not isinstance(sources, (list, tuple)):
            raise TypeError("routine sources must be a list")
        return cls(
            title=cast(str | None, value.get("title")),
            prompt=cast(str | None, value.get("prompt")),
            schedule=(
                None if schedule is None else FrozenJsonObject.from_mapping(schedule)
            ),
            sources=None if sources is None else tuple(cast(Iterable[str], sources)),
            run_immediately=cast(bool, value.get("run_immediately", False)),
        )


_INTENT_FIELDS = ("title", "prompt", "schedule", "sources", "run_immediately")


@dataclass(frozen=True, slots=True)
class ScheduledRoutine:
    routine_id: str
    agent_id: str
    conversation_id: str
    owner_principal_id: str
    title: str
    authorized_instruction: str
    instruction_digest: str
    schedule: RoutineSchedule
    schedule_interpreter_revision: int
    source_ids: tuple[str, ...]
    charged_tokens: int
    charged_cost_usd: Decimal
    attempt_count: int
    occurrence_count: int
    consecutive_failures: int
    next_due_at: datetime | None
    active_occurrence_id: str | None
    last_occurrence_id: str | None
    last_delivery_ids: tuple[str, ...]
    state: RoutineState
    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.routine_id, "routine_id"),
            (self.agent_id, "routine agent_id"),
            (self.conversation_id, "routine conversation_id"),
            (self.owner_principal_id, "routine owner_principal_id"),
        ):
            _text(value, name)
        _text(self.title, "routine title", maximum=MAX_ROUTINE_TITLE_CHARACTERS)
        _text(
            self.authorized_instruction,
            "routine authorized_instruction",
            maximum=MAX_ROUTINE_INSTRUCTION_BYTES,
            single_line=False,
        )
        if (
            len(self.authorized_instruction.encode("utf-8"))
            > MAX_ROUTINE_INSTRUCTION_BYTES
        ):
            raise ValueError("routine authorized_instruction exceeds its byte bound")
        _digest(self.instruction_digest, "routine instruction_digest")
        if self.instruction_digest != text_digest(self.authorized_instruction):
            raise ValueError("routine instruction digest does not match its content")
        if not isinstance(
            self.schedule, (OnceSchedule, IntervalSchedule, CalendarSchedule)
        ):
            raise TypeError("routine schedule is invalid")
        if self.schedule_interpreter_revision != SCHEDULE_INTERPRETER_REVISION:
            raise ValueError("routine schedule interpreter revision is unsupported")
        sources = _identities(self.source_ids, "routine source_ids")
        for count_value, count_name, maximum in (
            (self.charged_tokens, "charged token amount", None),
            (self.attempt_count, "routine attempt count", None),
            (self.occurrence_count, "routine occurrence count", None),
            (
                self.consecutive_failures,
                "consecutive failures",
                MAX_ROUTINE_CONSECUTIVE_FAILURES,
            ),
        ):
            _bounded_count(count_value, count_name, maximum=maximum)
        _money(self.charged_cost_usd, "charged cost amount", maximum=None)
        _utc(self.created_at, "routine created_at")
        _utc(self.updated_at, "routine updated_at")
        _optional_utc(self.next_due_at, "routine next_due_at")
        if self.updated_at < self.created_at:
            raise ValueError("routine updated_at precedes created_at")
        for optional_identity, identity_name in (
            (self.active_occurrence_id, "active occurrence_id"),
            (self.last_occurrence_id, "last occurrence_id"),
        ):
            if optional_identity is not None:
                _text(optional_identity, identity_name)
        last_delivery_ids = _delivery_ids(
            self.last_delivery_ids,
            "routine last_delivery_ids",
        )
        if not isinstance(self.state, RoutineState):
            raise TypeError("routine state is invalid")
        _positive_revision(self.revision, "routine revision")
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "last_delivery_ids", last_delivery_ids)


@dataclass(frozen=True, slots=True)
class RoutineOccurrence:
    occurrence_id: str
    agent_id: str
    routine_id: str
    routine_revision: int
    slot_kind: RoutineSlotKind
    slot_key: str
    scheduled_for: datetime
    claimed_at: datetime | None
    claim_token: str | None
    lease_expires_at: datetime | None
    execution_scope: ExecutionScope | None
    execution_scope_digest: str | None
    reserved_run_id: str | None
    charged_tokens: int
    charged_cost_usd: Decimal
    run_bound_at: datetime | None
    run_terminal_at: datetime | None
    conclusion_digest: str | None
    terminal_run_id: str | None
    delivery_ids: tuple[str, ...]
    attempt_count: int
    failure_code: str | None
    retry_at: datetime | None
    disposition: RoutineOccurrenceDisposition
    created_at: datetime
    updated_at: datetime
    effect_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for identity_value, identity_name in (
            (self.occurrence_id, "occurrence_id"),
            (self.agent_id, "occurrence agent_id"),
            (self.routine_id, "occurrence routine_id"),
            (self.slot_key, "occurrence slot_key"),
        ):
            _text(identity_value, identity_name)
        _positive_revision(self.routine_revision, "occurrence routine_revision")
        if not isinstance(self.slot_kind, RoutineSlotKind):
            raise TypeError("occurrence slot kind is invalid")
        if not self.slot_key.startswith(f"{self.slot_kind.value}:"):
            raise ValueError("occurrence slot key does not match its kind")
        _utc(self.scheduled_for, "occurrence scheduled_for")
        for optional_instant, instant_name in (
            (self.claimed_at, "occurrence claimed_at"),
            (self.lease_expires_at, "occurrence lease_expires_at"),
            (self.run_bound_at, "occurrence run_bound_at"),
            (self.run_terminal_at, "occurrence run_terminal_at"),
            (self.retry_at, "occurrence retry_at"),
        ):
            _optional_utc(optional_instant, instant_name)
        if (self.claimed_at is None) != (self.claim_token is None):
            raise ValueError("occurrence claim time and token must be present together")
        if self.claim_token is not None:
            _text(self.claim_token, "occurrence claim_token")
        if self.lease_expires_at is not None and self.claimed_at is None:
            raise ValueError("occurrence lease requires a claim")
        if (
            self.lease_expires_at is not None
            and self.claimed_at is not None
            and self.lease_expires_at <= self.claimed_at
        ):
            raise ValueError("occurrence lease must expire after claim")
        if (self.execution_scope is None) != (self.execution_scope_digest is None):
            raise ValueError(
                "occurrence execution scope and digest must be present together"
            )
        if (
            self.execution_scope is not None
            and self.execution_scope.digest != self.execution_scope_digest
        ):
            raise ValueError("occurrence execution scope digest does not match")
        for optional_identity, identity_name in (
            (self.reserved_run_id, "occurrence reserved_run_id"),
            (self.terminal_run_id, "occurrence terminal_run_id"),
        ):
            if optional_identity is not None:
                _text(optional_identity, identity_name)
        delivery_ids = _delivery_ids(self.delivery_ids, "occurrence delivery_ids")
        if self.run_bound_at is not None and self.reserved_run_id is None:
            raise ValueError("occurrence run binding requires a reserved run")
        if self.run_terminal_at is not None and self.terminal_run_id is None:
            raise ValueError("occurrence terminal time requires terminal run identity")
        if (
            self.terminal_run_id is not None
            and self.terminal_run_id != self.reserved_run_id
        ):
            raise ValueError("occurrence terminal run must be its reserved run")
        if self.conclusion_digest is not None:
            _digest(self.conclusion_digest, "occurrence conclusion_digest")
        for count_value, count_name, maximum in (
            (
                self.charged_tokens,
                "occurrence charged tokens",
                None,
            ),
            (self.attempt_count, "occurrence attempt count", MAX_ROUTINE_ATTEMPTS),
        ):
            _bounded_count(count_value, count_name, maximum=maximum)
        _money(
            self.charged_cost_usd,
            "occurrence charged cost",
            maximum=None,
        )
        if (
            self.failure_code is not None
            and _FAILURE_CODE.fullmatch(self.failure_code) is None
        ):
            raise ValueError("occurrence failure_code is invalid")
        if not isinstance(self.disposition, RoutineOccurrenceDisposition):
            raise TypeError("occurrence disposition is invalid")
        _utc(self.created_at, "occurrence created_at")
        _utc(self.updated_at, "occurrence updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("occurrence updated_at precedes created_at")
        receipts = tuple(self.effect_receipt_ids)
        if (
            len(receipts) > 256
            or len(set(receipts)) != len(receipts)
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"effect-receipt:sha256:[0-9a-f]{64}", item) is None
                for item in receipts
            )
        ):
            raise ValueError(
                "occurrence effect receipt references are invalid or exceed their bound"
            )
        object.__setattr__(self, "effect_receipt_ids", receipts)
        object.__setattr__(self, "delivery_ids", delivery_ids)


@dataclass(frozen=True, slots=True)
class ScheduledRoutineSummary:
    routine_id: str
    title: str
    state: RoutineState
    schedule_kind: ScheduleKind
    next_due_at: datetime | None
    revision: int
    occurrence_count: int
    consecutive_failures: int

    def __post_init__(self) -> None:
        _text(self.routine_id, "routine summary routine_id")
        _text(self.title, "routine summary title", maximum=MAX_ROUTINE_TITLE_CHARACTERS)
        if not isinstance(self.state, RoutineState):
            raise TypeError("routine summary state is invalid")
        if not isinstance(self.schedule_kind, ScheduleKind):
            raise TypeError("routine summary schedule kind is invalid")
        _optional_utc(self.next_due_at, "routine summary next_due_at")
        _positive_revision(self.revision, "routine summary revision")
        _bounded_count(
            self.occurrence_count, "routine summary occurrence count", maximum=None
        )
        _bounded_count(
            self.consecutive_failures,
            "routine summary failure count",
            maximum=MAX_ROUTINE_CONSECUTIVE_FAILURES,
        )


@dataclass(frozen=True, slots=True)
class ScheduledRoutineInspection:
    routine: ScheduledRoutine
    recent_occurrences: tuple[RoutineOccurrence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.routine, ScheduledRoutine):
            raise TypeError("routine inspection requires a routine")
        occurrences = tuple(self.recent_occurrences)
        if len(occurrences) > MAX_ROUTINE_HISTORY_PAGE_SIZE:
            raise ValueError("routine inspection history exceeds its bound")
        if any(
            not isinstance(item, RoutineOccurrence)
            or item.agent_id != self.routine.agent_id
            or item.routine_id != self.routine.routine_id
            for item in occurrences
        ):
            raise ValueError("routine inspection occurrence identity is invalid")
        object.__setattr__(self, "recent_occurrences", occurrences)
