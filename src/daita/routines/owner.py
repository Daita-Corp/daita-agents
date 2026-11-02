"""Own routine saves, controls and views for one agent."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from decimal import Decimal
from hashlib import sha256
from typing import Protocol
from zoneinfo import ZoneInfo

from .._json import FrozenJsonObject
from ..adapters.models import SourceRegistration
from ..capabilities import RecoveryKind, ToolRecovery, recovery_details
from ..distribution import Delivery, OutcomeArtifactReference
from ..errors import DaitaError, ErrorRetryability
from ..loop.models import LoopExit, Transcript
from ..scope import SOURCE_REF_PREFIX, ScopeRefs, scope_refs
from .models import (
    MAX_ROUTINE_HISTORY_PAGE_SIZE,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    SCHEDULE_INTERPRETER_REVISION,
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    NonexistentTimePolicy,
    OnceSchedule,
    RoutineControlAction,
    RoutineIntent,
    RoutineOccurrence,
    RoutineSchedule,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
    text_digest,
)
from .schedule import validate_schedule
from .temporal import next_weekday_utc, system_user_timezone

_SCOPE_REF_TOKEN = re.compile(r"\b[st][1-9][0-9]*\b")
MAX_RELATIVE_SCHEDULE_SECONDS = 30 * 24 * 60 * 60
SCHEDULE_KEYS = {
    "once": frozenset({"exact_at", "after_seconds"}),
    "interval": frozenset({"interval_seconds", "anchor_at", "anchor_after_seconds"}),
    "once_next_weekday": frozenset({"timezone", "weekday", "hour", "minute"}),
    "calendar": frozenset(
        {
            "timezone",
            "hour",
            "minute",
            "day_selector",
            "weekdays",
            "month_days",
            "months",
            "nonexistent_time_policy",
            "ambiguous_time_policy",
        }
    ),
}
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class RoutineError(DaitaError):
    def __init__(
        self, code: str, message: str, recovery: ToolRecovery | None = None
    ) -> None:
        self.code = code
        self.details = {} if recovery is None else recovery_details(recovery)
        super().__init__(
            message,
            error_code=code,
            retryability=ErrorRetryability.PERMANENT,
        )


class RoutineStore(Protocol):
    async def require_effects_unblocked(
        self, agent_id: str, *, run_id: str | None = None, routine_id: str | None = None
    ) -> None: ...

    async def admit_scheduled_routine(
        self, routine: ScheduledRoutine, *, run_immediately: bool = False
    ) -> ScheduledRoutine: ...

    async def load_scheduled_routine(
        self, agent_id: str, routine_id: str
    ) -> ScheduledRoutine | None: ...

    async def list_scheduled_routines(
        self,
        agent_id: str,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = MAX_ROUTINE_LIST_PAGE_SIZE,
    ) -> tuple[ScheduledRoutine, ...]: ...

    async def list_routine_occurrences(
        self,
        agent_id: str,
        routine_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrence, ...]: ...

    async def list_active_routine_occurrences(
        self,
        agent_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrence, ...]: ...

    async def load_routine_occurrences(
        self, agent_id: str, occurrence_ids: tuple[str, ...]
    ) -> tuple[RoutineOccurrence, ...]: ...

    async def revise_scheduled_routine(
        self,
        routine: ScheduledRoutine,
        *,
        expected_revision: int,
    ) -> ScheduledRoutine | None: ...

    async def transition_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        state: RoutineState,
        transitioned_at: datetime,
    ) -> ScheduledRoutine | None: ...

    async def claim_manual_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        authorized_control_call_id: str,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None: ...

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]: ...

    async def load(self, run_id: str) -> Transcript: ...


class RoutineSupervisorStore(Protocol):
    """Durable operations required by the routines supervisor."""

    async def claim_due_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        expected_due_at: datetime,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None: ...

    async def finalize_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
        failure_code: str | None = ...,
        artifact_references: tuple[OutcomeArtifactReference, ...] = ...,
        outcome_contract_failure_code: str | None = ...,
    ) -> tuple[RoutineOccurrence, Delivery | None] | None: ...

    async def list_scheduled_routines(
        self, agent_id: str, *, states: frozenset[RoutineState] = ..., limit: int = ...
    ) -> tuple[ScheduledRoutine, ...]: ...

    async def load_routine_occurrence(
        self, agent_id: str, occurrence_id: str
    ) -> RoutineOccurrence | None: ...

    async def load_scheduled_routine(
        self, agent_id: str, routine_id: str
    ) -> ScheduledRoutine | None: ...

    async def mark_routine_occurrence_run_terminal(
        self, agent_id: str, occurrence_id: str, *, run_id: str, terminal_at: datetime
    ) -> RoutineOccurrence | None: ...

    async def next_routine_deadline(self, agent_id: str) -> datetime | None: ...

    async def recover_stale_routine_occurrences(
        self,
        agent_id: str,
        *,
        recovered_at: datetime,
        claim_token_factory: Callable[[str], str],
    ) -> tuple[RoutineOccurrence, ...]: ...

    async def result(self, run_id: str) -> LoopExit | None: ...


@dataclass(frozen=True, slots=True)
class PreparedRoutine:
    """One validated routine save, not yet stored, and its confirm card."""

    routine: ScheduledRoutine
    expected_revision: int | None
    run_immediately: bool
    card: FrozenJsonObject


class RoutineOwner:
    """Validate routine saves and own agent-scoped routine transitions."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: RoutineStore,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("routine owner agent_id must be non-empty text")
        if not callable(clock):
            raise TypeError("routine owner clock must be callable")
        self.agent_id = agent_id
        self._store = store
        self._clock = clock
        self._wake: Callable[[], None] | None = None

    def bind_wake(self, wake: Callable[[], None]) -> None:
        if self._wake is not None:
            raise RuntimeError("routine owner wake callback is already bound")
        if not callable(wake):
            raise TypeError("routine wake callback must be callable")
        self._wake = wake

    def current_time(self) -> datetime:
        """The same trusted UTC clock used by routine admission and execution."""

        return self._clock()

    async def prepare(
        self,
        intent: RoutineIntent,
        *,
        conversation_id: str,
        identity: str,
        routine_id: str | None = None,
        expected_revision: int | None = None,
        origin_run_id: str | None = None,
        reference_at: datetime | None = None,
        relative_at: datetime | None = None,
    ) -> PreparedRoutine:
        """Validate one create (no ``routine_id``) or update and build its card.

        A named local instant resolves at ``reference_at`` (default: now), when
        the card was shown, so approval never moves it to a later week.
        Relative delays count from ``relative_at`` (default: ``reference_at``):
        the save after approval passes the moment it commits.
        """

        if not isinstance(intent, RoutineIntent):
            raise TypeError("routine intent must be RoutineIntent")
        now = self._clock()
        timing_at = reference_at or now
        saving_at = relative_at or timing_at
        current = None if routine_id is None else await self._load_owned(routine_id)
        if routine_id is not None and (
            current is None
            or (expected_revision is not None and current.revision != expected_revision)
        ):
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed or is no longer owned by this agent.",
            )
        if current is None:
            missing = [
                name
                for name in ("title", "prompt", "schedule")
                if getattr(intent, name) is None
            ]
            if missing:
                raise RoutineError(
                    "routine_argument_missing",
                    "A new routine needs " + ", ".join(missing) + ".",
                )
        elif intent.run_immediately:
            raise RoutineError(
                "routine_immediate_creation_only",
                "run_immediately is for creation; use run_now for an existing routine.",
            )
        prompt = (
            intent.prompt
            if intent.prompt is not None
            else current.authorized_instruction if current is not None else ""
        )
        refs: ScopeRefs | None = None
        if origin_run_id is not None:
            origin = await self._store.load(origin_run_id)
            _reject_scope_refs(prompt, origin)
            # The model names sources by the s# handles its own run shows it.
            refs = scope_refs(origin.run.resolved_source_scope)
        relative = isinstance(intent.schedule, FrozenJsonObject) and bool(
            {"after_seconds", "anchor_after_seconds"} & set(intent.schedule)
        )
        if isinstance(intent.schedule, FrozenJsonObject):
            schedule = parse_schedule(
                intent.schedule, reference_at=timing_at, relative_at=saving_at
            )
        elif intent.schedule is not None:
            schedule = intent.schedule
        else:
            assert current is not None
            schedule = current.schedule
        try:
            validate_schedule(schedule)
        except ValueError as error:
            raise RoutineError("routine_schedule_invalid", str(error)) from error
        if intent.run_immediately and isinstance(schedule, OnceSchedule):
            raise RoutineError(
                "routine_schedule_invalid",
                "run_immediately needs an interval or calendar schedule.",
            )
        if (
            intent.schedule is not None
            and isinstance(schedule, OnceSchedule)
            and schedule.exact_at <= saving_at
        ):
            raise RoutineError(
                "routine_schedule_passed",
                "The schedule's time passed before creation; choose a later time.",
            )
        registrations = tuple(
            item
            for item in await self._store.list_sources(self.agent_id)
            if item.active
        )
        source_ids = (
            _resolve_sources(intent.sources, registrations, refs)
            if intent.sources is not None
            else current.source_ids if current is not None else ()
        )
        title = (
            intent.title
            if intent.title is not None
            else current.title if current is not None else ""
        )
        if current is None:
            routine = ScheduledRoutine(
                routine_id="routine-"
                + sha256(f"{self.agent_id}\x00{identity}".encode()).hexdigest()[:32],
                agent_id=self.agent_id,
                conversation_id=conversation_id,
                owner_principal_id=f"agent:{self.agent_id}",
                title=title,
                authorized_instruction=prompt,
                instruction_digest=text_digest(prompt),
                schedule=schedule,
                schedule_interpreter_revision=SCHEDULE_INTERPRETER_REVISION,
                source_ids=source_ids,
                charged_tokens=0,
                charged_cost_usd=Decimal("0"),
                attempt_count=0,
                occurrence_count=0,
                consecutive_failures=0,
                next_due_at=None,
                active_occurrence_id=None,
                last_occurrence_id=None,
                last_delivery_ids=(),
                state=RoutineState.ACTIVE,
                revision=1,
                created_at=now,
                updated_at=now,
            )
        else:
            routine = replace(
                current,
                title=title,
                authorized_instruction=prompt,
                instruction_digest=text_digest(prompt),
                schedule=schedule,
                source_ids=source_ids,
                revision=current.revision + 1,
                updated_at=max(now, current.updated_at),
            )
        names = {item.id: item.display_name for item in registrations}
        card = FrozenJsonObject.from_mapping(
            {
                "title": routine.title,
                "schedule": schedule_text(
                    schedule,
                    run_immediately=intent.run_immediately,
                    from_approval=relative,
                ),
                "prompt": routine.authorized_instruction,
                "sources": (
                    tuple(names.get(item, item) for item in routine.source_ids)
                    if routine.source_ids
                    else "All current sources"
                ),
            }
        )
        return PreparedRoutine(
            routine=routine,
            expected_revision=None if current is None else current.revision,
            run_immediately=intent.run_immediately,
            card=card,
        )

    async def save(self, prepared: PreparedRoutine) -> ScheduledRoutine:
        """Store one prepared create or update."""

        try:
            if prepared.expected_revision is None:
                stored = await self._store.admit_scheduled_routine(
                    prepared.routine, run_immediately=prepared.run_immediately
                )
            else:
                revised = await self._store.revise_scheduled_routine(
                    prepared.routine, expected_revision=prepared.expected_revision
                )
                if revised is None:
                    raise RoutineError(
                        "routine_revision_changed",
                        "The routine changed or is no longer owned by this agent.",
                    )
                stored = revised
        except ValueError as error:
            raise self._store_error(error, "routine_save_failed") from error
        self._notify()
        return stored

    async def list(
        self,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = MAX_ROUTINE_LIST_PAGE_SIZE,
    ) -> tuple[ScheduledRoutineSummary, ...]:
        records = await self._store.list_scheduled_routines(
            self.agent_id,
            states=states,
            limit=limit,
        )
        return tuple(_summary(item) for item in records)

    async def inspect(self, routine_id: str) -> ScheduledRoutineInspection | None:
        routine = await self._load_owned(routine_id)
        if routine is None:
            return None
        occurrences = await self._store.list_routine_occurrences(
            self.agent_id,
            routine_id,
            limit=MAX_ROUTINE_HISTORY_PAGE_SIZE,
        )
        return ScheduledRoutineInspection(routine, tuple(occurrences))

    async def list_active_occurrences(
        self, *, limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE
    ) -> tuple[RoutineOccurrence, ...]:
        """Return this agent's queued or running occurrences, newest identity first."""

        return await self._store.list_active_routine_occurrences(
            self.agent_id, limit=limit
        )

    async def load_occurrences(
        self, occurrence_ids: tuple[str, ...]
    ) -> tuple[RoutineOccurrence, ...]:
        """Return the known occurrences among bounded exact identities."""

        return await self._store.load_routine_occurrences(self.agent_id, occurrence_ids)

    async def control(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        action: RoutineControlAction,
        authorized_control_call_id: str,
    ) -> ScheduledRoutine:
        current = await self._load_owned(routine_id)
        if current is None or current.revision != expected_revision:
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed or is no longer owned by this agent.",
            )
        now = self._clock()
        if action in {RoutineControlAction.RUN_NOW, RoutineControlAction.RESUME}:
            # An uncertain write stays unresolved until a person decides.
            await self._store.require_effects_unblocked(
                self.agent_id, routine_id=routine_id
            )
        if action is RoutineControlAction.RUN_NOW:
            if current.state is not RoutineState.ACTIVE:
                raise RoutineError(
                    "routine_not_active", "Only an active routine can run now."
                )
            claim_hash = sha256(
                f"{self.agent_id}\x00{routine_id}\x00{authorized_control_call_id}".encode(
                    "utf-8"
                )
            ).hexdigest()[:32]
            try:
                occurrence = await self._store.claim_manual_routine_occurrence(
                    self.agent_id,
                    routine_id,
                    expected_revision=expected_revision,
                    authorized_control_call_id=authorized_control_call_id,
                    claimed_at=now,
                    claim_token=f"routine-claim-{claim_hash}",
                )
            except ValueError as error:
                raise self._store_error(error, "routine_run_now_failed") from error
            if occurrence is None:
                raise RoutineError(
                    "routine_run_now_conflict",
                    "The routine cannot start a manual occurrence in its current state.",
                )
            self._notify()
            reloaded = await self._load_owned(routine_id)
            if reloaded is None:
                raise RoutineError("routine_not_found", "The routine no longer exists.")
            return reloaded
        target = {
            RoutineControlAction.PAUSE: RoutineState.PAUSED,
            RoutineControlAction.RESUME: RoutineState.ACTIVE,
            RoutineControlAction.DISABLE: RoutineState.DISABLED,
        }[action]
        if (
            action is RoutineControlAction.PAUSE
            and current.state is not RoutineState.ACTIVE
        ):
            raise RoutineError(
                "routine_not_active", "Only an active routine can be paused."
            )
        if action is RoutineControlAction.RESUME and current.state not in {
            RoutineState.PAUSED,
            RoutineState.NEEDS_ATTENTION,
        }:
            raise RoutineError("routine_not_paused", "This routine cannot be resumed.")
        if action is RoutineControlAction.DISABLE and current.state in {
            RoutineState.DISABLED,
            RoutineState.COMPLETED,
            RoutineState.EXPIRED,
        }:
            raise RoutineError("routine_terminal", "This routine is already terminal.")
        try:
            updated = await self._store.transition_scheduled_routine(
                self.agent_id,
                routine_id,
                expected_revision=expected_revision,
                state=target,
                transitioned_at=now,
            )
        except ValueError as error:
            raise self._store_error(error, "routine_control_failed") from error
        if updated is None:
            raise RoutineError(
                "routine_revision_changed", "The routine changed during control."
            )
        self._notify()
        return updated

    async def _load_owned(self, routine_id: str) -> ScheduledRoutine | None:
        if not isinstance(routine_id, str) or not routine_id:
            raise ValueError("routine_id must be non-empty text")
        return await self._store.load_scheduled_routine(self.agent_id, routine_id)

    @staticmethod
    def _store_error(error: ValueError, fallback: str) -> RoutineError:
        code = str(error)
        if not code.startswith("routine_"):
            code = fallback
        return RoutineError(code, "The routine transition failed its current contract.")

    def _notify(self) -> None:
        if self._wake is not None:
            self._wake()


def parse_schedule(
    value: Mapping[str, object],
    *,
    reference_at: datetime,
    relative_at: datetime | None = None,
) -> RoutineSchedule:
    """Turn the JSON schedule shape into a typed schedule.

    A named next local weekday resolves from ``reference_at``; relative
    timing (``after_seconds``, ``anchor_after_seconds``) counts from
    ``relative_at``, which defaults to ``reference_at``.
    """

    relative_at = relative_at or reference_at

    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in SCHEDULE_KEYS:
        raise RoutineError(
            "routine_schedule_invalid",
            "Schedule kind must be once, interval, once_next_weekday, or calendar.",
        )
    unexpected = set(value) - {"kind"} - SCHEDULE_KEYS[kind]
    if unexpected:
        raise RoutineError(
            "routine_schedule_invalid",
            f"Schedule kind {kind!r} does not accept: {', '.join(sorted(unexpected))}.",
        )
    if kind == "once" and ("exact_at" in value) == ("after_seconds" in value):
        raise RoutineError(
            "routine_schedule_invalid",
            "A once schedule needs exact_at or after_seconds.",
        )
    if kind == "interval" and ("anchor_at" in value) == (
        "anchor_after_seconds" in value
    ):
        raise RoutineError(
            "routine_schedule_invalid",
            "An interval schedule needs anchor_at or anchor_after_seconds.",
        )
    try:
        if kind == "once":
            return OnceSchedule(
                _relative(value, "after_seconds", relative_at)
                if "after_seconds" in value
                else _instant(value, "exact_at")
            )
        if kind == "interval":
            return IntervalSchedule(
                _integer(value, "interval_seconds"),
                (
                    _relative(value, "anchor_after_seconds", relative_at)
                    if "anchor_after_seconds" in value
                    else _instant(value, "anchor_at")
                ),
            )
        if kind == "once_next_weekday":
            return OnceSchedule(
                next_weekday_utc(
                    now=reference_at,
                    timezone=_string(value, "timezone"),
                    weekday=_integer(value, "weekday"),
                    hour=_integer(value, "hour"),
                    minute=_integer(value, "minute"),
                )
            )
        weekdays = _integers(value, "weekdays")
        month_days = _integers(value, "month_days")
        if "day_selector" in value:
            selector = CalendarDaySelector(_string(value, "day_selector"))
        elif weekdays and not month_days:
            selector = CalendarDaySelector.WEEKDAYS
        elif month_days and not weekdays:
            selector = CalendarDaySelector.MONTH_DAYS
        else:
            selector = CalendarDaySelector.EVERY_DAY
        return CalendarSchedule(
            timezone=_string(value, "timezone"),
            hour=_integer(value, "hour"),
            minute=_integer(value, "minute"),
            day_selector=selector,
            weekdays=weekdays,
            month_days=month_days,
            months=_integers(value, "months"),
            nonexistent_time_policy=NonexistentTimePolicy(
                str(value.get("nonexistent_time_policy", "skip"))
            ),
            ambiguous_time_policy=AmbiguousTimePolicy(
                str(value.get("ambiguous_time_policy", "first"))
            ),
        )
    except RoutineError:
        raise
    except (TypeError, ValueError) as error:
        message = str(error)
        if kind == "calendar":
            message += (
                ". Correct the schedule: day_selector weekdays takes weekdays, "
                "month_days takes month_days, every_day takes neither."
            )
        raise RoutineError("routine_schedule_invalid", message) from error


def schedule_text(
    schedule: RoutineSchedule,
    *,
    run_immediately: bool = False,
    from_approval: bool = False,
) -> str:
    """Describe a schedule in the person's local time."""

    zone_name = system_user_timezone()
    zone: tzinfo = UTC if zone_name is None else ZoneInfo(zone_name)
    if isinstance(schedule, OnceSchedule):
        text = "Once at " + _local(schedule.exact_at, zone)
    elif isinstance(schedule, IntervalSchedule):
        text = f"Every {_duration(schedule.interval_seconds)} from " + _local(
            schedule.anchor_at, zone
        )
    else:
        if schedule.day_selector is CalendarDaySelector.WEEKDAYS:
            days = "on " + ", ".join(_WEEKDAYS[day - 1] for day in schedule.weekdays)
        elif schedule.day_selector is CalendarDaySelector.MONTH_DAYS:
            days = "on day " + ", ".join(map(str, schedule.month_days))
        else:
            days = "every day"
        months = (
            " in months " + ", ".join(map(str, schedule.months))
            if schedule.months
            else ""
        )
        text = (
            f"Calendar {days}{months} at {schedule.hour:02d}:{schedule.minute:02d} "
            f"{schedule.timezone}"
        )
    if from_approval:
        text += " (timed from approval)"
    return ("Now, then " + text[0].lower() + text[1:]) if run_immediately else text


def schedule_payload(schedule: RoutineSchedule) -> dict[str, object]:
    payload: dict[str, object] = {"kind": schedule.kind.value}
    if isinstance(schedule, OnceSchedule):
        payload["exact_at"] = schedule.exact_at.isoformat()
    elif isinstance(schedule, IntervalSchedule):
        payload.update(
            {
                "interval_seconds": schedule.interval_seconds,
                "anchor_at": schedule.anchor_at.isoformat(),
            }
        )
    else:
        assert isinstance(schedule, CalendarSchedule)
        payload.update(
            {
                "timezone": schedule.timezone,
                "hour": schedule.hour,
                "minute": schedule.minute,
                "day_selector": schedule.day_selector.value,
                "weekdays": schedule.weekdays,
                "month_days": schedule.month_days,
                "months": schedule.months,
                "nonexistent_time_policy": schedule.nonexistent_time_policy.value,
                "ambiguous_time_policy": schedule.ambiguous_time_policy.value,
            }
        )
    return payload


def _summary(routine: ScheduledRoutine) -> ScheduledRoutineSummary:
    return ScheduledRoutineSummary(
        routine_id=routine.routine_id,
        title=routine.title,
        state=routine.state,
        schedule_kind=routine.schedule.kind,
        next_due_at=routine.next_due_at,
        revision=routine.revision,
        occurrence_count=routine.occurrence_count,
        consecutive_failures=routine.consecutive_failures,
    )


def _resolve_sources(
    requested: tuple[str, ...],
    registrations: tuple[SourceRegistration, ...],
    refs: ScopeRefs | None = None,
) -> tuple[str, ...]:
    """Resolve source handles, names or ids against the agent's current sources.

    Handles (``s1``, ``s2``...) mean something only with the origin run's refs.
    """

    resolved: set[str] = set()
    for requested_item in requested:
        item = (
            refs.resolve(SOURCE_REF_PREFIX, requested_item) if refs else None
        ) or requested_item
        matches = {
            source.id
            for source in registrations
            if item == source.id or item.casefold() == source.display_name.casefold()
        }
        if len(matches) != 1:
            names = ", ".join(sorted(source.display_name for source in registrations))
            raise RoutineError(
                "routine_source_unknown",
                f"No single current source is named {requested_item!r}.",
                ToolRecovery(
                    RecoveryKind.CORRECT_ARGUMENTS,
                    ("Current sources: " + names)[:512] if names else "No sources.",
                ),
            )
        resolved |= matches
    return tuple(sorted(resolved))


def _local(value: datetime, zone: tzinfo) -> str:
    return value.astimezone(zone).strftime("%a %Y-%m-%d %H:%M %Z")


def _duration(seconds: int) -> str:
    for size, unit in ((86_400, "day"), (3_600, "hour"), (60, "minute")):
        if seconds % size == 0:
            count = seconds // size
            return f"{count} {unit}" + ("" if count == 1 else "s")
    return f"{seconds} seconds"


def _relative(
    values: Mapping[str, object], name: str, reference_at: datetime
) -> datetime:
    seconds = _integer(values, name)
    if not 1 <= seconds <= MAX_RELATIVE_SCHEDULE_SECONDS:
        raise RoutineError(
            "routine_relative_time_invalid", "Relative timing is outside its bound."
        )
    return reference_at.astimezone(UTC) + timedelta(seconds=seconds)


def _instant(values: Mapping[str, object], name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_string(values, name))
    except ValueError as error:
        raise RoutineError(
            "routine_datetime_invalid", f"{name} must be ISO 8601 text."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RoutineError(
            "routine_datetime_invalid", f"{name} must include a UTC offset."
        )
    return parsed.astimezone(UTC)


def _string(values: Mapping[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise RoutineError("routine_schedule_invalid", f"{name} must be text.")
    return value


def _integer(values: Mapping[str, object], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RoutineError("routine_schedule_invalid", f"{name} must be an integer.")
    return value


def _integers(values: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = values.get(name, ())
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise RoutineError(
            "routine_schedule_invalid", f"{name} must be an integer array."
        )
    return tuple(value)


def _reject_scope_refs(instruction: str, origin: Transcript) -> None:
    """Refuse run-local s#/t# handles that a later scheduled run would renumber."""

    issued = scope_refs(origin.run.resolved_source_scope).ids
    named = sorted(
        {item for item in _SCOPE_REF_TOKEN.findall(instruction) if item in issued}
    )
    if named:
        raise RoutineError(
            "routine_instruction_scope_ref",
            f"prompt names this run's handles {', '.join(named)}, "
            "which later runs number differently.",
            ToolRecovery(
                RecoveryKind.CORRECT_ARGUMENTS,
                "Name each source or resource in prompt by its catalog name or full "
                "id: " + "; ".join(f"{item} is {issued[item]}" for item in named) + ".",
            ),
        )


__all__ = [
    "PreparedRoutine",
    "RoutineError",
    "RoutineOwner",
    "RoutineStore",
    "RoutineSupervisorStore",
    "parse_schedule",
    "schedule_payload",
    "schedule_text",
]
