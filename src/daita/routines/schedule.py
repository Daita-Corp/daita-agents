"""Pure deterministic schedule mathematics for D1 routines."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .._json import canonical_json
from .models import (
    MAX_ROUTINE_OCCURRENCES,
    MAX_SCHEDULE_LOOKAHEAD_DAYS,
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    MisfirePolicy,
    NonexistentTimePolicy,
    OnceSchedule,
    RoutineSchedule,
)


@dataclass(frozen=True, slots=True)
class DueSlotSelection:
    selected_at: datetime | None
    next_due_at: datetime | None
    skipped_slots: int


def validate_schedule(schedule: RoutineSchedule) -> None:
    """Validate zone availability and prove a calendar has a realizable date."""

    if isinstance(schedule, (OnceSchedule, IntervalSchedule)):
        return
    try:
        ZoneInfo(schedule.timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError("calendar timezone is not available") from error
    if not any(
        _date_matches(schedule, date(year, month, day))
        for year in range(2000, 2400)
        for month in (schedule.months or tuple(range(1, 13)))
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ):
        raise ValueError("calendar schedule cannot produce a valid local date")


def first_slot(
    schedule: RoutineSchedule,
    *,
    not_before: datetime,
    expires_at: datetime,
) -> datetime | None:
    """Return the first slot at or after ``not_before`` within expiration."""

    _utc(not_before, "schedule not_before")
    _utc(expires_at, "schedule expires_at")
    if expires_at < not_before:
        return None
    candidate = _next_slot(schedule, boundary=not_before, inclusive=True)
    if candidate is None or candidate > expires_at:
        return None
    return candidate


def next_slot(
    schedule: RoutineSchedule,
    *,
    after: datetime,
    expires_at: datetime,
) -> datetime | None:
    """Return the first slot strictly after ``after`` within expiration."""

    _utc(after, "schedule after")
    _utc(expires_at, "schedule expires_at")
    candidate = _next_slot(schedule, boundary=after, inclusive=False)
    if candidate is None or candidate > expires_at:
        return None
    return candidate


def select_due_slot(
    schedule: RoutineSchedule,
    *,
    materialized_due_at: datetime,
    now: datetime,
    expires_at: datetime,
    misfire_policy: MisfirePolicy,
) -> DueSlotSelection:
    """Select one bounded due slot and materialize the next future instant.

    A single outstanding slot is eligible under both policies. If multiple
    slots accumulated, ``skip`` omits all of them and ``latest_only`` selects
    only the most recent. This distinguishes an ordinary late wake from a
    missed backlog without introducing a wall-clock grace heuristic.
    """

    for value, name in (
        (materialized_due_at, "materialized due instant"),
        (now, "schedule now"),
        (expires_at, "schedule expires_at"),
    ):
        _utc(value, name)
    if not isinstance(misfire_policy, MisfirePolicy):
        raise TypeError("misfire policy is invalid")
    if materialized_due_at > now or materialized_due_at > expires_at:
        return DueSlotSelection(None, materialized_due_at, 0)

    due_count = 1
    latest = materialized_due_at
    cursor = materialized_due_at
    while due_count <= MAX_ROUTINE_OCCURRENCES:
        following = next_slot(schedule, after=cursor, expires_at=expires_at)
        if following is None or following > now:
            if due_count == 1:
                selected = latest
                skipped = 0
            elif misfire_policy is MisfirePolicy.LATEST_ONLY:
                selected = latest
                skipped = due_count - 1
            else:
                selected = None
                skipped = due_count
            return DueSlotSelection(selected, following, skipped)
        latest = following
        cursor = following
        due_count += 1
    raise ValueError("missed schedule exceeds the occurrence bound")


def scheduled_slot_key(
    routine_id: str,
    routine_revision: int,
    scheduled_for: datetime,
) -> str:
    _identity(routine_id, "routine_id")
    _positive_revision(routine_revision)
    _utc(scheduled_for, "scheduled_for")
    return (
        "scheduled:"
        + sha256(
            canonical_json(
                {
                    "routine_id": routine_id,
                    "routine_revision": routine_revision,
                    "scheduled_for": scheduled_for.isoformat(),
                }
            ).encode("utf-8")
        ).hexdigest()
    )


def manual_slot_key(
    routine_id: str,
    routine_revision: int,
    control_call_id: str,
) -> str:
    _identity(routine_id, "routine_id")
    _identity(control_call_id, "control_call_id")
    _positive_revision(routine_revision)
    return (
        "manual:"
        + sha256(
            canonical_json(
                {
                    "routine_id": routine_id,
                    "routine_revision": routine_revision,
                    "control_call_id": control_call_id,
                }
            ).encode("utf-8")
        ).hexdigest()
    )


def occurrence_id(routine_id: str, slot_key: str) -> str:
    _identity(routine_id, "routine_id")
    _identity(slot_key, "slot_key")
    return (
        "routine-occ-"
        + sha256(
            canonical_json({"routine_id": routine_id, "slot_key": slot_key}).encode(
                "utf-8"
            )
        ).hexdigest()[:32]
    )


def _next_slot(
    schedule: RoutineSchedule,
    *,
    boundary: datetime,
    inclusive: bool,
) -> datetime | None:
    validate_schedule(schedule)
    if isinstance(schedule, OnceSchedule):
        if schedule.exact_at > boundary or (
            inclusive and schedule.exact_at == boundary
        ):
            return schedule.exact_at
        return None
    if isinstance(schedule, IntervalSchedule):
        if boundary < schedule.anchor_at or (
            inclusive and boundary == schedule.anchor_at
        ):
            return schedule.anchor_at
        elapsed_us = _timedelta_microseconds(boundary - schedule.anchor_at)
        interval_us = schedule.interval_seconds * 1_000_000
        quotient, remainder = divmod(elapsed_us, interval_us)
        steps = quotient if inclusive and remainder == 0 else quotient + 1
        return schedule.anchor_at + timedelta(seconds=steps * schedule.interval_seconds)
    return _next_calendar_slot(schedule, boundary=boundary, inclusive=inclusive)


def _next_calendar_slot(
    schedule: CalendarSchedule,
    *,
    boundary: datetime,
    inclusive: bool,
) -> datetime | None:
    zone = ZoneInfo(schedule.timezone)
    local_boundary = boundary.astimezone(zone)
    start_date = local_boundary.date()
    for offset in range(MAX_SCHEDULE_LOOKAHEAD_DAYS + 1):
        local_date = start_date + timedelta(days=offset)
        if not _date_matches(schedule, local_date):
            continue
        naive = datetime.combine(local_date, time(schedule.hour, schedule.minute))
        resolved = _resolve_local(schedule, zone, naive)
        if resolved is None:
            continue
        candidate = resolved.astimezone(UTC)
        if candidate > boundary or (inclusive and candidate == boundary):
            return candidate
    raise ValueError("calendar schedule exceeds its maximum look-ahead")


def _date_matches(schedule: CalendarSchedule, value: date) -> bool:
    if schedule.months and value.month not in schedule.months:
        return False
    if schedule.day_selector is CalendarDaySelector.EVERY_DAY:
        return True
    if schedule.day_selector is CalendarDaySelector.WEEKDAYS:
        return value.isoweekday() in schedule.weekdays
    return value.day in schedule.month_days


def _resolve_local(
    schedule: CalendarSchedule,
    zone: ZoneInfo,
    naive: datetime,
) -> datetime | None:
    candidates = _valid_local_candidates(zone, naive)
    if candidates:
        if schedule.ambiguous_time_policy is AmbiguousTimePolicy.FIRST:
            return min(candidates, key=lambda item: item.astimezone(UTC))
        return max(candidates, key=lambda item: item.astimezone(UTC))
    if schedule.nonexistent_time_policy is NonexistentTimePolicy.SKIP:
        return None
    cursor = naive
    for _ in range(24 * 60):
        cursor += timedelta(minutes=1)
        if cursor.date() != naive.date():
            break
        candidates = _valid_local_candidates(zone, cursor)
        if candidates:
            return min(candidates, key=lambda item: item.astimezone(UTC))
    return None


def _valid_local_candidates(zone: ZoneInfo, naive: datetime) -> tuple[datetime, ...]:
    candidates: list[datetime] = []
    seen: set[datetime] = set()
    for fold in (0, 1):
        aware = naive.replace(tzinfo=zone, fold=fold)
        round_trip = aware.astimezone(UTC).astimezone(zone)
        if round_trip.replace(tzinfo=None) != naive:
            continue
        utc_value = aware.astimezone(UTC)
        if utc_value not in seen:
            candidates.append(aware)
            seen.add(utc_value)
    return tuple(candidates)


def _timedelta_microseconds(value: timedelta) -> int:
    return value.days * 86_400_000_000 + value.seconds * 1_000_000 + value.microseconds


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")


def _identity(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 1_024
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{name} is invalid")


def _positive_revision(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("routine revision must be positive")
