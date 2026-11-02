from __future__ import annotations

from datetime import UTC, datetime

import pytest

from daita.routines.models import (
    MAX_ROUTINE_INTERVAL_SECONDS,
    MIN_ROUTINE_INTERVAL_SECONDS,
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    NonexistentTimePolicy,
    OnceSchedule,
)
from daita.routines.schedule import (
    first_slot,
    manual_slot_key,
    next_slot,
    occurrence_id,
    scheduled_slot_key,
    select_due_slot,
    validate_schedule,
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def test_once_and_interval_boundaries_are_exact() -> None:
    instant = _utc("2026-08-27T12:00:00+00:00")
    once = OnceSchedule(instant)
    assert first_slot(once, not_before=instant) == instant
    assert next_slot(once, after=instant) is None

    minimum = IntervalSchedule(MIN_ROUTINE_INTERVAL_SECONDS, instant)
    assert next_slot(minimum, after=instant) == _utc("2026-08-27T12:01:00+00:00")
    maximum = IntervalSchedule(MAX_ROUTINE_INTERVAL_SECONDS, instant)
    assert maximum.interval_seconds == MAX_ROUTINE_INTERVAL_SECONDS
    with pytest.raises(ValueError, match="outside its bound"):
        IntervalSchedule(MIN_ROUTINE_INTERVAL_SECONDS - 1, instant)
    with pytest.raises(ValueError, match="outside its bound"):
        IntervalSchedule(MAX_ROUTINE_INTERVAL_SECONDS + 1, instant)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (NonexistentTimePolicy.SKIP, "2026-03-09T06:30:00+00:00"),
        (NonexistentTimePolicy.NEXT_VALID, "2026-03-08T07:00:00+00:00"),
    ],
)
def test_spring_gap_has_explicit_policy(
    policy: NonexistentTimePolicy,
    expected: str,
) -> None:
    schedule = CalendarSchedule(
        timezone="America/New_York",
        hour=2,
        minute=30,
        day_selector=CalendarDaySelector.EVERY_DAY,
        nonexistent_time_policy=policy,
    )
    assert first_slot(
        schedule,
        not_before=_utc("2026-03-08T05:00:00+00:00"),
    ) == _utc(expected)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (AmbiguousTimePolicy.FIRST, "2026-11-01T05:30:00+00:00"),
        (AmbiguousTimePolicy.SECOND, "2026-11-01T06:30:00+00:00"),
    ],
)
def test_autumn_overlap_has_explicit_policy(
    policy: AmbiguousTimePolicy,
    expected: str,
) -> None:
    schedule = CalendarSchedule(
        timezone="America/New_York",
        hour=1,
        minute=30,
        day_selector=CalendarDaySelector.EVERY_DAY,
        ambiguous_time_policy=policy,
    )
    assert first_slot(
        schedule,
        not_before=_utc("2026-11-01T04:00:00+00:00"),
    ) == _utc(expected)


def test_weekly_calendar_keeps_local_wall_clock_across_dst() -> None:
    schedule = CalendarSchedule(
        timezone="America/New_York",
        hour=9,
        minute=0,
        day_selector=CalendarDaySelector.WEEKDAYS,
        weekdays=(4,),
    )
    before = first_slot(
        schedule,
        not_before=_utc("2026-03-01T00:00:00+00:00"),
    )
    assert before is not None
    assert before == _utc("2026-03-05T14:00:00+00:00")
    assert next_slot(
        schedule,
        after=before,
    ) == _utc("2026-03-12T13:00:00+00:00")


def test_leap_year_and_month_end_selection() -> None:
    leap = CalendarSchedule(
        timezone="America/Chicago",
        hour=9,
        minute=0,
        day_selector=CalendarDaySelector.MONTH_DAYS,
        month_days=(29,),
        months=(2,),
    )
    assert first_slot(
        leap,
        not_before=_utc("2027-03-01T00:00:00+00:00"),
    ) == _utc("2028-02-29T15:00:00+00:00")

    impossible = CalendarSchedule(
        timezone="America/Chicago",
        hour=9,
        minute=0,
        day_selector=CalendarDaySelector.MONTH_DAYS,
        month_days=(30,),
        months=(2,),
    )
    with pytest.raises(ValueError, match="cannot produce"):
        validate_schedule(impossible)


@pytest.mark.parametrize("timezone", ["EST", "UTC", "Not/A_Zone"])
def test_ambiguous_or_invalid_timezone_is_rejected(timezone: str) -> None:
    if timezone in {"EST", "UTC"}:
        with pytest.raises(ValueError, match="exact IANA"):
            CalendarSchedule(
                timezone=timezone,
                hour=9,
                minute=0,
                day_selector=CalendarDaySelector.EVERY_DAY,
            )
        return
    schedule = CalendarSchedule(
        timezone=timezone,
        hour=9,
        minute=0,
        day_selector=CalendarDaySelector.EVERY_DAY,
    )
    with pytest.raises(ValueError, match="not available"):
        validate_schedule(schedule)


def test_missed_slots_are_skipped() -> None:
    schedule = IntervalSchedule(3_600, _utc("2026-08-27T00:00:00+00:00"))
    skipped = select_due_slot(
        schedule,
        materialized_due_at=_utc("2026-08-27T01:00:00+00:00"),
        now=_utc("2026-08-27T04:30:00+00:00"),
    )
    assert skipped.selected_at is None
    assert skipped.skipped_slots == 4
    assert skipped.next_due_at == _utc("2026-08-27T05:00:00+00:00")


@pytest.mark.parametrize(
    ("due", "now", "expected"),
    [
        # (selected, skipped, next due)
        ("01:00", "01:00", ("01:00", 0, "02:00")),
        ("01:00", "01:59", ("01:00", 0, "02:00")),
        ("01:00", "02:00", (None, 2, "03:00")),
        ("01:00", "04:30", (None, 4, "05:00")),
    ],
)
def test_interval_backlog_boundaries_match_hand_computed_slots(
    due: str,
    now: str,
    expected: tuple[str | None, int, str],
) -> None:
    def at(clock: str) -> datetime:
        return _utc(f"2026-08-27T{clock}:00+00:00")

    schedule = IntervalSchedule(3_600, at("00:00"))
    selected, skipped, following = expected
    selection = select_due_slot(schedule, materialized_due_at=at(due), now=at(now))
    assert selection.selected_at == (None if selected is None else at(selected))
    assert selection.skipped_slots == skipped
    assert selection.next_due_at == at(following)


def test_interval_backlog_beyond_the_occurrence_bound_stays_selectable() -> None:
    anchor = _utc("2026-08-27T00:00:00+00:00")
    schedule = IntervalSchedule(MIN_ROUTINE_INTERVAL_SECONDS, anchor)
    # A minute routine whose host was closed for a week misses 10,080 slots.
    skipped = select_due_slot(
        schedule,
        materialized_due_at=anchor,
        now=_utc("2026-09-03T00:00:30+00:00"),
    )
    assert skipped.selected_at is None
    assert skipped.skipped_slots == 7 * 24 * 60 + 1
    assert skipped.next_due_at == _utc("2026-09-03T00:01:00+00:00")


def test_pause_resume_and_schedule_revision_choose_future_identity() -> None:
    schedule = IntervalSchedule(3_600, _utc("2026-08-27T00:00:00+00:00"))
    resumed = next_slot(
        schedule,
        after=_utc("2026-08-27T04:30:00+00:00"),
    )
    assert resumed is not None
    assert resumed == _utc("2026-08-27T05:00:00+00:00")
    old = scheduled_slot_key("routine-1", 1, resumed)
    revised = scheduled_slot_key("routine-1", 2, resumed)
    assert old != revised
    assert old == scheduled_slot_key("routine-1", 1, resumed)


def test_duplicate_manual_requests_have_stable_occurrence_identity() -> None:
    first = manual_slot_key("routine-1", 3, "control-call-1")
    duplicate = manual_slot_key("routine-1", 3, "control-call-1")
    different = manual_slot_key("routine-1", 3, "control-call-2")
    assert first == duplicate
    assert first != different
    assert occurrence_id("routine-1", first) == occurrence_id("routine-1", duplicate)


def test_backward_clock_movement_cannot_recreate_completed_slot() -> None:
    schedule = IntervalSchedule(3_600, _utc("2026-08-27T00:00:00+00:00"))
    completed = _utc("2026-08-27T04:00:00+00:00")
    original_key = scheduled_slot_key("routine-1", 1, completed)
    assert next_slot(
        schedule,
        after=completed,
    ) == _utc("2026-08-27T05:00:00+00:00")
    assert scheduled_slot_key("routine-1", 1, completed) == original_key
