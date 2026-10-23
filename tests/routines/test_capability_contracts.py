"""Executable model-facing routine_save schema and schedule contracts."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    ToolOutputValidationError,
    _mode_input_schema,
    validate_tool_schema_value,
)
from daita.routines.capabilities import (
    ROUTINE_SAVE_DESCRIPTION,
    ROUTINE_SAVE_TOOL_NAME,
    _mutation_receipt,
    _mutation_receipt_schema,
    _save_schema,
    routine_capability_declarations,
    routine_projection,
)
from daita.routines.models import CalendarSchedule, OnceSchedule
from daita.routines.owner import RoutineError, RoutineOwner, parse_schedule
from tests.routines._storage_support import routine_record

NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


def _schedule_schema() -> Mapping[str, object]:
    properties = cast(Mapping[str, object], _save_schema(update=False)["properties"])
    return cast(Mapping[str, object], properties["schedule"])


def _parse(schedule: Mapping[str, object]):
    return parse_schedule(
        validate_tool_schema_value(_schedule_schema(), schedule), reference_at=NOW
    )


def test_run_immediately_is_a_creation_choice_only():
    create = cast(Mapping[str, object], _save_schema(update=False)["properties"])
    update = cast(Mapping[str, object], _save_schema(update=True)["properties"])
    assert "run_immediately" in create and "run_immediately" not in update
    assert _save_schema(update=False)["required"] == ["title", "prompt", "schedule"]
    assert _save_schema(update=True)["required"] == ["routine_id"]


@pytest.mark.parametrize(
    "schedule",
    (
        {"kind": "once", "exact_at": "2026-09-07T14:00:00+00:00"},
        {
            "kind": "interval",
            "interval_seconds": 3600,
            "anchor_at": "2026-09-07T14:00:00+00:00",
        },
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "weekdays": [1],
        },
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "weekdays": [],
            "month_days": [],
            "months": [],
        },
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "month_days": [1, 15],
            "months": [1, 7],
        },
    ),
)
def test_declared_schedule_shapes_reach_the_typed_parser(schedule):
    assert _parse(schedule).kind.value == schedule["kind"]


@pytest.mark.parametrize(
    "schedule",
    (
        {"kind": "cron", "expression": "0 9 * * 1", "timezone": "America/Chicago"},
        {
            "kind": "calendar",
            "rrule": "FREQ=WEEKLY;BYDAY=MO",
            "timezone": "America/Chicago",
        },
        {
            "kind": "once",
            "exact_at": "2026-09-07T14:00:00+00:00",
            "rrule": "FREQ=DAILY",
        },
        {"kind": "interval", "anchor_at": "2026-09-07T14:00:00+00:00"},
        {"kind": "once", "exact_at": "2026-09-07T14:00:00+00:00", "hour": 9},
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "weekdays": [1],
            "month_days": [1],
        },
    ),
)
def test_invented_or_incomplete_schedule_shapes_fail_the_declared_schema(schedule):
    # The flat schema rejects unknown keys; the owner rejects keys of another
    # kind and incomplete or contradictory selections.
    with pytest.raises((ToolOutputValidationError, RoutineError)):
        _parse(schedule)


@pytest.mark.parametrize(
    "filters,selector",
    (
        ({}, "every_day"),
        ({"weekdays": [], "month_days": []}, "every_day"),
        ({"weekdays": [1, 3]}, "weekdays"),
        ({"month_days": [1, 15], "months": [2]}, "month_days"),
    ),
)
def test_calendar_day_selector_follows_the_given_day_filters(filters, selector):
    schedule = {
        "kind": "calendar",
        "timezone": "America/Chicago",
        "hour": 9,
        "minute": 0,
        **filters,
    }
    parsed = _parse(schedule)
    assert isinstance(parsed, CalendarSchedule)
    assert parsed.day_selector.value == selector


def test_contradictory_calendar_says_how_to_correct_it():
    with pytest.raises(RoutineError, match="Correct the schedule"):
        parse_schedule(
            FrozenJsonObject.from_mapping(
                {
                    "kind": "calendar",
                    "timezone": "America/Chicago",
                    "hour": 9,
                    "minute": 0,
                    "day_selector": "every_day",
                    "weekdays": (1,),
                }
            ),
            reference_at=NOW,
        )


def test_null_keys_of_other_schedule_kinds_are_omitted_before_checks():
    from daita.capabilities import omit_unset_arguments

    # routine_save offers both routes, so both schemas are candidates.
    schemas = (_save_schema(update=False), _save_schema(update=True))
    # The flat schedule lists every kind's keys; models send unused ones null.
    arguments: dict[str, object] = {
        "title": "Daily",
        "routine_id": None,
        "schedule": {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "weekdays": [],
            "exact_at": None,
            "after_seconds": None,
            "interval_seconds": None,
        },
    }
    normalized = omit_unset_arguments(arguments, schemas)
    assert "routine_id" not in normalized
    schedule = cast(Mapping[str, object], normalized["schedule"])
    assert set(schedule) == {"kind", "timezone", "hour", "minute", "weekdays"}
    parsed = parse_schedule(FrozenJsonObject.from_mapping(schedule), reference_at=NOW)
    assert isinstance(parsed, CalendarSchedule)
    assert parsed.day_selector.value == "every_day"


def test_one_of_rejects_ambiguous_shapes_and_enforces_output_rules():
    rule = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    with pytest.raises(ToolOutputValidationError, match="exactly one"):
        validate_tool_schema_value({"type": "object", "oneOf": [rule, rule]}, {"x": 1})
    with pytest.raises(ValueError, match="one to eight"):
        validate_tool_schema_value({"type": "object", "oneOf": [rule] * 9}, {"x": 1})


@pytest.mark.parametrize("selector", ["every_day", "weekdays", "month_days"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("hour", 24),
        ("minute", -1),
        ("timezone", ""),
        ("months", [13]),
        ("months", [1, 1]),
        ("weekdays", [True]),
        ("month_days", [32]),
        ("nonexistent_time_policy", "invented"),
        ("unexpected", "value"),
    ],
)
def test_shared_calendar_constraints_reject_invalid_values_in_every_selector(
    selector, field, value
):
    schedule: dict[str, object] = {
        "kind": "calendar",
        "timezone": "America/Chicago",
        "hour": 9,
        "minute": 0,
    }
    if selector != "every_day":
        schedule[selector] = [1]
    schedule[field] = value
    with pytest.raises((ToolOutputValidationError, RoutineError)):
        _parse(schedule)


def test_routine_save_is_at_most_1_5k_characters_as_sent():
    declarations = routine_capability_declarations(
        RoutineOwner.__new__(RoutineOwner)  # schemas only; no store is touched
    )
    view = next(
        item for item in declarations.tool_views if item.name == ROUTINE_SAVE_TOOL_NAME
    )
    schemas = {item.id: item.input_schema for item in declarations.capabilities}
    merged = _mode_input_schema(
        view,
        tuple(
            (mode, schemas[capability]) for mode, capability in view.mode_capability_ids
        ),
    )
    sent = canonical_json(
        {
            "name": ROUTINE_SAVE_TOOL_NAME,
            "description": ROUTINE_SAVE_DESCRIPTION,
            "input_schema": merged,
        }
    )
    assert len(sent) <= 1_500


@pytest.mark.parametrize(
    "schedule",
    (
        {"kind": "once", "after_seconds": 300},
        {"kind": "interval", "interval_seconds": 300, "anchor_after_seconds": 300},
        {
            "kind": "once_next_weekday",
            "timezone": "America/Chicago",
            "weekday": 3,
            "hour": 17,
            "minute": 0,
        },
    ),
)
def test_create_and_update_accept_bounded_temporal_intents(schedule):
    for update in (False, True):
        properties = cast(
            Mapping[str, object], _save_schema(update=update)["properties"]
        )
        validate_tool_schema_value(
            cast(Mapping[str, object], properties["schedule"]), schedule
        )
    relative = parse_schedule(
        FrozenJsonObject.from_mapping(schedule),
        reference_at=NOW,
        relative_at=datetime(2026, 9, 7, 13, tzinfo=UTC),
    )
    if schedule["kind"] == "once_next_weekday":
        # A named local instant resolves from the card's time, not the save's.
        assert relative == OnceSchedule(datetime(2026, 9, 9, 22, tzinfo=UTC))
    else:
        # Relative delays count from the save.
        first = relative.exact_at if isinstance(relative, OnceSchedule) else None
        anchor = getattr(relative, "anchor_at", first)
        assert anchor == datetime(2026, 9, 7, 13, 5, tzinfo=UTC)


def test_exact_offset_schedule_is_normalized_to_utc():
    schedule = parse_schedule(
        {"kind": "once", "exact_at": "2026-09-23T17:00:00-05:00"}, reference_at=NOW
    )
    assert isinstance(schedule, OnceSchedule)
    assert schedule.exact_at == datetime(2026, 9, 23, 22, tzinfo=UTC)


@pytest.mark.parametrize(
    "action", ["create", "update", "pause", "resume", "run_now", "disable"]
)
def test_mutation_receipt_is_bounded_and_full_record_stays_inspectable(action):
    routine = routine_record()
    receipt = _mutation_receipt(action, routine)
    validate_tool_schema_value(_mutation_receipt_schema(), receipt)
    assert len(canonical_json(receipt).encode()) < 1600
    compact = receipt["routine"]
    assert isinstance(compact, dict)
    assert compact["routine_id"] == routine.routine_id
    assert compact["revision"] == routine.revision
    full = routine_projection(routine)
    for key in ("prompt", "source_ids", "conversation_id"):
        assert key in full and key not in compact
    with pytest.raises(ToolOutputValidationError):
        validate_tool_schema_value(
            _mutation_receipt_schema(), {**receipt, "routine": full}
        )
