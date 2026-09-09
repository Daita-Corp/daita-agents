"""Executable model-facing schedule and automation-discovery contracts."""

from typing import cast
from collections.abc import Mapping

import pytest

from daita.capabilities import ToolOutputValidationError, validate_tool_schema_value
from daita.routines.capabilities import _parse_schedule, _spec_schema
from daita._json import canonical_json


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
            "day_selector": "weekdays",
            "weekdays": [1],
        },
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "day_selector": "every_day",
            "weekdays": [],
            "month_days": [],
            "months": [],
        },
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "day_selector": "month_days",
            "month_days": [1, 15],
            "months": [1, 7],
        },
    ),
)
def test_declared_schedule_shapes_reach_the_typed_parser(schedule):
    schema = cast(
        Mapping[str, object],
        cast(Mapping[str, object], _spec_schema(update=False)["properties"])[
            "schedule"
        ],
    )
    validated = validate_tool_schema_value(schema, schedule)
    parsed = _parse_schedule(validated)
    assert parsed.kind.value == schedule["kind"]


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
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "day_selector": "weekdays",
            "weekdays": [],
        },
        {
            "kind": "calendar",
            "timezone": "America/Chicago",
            "hour": 9,
            "minute": 0,
            "day_selector": "every_day",
            "weekdays": [1],
        },
    ),
)
def test_invented_or_incomplete_schedule_shapes_fail_the_declared_schema(schedule):
    schema = cast(
        Mapping[str, object],
        cast(Mapping[str, object], _spec_schema(update=False)["properties"])[
            "schedule"
        ],
    )
    with pytest.raises(ToolOutputValidationError):
        validate_tool_schema_value(schema, schedule)


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


def test_mcp_binding_revision_is_not_a_resource_precheck():
    schema = cast(
        Mapping[str, object],
        cast(Mapping[str, object], _spec_schema(update=False)["properties"])[
            "precheck"
        ],
    )
    with pytest.raises(ToolOutputValidationError):
        validate_tool_schema_value(
            schema, {"connector_binding_revisions": {"binding": 1}}
        )


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
    schedule = {
        "kind": "calendar",
        "timezone": "America/Chicago",
        "hour": 9,
        "minute": 0,
        "day_selector": selector,
    }
    if selector != "every_day":
        schedule[selector] = [1]
    schedule[field] = value
    schema = cast(Mapping[str, object], _spec_schema(update=False)["properties"])[
        "schedule"
    ]
    with pytest.raises(ToolOutputValidationError):
        validate_tool_schema_value(cast(Mapping[str, object], schema), schedule)


def test_routine_schema_retains_typed_constraints_with_a_bounded_wire_footprint():
    schema = _spec_schema(update=False)
    assert len(canonical_json(schema).encode()) < 8600
    properties = cast(Mapping[str, object], schema["properties"])
    assert len(canonical_json(properties["schedule"]).encode()) < 2600


@pytest.mark.parametrize(
    "action", ["create", "update", "pause", "resume", "run_now", "disable"]
)
async def test_mutation_receipt_is_bounded_and_full_contract_stays_inspectable(action):
    from test_routine_owner import _Store, _owner, _proposal
    from daita.routines.capabilities import (
        _mutation_receipt,
        _mutation_receipt_schema,
        routine_projection,
    )

    store = _Store()
    owner = _owner(store)
    proposal = await _proposal(owner)
    receipt = _mutation_receipt(action, proposal)
    validate_tool_schema_value(_mutation_receipt_schema(), receipt)
    assert len(canonical_json(receipt).encode()) < 1600
    compact = receipt["routine"]
    assert isinstance(compact, dict)
    assert compact["routine_id"] == proposal.routine_id
    assert compact["revision"] == proposal.revision
    full = routine_projection(proposal)
    for key in (
        "contract_bindings",
        "authorized_instruction",
        "capability_grants",
        "distribution_plan",
    ):
        assert key in full and key not in compact
    with pytest.raises(ToolOutputValidationError):
        validate_tool_schema_value(
            _mutation_receipt_schema(), {**receipt, "routine": full}
        )
