"""Executable model-facing schedule and automation-discovery contracts."""

from collections.abc import Mapping
from typing import Any, cast

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import ToolOutputValidationError, validate_tool_schema_value
from daita.routines.capabilities import _parse_schedule, _spec_schema
from daita.routines.models import OnceSchedule


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("immediate", [None, False, True])
async def test_model_immediate_choice_is_explicit_on_create_and_cannot_run_a_revision(
    update, immediate
):
    from daita.routines.capabilities import _parsed_spec
    from daita.routines.owner import _routine_proposal_payload
    from tests.routines._owner_support import _owner, _proposal, _Store

    proposal = await _proposal(_owner(_Store()))
    assert proposal.run_immediately is False
    schema = _spec_schema(update=update)
    properties = cast(Mapping[str, object], schema["properties"])
    arguments = {
        key: value
        for key, value in _routine_proposal_payload(proposal).items()
        if key in properties and value is not None
    }
    arguments.update(skill_names=(), distribution_destination_id="destination")
    if update:
        arguments.update(routine_id=proposal.routine_id, expected_revision=1)
    if immediate is None:
        arguments.pop("run_immediately", None)
    else:
        arguments["run_immediately"] = immediate
    rejected = (not update and immediate is None) or (update and immediate is True)
    if rejected:
        with pytest.raises(ToolOutputValidationError):
            validate_tool_schema_value(schema, arguments)
    else:
        validated = validate_tool_schema_value(schema, arguments)
        assert _parsed_spec(validated)["run_immediately"] is bool(immediate)


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
    assert len(canonical_json(schema).encode()) < 11000
    properties = cast(Mapping[str, object], schema["properties"])
    assert len(canonical_json(properties["schedule"]).encode()) < 4200


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
def test_create_accepts_bounded_temporal_intents_but_update_requires_exact_time(
    schedule,
):
    create_schema = cast(
        Mapping[str, object], _spec_schema(update=False)["properties"]
    )["schedule"]
    update_schema = cast(Mapping[str, object], _spec_schema(update=True)["properties"])[
        "schedule"
    ]
    validate_tool_schema_value(cast(Mapping[str, object], create_schema), schedule)
    with pytest.raises(ToolOutputValidationError):
        validate_tool_schema_value(cast(Mapping[str, object], update_schema), schedule)


def test_exact_offset_schedule_is_normalized_to_utc():
    from datetime import UTC, datetime

    schedule = _parse_schedule(
        {"kind": "once", "exact_at": "2026-09-23T17:00:00-05:00"}
    )
    assert isinstance(schedule, OnceSchedule)
    assert schedule.exact_at == datetime(2026, 9, 23, 22, tzinfo=UTC)


async def test_named_local_instant_does_not_jump_to_next_week_after_approval(
    monkeypatch,
):
    from datetime import UTC, datetime

    from daita.capabilities import ToolExecution
    from daita.routines.capabilities import _create_proposal
    from daita.routines.owner import RoutineError, _routine_proposal_payload
    from tests.routines._owner_support import _owner, _proposal, _Store

    owner = _owner(_Store())
    proposal = await _proposal(owner)
    properties = cast(Mapping[str, object], _spec_schema(update=False)["properties"])
    arguments = {
        key: value
        for key, value in _routine_proposal_payload(proposal).items()
        if key in properties and value is not None
    }
    arguments.update(
        schedule={
            "kind": "once_next_weekday",
            "timezone": "America/Chicago",
            "weekday": 3,
            "hour": 17,
            "minute": 0,
        },
        expires_after_seconds=86400,
        skill_names=(),
        distribution_destination_id="destination",
        _timing_reference_at="2026-09-23T21:59:00+00:00",
    )
    arguments.pop("expires_at")
    monkeypatch.setattr(
        owner, "current_time", lambda: datetime(2026, 9, 23, 22, 1, tzinfo=UTC)
    )
    execution = ToolExecution(
        run_id="run-local-passed",
        call_id="call-local-passed",
        capability_id="routines.create",
        arguments=arguments,
        conversation_id="conversation-local-passed",
    )
    with pytest.raises(RoutineError, match="passed before creation"):
        await _create_proposal(owner, execution, commit_time=True)


@pytest.mark.parametrize(
    "action", ["create", "update", "pause", "resume", "run_now", "disable"]
)
async def test_mutation_receipt_is_bounded_and_full_contract_stays_inspectable(action):
    from daita.routines.capabilities import (
        _mutation_receipt,
        _mutation_receipt_schema,
        routine_projection,
    )
    from tests.routines._owner_support import _owner, _proposal, _Store

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


@pytest.mark.parametrize("budget", ["tokens", "cost_usd"])
async def test_draft_and_record_share_budget_relationship_and_equal_boundary(budget):
    from dataclasses import replace
    from decimal import Decimal

    from daita.routines.capabilities import _parsed_spec
    from daita.routines.models import ScheduledRoutineDraft
    from daita.routines.owner import _routine_proposal_payload
    from tests.routines._owner_support import _owner, _proposal, _Store

    proposal = await _proposal(_owner(_Store()))
    properties = cast(Mapping[str, object], _spec_schema(update=False)["properties"])
    arguments = {
        key: value
        for key, value in _routine_proposal_payload(proposal).items()
        if key in properties and value is not None
    }
    arguments.update(skill_names=(), distribution_destination_id="destination")
    parsed = dict(_parsed_spec(FrozenJsonObject.from_mapping(arguments)))
    parsed.pop("basis_run_id")
    draft = ScheduledRoutineDraft(origin_run_id="run-origin", **cast(Any, parsed))
    for item in (draft, proposal):
        field = f"per_run_max_{budget}"
        ceiling = f"cumulative_max_{budget}"
        value = getattr(item, field)
        equal = replace(item, **{ceiling: value})
        assert getattr(equal, field) == getattr(equal, ceiling)
        with pytest.raises(ValueError, match=f"{field} must not exceed {ceiling}"):
            replace(
                item,
                **{ceiling: value - (1 if budget == "tokens" else Decimal("0.01"))},
            )
        if budget == "cost_usd":
            zero = replace(
                item,
                per_run_max_cost_usd=Decimal(0),
                cumulative_max_cost_usd=Decimal(0),
            )
            assert zero.per_run_max_cost_usd == zero.cumulative_max_cost_usd == 0
