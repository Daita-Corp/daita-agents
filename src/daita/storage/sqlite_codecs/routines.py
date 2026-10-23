"""Encode and decode the current canonical routine aggregates."""

from __future__ import annotations

from ...routines.models import (
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    NonexistentTimePolicy,
    OnceSchedule,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineSchedule,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutine,
)
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    decimal_decode,
    decimal_encode,
    dump_payload,
    integer,
    load_payload,
    mapping,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    record,
    record_fields,
    sequence,
    text,
)
from .execution_scope import decode_execution_scope, encode_execution_scope


def encode_scheduled_routine(value: ScheduledRoutine) -> str:
    if not isinstance(value, ScheduledRoutine):
        raise TypeError("routine codec requires ScheduledRoutine")
    return dump_payload(
        record(
            "ScheduledRoutine",
            {
                "conversation_id": value.conversation_id,
                "owner_principal_id": value.owner_principal_id,
                "title": value.title,
                "authorized_instruction": value.authorized_instruction,
                "instruction_digest": value.instruction_digest,
                "schedule": _encode_schedule(value.schedule),
                "schedule_interpreter_revision": value.schedule_interpreter_revision,
                "source_ids": list(value.source_ids),
                "charged_tokens": value.charged_tokens,
                "charged_cost_usd": decimal_encode(value.charged_cost_usd),
                "attempt_count": value.attempt_count,
                "occurrence_count": value.occurrence_count,
                "consecutive_failures": value.consecutive_failures,
                "next_due_at": optional_datetime_encode(value.next_due_at),
                "active_occurrence_id": value.active_occurrence_id,
                "last_occurrence_id": value.last_occurrence_id,
                "last_delivery_ids": list(value.last_delivery_ids),
                "state": value.state.value,
                "revision": value.revision,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
            },
        )
    )


def decode_scheduled_routine(
    value: str,
    *,
    agent_id: str,
    routine_id: str,
) -> ScheduledRoutine:
    fields = record_fields(
        load_payload(value),
        "ScheduledRoutine",
        (
            "conversation_id",
            "owner_principal_id",
            "title",
            "authorized_instruction",
            "instruction_digest",
            "schedule",
            "schedule_interpreter_revision",
            "source_ids",
            "charged_tokens",
            "charged_cost_usd",
            "attempt_count",
            "occurrence_count",
            "consecutive_failures",
            "next_due_at",
            "active_occurrence_id",
            "last_occurrence_id",
            "last_delivery_ids",
            "state",
            "revision",
            "created_at",
            "updated_at",
        ),
    )
    try:
        state = RoutineState(text(fields["state"], "routine state"))
    except ValueError:
        raise ValueError("stored routine enum is invalid") from None
    return ScheduledRoutine(
        routine_id=routine_id,
        agent_id=agent_id,
        conversation_id=text(fields["conversation_id"], "routine conversation"),
        owner_principal_id=text(fields["owner_principal_id"], "routine principal"),
        title=text(fields["title"], "routine title"),
        authorized_instruction=text(
            fields["authorized_instruction"],
            "routine instruction",
        ),
        instruction_digest=text(
            fields["instruction_digest"],
            "routine instruction digest",
        ),
        schedule=_decode_schedule(fields["schedule"]),
        schedule_interpreter_revision=integer(
            fields["schedule_interpreter_revision"],
            "routine schedule interpreter revision",
        ),
        source_ids=_text_sequence(fields["source_ids"], "source ids"),
        charged_tokens=integer(fields["charged_tokens"], "routine charged tokens"),
        charged_cost_usd=decimal_decode(fields["charged_cost_usd"]),
        attempt_count=integer(fields["attempt_count"], "routine attempts"),
        occurrence_count=integer(fields["occurrence_count"], "routine occurrences"),
        consecutive_failures=integer(
            fields["consecutive_failures"],
            "routine consecutive failures",
        ),
        next_due_at=optional_datetime_decode(fields["next_due_at"]),
        active_occurrence_id=optional_text(
            fields["active_occurrence_id"],
            "routine active occurrence",
        ),
        last_occurrence_id=optional_text(
            fields["last_occurrence_id"],
            "routine last occurrence",
        ),
        last_delivery_ids=_text_sequence(
            fields["last_delivery_ids"],
            "last delivery ids",
        ),
        state=state,
        revision=integer(fields["revision"], "routine revision"),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
    )


def encode_routine_occurrence(value: RoutineOccurrence) -> str:
    if not isinstance(value, RoutineOccurrence):
        raise TypeError("routine codec requires RoutineOccurrence")
    return dump_payload(
        record(
            "RoutineOccurrence",
            {
                "routine_id": value.routine_id,
                "routine_revision": value.routine_revision,
                "slot_kind": value.slot_kind.value,
                "slot_key": value.slot_key,
                "scheduled_for": datetime_encode(value.scheduled_for),
                "claimed_at": optional_datetime_encode(value.claimed_at),
                "claim_token": value.claim_token,
                "lease_expires_at": optional_datetime_encode(value.lease_expires_at),
                "execution_scope": (
                    None
                    if value.execution_scope is None
                    else encode_execution_scope(value.execution_scope)
                ),
                "execution_scope_digest": value.execution_scope_digest,
                "reserved_run_id": value.reserved_run_id,
                "charged_tokens": value.charged_tokens,
                "charged_cost_usd": decimal_encode(value.charged_cost_usd),
                "run_bound_at": optional_datetime_encode(value.run_bound_at),
                "run_terminal_at": optional_datetime_encode(value.run_terminal_at),
                "conclusion_digest": value.conclusion_digest,
                "terminal_run_id": value.terminal_run_id,
                "delivery_ids": list(value.delivery_ids),
                "effect_receipt_ids": list(value.effect_receipt_ids),
                "attempt_count": value.attempt_count,
                "failure_code": value.failure_code,
                "retry_at": optional_datetime_encode(value.retry_at),
                "disposition": value.disposition.value,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
            },
        )
    )


def decode_routine_occurrence(
    value: str,
    *,
    agent_id: str,
    occurrence_id: str,
) -> RoutineOccurrence:
    fields = record_fields(
        load_payload(value),
        "RoutineOccurrence",
        (
            "routine_id",
            "routine_revision",
            "slot_kind",
            "slot_key",
            "scheduled_for",
            "claimed_at",
            "claim_token",
            "lease_expires_at",
            "execution_scope",
            "execution_scope_digest",
            "reserved_run_id",
            "charged_tokens",
            "charged_cost_usd",
            "run_bound_at",
            "run_terminal_at",
            "conclusion_digest",
            "terminal_run_id",
            "delivery_ids",
            "effect_receipt_ids",
            "attempt_count",
            "failure_code",
            "retry_at",
            "disposition",
            "created_at",
            "updated_at",
        ),
    )
    try:
        slot_kind = RoutineSlotKind(text(fields["slot_kind"], "slot kind"))
        disposition = RoutineOccurrenceDisposition(
            text(fields["disposition"], "occurrence disposition")
        )
    except ValueError:
        raise ValueError("stored occurrence enum is invalid") from None
    scope = fields["execution_scope"]
    return RoutineOccurrence(
        effect_receipt_ids=_text_sequence(
            fields["effect_receipt_ids"], "occurrence effect receipt IDs"
        ),
        occurrence_id=occurrence_id,
        agent_id=agent_id,
        routine_id=text(fields["routine_id"], "occurrence routine id"),
        routine_revision=integer(
            fields["routine_revision"],
            "occurrence routine revision",
        ),
        slot_kind=slot_kind,
        slot_key=text(fields["slot_key"], "occurrence slot key"),
        scheduled_for=datetime_decode(fields["scheduled_for"]),
        claimed_at=optional_datetime_decode(fields["claimed_at"]),
        claim_token=optional_text(fields["claim_token"], "occurrence claim token"),
        lease_expires_at=optional_datetime_decode(fields["lease_expires_at"]),
        execution_scope=None if scope is None else decode_execution_scope(scope),
        execution_scope_digest=optional_text(
            fields["execution_scope_digest"],
            "occurrence scope digest",
        ),
        reserved_run_id=optional_text(
            fields["reserved_run_id"],
            "occurrence reserved run id",
        ),
        charged_tokens=integer(
            fields["charged_tokens"],
            "occurrence charged tokens",
        ),
        charged_cost_usd=decimal_decode(fields["charged_cost_usd"]),
        run_bound_at=optional_datetime_decode(fields["run_bound_at"]),
        run_terminal_at=optional_datetime_decode(fields["run_terminal_at"]),
        conclusion_digest=optional_text(
            fields["conclusion_digest"],
            "occurrence conclusion digest",
        ),
        terminal_run_id=optional_text(
            fields["terminal_run_id"],
            "occurrence terminal run id",
        ),
        delivery_ids=_text_sequence(fields["delivery_ids"], "delivery ids"),
        attempt_count=integer(fields["attempt_count"], "occurrence attempts"),
        failure_code=optional_text(fields["failure_code"], "occurrence failure code"),
        retry_at=optional_datetime_decode(fields["retry_at"]),
        disposition=disposition,
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
    )


def _encode_schedule(value: RoutineSchedule) -> JsonValue:
    if isinstance(value, OnceSchedule):
        return record("OnceSchedule", {"exact_at": datetime_encode(value.exact_at)})
    if isinstance(value, IntervalSchedule):
        return record(
            "IntervalSchedule",
            {
                "interval_seconds": value.interval_seconds,
                "anchor_at": datetime_encode(value.anchor_at),
            },
        )
    if isinstance(value, CalendarSchedule):
        return record(
            "CalendarSchedule",
            {
                "timezone": value.timezone,
                "hour": value.hour,
                "minute": value.minute,
                "day_selector": value.day_selector.value,
                "weekdays": list(value.weekdays),
                "month_days": list(value.month_days),
                "months": list(value.months),
                "nonexistent_time_policy": value.nonexistent_time_policy.value,
                "ambiguous_time_policy": value.ambiguous_time_policy.value,
            },
        )
    raise TypeError("routine schedule is invalid")


def _decode_schedule(value: JsonValue) -> RoutineSchedule:
    envelope = mapping(value, "routine schedule")
    name = envelope.get("__record__")
    if name == "OnceSchedule":
        fields = record_fields(value, "OnceSchedule", ("exact_at",))
        return OnceSchedule(datetime_decode(fields["exact_at"]))
    if name == "IntervalSchedule":
        fields = record_fields(
            value,
            "IntervalSchedule",
            ("interval_seconds", "anchor_at"),
        )
        return IntervalSchedule(
            integer(fields["interval_seconds"], "interval seconds"),
            datetime_decode(fields["anchor_at"]),
        )
    if name != "CalendarSchedule":
        raise ValueError("stored routine schedule kind is invalid")
    fields = record_fields(
        value,
        "CalendarSchedule",
        (
            "timezone",
            "hour",
            "minute",
            "day_selector",
            "weekdays",
            "month_days",
            "months",
            "nonexistent_time_policy",
            "ambiguous_time_policy",
        ),
    )
    try:
        selector = CalendarDaySelector(
            text(fields["day_selector"], "calendar selector")
        )
        nonexistent = NonexistentTimePolicy(
            text(fields["nonexistent_time_policy"], "nonexistent-time policy")
        )
        ambiguous = AmbiguousTimePolicy(
            text(fields["ambiguous_time_policy"], "ambiguous-time policy")
        )
    except ValueError:
        raise ValueError("stored calendar enum is invalid") from None
    return CalendarSchedule(
        timezone=text(fields["timezone"], "calendar timezone"),
        hour=integer(fields["hour"], "calendar hour"),
        minute=integer(fields["minute"], "calendar minute"),
        day_selector=selector,
        weekdays=_integer_sequence(fields["weekdays"], "calendar weekdays"),
        month_days=_integer_sequence(fields["month_days"], "calendar month days"),
        months=_integer_sequence(fields["months"], "calendar months"),
        nonexistent_time_policy=nonexistent,
        ambiguous_time_policy=ambiguous,
    )


def _text_sequence(value: JsonValue, name: str) -> tuple[str, ...]:
    return tuple(text(item, name) for item in sequence(value, name))


def _integer_sequence(value: JsonValue, name: str) -> tuple[int, ...]:
    return tuple(integer(item, name) for item in sequence(value, name))


__all__ = [
    "decode_routine_occurrence",
    "decode_scheduled_routine",
    "encode_routine_occurrence",
    "encode_scheduled_routine",
]
