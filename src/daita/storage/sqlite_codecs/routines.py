"""Encode and decode the two current D1 codec-v1 routine aggregates."""

from __future__ import annotations

from ...capabilities import AccessMode, OperationalEffect
from ...llm.models import ModelSensitivity
from ...routines.models import (
    AmbiguousTimePolicy,
    CalendarDaySelector,
    CalendarSchedule,
    IntervalSchedule,
    MisfirePolicy,
    NonexistentTimePolicy,
    OnceSchedule,
    ReportingMode,
    ResourceRevisionObservation,
    ResourceRevisionPrecheck,
    RoutineOccurrenceDisposition,
    RoutineOccurrenceV1,
    RoutinePromotionEvidence,
    RoutineSchedule,
    RoutineSkillBinding,
    RoutineSlotKind,
    RoutineState,
    ScheduledRoutineV1,
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
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)
from .execution_scope import decode_execution_scope, encode_execution_scope

_ROUTINE_VERSION = 1
_OCCURRENCE_VERSION = 1


def encode_scheduled_routine(value: ScheduledRoutineV1) -> str:
    if not isinstance(value, ScheduledRoutineV1):
        raise TypeError("routine codec requires ScheduledRoutineV1")
    return dump_payload(
        record(
            "ScheduledRoutine",
            {
                "version": _ROUTINE_VERSION,
                "conversation_id": value.conversation_id,
                "owner_principal_id": value.owner_principal_id,
                "title": value.title,
                "authorized_instruction": value.authorized_instruction,
                "instruction_digest": value.instruction_digest,
                "schedule": _encode_schedule(value.schedule),
                "schedule_interpreter_revision": (value.schedule_interpreter_revision),
                "misfire_policy": value.misfire_policy.value,
                "reporting_mode": value.reporting_mode.value,
                "precheck": (
                    None if value.precheck is None else _encode_precheck(value.precheck)
                ),
                "last_acknowledged_precheck_observation": (
                    None
                    if value.last_acknowledged_precheck_observation is None
                    else _encode_observation(
                        value.last_acknowledged_precheck_observation
                    )
                ),
                "allowed_source_ids": list(value.allowed_source_ids),
                "allowed_connector_binding_ids": list(
                    value.allowed_connector_binding_ids
                ),
                "allowed_resource_ids": list(value.allowed_resource_ids),
                "allowed_capability_ids": list(value.allowed_capability_ids),
                "allowed_access_modes": plain_encode(
                    tuple(sorted(item.value for item in value.allowed_access_modes))
                ),
                "allowed_operational_effects": plain_encode(
                    tuple(
                        sorted(item.value for item in value.allowed_operational_effects)
                    )
                ),
                "sensitivity_ceiling": value.sensitivity_ceiling.value,
                "eligible_model_routes": list(value.eligible_model_routes),
                "skill_bindings": [
                    _encode_skill_binding(item) for item in value.skill_bindings
                ],
                "delivery_destination": value.delivery_destination,
                "per_run_max_tokens": value.per_run_max_tokens,
                "per_run_max_cost_usd": decimal_encode(value.per_run_max_cost_usd),
                "cumulative_max_tokens": value.cumulative_max_tokens,
                "cumulative_max_cost_usd": decimal_encode(
                    value.cumulative_max_cost_usd
                ),
                "cumulative_max_attempts": value.cumulative_max_attempts,
                "cumulative_max_occurrences": value.cumulative_max_occurrences,
                "reserved_tokens": value.reserved_tokens,
                "reserved_cost_usd": decimal_encode(value.reserved_cost_usd),
                "charged_tokens": value.charged_tokens,
                "charged_cost_usd": decimal_encode(value.charged_cost_usd),
                "attempt_count": value.attempt_count,
                "occurrence_count": value.occurrence_count,
                "maximum_consecutive_failures": (value.maximum_consecutive_failures),
                "consecutive_failures": value.consecutive_failures,
                "expires_at": datetime_encode(value.expires_at),
                "next_due_at": optional_datetime_encode(value.next_due_at),
                "active_occurrence_id": value.active_occurrence_id,
                "last_occurrence_id": value.last_occurrence_id,
                "last_delivery_id": value.last_delivery_id,
                "promotion_evidence": (
                    None
                    if value.promotion_evidence is None
                    else _encode_promotion(value.promotion_evidence)
                ),
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
) -> ScheduledRoutineV1:
    fields = record_fields(
        load_payload(value),
        "ScheduledRoutine",
        (
            "version",
            "conversation_id",
            "owner_principal_id",
            "title",
            "authorized_instruction",
            "instruction_digest",
            "schedule",
            "schedule_interpreter_revision",
            "misfire_policy",
            "reporting_mode",
            "precheck",
            "last_acknowledged_precheck_observation",
            "allowed_source_ids",
            "allowed_connector_binding_ids",
            "allowed_resource_ids",
            "allowed_capability_ids",
            "allowed_access_modes",
            "allowed_operational_effects",
            "sensitivity_ceiling",
            "eligible_model_routes",
            "skill_bindings",
            "delivery_destination",
            "per_run_max_tokens",
            "per_run_max_cost_usd",
            "cumulative_max_tokens",
            "cumulative_max_cost_usd",
            "cumulative_max_attempts",
            "cumulative_max_occurrences",
            "reserved_tokens",
            "reserved_cost_usd",
            "charged_tokens",
            "charged_cost_usd",
            "attempt_count",
            "occurrence_count",
            "maximum_consecutive_failures",
            "consecutive_failures",
            "expires_at",
            "next_due_at",
            "active_occurrence_id",
            "last_occurrence_id",
            "last_delivery_id",
            "promotion_evidence",
            "state",
            "revision",
            "created_at",
            "updated_at",
        ),
    )
    if integer(fields["version"], "routine version") != _ROUTINE_VERSION:
        raise ValueError("stored routine version is unsupported")
    try:
        misfire = MisfirePolicy(text(fields["misfire_policy"], "misfire policy"))
        reporting = ReportingMode(text(fields["reporting_mode"], "reporting mode"))
        sensitivity = ModelSensitivity(
            text(fields["sensitivity_ceiling"], "routine sensitivity")
        )
        state = RoutineState(text(fields["state"], "routine state"))
        access_modes = frozenset(
            AccessMode(text(item, "routine access mode"))
            for item in sequence(fields["allowed_access_modes"], "access modes")
        )
        effects = frozenset(
            OperationalEffect(text(item, "routine operational effect"))
            for item in sequence(
                fields["allowed_operational_effects"],
                "operational effects",
            )
        )
    except ValueError:
        raise ValueError("stored routine enum is invalid") from None
    precheck = fields["precheck"]
    observation = fields["last_acknowledged_precheck_observation"]
    promotion = fields["promotion_evidence"]
    return ScheduledRoutineV1(
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
        misfire_policy=misfire,
        reporting_mode=reporting,
        precheck=None if precheck is None else _decode_precheck(precheck),
        last_acknowledged_precheck_observation=(
            None if observation is None else _decode_observation(observation)
        ),
        allowed_source_ids=_text_sequence(fields["allowed_source_ids"], "source ids"),
        allowed_connector_binding_ids=_text_sequence(
            fields["allowed_connector_binding_ids"],
            "connector binding ids",
        ),
        allowed_resource_ids=_text_sequence(
            fields["allowed_resource_ids"],
            "resource ids",
        ),
        allowed_capability_ids=_text_sequence(
            fields["allowed_capability_ids"],
            "capability ids",
        ),
        allowed_access_modes=access_modes,
        allowed_operational_effects=effects,
        sensitivity_ceiling=sensitivity,
        eligible_model_routes=_text_sequence(
            fields["eligible_model_routes"],
            "model routes",
        ),
        skill_bindings=tuple(
            _decode_skill_binding(item)
            for item in sequence(fields["skill_bindings"], "skill bindings")
        ),
        delivery_destination=text(
            fields["delivery_destination"],
            "routine destination",
        ),
        per_run_max_tokens=integer(
            fields["per_run_max_tokens"],
            "routine per-run tokens",
        ),
        per_run_max_cost_usd=decimal_decode(fields["per_run_max_cost_usd"]),
        cumulative_max_tokens=integer(
            fields["cumulative_max_tokens"],
            "routine cumulative tokens",
        ),
        cumulative_max_cost_usd=decimal_decode(fields["cumulative_max_cost_usd"]),
        cumulative_max_attempts=integer(
            fields["cumulative_max_attempts"],
            "routine cumulative attempts",
        ),
        cumulative_max_occurrences=integer(
            fields["cumulative_max_occurrences"],
            "routine cumulative occurrences",
        ),
        reserved_tokens=integer(fields["reserved_tokens"], "routine reserved tokens"),
        reserved_cost_usd=decimal_decode(fields["reserved_cost_usd"]),
        charged_tokens=integer(fields["charged_tokens"], "routine charged tokens"),
        charged_cost_usd=decimal_decode(fields["charged_cost_usd"]),
        attempt_count=integer(fields["attempt_count"], "routine attempts"),
        occurrence_count=integer(fields["occurrence_count"], "routine occurrences"),
        maximum_consecutive_failures=integer(
            fields["maximum_consecutive_failures"],
            "routine maximum failures",
        ),
        consecutive_failures=integer(
            fields["consecutive_failures"],
            "routine consecutive failures",
        ),
        expires_at=datetime_decode(fields["expires_at"]),
        next_due_at=optional_datetime_decode(fields["next_due_at"]),
        active_occurrence_id=optional_text(
            fields["active_occurrence_id"],
            "routine active occurrence",
        ),
        last_occurrence_id=optional_text(
            fields["last_occurrence_id"],
            "routine last occurrence",
        ),
        last_delivery_id=optional_text(
            fields["last_delivery_id"],
            "routine last delivery",
        ),
        promotion_evidence=(
            None if promotion is None else _decode_promotion(promotion)
        ),
        state=state,
        revision=integer(fields["revision"], "routine revision"),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
    )


def encode_routine_occurrence(value: RoutineOccurrenceV1) -> str:
    if not isinstance(value, RoutineOccurrenceV1):
        raise TypeError("routine codec requires RoutineOccurrenceV1")
    return dump_payload(
        record(
            "RoutineOccurrence",
            {
                "version": _OCCURRENCE_VERSION,
                "routine_id": value.routine_id,
                "routine_revision": value.routine_revision,
                "slot_kind": value.slot_kind.value,
                "slot_key": value.slot_key,
                "scheduled_for": datetime_encode(value.scheduled_for),
                "claimed_at": optional_datetime_encode(value.claimed_at),
                "claim_token": value.claim_token,
                "lease_expires_at": optional_datetime_encode(value.lease_expires_at),
                "precheck_observation": (
                    None
                    if value.precheck_observation is None
                    else _encode_observation(value.precheck_observation)
                ),
                "execution_scope": (
                    None
                    if value.execution_scope is None
                    else encode_execution_scope(value.execution_scope)
                ),
                "execution_scope_digest": value.execution_scope_digest,
                "reserved_run_id": value.reserved_run_id,
                "reserved_tokens": value.reserved_tokens,
                "reserved_cost_usd": decimal_encode(value.reserved_cost_usd),
                "charged_tokens": value.charged_tokens,
                "charged_cost_usd": decimal_encode(value.charged_cost_usd),
                "run_bound_at": optional_datetime_encode(value.run_bound_at),
                "run_terminal_at": optional_datetime_encode(value.run_terminal_at),
                "conclusion_digest": value.conclusion_digest,
                "terminal_run_id": value.terminal_run_id,
                "delivery_id": value.delivery_id,
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
) -> RoutineOccurrenceV1:
    fields = record_fields(
        load_payload(value),
        "RoutineOccurrence",
        (
            "version",
            "routine_id",
            "routine_revision",
            "slot_kind",
            "slot_key",
            "scheduled_for",
            "claimed_at",
            "claim_token",
            "lease_expires_at",
            "precheck_observation",
            "execution_scope",
            "execution_scope_digest",
            "reserved_run_id",
            "reserved_tokens",
            "reserved_cost_usd",
            "charged_tokens",
            "charged_cost_usd",
            "run_bound_at",
            "run_terminal_at",
            "conclusion_digest",
            "terminal_run_id",
            "delivery_id",
            "attempt_count",
            "failure_code",
            "retry_at",
            "disposition",
            "created_at",
            "updated_at",
        ),
    )
    if integer(fields["version"], "occurrence version") != _OCCURRENCE_VERSION:
        raise ValueError("stored occurrence version is unsupported")
    try:
        slot_kind = RoutineSlotKind(text(fields["slot_kind"], "slot kind"))
        disposition = RoutineOccurrenceDisposition(
            text(fields["disposition"], "occurrence disposition")
        )
    except ValueError:
        raise ValueError("stored occurrence enum is invalid") from None
    observation = fields["precheck_observation"]
    scope = fields["execution_scope"]
    return RoutineOccurrenceV1(
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
        precheck_observation=(
            None if observation is None else _decode_observation(observation)
        ),
        execution_scope=None if scope is None else decode_execution_scope(scope),
        execution_scope_digest=optional_text(
            fields["execution_scope_digest"],
            "occurrence scope digest",
        ),
        reserved_run_id=optional_text(
            fields["reserved_run_id"],
            "occurrence reserved run id",
        ),
        reserved_tokens=integer(
            fields["reserved_tokens"],
            "occurrence reserved tokens",
        ),
        reserved_cost_usd=decimal_decode(fields["reserved_cost_usd"]),
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
        delivery_id=optional_text(fields["delivery_id"], "occurrence delivery id"),
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


def _encode_precheck(value: ResourceRevisionPrecheck) -> JsonValue:
    return record(
        "ResourceRevisionPrecheck",
        {
            "capability_id": value.capability_id,
            "contract_digest": value.contract_digest,
            "source_id": value.source_id,
            "resource_id": value.resource_id,
        },
    )


def _decode_precheck(value: JsonValue) -> ResourceRevisionPrecheck:
    fields = record_fields(
        value,
        "ResourceRevisionPrecheck",
        ("capability_id", "contract_digest", "source_id", "resource_id"),
    )
    return ResourceRevisionPrecheck(
        capability_id=text(fields["capability_id"], "precheck capability id"),
        contract_digest=text(fields["contract_digest"], "precheck contract digest"),
        source_id=text(fields["source_id"], "precheck source id"),
        resource_id=text(fields["resource_id"], "precheck resource id"),
    )


def _encode_observation(value: ResourceRevisionObservation) -> JsonValue:
    return record(
        "ResourceRevisionObservation",
        {
            "source_id": value.source_id,
            "resource_id": value.resource_id,
            "resource_revision": value.resource_revision,
            "catalog_revision": value.catalog_revision,
            "observed_at": datetime_encode(value.observed_at),
        },
    )


def _decode_observation(value: JsonValue) -> ResourceRevisionObservation:
    fields = record_fields(
        value,
        "ResourceRevisionObservation",
        (
            "source_id",
            "resource_id",
            "resource_revision",
            "catalog_revision",
            "observed_at",
        ),
    )
    return ResourceRevisionObservation(
        source_id=text(fields["source_id"], "observation source id"),
        resource_id=text(fields["resource_id"], "observation resource id"),
        resource_revision=text(
            fields["resource_revision"],
            "observation resource revision",
        ),
        catalog_revision=text(
            fields["catalog_revision"],
            "observation catalog revision",
        ),
        observed_at=datetime_decode(fields["observed_at"]),
    )


def _encode_skill_binding(value: RoutineSkillBinding) -> JsonValue:
    return record(
        "RoutineSkillBinding",
        {
            "skill_name": value.skill_name,
            "skill_revision": value.skill_revision,
            "content_digest": value.content_digest,
            "attached_by_principal": value.attached_by_principal,
            "attached_at": datetime_encode(value.attached_at),
        },
    )


def _decode_skill_binding(value: JsonValue) -> RoutineSkillBinding:
    fields = record_fields(
        value,
        "RoutineSkillBinding",
        (
            "skill_name",
            "skill_revision",
            "content_digest",
            "attached_by_principal",
            "attached_at",
        ),
    )
    return RoutineSkillBinding(
        skill_name=text(fields["skill_name"], "routine skill name"),
        skill_revision=integer(fields["skill_revision"], "routine skill revision"),
        content_digest=text(fields["content_digest"], "routine skill digest"),
        attached_by_principal=text(
            fields["attached_by_principal"],
            "routine skill principal",
        ),
        attached_at=datetime_decode(fields["attached_at"]),
    )


def _encode_promotion(value: RoutinePromotionEvidence) -> JsonValue:
    return record(
        "RoutinePromotionEvidence",
        {
            "basis_run_id": value.basis_run_id,
            "terminal_result_digest": value.terminal_result_digest,
            "executed_capability_ids": list(value.executed_capability_ids),
        },
    )


def _decode_promotion(value: JsonValue) -> RoutinePromotionEvidence:
    fields = record_fields(
        value,
        "RoutinePromotionEvidence",
        ("basis_run_id", "terminal_result_digest", "executed_capability_ids"),
    )
    return RoutinePromotionEvidence(
        basis_run_id=text(fields["basis_run_id"], "routine basis run id"),
        terminal_result_digest=text(
            fields["terminal_result_digest"],
            "routine terminal result digest",
        ),
        executed_capability_ids=_text_sequence(
            fields["executed_capability_ids"],
            "executed capability ids",
        ),
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
