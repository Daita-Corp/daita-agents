"""Encode the current codec-v1 Stage C follow-up and inbox record families."""

from __future__ import annotations

from ...autonomy import (
    AutonomousFollowup,
    DeliverySubject,
    DeliverySubjectKind,
    DeliveryState,
    FollowupConclusionEvidence,
    FollowupDisposition,
    FollowupGrant,
    FollowupObservationSource,
    InboxItem,
)
from ...capabilities import AccessMode, OperationalEffect
from ...llm.models import ModelSensitivity
from .common import (
    datetime_decode,
    datetime_encode,
    decimal_decode,
    decimal_encode,
    dump_payload,
    integer,
    load_payload,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)
from .execution_scope import decode_execution_scope, encode_execution_scope

_FOLLOWUP_VERSION = 1
_INBOX_VERSION = 1


def encode_autonomous_followup(value: AutonomousFollowup) -> str:
    if not isinstance(value, AutonomousFollowup):
        raise TypeError("follow-up codec requires AutonomousFollowup")
    return dump_payload(
        record(
            "AutonomousFollowup",
            {
                "version": _FOLLOWUP_VERSION,
                "conversation_id": value.conversation_id,
                "event_id": value.event_id,
                "observation_source": value.observation_source.value,
                "job_id": value.job_id,
                "job_terminal_revision": value.job_terminal_revision,
                "event_type": value.event_type,
                "event_payload": plain_encode(value.event_payload),
                "payload_digest": value.payload_digest,
                "received_at": datetime_encode(value.received_at),
                "grant": _encode_grant(value.grant),
                "execution_scope": encode_execution_scope(value.execution_scope),
                "disposition": value.disposition.value,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "revision": value.revision,
                "attempt_count": value.attempt_count,
                "claim_token": value.claim_token,
                "lease_expires_at": optional_datetime_encode(value.lease_expires_at),
                "reserved_run_id": value.reserved_run_id,
                "reserved_cost_usd": decimal_encode(value.reserved_cost_usd),
                "reserved_tokens": value.reserved_tokens,
                "charged_cost_usd": decimal_encode(value.charged_cost_usd),
                "charged_tokens": value.charged_tokens,
                "run_bound_at": optional_datetime_encode(value.run_bound_at),
                "run_terminal_at": optional_datetime_encode(value.run_terminal_at),
                "audit_context": plain_encode(value.audit_context),
                "grant_consumed_at": optional_datetime_encode(value.grant_consumed_at),
                "conclusion_evidence": (
                    None
                    if value.conclusion_evidence is None
                    else _encode_conclusion_evidence(value.conclusion_evidence)
                ),
                "delivery_id": value.delivery_id,
                "failure_code": value.failure_code,
            },
        )
    )


def decode_autonomous_followup(
    value: str,
    *,
    agent_id: str,
    followup_id: str,
) -> AutonomousFollowup:
    fields = record_fields(
        load_payload(value),
        "AutonomousFollowup",
        (
            "version",
            "conversation_id",
            "event_id",
            "observation_source",
            "job_id",
            "job_terminal_revision",
            "event_type",
            "event_payload",
            "payload_digest",
            "received_at",
            "grant",
            "execution_scope",
            "disposition",
            "created_at",
            "updated_at",
            "revision",
            "attempt_count",
            "claim_token",
            "lease_expires_at",
            "reserved_run_id",
            "reserved_cost_usd",
            "reserved_tokens",
            "charged_cost_usd",
            "charged_tokens",
            "run_bound_at",
            "run_terminal_at",
            "audit_context",
            "grant_consumed_at",
            "conclusion_evidence",
            "delivery_id",
            "failure_code",
        ),
    )
    if integer(fields["version"], "follow-up version") != _FOLLOWUP_VERSION:
        raise ValueError("stored follow-up version is unsupported")
    try:
        source = FollowupObservationSource(
            text(fields["observation_source"], "follow-up observation source")
        )
        disposition = FollowupDisposition(
            text(fields["disposition"], "follow-up disposition")
        )
    except ValueError:
        raise ValueError("stored follow-up enum is invalid") from None
    payload = plain_decode(fields["event_payload"])
    audit = plain_decode(fields["audit_context"])
    if not isinstance(payload, dict) or not isinstance(audit, dict):
        raise ValueError("stored follow-up payloads must be objects")
    return AutonomousFollowup(
        followup_id=followup_id,
        agent_id=agent_id,
        conversation_id=text(fields["conversation_id"], "follow-up conversation"),
        event_id=text(fields["event_id"], "follow-up event id"),
        observation_source=source,
        job_id=text(fields["job_id"], "follow-up job id"),
        job_terminal_revision=integer(
            fields["job_terminal_revision"],
            "follow-up terminal revision",
        ),
        event_type=text(fields["event_type"], "follow-up event type"),
        event_payload=payload,
        payload_digest=text(fields["payload_digest"], "follow-up payload digest"),
        received_at=datetime_decode(fields["received_at"]),
        grant=_decode_grant(fields["grant"]),
        execution_scope=decode_execution_scope(fields["execution_scope"]),
        disposition=disposition,
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        revision=integer(fields["revision"], "follow-up revision"),
        attempt_count=integer(fields["attempt_count"], "follow-up attempts"),
        claim_token=optional_text(fields["claim_token"], "follow-up claim token"),
        lease_expires_at=optional_datetime_decode(fields["lease_expires_at"]),
        reserved_run_id=optional_text(
            fields["reserved_run_id"],
            "follow-up reserved run id",
        ),
        reserved_cost_usd=decimal_decode(fields["reserved_cost_usd"]),
        reserved_tokens=integer(fields["reserved_tokens"], "follow-up reserved tokens"),
        charged_cost_usd=decimal_decode(fields["charged_cost_usd"]),
        charged_tokens=integer(fields["charged_tokens"], "follow-up charged tokens"),
        run_bound_at=optional_datetime_decode(fields["run_bound_at"]),
        run_terminal_at=optional_datetime_decode(fields["run_terminal_at"]),
        audit_context=audit,
        grant_consumed_at=optional_datetime_decode(fields["grant_consumed_at"]),
        conclusion_evidence=(
            None
            if fields["conclusion_evidence"] is None
            else _decode_conclusion_evidence(fields["conclusion_evidence"])
        ),
        delivery_id=optional_text(fields["delivery_id"], "follow-up delivery id"),
        failure_code=optional_text(fields["failure_code"], "follow-up failure code"),
    )


def encode_inbox_item(value: InboxItem) -> str:
    if not isinstance(value, InboxItem):
        raise TypeError("inbox codec requires InboxItem")
    return dump_payload(
        record(
            "InboxItem",
            {
                "version": _INBOX_VERSION,
                "conversation_id": value.conversation_id,
                "subject": _encode_delivery_subject(value.subject),
                "resulting_run_id": value.resulting_run_id,
                "grant_id": value.grant_id,
                "logical_key": value.logical_key,
                "conclusion_digest": value.conclusion_digest,
                "payload": plain_encode(value.payload),
                "sensitivity": value.sensitivity.value,
                "destination": value.destination,
                "destination_sensitivity_ceiling": (
                    value.destination_sensitivity_ceiling.value
                ),
                "state": value.state.value,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
                "attempt_count": value.attempt_count,
                "acknowledged_at": optional_datetime_encode(value.acknowledged_at),
                "terminal_error": value.terminal_error,
            },
        )
    )


def decode_inbox_item(
    value: str,
    *,
    agent_id: str,
    delivery_id: str,
) -> InboxItem:
    fields = record_fields(
        load_payload(value),
        "InboxItem",
        (
            "version",
            "conversation_id",
            "subject",
            "resulting_run_id",
            "grant_id",
            "logical_key",
            "conclusion_digest",
            "payload",
            "sensitivity",
            "destination",
            "destination_sensitivity_ceiling",
            "state",
            "created_at",
            "updated_at",
            "attempt_count",
            "acknowledged_at",
            "terminal_error",
        ),
    )
    if integer(fields["version"], "inbox version") != _INBOX_VERSION:
        raise ValueError("stored inbox version is unsupported")
    try:
        sensitivity = ModelSensitivity(text(fields["sensitivity"], "inbox sensitivity"))
        ceiling = ModelSensitivity(
            text(
                fields["destination_sensitivity_ceiling"],
                "inbox destination sensitivity ceiling",
            )
        )
        state = DeliveryState(text(fields["state"], "inbox state"))
    except ValueError:
        raise ValueError("stored inbox enum is invalid") from None
    payload = plain_decode(fields["payload"])
    if not isinstance(payload, dict):
        raise ValueError("stored inbox payload must be an object")
    return InboxItem(
        delivery_id=delivery_id,
        agent_id=agent_id,
        conversation_id=text(fields["conversation_id"], "inbox conversation id"),
        subject=_decode_delivery_subject(fields["subject"]),
        resulting_run_id=text(fields["resulting_run_id"], "inbox run id"),
        grant_id=text(fields["grant_id"], "inbox grant id"),
        logical_key=text(fields["logical_key"], "inbox logical key"),
        conclusion_digest=text(
            fields["conclusion_digest"],
            "inbox conclusion digest",
        ),
        payload=payload,
        sensitivity=sensitivity,
        destination=text(fields["destination"], "inbox destination"),
        destination_sensitivity_ceiling=ceiling,
        state=state,
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
        attempt_count=integer(fields["attempt_count"], "inbox attempt count"),
        acknowledged_at=optional_datetime_decode(fields["acknowledged_at"]),
        terminal_error=optional_text(fields["terminal_error"], "inbox terminal error"),
    )


def _encode_delivery_subject(value: DeliverySubject):
    return record(
        "DeliverySubject",
        {
            "kind": value.kind.value,
            "subject_id": value.subject_id,
        },
    )


def _decode_delivery_subject(value) -> DeliverySubject:
    fields = record_fields(
        value,
        "DeliverySubject",
        ("kind", "subject_id"),
    )
    try:
        kind = DeliverySubjectKind(text(fields["kind"], "delivery subject kind"))
    except ValueError:
        raise ValueError("stored delivery subject kind is invalid") from None
    return DeliverySubject(
        kind=kind,
        subject_id=text(fields["subject_id"], "delivery subject id"),
    )


def _encode_conclusion_evidence(value: FollowupConclusionEvidence):
    return record(
        "FollowupConclusionEvidence",
        {
            "run_id": value.run_id,
            "job_id": value.job_id,
            "job_revision": value.job_revision,
            "inspection_call_id": value.inspection_call_id,
            "inspection_result_digest": value.inspection_result_digest,
            "result_call_id": value.result_call_id,
            "result_result_digest": value.result_result_digest,
            "job_result_id": value.job_result_id,
            "report_digest": value.report_digest,
        },
    )


def _decode_conclusion_evidence(value) -> FollowupConclusionEvidence:
    fields = record_fields(
        value,
        "FollowupConclusionEvidence",
        (
            "run_id",
            "job_id",
            "job_revision",
            "inspection_call_id",
            "inspection_result_digest",
            "result_call_id",
            "result_result_digest",
            "job_result_id",
            "report_digest",
        ),
    )
    return FollowupConclusionEvidence(
        run_id=text(fields["run_id"], "follow-up evidence run id"),
        job_id=text(fields["job_id"], "follow-up evidence job id"),
        job_revision=integer(
            fields["job_revision"],
            "follow-up evidence job revision",
        ),
        inspection_call_id=text(
            fields["inspection_call_id"],
            "follow-up evidence inspection call id",
        ),
        inspection_result_digest=text(
            fields["inspection_result_digest"],
            "follow-up evidence inspection digest",
        ),
        result_call_id=text(
            fields["result_call_id"],
            "follow-up evidence result call id",
        ),
        result_result_digest=text(
            fields["result_result_digest"],
            "follow-up evidence result digest",
        ),
        job_result_id=optional_text(
            fields["job_result_id"],
            "follow-up evidence job result id",
        ),
        report_digest=text(
            fields["report_digest"],
            "follow-up evidence report digest",
        ),
    )


def _encode_grant(value: FollowupGrant):
    return record(
        "FollowupGrant",
        {
            "grant_id": value.grant_id,
            "job_id": value.job_id,
            "agent_id": value.agent_id,
            "conversation_id": value.conversation_id,
            "authorizing_principal": value.authorizing_principal,
            "allowed_terminal_job_observation": (
                value.allowed_terminal_job_observation
            ),
            "allowed_source_ids": list(value.allowed_source_ids),
            "allowed_resource_ids": list(value.allowed_resource_ids),
            "allowed_capability_ids": list(value.allowed_capability_ids),
            "allowed_access_modes": plain_encode(
                tuple(sorted(item.value for item in value.allowed_access_modes))
            ),
            "allowed_operational_effects": plain_encode(
                tuple(sorted(item.value for item in value.allowed_operational_effects))
            ),
            "instruction_id": value.instruction_id,
            "instruction_digest": value.instruction_digest,
            "sensitivity_ceiling": value.sensitivity_ceiling.value,
            "delivery_sensitivity_ceiling": (value.delivery_sensitivity_ceiling.value),
            "eligible_model_routes": list(value.eligible_model_routes),
            "delivery_destination": value.delivery_destination,
            "max_successful_runs": value.max_successful_runs,
            "max_attempts": value.max_attempts,
            "per_run_max_cost_usd": decimal_encode(value.per_run_max_cost_usd),
            "per_run_max_tokens": value.per_run_max_tokens,
            "cumulative_max_cost_usd": decimal_encode(value.cumulative_max_cost_usd),
            "cumulative_max_tokens": value.cumulative_max_tokens,
            "expires_at": datetime_encode(value.expires_at),
        },
    )


def _decode_grant(value) -> FollowupGrant:
    fields = record_fields(
        value,
        "FollowupGrant",
        (
            "grant_id",
            "job_id",
            "agent_id",
            "conversation_id",
            "authorizing_principal",
            "allowed_terminal_job_observation",
            "allowed_source_ids",
            "allowed_resource_ids",
            "allowed_capability_ids",
            "allowed_access_modes",
            "allowed_operational_effects",
            "instruction_id",
            "instruction_digest",
            "sensitivity_ceiling",
            "delivery_sensitivity_ceiling",
            "eligible_model_routes",
            "delivery_destination",
            "max_successful_runs",
            "max_attempts",
            "per_run_max_cost_usd",
            "per_run_max_tokens",
            "cumulative_max_cost_usd",
            "cumulative_max_tokens",
            "expires_at",
        ),
    )
    try:
        access_modes = frozenset(
            AccessMode(text(item, "follow-up access mode"))
            for item in sequence(
                fields["allowed_access_modes"], "follow-up access modes"
            )
        )
        effects = frozenset(
            OperationalEffect(text(item, "follow-up effect"))
            for item in sequence(
                fields["allowed_operational_effects"],
                "follow-up effects",
            )
        )
        sensitivity = ModelSensitivity(
            text(fields["sensitivity_ceiling"], "follow-up sensitivity")
        )
        delivery_sensitivity = ModelSensitivity(
            text(
                fields["delivery_sensitivity_ceiling"],
                "follow-up delivery sensitivity",
            )
        )
    except ValueError:
        raise ValueError("stored follow-up grant enum is invalid") from None
    return FollowupGrant(
        grant_id=text(fields["grant_id"], "follow-up grant id"),
        job_id=text(fields["job_id"], "follow-up grant job id"),
        agent_id=text(fields["agent_id"], "follow-up grant agent id"),
        conversation_id=text(
            fields["conversation_id"], "follow-up grant conversation id"
        ),
        authorizing_principal=text(
            fields["authorizing_principal"],
            "follow-up authorizing principal",
        ),
        allowed_terminal_job_observation=text(
            fields["allowed_terminal_job_observation"],
            "follow-up allowed terminal observation",
        ),
        allowed_source_ids=_text_sequence(fields["allowed_source_ids"], "source id"),
        allowed_resource_ids=_text_sequence(
            fields["allowed_resource_ids"], "resource id"
        ),
        allowed_capability_ids=_text_sequence(
            fields["allowed_capability_ids"], "capability id"
        ),
        allowed_access_modes=access_modes,
        allowed_operational_effects=effects,
        instruction_id=text(fields["instruction_id"], "follow-up instruction id"),
        instruction_digest=text(
            fields["instruction_digest"], "follow-up instruction digest"
        ),
        sensitivity_ceiling=sensitivity,
        delivery_sensitivity_ceiling=delivery_sensitivity,
        eligible_model_routes=_text_sequence(
            fields["eligible_model_routes"], "model route"
        ),
        delivery_destination=text(
            fields["delivery_destination"],
            "follow-up delivery destination",
        ),
        max_successful_runs=integer(
            fields["max_successful_runs"], "follow-up successful runs"
        ),
        max_attempts=integer(fields["max_attempts"], "follow-up max attempts"),
        per_run_max_cost_usd=decimal_decode(fields["per_run_max_cost_usd"]),
        per_run_max_tokens=integer(
            fields["per_run_max_tokens"], "follow-up per-run tokens"
        ),
        cumulative_max_cost_usd=decimal_decode(fields["cumulative_max_cost_usd"]),
        cumulative_max_tokens=integer(
            fields["cumulative_max_tokens"], "follow-up cumulative tokens"
        ),
        expires_at=datetime_decode(fields["expires_at"]),
    )


def _text_sequence(value, name: str) -> tuple[str, ...]:
    return tuple(text(item, f"follow-up {name}") for item in sequence(value, name))


__all__ = [
    "decode_autonomous_followup",
    "decode_inbox_item",
    "encode_autonomous_followup",
    "encode_inbox_item",
]
