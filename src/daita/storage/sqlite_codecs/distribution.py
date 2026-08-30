"""Encode the sole current codec-v1 outcome and logical-delivery families."""

from __future__ import annotations

from ...artifacts.models import ArtifactAuthorship
from ...distribution.models import (
    ArtifactRequirement,
    ConversationInboxTarget,
    Delivery,
    DeliveryState,
    DeliverySubjectKind,
    DistributionPlan,
    OutcomeArtifactReference,
    OutcomeConclusionKind,
    OutcomeContract,
    OutcomeReference,
    OutcomeState,
)
from ...llm.models import ModelSensitivity
from .common import (
    datetime_decode,
    datetime_encode,
    dump_payload,
    integer,
    load_payload,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    record,
    record_fields,
    sequence,
    text,
)

_DELIVERY_VERSION = 1


def encode_delivery(value: Delivery) -> str:
    if not isinstance(value, Delivery):
        raise TypeError("delivery codec requires Delivery")
    return dump_payload(
        record(
            "Delivery",
            {
                "version": _DELIVERY_VERSION,
                "conversation_id": value.conversation_id,
                "subject_kind": value.subject_kind.value,
                "subject_id": value.subject_id,
                "logical_key": value.logical_key,
                "target": encode_conversation_inbox_target(value.target),
                "outcome": encode_outcome_reference(value.outcome),
                "visibility_state": value.visibility_state.value,
                "acknowledged_at": optional_datetime_encode(value.acknowledged_at),
                "blocked_reason_code": value.blocked_reason_code,
                "created_at": datetime_encode(value.created_at),
                "updated_at": datetime_encode(value.updated_at),
            },
        )
    )


def decode_delivery(
    value: str,
    *,
    agent_id: str,
    delivery_id: str,
) -> Delivery:
    fields = record_fields(
        load_payload(value),
        "Delivery",
        (
            "version",
            "conversation_id",
            "subject_kind",
            "subject_id",
            "logical_key",
            "target",
            "outcome",
            "visibility_state",
            "acknowledged_at",
            "blocked_reason_code",
            "created_at",
            "updated_at",
        ),
    )
    if integer(fields["version"], "delivery version") != _DELIVERY_VERSION:
        raise ValueError("stored delivery version is unsupported")
    try:
        subject_kind = DeliverySubjectKind(
            text(fields["subject_kind"], "delivery subject kind")
        )
        state = DeliveryState(
            text(fields["visibility_state"], "delivery visibility state")
        )
    except ValueError:
        raise ValueError("stored delivery enum is invalid") from None
    return Delivery(
        delivery_id=delivery_id,
        agent_id=agent_id,
        conversation_id=text(fields["conversation_id"], "delivery conversation id"),
        subject_kind=subject_kind,
        subject_id=text(fields["subject_id"], "delivery subject id"),
        logical_key=text(fields["logical_key"], "delivery logical key"),
        target=decode_conversation_inbox_target(fields["target"]),
        outcome=decode_outcome_reference(fields["outcome"]),
        visibility_state=state,
        acknowledged_at=optional_datetime_decode(fields["acknowledged_at"]),
        blocked_reason_code=optional_text(
            fields["blocked_reason_code"],
            "delivery blocked reason code",
        ),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
    )


def encode_outcome_contract(value: OutcomeContract):
    if not isinstance(value, OutcomeContract):
        raise TypeError("outcome contract codec requires OutcomeContract")
    return record(
        "OutcomeContract",
        {
            "require_terminal_conclusion": value.require_terminal_conclusion,
            "artifact_requirements": [
                _encode_artifact_requirement(item)
                for item in value.artifact_requirements
            ],
            "maximum_total_artifact_bytes": value.maximum_total_artifact_bytes,
            "maximum_effective_sensitivity": (
                value.maximum_effective_sensitivity.value
            ),
            "require_current_run_provenance": value.require_current_run_provenance,
            "require_exact_source_bindings": value.require_exact_source_bindings,
        },
    )


def decode_outcome_contract(value) -> OutcomeContract:
    fields = record_fields(
        value,
        "OutcomeContract",
        (
            "require_terminal_conclusion",
            "artifact_requirements",
            "maximum_total_artifact_bytes",
            "maximum_effective_sensitivity",
            "require_current_run_provenance",
            "require_exact_source_bindings",
        ),
    )
    try:
        sensitivity = ModelSensitivity(
            text(
                fields["maximum_effective_sensitivity"],
                "outcome contract sensitivity",
            )
        )
    except ValueError:
        raise ValueError("stored outcome contract sensitivity is invalid") from None
    return OutcomeContract(
        require_terminal_conclusion=_boolean(
            fields["require_terminal_conclusion"],
            "outcome require_terminal_conclusion",
        ),
        artifact_requirements=tuple(
            _decode_artifact_requirement(item)
            for item in sequence(
                fields["artifact_requirements"],
                "outcome artifact requirements",
            )
        ),
        maximum_total_artifact_bytes=integer(
            fields["maximum_total_artifact_bytes"],
            "outcome maximum artifact bytes",
        ),
        maximum_effective_sensitivity=sensitivity,
        require_current_run_provenance=_boolean(
            fields["require_current_run_provenance"],
            "outcome require_current_run_provenance",
        ),
        require_exact_source_bindings=_boolean(
            fields["require_exact_source_bindings"],
            "outcome require_exact_source_bindings",
        ),
    )


def encode_distribution_plan(value: DistributionPlan):
    if not isinstance(value, DistributionPlan):
        raise TypeError("distribution plan codec requires DistributionPlan")
    return record(
        "DistributionPlan",
        {
            "targets": [
                encode_conversation_inbox_target(item) for item in value.targets
            ],
            "required_target_count": value.required_target_count,
            "plan_digest": value.plan_digest,
        },
    )


def decode_distribution_plan(value) -> DistributionPlan:
    fields = record_fields(
        value,
        "DistributionPlan",
        ("targets", "required_target_count", "plan_digest"),
    )
    return DistributionPlan(
        targets=tuple(
            decode_conversation_inbox_target(item)
            for item in sequence(fields["targets"], "distribution targets")
        ),
        required_target_count=integer(
            fields["required_target_count"],
            "distribution required target count",
        ),
        plan_digest=text(fields["plan_digest"], "distribution plan digest"),
    )


def encode_conversation_inbox_target(value: ConversationInboxTarget):
    if not isinstance(value, ConversationInboxTarget):
        raise TypeError("target codec requires ConversationInboxTarget")
    return record(
        "ConversationInboxTarget",
        {
            "conversation_id": value.conversation_id,
            "destination_id": value.destination_id,
            "destination_revision": value.destination_revision,
            "sensitivity_ceiling": value.sensitivity_ceiling.value,
            "target_fingerprint": value.target_fingerprint,
        },
    )


def decode_conversation_inbox_target(value) -> ConversationInboxTarget:
    fields = record_fields(
        value,
        "ConversationInboxTarget",
        (
            "conversation_id",
            "destination_id",
            "destination_revision",
            "sensitivity_ceiling",
            "target_fingerprint",
        ),
    )
    try:
        sensitivity = ModelSensitivity(
            text(fields["sensitivity_ceiling"], "target sensitivity ceiling")
        )
    except ValueError:
        raise ValueError("stored target sensitivity is invalid") from None
    return ConversationInboxTarget(
        conversation_id=text(fields["conversation_id"], "target conversation id"),
        destination_id=text(fields["destination_id"], "target destination id"),
        destination_revision=integer(
            fields["destination_revision"],
            "target destination revision",
        ),
        sensitivity_ceiling=sensitivity,
        target_fingerprint=text(
            fields["target_fingerprint"],
            "target fingerprint",
        ),
    )


def encode_outcome_reference(value: OutcomeReference):
    if not isinstance(value, OutcomeReference):
        raise TypeError("outcome reference codec requires OutcomeReference")
    return record(
        "OutcomeReference",
        {
            "conclusion_kind": value.conclusion_kind.value,
            "conclusion_state": value.conclusion_state.value,
            "conclusion_id": value.conclusion_id,
            "conclusion_digest": value.conclusion_digest,
            "conclusion_preview": value.conclusion_preview,
            "conclusion_preview_truncated": value.conclusion_preview_truncated,
            "resulting_run_id": value.resulting_run_id,
            "artifact_references": [
                _encode_outcome_artifact_reference(item)
                for item in value.artifact_references
            ],
            "effective_sensitivity": value.effective_sensitivity.value,
            "provenance_digest": value.provenance_digest,
            "failure_code": value.failure_code,
            "observed_at": datetime_encode(value.observed_at),
        },
    )


def decode_outcome_reference(value) -> OutcomeReference:
    fields = record_fields(
        value,
        "OutcomeReference",
        (
            "conclusion_kind",
            "conclusion_state",
            "conclusion_id",
            "conclusion_digest",
            "conclusion_preview",
            "conclusion_preview_truncated",
            "resulting_run_id",
            "artifact_references",
            "effective_sensitivity",
            "provenance_digest",
            "failure_code",
            "observed_at",
        ),
    )
    try:
        kind = OutcomeConclusionKind(
            text(fields["conclusion_kind"], "outcome conclusion kind")
        )
        state = OutcomeState(text(fields["conclusion_state"], "outcome state"))
        sensitivity = ModelSensitivity(
            text(fields["effective_sensitivity"], "outcome sensitivity")
        )
    except ValueError:
        raise ValueError("stored outcome reference enum is invalid") from None
    return OutcomeReference(
        conclusion_kind=kind,
        conclusion_state=state,
        conclusion_id=text(fields["conclusion_id"], "outcome conclusion id"),
        conclusion_digest=text(
            fields["conclusion_digest"],
            "outcome conclusion digest",
        ),
        conclusion_preview=_string(
            fields["conclusion_preview"],
            "outcome conclusion preview",
        ),
        conclusion_preview_truncated=_boolean(
            fields["conclusion_preview_truncated"],
            "outcome conclusion preview truncated",
        ),
        resulting_run_id=optional_text(
            fields["resulting_run_id"],
            "outcome resulting run id",
        ),
        artifact_references=tuple(
            _decode_outcome_artifact_reference(item)
            for item in sequence(
                fields["artifact_references"],
                "outcome artifact references",
            )
        ),
        effective_sensitivity=sensitivity,
        provenance_digest=text(
            fields["provenance_digest"],
            "outcome provenance digest",
        ),
        failure_code=optional_text(fields["failure_code"], "outcome failure code"),
        observed_at=datetime_decode(fields["observed_at"]),
    )


def _encode_artifact_requirement(value: ArtifactRequirement):
    return record(
        "ArtifactRequirement",
        {
            "required": value.required,
            "minimum_count": value.minimum_count,
            "maximum_count": value.maximum_count,
            "allowed_media_types": list(value.allowed_media_types),
            "allowed_authorships": [item.value for item in value.allowed_authorships],
            "allowed_producer_capability_ids": list(
                value.allowed_producer_capability_ids
            ),
            "maximum_artifact_bytes": value.maximum_artifact_bytes,
            "maximum_total_bytes": value.maximum_total_bytes,
            "maximum_sensitivity": value.maximum_sensitivity.value,
        },
    )


def _decode_artifact_requirement(value) -> ArtifactRequirement:
    fields = record_fields(
        value,
        "ArtifactRequirement",
        (
            "required",
            "minimum_count",
            "maximum_count",
            "allowed_media_types",
            "allowed_authorships",
            "allowed_producer_capability_ids",
            "maximum_artifact_bytes",
            "maximum_total_bytes",
            "maximum_sensitivity",
        ),
    )
    try:
        authorships = tuple(
            ArtifactAuthorship(text(item, "artifact requirement authorship"))
            for item in sequence(
                fields["allowed_authorships"],
                "artifact requirement authorships",
            )
        )
        sensitivity = ModelSensitivity(
            text(fields["maximum_sensitivity"], "artifact maximum sensitivity")
        )
    except ValueError:
        raise ValueError("stored artifact requirement enum is invalid") from None
    return ArtifactRequirement(
        required=_boolean(fields["required"], "artifact requirement required"),
        minimum_count=integer(
            fields["minimum_count"],
            "artifact requirement minimum count",
        ),
        maximum_count=integer(
            fields["maximum_count"],
            "artifact requirement maximum count",
        ),
        allowed_media_types=_text_sequence(
            fields["allowed_media_types"],
            "artifact requirement media type",
        ),
        allowed_authorships=authorships,
        allowed_producer_capability_ids=_text_sequence(
            fields["allowed_producer_capability_ids"],
            "artifact requirement producer capability id",
        ),
        maximum_artifact_bytes=integer(
            fields["maximum_artifact_bytes"],
            "artifact requirement maximum artifact bytes",
        ),
        maximum_total_bytes=integer(
            fields["maximum_total_bytes"],
            "artifact requirement maximum total bytes",
        ),
        maximum_sensitivity=sensitivity,
    )


def _encode_outcome_artifact_reference(value: OutcomeArtifactReference):
    return record(
        "OutcomeArtifactReference",
        {
            "artifact_id": value.artifact_id,
            "producing_run_id": value.producing_run_id,
            "producing_call_id": value.producing_call_id,
            "producer_capability_id": value.producer_capability_id,
            "sha256": value.sha256,
            "media_type": value.media_type,
            "byte_size": value.byte_size,
            "sensitivity": value.sensitivity.value,
            "provenance_digest": value.provenance_digest,
            "authorship": value.authorship.value,
        },
    )


def _decode_outcome_artifact_reference(value) -> OutcomeArtifactReference:
    fields = record_fields(
        value,
        "OutcomeArtifactReference",
        (
            "artifact_id",
            "producing_run_id",
            "producing_call_id",
            "producer_capability_id",
            "sha256",
            "media_type",
            "byte_size",
            "sensitivity",
            "provenance_digest",
            "authorship",
        ),
    )
    try:
        sensitivity = ModelSensitivity(
            text(fields["sensitivity"], "outcome artifact sensitivity")
        )
        authorship = ArtifactAuthorship(
            text(fields["authorship"], "outcome artifact authorship")
        )
    except ValueError:
        raise ValueError("stored outcome artifact enum is invalid") from None
    return OutcomeArtifactReference(
        artifact_id=text(fields["artifact_id"], "outcome artifact id"),
        producing_run_id=text(
            fields["producing_run_id"],
            "outcome artifact producing run id",
        ),
        producing_call_id=text(
            fields["producing_call_id"],
            "outcome artifact producing call id",
        ),
        producer_capability_id=text(
            fields["producer_capability_id"],
            "outcome artifact producer capability id",
        ),
        sha256=text(fields["sha256"], "outcome artifact sha256"),
        media_type=text(fields["media_type"], "outcome artifact media type"),
        byte_size=integer(fields["byte_size"], "outcome artifact byte size"),
        sensitivity=sensitivity,
        provenance_digest=text(
            fields["provenance_digest"],
            "outcome artifact provenance digest",
        ),
        authorship=authorship,
    )


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _string(value, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _text_sequence(value, name: str) -> tuple[str, ...]:
    return tuple(text(item, name) for item in sequence(value, name))


__all__ = [
    "decode_conversation_inbox_target",
    "decode_delivery",
    "decode_distribution_plan",
    "decode_outcome_contract",
    "decode_outcome_reference",
    "encode_conversation_inbox_target",
    "encode_delivery",
    "encode_distribution_plan",
    "encode_outcome_contract",
    "encode_outcome_reference",
]
