"""Immutable revision-1 autonomous-follow-up delivery records for migration tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256

from ..._json import canonical_json
from ...distribution.models import (
    ConversationInboxTarget,
    DeliveryState,
    OutcomeReference,
)
from ..sqlite_codecs.common import (
    datetime_decode,
    datetime_encode,
    dump_payload,
    load_payload,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    record,
    record_fields,
    text,
)
from ..sqlite_codecs.distribution import (
    decode_conversation_inbox_target,
    decode_outcome_reference,
    encode_conversation_inbox_target,
    encode_outcome_reference,
)


class Revision1DeliverySubjectKind(str, Enum):
    AUTONOMOUS_FOLLOWUP = "autonomous_followup"
    ROUTINE_OCCURRENCE = "routine_occurrence"


def revision_1_logical_delivery_key(
    *,
    agent_id: str,
    subject_kind: Revision1DeliverySubjectKind,
    subject_id: str,
    target_fingerprint: str,
) -> str:
    material = {
        "agent_id": agent_id,
        "subject_kind": subject_kind.value,
        "subject_id": subject_id,
        "target_fingerprint": target_fingerprint,
    }
    return "delivery:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Revision1Delivery:
    delivery_id: str
    agent_id: str
    conversation_id: str
    subject_kind: Revision1DeliverySubjectKind
    subject_id: str
    logical_key: str
    target: ConversationInboxTarget
    outcome: OutcomeReference
    visibility_state: DeliveryState
    acknowledged_at: datetime | None
    blocked_reason_code: str | None
    created_at: datetime
    updated_at: datetime


def encode_revision_1_delivery(value: Revision1Delivery) -> str:
    if not isinstance(value, Revision1Delivery):
        raise TypeError("revision-1 delivery codec requires Revision1Delivery")
    return dump_payload(
        record(
            "Delivery",
            {
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


def decode_revision_1_delivery(
    value: str,
    *,
    agent_id: str,
    delivery_id: str,
) -> Revision1Delivery:
    fields = record_fields(
        load_payload(value),
        "Delivery",
        (
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
    try:
        subject_kind = Revision1DeliverySubjectKind(
            text(fields["subject_kind"], "revision-1 delivery subject kind")
        )
        visibility = DeliveryState(
            text(fields["visibility_state"], "revision-1 delivery state")
        )
    except ValueError:
        raise ValueError("stored revision-1 delivery enum is invalid") from None
    result = Revision1Delivery(
        delivery_id=delivery_id,
        agent_id=agent_id,
        conversation_id=text(
            fields["conversation_id"], "revision-1 delivery conversation"
        ),
        subject_kind=subject_kind,
        subject_id=text(fields["subject_id"], "revision-1 delivery subject ID"),
        logical_key=text(fields["logical_key"], "revision-1 delivery logical key"),
        target=decode_conversation_inbox_target(fields["target"]),
        outcome=decode_outcome_reference(fields["outcome"]),
        visibility_state=visibility,
        acknowledged_at=optional_datetime_decode(fields["acknowledged_at"]),
        blocked_reason_code=optional_text(
            fields["blocked_reason_code"], "revision-1 delivery blocked reason"
        ),
        created_at=datetime_decode(fields["created_at"]),
        updated_at=datetime_decode(fields["updated_at"]),
    )
    if result.target.conversation_id != result.conversation_id:
        raise ValueError("stored revision-1 delivery target differs")
    if result.logical_key != revision_1_logical_delivery_key(
        agent_id=agent_id,
        subject_kind=subject_kind,
        subject_id=result.subject_id,
        target_fingerprint=result.target.target_fingerprint,
    ):
        raise ValueError("stored revision-1 delivery logical key differs")
    return result


__all__ = [
    "Revision1Delivery",
    "Revision1DeliverySubjectKind",
    "decode_revision_1_delivery",
    "encode_revision_1_delivery",
    "revision_1_logical_delivery_key",
]
