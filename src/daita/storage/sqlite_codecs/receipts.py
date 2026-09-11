"""Encode the current effect observation and its separate human resolution."""

from __future__ import annotations

from ..._json import FrozenJsonObject
from ...capabilities import EffectEvidenceBasis, EffectOutcome
from ...llm.models import ModelSensitivity
from ..sqlite_records import EffectReceipt, EffectResolution, EffectResolutionDecision
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    load_payload,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_integer,
    optional_text,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)

_FIELDS = (
    "receipt_id",
    "receipt_kind",
    "agent_id",
    "run_id",
    "call_id",
    "capability_id",
    "domain_owner_id",
    "capability_contract_digest",
    "routine_id",
    "routine_revision",
    "occurrence_id",
    "capability_grant_digest",
    "operation_key",
    "argument_fingerprint",
    "outcome",
    "evidence_basis",
    "sensitivity",
    "payload",
    "started_at",
    "finished_at",
    "receipt_digest",
    "resolution",
)
_RESOLUTION_FIELDS = (
    "receipt_id",
    "receipt_digest",
    "decision",
    "approving_principal_id",
    "control_id",
    "resolved_at",
    "note",
    "evidence_references",
)


def encode_receipt(value: EffectReceipt) -> str:
    if not isinstance(value, EffectReceipt):
        raise TypeError("receipt codec requires EffectReceipt")
    fields = {key: plain_encode(item) for key, item in value.material().items()}
    fields["started_at"] = datetime_encode(value.started_at)
    fields["finished_at"] = optional_datetime_encode(value.finished_at)
    fields["receipt_digest"] = value.receipt_digest
    resolution = value.resolution
    fields["resolution"] = (
        None
        if resolution is None
        else record(
            "EffectResolution",
            {
                "receipt_id": resolution.receipt_id,
                "receipt_digest": resolution.receipt_digest,
                "decision": resolution.decision.value,
                "approving_principal_id": resolution.approving_principal_id,
                "control_id": resolution.control_id,
                "resolved_at": datetime_encode(resolution.resolved_at),
                "note": resolution.note,
                "evidence_references": list(resolution.evidence_references),
            },
        )
    )
    encoded = dump_payload(record("EffectReceipt", fields))
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise ValueError("effect receipt exceeds its encoded byte bound")
    return encoded


def decode_receipt(value: str) -> EffectReceipt:
    if len(value.encode("utf-8")) > 64 * 1024:
        raise ValueError("effect receipt exceeds its encoded byte bound")
    fields = record_fields(load_payload(value), "EffectReceipt", _FIELDS)
    resolution = _decode_resolution(fields["resolution"])
    raw_payload = fields["payload"]
    if raw_payload is not None and not isinstance(raw_payload, dict):
        raise ValueError("effect payload must be an object or null")
    receipt = EffectReceipt(
        receipt_id=text(fields["receipt_id"], "receipt id"),
        receipt_kind=text(fields["receipt_kind"], "receipt kind"),
        agent_id=text(fields["agent_id"], "receipt agent"),
        run_id=text(fields["run_id"], "receipt run"),
        call_id=text(fields["call_id"], "receipt call"),
        capability_id=text(fields["capability_id"], "receipt capability"),
        domain_owner_id=text(fields["domain_owner_id"], "receipt domain"),
        capability_contract_digest=text(
            fields["capability_contract_digest"], "receipt contract"
        ),
        routine_id=optional_text(fields["routine_id"], "receipt routine"),
        routine_revision=optional_integer(
            fields["routine_revision"], "receipt routine revision"
        ),
        occurrence_id=optional_text(fields["occurrence_id"], "receipt occurrence"),
        capability_grant_digest=optional_text(
            fields["capability_grant_digest"], "receipt grant"
        ),
        operation_key=text(fields["operation_key"], "receipt operation key"),
        argument_fingerprint=text(
            fields["argument_fingerprint"], "receipt argument fingerprint"
        ),
        outcome=EffectOutcome(text(fields["outcome"], "receipt outcome")),
        evidence_basis=EffectEvidenceBasis(
            text(fields["evidence_basis"], "receipt basis")
        ),
        sensitivity=ModelSensitivity(
            text(fields["sensitivity"], "receipt sensitivity")
        ),
        payload=(
            None if raw_payload is None else FrozenJsonObject.from_mapping(raw_payload)
        ),
        started_at=datetime_decode(fields["started_at"]),
        finished_at=optional_datetime_decode(fields["finished_at"]),
        resolution=resolution,
    )
    if receipt.receipt_digest != text(fields["receipt_digest"], "receipt digest"):
        raise ValueError("receipt digest does not match its observation")
    return receipt


def _decode_resolution(value: JsonValue) -> EffectResolution | None:
    if value is None:
        return None
    fields = record_fields(value, "EffectResolution", _RESOLUTION_FIELDS)
    return EffectResolution(
        receipt_id=text(fields["receipt_id"], "resolution receipt"),
        receipt_digest=text(fields["receipt_digest"], "resolution observation digest"),
        decision=EffectResolutionDecision(
            text(fields["decision"], "resolution decision")
        ),
        approving_principal_id=text(
            fields["approving_principal_id"], "resolution principal"
        ),
        control_id=text(fields["control_id"], "resolution control"),
        resolved_at=datetime_decode(fields["resolved_at"]),
        note=text(fields["note"], "resolution note"),
        evidence_references=tuple(
            text(item, "evidence reference")
            for item in sequence(fields["evidence_references"], "evidence references")
        ),
    )
