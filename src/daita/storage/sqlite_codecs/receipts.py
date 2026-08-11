"""Explicit SQLite codec for database-write receipt records."""

from __future__ import annotations

from ..sqlite_records import DatabaseWriteOutcome, DatabaseWriteReceipt
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    enum_decode,
    enum_encode,
    load_payload,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_integer,
    optional_text,
    record,
    record_fields,
    text,
)


def encode_receipt(value: DatabaseWriteReceipt) -> str:
    if not isinstance(value, DatabaseWriteReceipt):
        raise TypeError("receipt codec requires DatabaseWriteReceipt")
    return dump_payload(_encode_receipt(value))


def decode_receipt(value: str) -> DatabaseWriteReceipt:
    return _decode_receipt(load_payload(value))


def _encode_receipt(value: DatabaseWriteReceipt) -> dict[str, JsonValue]:
    return record(
        "DatabaseWriteReceipt",
        {
            "receipt_id": value.receipt_id,
            "agent_id": value.agent_id,
            "run_id": value.run_id,
            "call_id": value.call_id,
            "capability_id": value.capability_id,
            "source_id": value.source_id,
            "resource_id": value.resource_id,
            "intent_sha256": value.intent_sha256,
            "preview_fingerprint": value.preview_fingerprint,
            "outcome": enum_encode(value.outcome, "DatabaseWriteOutcome"),
            "affected_rows": value.affected_rows,
            "normalized_error_code": value.normalized_error_code,
            "started_at": datetime_encode(value.started_at),
            "completed_at": optional_datetime_encode(value.completed_at),
        },
    )


def _decode_receipt(value: JsonValue) -> DatabaseWriteReceipt:
    fields = record_fields(
        value,
        "DatabaseWriteReceipt",
        (
            "receipt_id",
            "agent_id",
            "run_id",
            "call_id",
            "capability_id",
            "source_id",
            "resource_id",
            "intent_sha256",
            "preview_fingerprint",
            "outcome",
            "affected_rows",
            "normalized_error_code",
            "started_at",
            "completed_at",
        ),
    )
    return DatabaseWriteReceipt(
        receipt_id=text(fields["receipt_id"], "receipt id"),
        agent_id=text(fields["agent_id"], "receipt agent_id"),
        run_id=text(fields["run_id"], "receipt run_id"),
        call_id=text(fields["call_id"], "receipt call_id"),
        capability_id=text(fields["capability_id"], "receipt capability_id"),
        source_id=text(fields["source_id"], "receipt source_id"),
        resource_id=text(fields["resource_id"], "receipt resource_id"),
        intent_sha256=text(fields["intent_sha256"], "receipt intent hash"),
        preview_fingerprint=text(
            fields["preview_fingerprint"], "receipt preview fingerprint"
        ),
        outcome=enum_decode(
            fields["outcome"], DatabaseWriteOutcome, "DatabaseWriteOutcome"
        ),
        affected_rows=optional_integer(fields["affected_rows"], "affected rows"),
        normalized_error_code=optional_text(
            fields["normalized_error_code"], "normalized error code"
        ),
        started_at=datetime_decode(fields["started_at"]),
        completed_at=optional_datetime_decode(fields["completed_at"]),
    )
