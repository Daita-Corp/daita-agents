"""Encode and decode persistent agent identities and identifiers."""

from __future__ import annotations

from ...identity import AgentIdentity
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    load_payload,
    record,
    record_fields,
    text,
)


def encode_identity(value: AgentIdentity) -> str:
    if not isinstance(value, AgentIdentity):
        raise TypeError("identity codec requires AgentIdentity")
    return dump_payload(_encode_identity(value))


def decode_identity(value: str) -> AgentIdentity:
    return _decode_identity(load_payload(value))


def encode_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("identifier codec requires text")
    return dump_payload(value)


def decode_identifier(value: str) -> str:
    return text(load_payload(value), "stored identifier")


def _encode_identity(value: AgentIdentity) -> dict[str, JsonValue]:
    return record(
        "AgentIdentity",
        {
            "id": value.id,
            "display_name": value.display_name,
            "created_at": datetime_encode(value.created_at),
        },
    )


def _decode_identity(value: JsonValue) -> AgentIdentity:
    fields = record_fields(
        value,
        "AgentIdentity",
        ("id", "display_name", "created_at"),
    )
    return AgentIdentity(
        id=text(fields["id"], "identity id"),
        display_name=text(fields["display_name"], "identity display_name"),
        created_at=datetime_decode(fields["created_at"]),
    )
