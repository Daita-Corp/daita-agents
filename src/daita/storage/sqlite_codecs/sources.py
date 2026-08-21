"""Explicit SQLite codecs for source registration records."""

from __future__ import annotations

from ...adapters.models import SourceRegistration
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    load_payload,
    mapping,
    optional_datetime_decode,
    optional_datetime_encode,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    text,
)


def _require_current_source(value: SourceRegistration) -> SourceRegistration:
    if not isinstance(value, SourceRegistration):
        raise TypeError("source codec requires SourceRegistration")
    if value.adapter_id == "postgresql" and "write_access" in value.configuration:
        raise ValueError("PostgreSQL source contains embedded write admission")
    return value


def encode_source(value: SourceRegistration) -> str:
    return dump_payload(_encode_source(_require_current_source(value)))


def decode_source(value: str) -> SourceRegistration:
    decoded = _decode_source(load_payload(value))
    if decoded.adapter_id == "postgresql" and "write_access" in decoded.configuration:
        raise ValueError("stored PostgreSQL source contains embedded write admission")
    return decoded


def _encode_source(value: SourceRegistration) -> dict[str, JsonValue]:
    return record(
        "SourceRegistration",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "adapter_id": value.adapter_id,
            "native_identity": value.native_identity,
            "display_name": value.display_name,
            "configuration": plain_encode(value.configuration),
            "attached_at": datetime_encode(value.attached_at),
            "detached_at": optional_datetime_encode(value.detached_at),
        },
    )


def _decode_source(value: JsonValue) -> SourceRegistration:
    fields = record_fields(
        value,
        "SourceRegistration",
        (
            "id",
            "agent_id",
            "adapter_id",
            "native_identity",
            "display_name",
            "configuration",
            "attached_at",
            "detached_at",
        ),
    )
    raw_configuration = mapping(fields["configuration"], "source configuration")
    configuration = plain_decode(raw_configuration)
    if not isinstance(configuration, dict):
        raise ValueError("stored source configuration is invalid")
    adapter_id = text(fields["adapter_id"], "source adapter_id")
    if adapter_id == "postgresql" and "write_access" in configuration:
        raise ValueError("stored PostgreSQL source contains embedded write admission")
    return SourceRegistration(
        id=text(fields["id"], "source id"),
        agent_id=text(fields["agent_id"], "source agent_id"),
        adapter_id=adapter_id,
        native_identity=text(fields["native_identity"], "source native_identity"),
        display_name=text(fields["display_name"], "source display_name"),
        configuration=configuration,
        attached_at=datetime_decode(fields["attached_at"]),
        detached_at=optional_datetime_decode(fields["detached_at"]),
    )
