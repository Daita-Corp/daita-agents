"""Explicit SQLite codecs for source registration records."""

from __future__ import annotations

from dataclasses import replace

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


def persisted_source(value: SourceRegistration) -> SourceRegistration:
    """Return the registration identity stored independently from admission."""

    if not isinstance(value, SourceRegistration):
        raise TypeError("source codec requires SourceRegistration")
    if value.adapter_id != "postgresql" or "write_access" not in value.configuration:
        return value
    if not isinstance(value.configuration["write_access"], bool):
        raise ValueError("PostgreSQL write admission projection must be boolean")
    configuration = dict(value.configuration)
    del configuration["write_access"]
    return replace(value, configuration=configuration)


def project_source_admission(
    value: SourceRegistration,
    enabled: bool,
) -> SourceRegistration:
    if not isinstance(value, SourceRegistration):
        raise TypeError("source codec requires SourceRegistration")
    if not isinstance(enabled, bool):
        raise TypeError("source admission projection must be boolean")
    if value.adapter_id != "postgresql":
        return value
    configuration = dict(value.configuration)
    configuration["write_access"] = enabled
    return replace(value, configuration=configuration)


def encode_source(value: SourceRegistration) -> str:
    return dump_payload(_encode_source(persisted_source(value)))


def decode_source(value: str) -> SourceRegistration:
    decoded = _decode_source(load_payload(value), legacy_admission=False)
    if decoded.adapter_id == "postgresql" and "write_access" in decoded.configuration:
        raise ValueError("stored PostgreSQL source contains embedded write admission")
    return decoded


def decode_preledger_source(value: str) -> SourceRegistration:
    """Decode the one legacy source shape admitted only by the pre-ledger bridge."""

    return _decode_source(load_payload(value), legacy_admission=True)


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


def _decode_source(value: JsonValue, *, legacy_admission: bool) -> SourceRegistration:
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
        ),
        optional={"detached_at": None},
    )
    raw_configuration = mapping(fields["configuration"], "source configuration")
    configuration = plain_decode(raw_configuration)
    if not isinstance(configuration, dict):
        raise ValueError("stored source configuration is invalid")
    adapter_id = text(fields["adapter_id"], "source adapter_id")
    if adapter_id == "postgresql" and "write_access" in configuration:
        admission = configuration["write_access"]
        if not legacy_admission:
            raise ValueError(
                "stored PostgreSQL source contains embedded write admission"
            )
        if not isinstance(admission, bool):
            raise ValueError("stored PostgreSQL write admission is invalid")
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
