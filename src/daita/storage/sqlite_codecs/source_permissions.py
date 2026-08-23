"""Encode and decode source read scopes and PostgreSQL update scopes."""

from __future__ import annotations

from ..sqlite_records import PostgreSQLUpdateScope, SourceReadMode, SourceReadScope
from .common import (
    dump_payload,
    integer,
    load_payload,
    record,
    record_fields,
    sequence,
    text,
)

_SOURCE_READ_SCOPE_VERSION = 1
_POSTGRESQL_UPDATE_SCOPE_VERSION = 1


def encode_source_read_scope(value: SourceReadScope) -> str:
    if not isinstance(value, SourceReadScope):
        raise TypeError("read-scope codec requires SourceReadScope")
    return dump_payload(
        record(
            "SourceReadScope",
            {
                "version": _SOURCE_READ_SCOPE_VERSION,
                "mode": value.mode.value,
                "resource_ids": list(value.resource_ids),
            },
        )
    )


def decode_source_read_scope(
    value: str,
    *,
    agent_id: str,
    source_id: str,
) -> SourceReadScope:
    fields = record_fields(
        load_payload(value),
        "SourceReadScope",
        ("version", "mode", "resource_ids"),
    )
    if integer(fields["version"], "read scope version") != _SOURCE_READ_SCOPE_VERSION:
        raise ValueError("stored read scope version is unsupported")
    try:
        mode = SourceReadMode(text(fields["mode"], "read scope mode"))
    except ValueError:
        raise ValueError("stored read scope mode is invalid") from None
    resource_ids = tuple(
        text(item, "read scope resource id")
        for item in sequence(fields["resource_ids"], "read scope resource_ids")
    )
    return SourceReadScope(
        agent_id=agent_id,
        source_id=source_id,
        mode=mode,
        resource_ids=resource_ids,
    )


def encode_postgresql_update_scope(value: PostgreSQLUpdateScope) -> str:
    if not isinstance(value, PostgreSQLUpdateScope):
        raise TypeError("update-scope codec requires PostgreSQLUpdateScope")
    return dump_payload(
        record(
            "PostgreSQLUpdateScope",
            {
                "version": _POSTGRESQL_UPDATE_SCOPE_VERSION,
                "allowed_assignment_columns": list(value.allowed_assignment_columns),
            },
        )
    )


def decode_postgresql_update_scope(
    value: str,
    *,
    agent_id: str,
    source_id: str,
    resource_id: str,
    authorization_fingerprint: str,
) -> PostgreSQLUpdateScope:
    fields = record_fields(
        load_payload(value),
        "PostgreSQLUpdateScope",
        ("version", "allowed_assignment_columns"),
    )
    if (
        integer(fields["version"], "update scope version")
        != _POSTGRESQL_UPDATE_SCOPE_VERSION
    ):
        raise ValueError("stored PostgreSQL update scope version is unsupported")
    columns = tuple(
        text(item, "update scope assignment column")
        for item in sequence(
            fields["allowed_assignment_columns"],
            "update scope allowed_assignment_columns",
        )
    )
    return PostgreSQLUpdateScope(
        agent_id=agent_id,
        source_id=source_id,
        resource_id=resource_id,
        allowed_assignment_columns=columns,
        authorization_fingerprint=authorization_fingerprint,
    )


__all__ = [
    "decode_postgresql_update_scope",
    "decode_source_read_scope",
    "encode_postgresql_update_scope",
    "encode_source_read_scope",
]
