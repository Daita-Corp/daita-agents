"""Encode and decode source read scopes and PostgreSQL update scopes."""

from __future__ import annotations

from ..sqlite_records import RelationalWriteScope, SourceReadMode, SourceReadScope
from .common import (
    dump_payload,
    integer,
    load_payload,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)


def encode_source_read_scope(value: SourceReadScope) -> str:
    if not isinstance(value, SourceReadScope):
        raise TypeError("read-scope codec requires SourceReadScope")
    return dump_payload(
        record(
            "SourceReadScope",
            {
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
        ("mode", "resource_ids"),
    )
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


def encode_relational_write_scope(value: RelationalWriteScope) -> str:
    if not isinstance(value, RelationalWriteScope):
        raise TypeError("write-scope codec requires RelationalWriteScope")
    return dump_payload(
        record(
            "RelationalWriteScope",
            {
                **{
                    key: plain_encode(item) for key, item in value.constraints().items()
                },
            },
        )
    )


def decode_relational_write_scope(
    value: str,
    *,
    agent_id: str,
    source_id: str,
    resource_id: str,
    authorization_fingerprint: str,
) -> RelationalWriteScope:
    fields = record_fields(
        load_payload(value),
        "RelationalWriteScope",
        (
            "resource_revision",
            "allowed_operations",
            "allowed_insert_columns",
            "allowed_update_columns",
            "key_columns",
            "generated_identity_columns",
            "max_rows",
        ),
    )

    def names(name: str) -> tuple[str, ...]:
        return tuple(text(item, name) for item in sequence(fields[name], name))

    return RelationalWriteScope(
        agent_id=agent_id,
        source_id=source_id,
        resource_id=resource_id,
        authorization_fingerprint=authorization_fingerprint,
        resource_revision=text(fields["resource_revision"], "resource_revision"),
        allowed_operations=names("allowed_operations"),
        allowed_insert_columns=names("allowed_insert_columns"),
        allowed_update_columns=names("allowed_update_columns"),
        key_columns=names("key_columns"),
        generated_identity_columns=names("generated_identity_columns"),
        max_rows=integer(fields["max_rows"], "max_rows"),
    )


__all__ = [
    "decode_relational_write_scope",
    "decode_source_read_scope",
    "encode_relational_write_scope",
    "encode_source_read_scope",
]
