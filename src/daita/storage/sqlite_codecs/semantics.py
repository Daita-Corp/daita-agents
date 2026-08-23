"""Encode and decode semantic annotations, evidence, subjects, and bindings."""

from __future__ import annotations

from ...semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
)
from .common import (
    JsonValue,
    datetime_decode,
    datetime_encode,
    dump_payload,
    enum_decode,
    enum_encode,
    integer,
    load_payload,
    optional_text,
    record,
    record_fields,
    sequence,
    text,
)


def encode_semantic_annotation(value: SemanticAnnotation) -> str:
    if not isinstance(value, SemanticAnnotation):
        raise TypeError("semantic codec requires SemanticAnnotation")
    return dump_payload(_encode_annotation(value))


def decode_semantic_annotation(value: str) -> SemanticAnnotation:
    return _decode_annotation(load_payload(value))


def encode_field_reference(value: SemanticFieldReference) -> dict[str, JsonValue]:
    return record(
        "SemanticFieldReference",
        {"resource_id": value.resource_id, "field_name": value.field_name},
    )


def decode_field_reference(value: JsonValue) -> SemanticFieldReference:
    fields = record_fields(
        value,
        "SemanticFieldReference",
        ("resource_id", "field_name"),
    )
    return SemanticFieldReference(
        resource_id=text(fields["resource_id"], "semantic field resource_id"),
        field_name=text(fields["field_name"], "semantic field name"),
    )


def encode_revision_binding(value: ResourceRevisionBinding) -> dict[str, JsonValue]:
    return record(
        "ResourceRevisionBinding",
        {"resource_id": value.resource_id, "revision": value.revision},
    )


def decode_revision_binding(value: JsonValue) -> ResourceRevisionBinding:
    fields = record_fields(
        value,
        "ResourceRevisionBinding",
        ("resource_id", "revision"),
    )
    return ResourceRevisionBinding(
        resource_id=text(fields["resource_id"], "semantic revision resource_id"),
        revision=text(fields["revision"], "semantic revision"),
    )


def encode_subject(value: SemanticSubject) -> dict[str, JsonValue]:
    return record(
        "SemanticSubject",
        {
            "source_ids": list(value.source_ids),
            "resource_ids": list(value.resource_ids),
            "fields": [encode_field_reference(item) for item in value.fields],
        },
    )


def decode_subject(value: JsonValue) -> SemanticSubject:
    fields = record_fields(
        value,
        "SemanticSubject",
        ("source_ids", "resource_ids", "fields"),
    )
    return SemanticSubject(
        source_ids=tuple(
            text(item, "semantic source_id")
            for item in sequence(fields["source_ids"], "semantic source_ids")
        ),
        resource_ids=tuple(
            text(item, "semantic resource_id")
            for item in sequence(fields["resource_ids"], "semantic resource_ids")
        ),
        fields=tuple(
            decode_field_reference(item)
            for item in sequence(fields["fields"], "semantic fields")
        ),
    )


def _encode_evidence(value: SemanticEvidence) -> dict[str, JsonValue]:
    return record(
        "SemanticEvidence",
        {
            "kind": enum_encode(value.kind, "SemanticEvidenceKind"),
            "run_id": value.run_id,
            "message_position": value.message_position,
            "tool_call_id": value.tool_call_id,
            "note": value.note,
        },
    )


def _decode_evidence(value: JsonValue) -> SemanticEvidence:
    fields = record_fields(
        value,
        "SemanticEvidence",
        ("kind", "run_id", "message_position", "tool_call_id", "note"),
    )
    return SemanticEvidence(
        kind=enum_decode(fields["kind"], SemanticEvidenceKind, "SemanticEvidenceKind"),
        run_id=text(fields["run_id"], "semantic evidence run_id"),
        message_position=integer(
            fields["message_position"], "semantic evidence message position"
        ),
        tool_call_id=optional_text(
            fields["tool_call_id"], "semantic evidence tool call id"
        ),
        note=optional_text(fields["note"], "semantic evidence note"),
    )


def _encode_annotation(value: SemanticAnnotation) -> dict[str, JsonValue]:
    return record(
        "SemanticAnnotation",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "subject": encode_subject(value.subject),
            "kind": enum_encode(value.kind, "SemanticKind"),
            "statement": value.statement,
            "evidence": [_encode_evidence(item) for item in value.evidence],
            "catalog_revisions": [
                encode_revision_binding(item) for item in value.catalog_revisions
            ],
            "created_at": datetime_encode(value.created_at),
            "confirmed_at": datetime_encode(value.confirmed_at),
            "confirmed_by": value.confirmed_by,
            "supersedes_id": value.supersedes_id,
        },
    )


def _decode_annotation(value: JsonValue) -> SemanticAnnotation:
    fields = record_fields(
        value,
        "SemanticAnnotation",
        (
            "id",
            "agent_id",
            "subject",
            "kind",
            "statement",
            "evidence",
            "catalog_revisions",
            "created_at",
            "confirmed_at",
            "confirmed_by",
            "supersedes_id",
        ),
    )
    return SemanticAnnotation(
        id=text(fields["id"], "semantic annotation id"),
        agent_id=text(fields["agent_id"], "semantic annotation agent_id"),
        subject=decode_subject(fields["subject"]),
        kind=enum_decode(fields["kind"], SemanticKind, "SemanticKind"),
        statement=text(fields["statement"], "semantic statement"),
        evidence=tuple(
            _decode_evidence(item)
            for item in sequence(fields["evidence"], "semantic evidence")
        ),
        catalog_revisions=tuple(
            decode_revision_binding(item)
            for item in sequence(
                fields["catalog_revisions"], "semantic catalog revisions"
            )
        ),
        created_at=datetime_decode(fields["created_at"]),
        confirmed_at=datetime_decode(fields["confirmed_at"]),
        confirmed_by=text(fields["confirmed_by"], "semantic confirmed_by"),
        supersedes_id=optional_text(fields["supersedes_id"], "semantic supersedes id"),
    )
