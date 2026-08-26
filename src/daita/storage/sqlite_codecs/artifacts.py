"""Encode and decode artifact bindings, provenance, references, and receipts."""

from __future__ import annotations

from ...artifacts.models import (
    ArtifactAuthorship,
    ArtifactDeliveryMode,
    ArtifactDeliveryOutcome,
    ArtifactDeliveryReceipt,
    ArtifactLocalFileBinding,
    ArtifactProvenance,
    ArtifactRef,
    ArtifactResourceBinding,
    ArtifactTextChangeSummary,
    artifact_text_change_summary_to_mapping,
)
from ...catalog.models import Sensitivity
from .common import (
    JsonValue,
    boolean,
    datetime_decode,
    datetime_encode,
    enum_decode,
    enum_encode,
    integer,
    optional_integer,
    optional_text,
    record,
    record_fields,
    sequence,
    text,
)


def encode_artifact_binding(value: ArtifactResourceBinding) -> dict[str, JsonValue]:
    return record(
        "ArtifactResourceBinding",
        {
            "source_id": value.source_id,
            "source_revision": value.source_revision,
            "resource_id": value.resource_id,
            "resource_revision": value.resource_revision,
        },
    )


def decode_artifact_binding(value: JsonValue) -> ArtifactResourceBinding:
    fields = record_fields(
        value,
        "ArtifactResourceBinding",
        ("source_id", "source_revision", "resource_id", "resource_revision"),
    )
    return ArtifactResourceBinding(
        source_id=text(fields["source_id"], "artifact source_id"),
        source_revision=text(fields["source_revision"], "artifact source_revision"),
        resource_id=text(fields["resource_id"], "artifact resource_id"),
        resource_revision=text(
            fields["resource_revision"], "artifact resource_revision"
        ),
    )


def encode_local_file_binding(
    value: ArtifactLocalFileBinding,
) -> dict[str, JsonValue]:
    return record(
        "ArtifactLocalFileBinding",
        {
            "workspace_id": value.workspace_id,
            "relative_path": value.relative_path,
            "original_physical_revision": value.original_physical_revision,
            "observed_content_sha256": value.observed_content_sha256,
            "source_byte_size": value.source_byte_size,
            "change_summary": record(
                "ArtifactTextChangeSummary",
                artifact_text_change_summary_to_mapping(value.change_summary),
            ),
        },
    )


def decode_local_file_binding(value: JsonValue) -> ArtifactLocalFileBinding:
    fields = record_fields(
        value,
        "ArtifactLocalFileBinding",
        (
            "workspace_id",
            "relative_path",
            "original_physical_revision",
            "observed_content_sha256",
            "source_byte_size",
            "change_summary",
        ),
    )
    summary = record_fields(
        fields["change_summary"],
        "ArtifactTextChangeSummary",
        (
            "operation_count",
            "replacement_count",
            "insertion_count",
            "deletion_count",
            "occurrence_count",
            "bytes_removed",
            "bytes_added",
            "description",
        ),
    )
    return ArtifactLocalFileBinding(
        workspace_id=text(fields["workspace_id"], "artifact workspace id"),
        relative_path=text(fields["relative_path"], "artifact relative path"),
        original_physical_revision=text(
            fields["original_physical_revision"],
            "artifact original physical revision",
        ),
        observed_content_sha256=text(
            fields["observed_content_sha256"], "artifact observed content hash"
        ),
        source_byte_size=integer(
            fields["source_byte_size"], "artifact source byte size"
        ),
        change_summary=ArtifactTextChangeSummary(
            operation_count=integer(
                summary["operation_count"], "artifact edit operation count"
            ),
            replacement_count=integer(
                summary["replacement_count"], "artifact replacement count"
            ),
            insertion_count=integer(
                summary["insertion_count"], "artifact insertion count"
            ),
            deletion_count=integer(
                summary["deletion_count"], "artifact deletion count"
            ),
            occurrence_count=integer(
                summary["occurrence_count"], "artifact occurrence count"
            ),
            bytes_removed=integer(
                summary["bytes_removed"], "artifact removed byte count"
            ),
            bytes_added=integer(summary["bytes_added"], "artifact added byte count"),
            description=text(summary["description"], "artifact change description"),
        ),
    )


def encode_artifact_provenance(value: ArtifactProvenance) -> dict[str, JsonValue]:
    return record(
        "ArtifactProvenance",
        {
            "authorship": enum_encode(value.authorship, "ArtifactAuthorship"),
            "evidence_call_ids": list(value.evidence_call_ids),
            "derived_from_artifact_id": value.derived_from_artifact_id,
            "resource_bindings": [
                encode_artifact_binding(item) for item in value.resource_bindings
            ],
            "local_file_binding": (
                None
                if value.local_file_binding is None
                else encode_local_file_binding(value.local_file_binding)
            ),
            "sql_fingerprint": value.sql_fingerprint,
            "parameters_sha256": value.parameters_sha256,
            "columns": list(value.columns),
            "row_count": value.row_count,
        },
    )


def decode_artifact_provenance(value: JsonValue) -> ArtifactProvenance:
    fields = record_fields(
        value,
        "ArtifactProvenance",
        (
            "authorship",
            "evidence_call_ids",
            "derived_from_artifact_id",
            "resource_bindings",
            "local_file_binding",
            "sql_fingerprint",
            "parameters_sha256",
            "columns",
            "row_count",
        ),
    )
    return ArtifactProvenance(
        authorship=enum_decode(
            fields["authorship"], ArtifactAuthorship, "ArtifactAuthorship"
        ),
        evidence_call_ids=tuple(
            text(item, "artifact evidence call id")
            for item in sequence(
                fields["evidence_call_ids"], "artifact evidence call ids"
            )
        ),
        derived_from_artifact_id=optional_text(
            fields["derived_from_artifact_id"], "derived artifact id"
        ),
        resource_bindings=tuple(
            decode_artifact_binding(item)
            for item in sequence(
                fields["resource_bindings"], "artifact resource bindings"
            )
        ),
        local_file_binding=(
            None
            if fields["local_file_binding"] is None
            else decode_local_file_binding(fields["local_file_binding"])
        ),
        sql_fingerprint=optional_text(
            fields["sql_fingerprint"], "artifact SQL fingerprint"
        ),
        parameters_sha256=optional_text(
            fields["parameters_sha256"], "artifact parameters hash"
        ),
        columns=tuple(
            text(item, "artifact column")
            for item in sequence(fields["columns"], "artifact columns")
        ),
        row_count=optional_integer(fields["row_count"], "artifact row count"),
    )


def encode_artifact_ref(value: ArtifactRef) -> dict[str, JsonValue]:
    return record(
        "ArtifactRef",
        {
            "artifact_id": value.artifact_id,
            "run_id": value.run_id,
            "conversation_id": value.conversation_id,
            "call_id": value.call_id,
            "capability_id": value.capability_id,
            "filename": value.filename,
            "media_type": value.media_type,
            "byte_size": value.byte_size,
            "sha256": value.sha256,
            "sensitivity": enum_encode(value.sensitivity, "Sensitivity"),
            "provenance": encode_artifact_provenance(value.provenance),
            "created_at": datetime_encode(value.created_at),
        },
    )


def decode_artifact_ref(value: JsonValue) -> ArtifactRef:
    fields = record_fields(
        value,
        "ArtifactRef",
        (
            "artifact_id",
            "run_id",
            "conversation_id",
            "call_id",
            "capability_id",
            "filename",
            "media_type",
            "byte_size",
            "sha256",
            "sensitivity",
            "provenance",
            "created_at",
        ),
    )
    return ArtifactRef(
        artifact_id=text(fields["artifact_id"], "artifact id"),
        run_id=text(fields["run_id"], "artifact run_id"),
        conversation_id=text(fields["conversation_id"], "artifact conversation_id"),
        call_id=text(fields["call_id"], "artifact call_id"),
        capability_id=text(fields["capability_id"], "artifact capability_id"),
        filename=text(fields["filename"], "artifact filename"),
        media_type=text(fields["media_type"], "artifact media_type"),
        byte_size=integer(fields["byte_size"], "artifact byte_size"),
        sha256=text(fields["sha256"], "artifact sha256"),
        sensitivity=enum_decode(fields["sensitivity"], Sensitivity, "Sensitivity"),
        provenance=decode_artifact_provenance(fields["provenance"]),
        created_at=datetime_decode(fields["created_at"]),
    )


def encode_delivery_receipt(
    value: ArtifactDeliveryReceipt,
) -> dict[str, JsonValue]:
    return record(
        "ArtifactDeliveryReceipt",
        {
            "artifact_id": value.artifact_id,
            "destination_id": value.destination_id,
            "filename": value.filename,
            "saved_path": value.saved_path,
            "byte_size": value.byte_size,
            "sha256": value.sha256,
            "renamed_for_collision": value.renamed_for_collision,
            "delivered_at": datetime_encode(value.delivered_at),
            "mode": enum_encode(value.mode, "ArtifactDeliveryMode"),
            "outcome": enum_encode(value.outcome, "ArtifactDeliveryOutcome"),
            "workspace_id": value.workspace_id,
            "relative_path": value.relative_path,
            "prior_physical_revision": value.prior_physical_revision,
            "result_physical_revision": value.result_physical_revision,
            "failure_code": value.failure_code,
        },
    )


def decode_delivery_receipt(value: JsonValue) -> ArtifactDeliveryReceipt:
    fields = record_fields(
        value,
        "ArtifactDeliveryReceipt",
        (
            "artifact_id",
            "destination_id",
            "filename",
            "saved_path",
            "byte_size",
            "sha256",
            "renamed_for_collision",
            "delivered_at",
            "mode",
            "outcome",
            "workspace_id",
            "relative_path",
            "prior_physical_revision",
            "result_physical_revision",
            "failure_code",
        ),
    )
    return ArtifactDeliveryReceipt(
        artifact_id=text(fields["artifact_id"], "delivery artifact_id"),
        destination_id=text(fields["destination_id"], "delivery destination_id"),
        filename=text(fields["filename"], "delivery filename"),
        saved_path=text(fields["saved_path"], "delivery saved_path"),
        byte_size=integer(fields["byte_size"], "delivery byte_size"),
        sha256=text(fields["sha256"], "delivery sha256"),
        renamed_for_collision=boolean(
            fields["renamed_for_collision"], "delivery collision flag"
        ),
        delivered_at=datetime_decode(fields["delivered_at"]),
        mode=enum_decode(fields["mode"], ArtifactDeliveryMode, "ArtifactDeliveryMode"),
        outcome=enum_decode(
            fields["outcome"], ArtifactDeliveryOutcome, "ArtifactDeliveryOutcome"
        ),
        workspace_id=optional_text(fields["workspace_id"], "delivery workspace id"),
        relative_path=optional_text(fields["relative_path"], "delivery relative path"),
        prior_physical_revision=optional_text(
            fields["prior_physical_revision"], "delivery prior revision"
        ),
        result_physical_revision=optional_text(
            fields["result_physical_revision"], "delivery result revision"
        ),
        failure_code=optional_text(fields["failure_code"], "delivery failure code"),
    )
