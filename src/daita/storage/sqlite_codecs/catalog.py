"""Encode and decode catalog synchronizations, snapshots, facets, and revisions."""

from __future__ import annotations

from ...catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    FacetKind,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
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
    mapping,
    number,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)


def encode_catalog_sync(value: CatalogSync) -> str:
    if not isinstance(value, CatalogSync):
        raise TypeError("catalog sync codec requires CatalogSync")
    return dump_payload(_encode_catalog_sync(value))


def decode_catalog_sync(value: str) -> CatalogSync:
    return _decode_catalog_sync(load_payload(value))


def encode_catalog_snapshot(value: SourceCatalogSnapshot) -> str:
    if not isinstance(value, SourceCatalogSnapshot):
        raise TypeError("catalog snapshot codec requires SourceCatalogSnapshot")
    return dump_payload(_encode_catalog_snapshot(value))


def decode_catalog_snapshot(value: str) -> SourceCatalogSnapshot:
    return _decode_catalog_snapshot(load_payload(value))


def _encode_catalog_facet(value: CatalogFacet) -> dict[str, JsonValue]:
    return record(
        "CatalogFacet",
        {
            "resource_id": value.resource_id,
            "sync_id": value.sync_id,
            "kind": enum_encode(value.kind, "FacetKind"),
            "revision": value.revision,
            "payload": plain_encode(value.payload),
            "observed_at": datetime_encode(value.observed_at),
        },
    )


def _decode_catalog_facet(value: JsonValue) -> CatalogFacet:
    fields = record_fields(
        value,
        "CatalogFacet",
        ("resource_id", "sync_id", "kind", "revision", "payload", "observed_at"),
    )
    payload = plain_decode(mapping(fields["payload"], "catalog facet payload"))
    if not isinstance(payload, dict):
        raise ValueError("stored catalog facet payload is invalid")
    return CatalogFacet(
        resource_id=text(fields["resource_id"], "catalog facet resource_id"),
        sync_id=text(fields["sync_id"], "catalog facet sync_id"),
        kind=enum_decode(fields["kind"], FacetKind, "FacetKind"),
        revision=text(fields["revision"], "catalog facet revision"),
        payload=payload,
        observed_at=datetime_decode(fields["observed_at"]),
    )


def _encode_field_pair(value: RelationshipFieldPair) -> dict[str, JsonValue]:
    return record(
        "RelationshipFieldPair",
        {
            "source_field": value.source_field,
            "target_field": value.target_field,
            "ordinal": value.ordinal,
        },
    )


def _decode_field_pair(value: JsonValue) -> RelationshipFieldPair:
    fields = record_fields(
        value,
        "RelationshipFieldPair",
        ("source_field", "target_field", "ordinal"),
    )
    return RelationshipFieldPair(
        source_field=text(fields["source_field"], "relationship source_field"),
        target_field=text(fields["target_field"], "relationship target_field"),
        ordinal=integer(fields["ordinal"], "relationship ordinal"),
    )


def _encode_relationship(value: CatalogRelationship) -> dict[str, JsonValue]:
    return record(
        "CatalogRelationship",
        {
            "id": value.id,
            "revision": value.revision,
            "source_id": value.source_id,
            "from_resource_id": value.from_resource_id,
            "to_resource_id": value.to_resource_id,
            "kind": enum_encode(value.kind, "RelationshipKind"),
            "provenance": enum_encode(value.provenance, "RelationshipProvenance"),
            "confidence": value.confidence,
            "sync_id": value.sync_id,
            "observed_at": datetime_encode(value.observed_at),
            "field_pairs": [_encode_field_pair(item) for item in value.field_pairs],
            "attributes": plain_encode(value.attributes),
        },
    )


def _decode_relationship(value: JsonValue) -> CatalogRelationship:
    fields = record_fields(
        value,
        "CatalogRelationship",
        (
            "id",
            "revision",
            "source_id",
            "from_resource_id",
            "to_resource_id",
            "kind",
            "provenance",
            "confidence",
            "sync_id",
            "observed_at",
            "field_pairs",
            "attributes",
        ),
    )
    attributes = plain_decode(mapping(fields["attributes"], "relationship attributes"))
    if not isinstance(attributes, dict):
        raise ValueError("stored relationship attributes are invalid")
    return CatalogRelationship(
        id=text(fields["id"], "relationship id"),
        revision=text(fields["revision"], "relationship revision"),
        source_id=text(fields["source_id"], "relationship source_id"),
        from_resource_id=text(
            fields["from_resource_id"], "relationship from_resource_id"
        ),
        to_resource_id=text(fields["to_resource_id"], "relationship to_resource_id"),
        kind=enum_decode(fields["kind"], RelationshipKind, "RelationshipKind"),
        provenance=enum_decode(
            fields["provenance"],
            RelationshipProvenance,
            "RelationshipProvenance",
        ),
        confidence=number(fields["confidence"], "relationship confidence"),
        sync_id=text(fields["sync_id"], "relationship sync_id"),
        observed_at=datetime_decode(fields["observed_at"]),
        field_pairs=tuple(
            _decode_field_pair(item)
            for item in sequence(fields["field_pairs"], "relationship field_pairs")
        ),
        attributes=attributes,
    )


def _encode_revision(value: CatalogResourceRevision) -> dict[str, JsonValue]:
    return record(
        "CatalogResourceRevision",
        {
            "resource_id": value.resource_id,
            "revision": value.revision,
            "sync_id": value.sync_id,
            "observed_at": datetime_encode(value.observed_at),
            "facet_revisions": list(value.facet_revisions),
            "relationship_revisions": list(value.relationship_revisions),
            "source_revision": value.source_revision,
        },
    )


def _decode_revision(value: JsonValue) -> CatalogResourceRevision:
    fields = record_fields(
        value,
        "CatalogResourceRevision",
        (
            "resource_id",
            "revision",
            "sync_id",
            "observed_at",
            "facet_revisions",
            "relationship_revisions",
            "source_revision",
        ),
    )
    return CatalogResourceRevision(
        resource_id=text(fields["resource_id"], "catalog revision resource_id"),
        revision=text(fields["revision"], "catalog revision"),
        sync_id=text(fields["sync_id"], "catalog revision sync_id"),
        observed_at=datetime_decode(fields["observed_at"]),
        facet_revisions=tuple(
            text(item, "facet revision")
            for item in sequence(fields["facet_revisions"], "facet revisions")
        ),
        relationship_revisions=tuple(
            text(item, "relationship revision")
            for item in sequence(
                fields["relationship_revisions"], "relationship revisions"
            )
        ),
        source_revision=optional_text(
            fields["source_revision"], "catalog source revision"
        ),
    )


def _encode_resource(value: CatalogResource) -> dict[str, JsonValue]:
    return record(
        "CatalogResource",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "source_id": value.source_id,
            "native_identity": value.native_identity,
            "external_uri": value.external_uri,
            "kind": enum_encode(value.kind, "ResourceKind"),
            "name": value.name,
            "sensitivity": enum_encode(value.sensitivity, "Sensitivity"),
            "current_revision": value.current_revision,
            "current_sync_id": value.current_sync_id,
            "first_observed_at": datetime_encode(value.first_observed_at),
            "last_observed_at": datetime_encode(value.last_observed_at),
        },
    )


def _decode_resource(value: JsonValue) -> CatalogResource:
    fields = record_fields(
        value,
        "CatalogResource",
        (
            "id",
            "agent_id",
            "source_id",
            "native_identity",
            "external_uri",
            "kind",
            "name",
            "sensitivity",
            "current_revision",
            "current_sync_id",
            "first_observed_at",
            "last_observed_at",
        ),
    )
    return CatalogResource(
        id=text(fields["id"], "catalog resource id"),
        agent_id=text(fields["agent_id"], "catalog resource agent_id"),
        source_id=text(fields["source_id"], "catalog resource source_id"),
        native_identity=text(
            fields["native_identity"], "catalog resource native_identity"
        ),
        external_uri=text(fields["external_uri"], "catalog resource external_uri"),
        kind=enum_decode(fields["kind"], ResourceKind, "ResourceKind"),
        name=text(fields["name"], "catalog resource name"),
        sensitivity=enum_decode(fields["sensitivity"], Sensitivity, "Sensitivity"),
        current_revision=text(
            fields["current_revision"], "catalog resource current_revision"
        ),
        current_sync_id=text(
            fields["current_sync_id"], "catalog resource current_sync_id"
        ),
        first_observed_at=datetime_decode(fields["first_observed_at"]),
        last_observed_at=datetime_decode(fields["last_observed_at"]),
    )


def _encode_catalog_sync(value: CatalogSync) -> dict[str, JsonValue]:
    return record(
        "CatalogSync",
        {
            "id": value.id,
            "agent_id": value.agent_id,
            "source_id": value.source_id,
            "adapter_id": value.adapter_id,
            "status": enum_encode(value.status, "CatalogSyncStatus"),
            "started_at": datetime_encode(value.started_at),
            "completed_at": optional_datetime_encode(value.completed_at),
            "source_revision": value.source_revision,
            "resource_count": value.resource_count,
            "relationship_count": value.relationship_count,
            "error_code": value.error_code,
        },
    )


def _decode_catalog_sync(value: JsonValue) -> CatalogSync:
    fields = record_fields(
        value,
        "CatalogSync",
        (
            "id",
            "agent_id",
            "source_id",
            "adapter_id",
            "status",
            "started_at",
            "completed_at",
            "source_revision",
            "resource_count",
            "relationship_count",
            "error_code",
        ),
    )
    return CatalogSync(
        id=text(fields["id"], "catalog sync id"),
        agent_id=text(fields["agent_id"], "catalog sync agent_id"),
        source_id=text(fields["source_id"], "catalog sync source_id"),
        adapter_id=text(fields["adapter_id"], "catalog sync adapter_id"),
        status=enum_decode(fields["status"], CatalogSyncStatus, "CatalogSyncStatus"),
        started_at=datetime_decode(fields["started_at"]),
        completed_at=optional_datetime_decode(fields["completed_at"]),
        source_revision=optional_text(
            fields["source_revision"], "catalog sync source_revision"
        ),
        resource_count=integer(fields["resource_count"], "catalog resource_count"),
        relationship_count=integer(
            fields["relationship_count"], "catalog relationship_count"
        ),
        error_code=optional_text(fields["error_code"], "catalog sync error_code"),
    )


def _encode_catalog_snapshot(value: SourceCatalogSnapshot) -> dict[str, JsonValue]:
    return record(
        "SourceCatalogSnapshot",
        {
            "sync": _encode_catalog_sync(value.sync),
            "resources": [_encode_resource(item) for item in value.resources],
            "revisions": [_encode_revision(item) for item in value.revisions],
            "facets": [_encode_catalog_facet(item) for item in value.facets],
            "relationships": [
                _encode_relationship(item) for item in value.relationships
            ],
        },
    )


def _decode_catalog_snapshot(value: JsonValue) -> SourceCatalogSnapshot:
    fields = record_fields(
        value,
        "SourceCatalogSnapshot",
        ("sync", "resources", "revisions", "facets", "relationships"),
    )
    return SourceCatalogSnapshot(
        sync=_decode_catalog_sync(fields["sync"]),
        resources=tuple(
            _decode_resource(item)
            for item in sequence(fields["resources"], "catalog resources")
        ),
        revisions=tuple(
            _decode_revision(item)
            for item in sequence(fields["revisions"], "catalog revisions")
        ),
        facets=tuple(
            _decode_catalog_facet(item)
            for item in sequence(fields["facets"], "catalog facets")
        ),
        relationships=tuple(
            _decode_relationship(item)
            for item in sequence(fields["relationships"], "catalog relationships")
        ),
    )
