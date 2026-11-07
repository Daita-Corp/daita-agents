"""Encode and decode catalog entries, relationships, source state, changes and links."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, cast

from ..._json import FrozenJsonObject, thaw_json
from ...catalog.models import (
    CatalogChangeSide,
    CatalogEntry,
    CatalogFacet,
    CatalogLink,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSourceState,
    CatalogSync,
    CatalogSyncStatus,
    FacetKind,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    SearchPosting,
    Sensitivity,
    SourceCatalogSnapshot,
)
from ...catalog.search_text import posting_field_name
from .common import (
    JsonValue,
    boolean,
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
    optional_integer,
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


def encode_catalog_entry(value: CatalogEntry) -> str:
    if not isinstance(value, CatalogEntry):
        raise TypeError("catalog entry codec requires CatalogEntry")
    return dump_payload(
        record(
            "CatalogEntry",
            {
                "resource": _encode_resource(value.resource),
                "revision": _encode_revision(value.revision),
                "facets": [_encode_catalog_facet(item) for item in value.facets],
                "detailed": value.detailed,
                "source_object_id": value.source_object_id,
            },
        )
    )


def decode_catalog_entry(value: str) -> CatalogEntry:
    fields = record_fields(
        load_payload(value),
        "CatalogEntry",
        ("resource", "revision", "facets", "detailed", "source_object_id"),
    )
    return CatalogEntry(
        resource=_decode_resource(fields["resource"]),
        revision=_decode_revision(fields["revision"]),
        facets=tuple(
            _decode_catalog_facet(item)
            for item in sequence(fields["facets"], "catalog entry facets")
        ),
        detailed=boolean(fields["detailed"], "catalog entry detailed"),
        source_object_id=optional_text(
            fields["source_object_id"], "catalog entry source_object_id"
        ),
    )


def encode_catalog_relationship(value: CatalogRelationship) -> str:
    if not isinstance(value, CatalogRelationship):
        raise TypeError("catalog relationship codec requires CatalogRelationship")
    return dump_payload(_encode_relationship(value))


def decode_catalog_relationship(value: str) -> CatalogRelationship:
    return _decode_relationship(load_payload(value))


def encode_catalog_source_state(value: CatalogSourceState) -> str:
    if not isinstance(value, CatalogSourceState):
        raise TypeError("catalog source state codec requires CatalogSourceState")
    return dump_payload(
        record(
            "CatalogSourceState",
            {
                "agent_id": value.agent_id,
                "source_id": value.source_id,
                "current_sync_id": value.current_sync_id,
                "probe_tokens": dict(value.probe_tokens),
                "probe_cursor": value.probe_cursor,
                "last_probe_at": optional_datetime_encode(value.last_probe_at),
                "last_change_at": optional_datetime_encode(value.last_change_at),
                "last_full_check_at": optional_datetime_encode(
                    value.last_full_check_at
                ),
                "last_missed": value.last_missed,
                "last_error_code": value.last_error_code,
                "keep_current": value.keep_current,
            },
        )
    )


def decode_catalog_source_state(value: str) -> CatalogSourceState:
    fields = record_fields(
        load_payload(value),
        "CatalogSourceState",
        (
            "agent_id",
            "source_id",
            "current_sync_id",
            "probe_tokens",
            "probe_cursor",
            "last_probe_at",
            "last_change_at",
            "last_full_check_at",
            "last_missed",
            "last_error_code",
            "keep_current",
        ),
    )
    return CatalogSourceState(
        agent_id=text(fields["agent_id"], "source state agent_id"),
        source_id=text(fields["source_id"], "source state source_id"),
        current_sync_id=text(fields["current_sync_id"], "source state sync id"),
        probe_tokens={
            key: text(item, "source state probe token")
            for key, item in mapping(
                fields["probe_tokens"], "source state probe tokens"
            ).items()
        },
        probe_cursor=optional_text(fields["probe_cursor"], "source state cursor"),
        last_probe_at=optional_datetime_decode(fields["last_probe_at"]),
        last_change_at=optional_datetime_decode(fields["last_change_at"]),
        last_full_check_at=optional_datetime_decode(fields["last_full_check_at"]),
        last_missed=optional_integer(fields["last_missed"], "source state missed"),
        last_error_code=optional_text(
            fields["last_error_code"], "source state error code"
        ),
        keep_current=boolean(fields["keep_current"], "source state keep_current"),
    )


def encode_catalog_change_data(
    before: CatalogChangeSide | None, after: CatalogChangeSide | None
) -> str:
    for side in (before, after):
        if side is not None and not isinstance(side, CatalogChangeSide):
            raise TypeError("catalog change codec requires CatalogChangeSide")
    return dump_payload(
        record(
            "CatalogChangeData",
            {
                "before": None if before is None else _encode_change_side(before),
                "after": None if after is None else _encode_change_side(after),
            },
        )
    )


def decode_catalog_change_data(
    value: str,
) -> tuple[CatalogChangeSide | None, CatalogChangeSide | None]:
    fields = record_fields(
        load_payload(value), "CatalogChangeData", ("before", "after")
    )
    return (
        None if fields["before"] is None else _decode_change_side(fields["before"]),
        None if fields["after"] is None else _decode_change_side(fields["after"]),
    )


def encode_search_postings(value: tuple[SearchPosting, ...]) -> str:
    if any(not isinstance(item, SearchPosting) for item in value):
        raise TypeError("search posting codec requires SearchPosting records")
    # The field name is derivable from kind and value, so it is not stored.
    return dump_payload([[item.field_kind, item.value] for item in value])


@lru_cache(maxsize=262_144)  # column names repeat across every table
def _posting(kind: str, value: str) -> SearchPosting:
    return SearchPosting(kind, posting_field_name(kind, value), value)


def decode_search_postings(value: str) -> tuple[SearchPosting, ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("stored search postings are invalid")
    result: list[SearchPosting] = []
    for item in parsed:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError("stored search posting is invalid")
        result.append(_posting(item[0], item[1]))
    return tuple(result)


def _encode_change_side(value: CatalogChangeSide) -> dict[str, JsonValue]:
    return record(
        "CatalogChangeSide",
        {
            "name": value.name,
            "kind": enum_encode(value.kind, "ResourceKind"),
            "revision": value.revision,
            "columns": [[name, native_type] for name, native_type in value.columns],
        },
    )


def _decode_change_side(value: JsonValue) -> CatalogChangeSide:
    fields = record_fields(
        value, "CatalogChangeSide", ("name", "kind", "revision", "columns")
    )
    columns: list[tuple[str, str]] = []
    for item in sequence(fields["columns"], "change side columns"):
        pair = sequence(item, "change side column")
        if len(pair) != 2:
            raise ValueError("stored change side column is invalid")
        columns.append(
            (
                text(pair[0], "change side column name"),
                text(pair[1], "change side column type"),
            )
        )
    return CatalogChangeSide(
        name=text(fields["name"], "change side name"),
        kind=enum_decode(fields["kind"], ResourceKind, "ResourceKind"),
        revision=text(fields["revision"], "change side revision"),
        columns=tuple(columns),
    )


def encode_catalog_link(value: CatalogLink) -> str:
    if not isinstance(value, CatalogLink):
        raise TypeError("catalog link codec requires CatalogLink")
    return dump_payload(
        record(
            "CatalogLink",
            {
                "id": value.id,
                "agent_id": value.agent_id,
                "from_source_id": value.from_source_id,
                "from_resource_id": value.from_resource_id,
                "to_source_id": value.to_source_id,
                "to_resource_id": value.to_resource_id,
                "field_pairs": [_encode_field_pair(item) for item in value.field_pairs],
                "declared_at": datetime_encode(value.declared_at),
                "declared_revisions": list(value.declared_revisions),
                "note": value.note,
            },
        )
    )


def decode_catalog_link(value: str) -> CatalogLink:
    fields = record_fields(
        load_payload(value),
        "CatalogLink",
        (
            "id",
            "agent_id",
            "from_source_id",
            "from_resource_id",
            "to_source_id",
            "to_resource_id",
            "field_pairs",
            "declared_at",
            "declared_revisions",
            "note",
        ),
    )
    revisions = sequence(fields["declared_revisions"], "catalog link revisions")
    if len(revisions) != 2:
        raise ValueError("stored catalog link revisions are invalid")
    return CatalogLink(
        id=text(fields["id"], "catalog link id"),
        agent_id=text(fields["agent_id"], "catalog link agent_id"),
        from_source_id=text(fields["from_source_id"], "catalog link from_source_id"),
        from_resource_id=text(
            fields["from_resource_id"], "catalog link from_resource_id"
        ),
        to_source_id=text(fields["to_source_id"], "catalog link to_source_id"),
        to_resource_id=text(fields["to_resource_id"], "catalog link to_resource_id"),
        field_pairs=tuple(
            _decode_field_pair(item)
            for item in sequence(fields["field_pairs"], "catalog link field_pairs")
        ),
        declared_at=datetime_decode(fields["declared_at"]),
        declared_revisions=(
            text(revisions[0], "catalog link revision"),
            text(revisions[1], "catalog link revision"),
        ),
        note=optional_text(fields["note"], "catalog link note"),
    )


def _frozen_plain(value: object) -> JsonValue:
    # plain_encode on a frozen object re-scans its items per key (quadratic).
    if isinstance(value, FrozenJsonObject):
        return cast(JsonValue, thaw_json(value))
    return plain_encode(value)


# A tabular column payload repeats the same twelve keys, most of them at their
# default. Columns with exactly that shape are stored as [name, native_type,
# ordinal, nullable, {non-default extras}?]; anything else is stored as it is.
_COLUMN_CORE = ("name", "native_type", "ordinal", "nullable")
_COLUMN_DEFAULTS: dict[str, JsonValue] = {
    "collation": None,
    "default_expression": None,
    "generated": False,
    "identity": False,
    "native_type_name": None,
    "native_type_namespace": None,
    "primary_key_ordinal": None,
    "updatable": False,
}
_PACKED = "__packed_columns__"


def _pack_payload(kind: FacetKind, payload: object) -> JsonValue:
    if kind is FacetKind.TABULAR and isinstance(payload, FrozenJsonObject):
        items = dict(payload._items)
        packed = _pack_columns(items.get("columns"))
        if packed is not None:
            return cast(
                JsonValue,
                {
                    key: {_PACKED: packed} if key == "columns" else thaw_json(item)
                    for key, item in items.items()
                },
            )
    return _frozen_plain(payload)


def _pack_columns(columns: object) -> list[Any] | None:
    if not isinstance(columns, tuple):
        return None
    expected = {*_COLUMN_CORE, *_COLUMN_DEFAULTS}
    packed: list[Any] = []
    for column in columns:
        if not isinstance(column, FrozenJsonObject):
            return None
        fields = dict(column._items)
        if not expected <= set(fields) <= {*expected, "comment"}:
            return None
        extras = {
            key: thaw_json(item)
            for key, item in fields.items()
            if key not in _COLUMN_CORE and item is not _COLUMN_DEFAULTS.get(key, ...)
        }
        row = [thaw_json(fields[key]) for key in _COLUMN_CORE]
        packed.append([*row, extras] if extras else row)
    return packed


def _unpack_payload(kind: FacetKind, payload: JsonValue) -> JsonValue:
    if kind is not FacetKind.TABULAR or not isinstance(payload, dict):
        return payload
    columns = payload.get("columns")
    if not isinstance(columns, dict) or set(columns) != {_PACKED}:
        return payload
    unpacked: list[JsonValue] = []
    for row in sequence(columns[_PACKED], "packed columns"):
        parts = sequence(row, "packed column")
        if len(parts) not in (4, 5) or (
            len(parts) == 5 and not isinstance(parts[4], dict)
        ):
            raise ValueError("stored packed column is invalid")
        extras = cast(dict[str, JsonValue], parts[4] if len(parts) == 5 else {})
        unpacked.append(
            {
                **_COLUMN_DEFAULTS,
                **dict(zip(_COLUMN_CORE, parts[:4], strict=True)),
                **extras,
            }
        )
    return {**payload, "columns": unpacked}


def _encode_catalog_facet(value: CatalogFacet) -> dict[str, JsonValue]:
    return record(
        "CatalogFacet",
        {
            "resource_id": value.resource_id,
            "sync_id": value.sync_id,
            "kind": enum_encode(value.kind, "FacetKind"),
            "revision": value.revision,
            "payload": _pack_payload(value.kind, value.payload),
            "observed_at": datetime_encode(value.observed_at),
        },
    )


def _decode_catalog_facet(value: JsonValue) -> CatalogFacet:
    fields = record_fields(
        value,
        "CatalogFacet",
        ("resource_id", "sync_id", "kind", "revision", "payload", "observed_at"),
    )
    kind = enum_decode(fields["kind"], FacetKind, "FacetKind")
    payload = plain_decode(
        _unpack_payload(kind, mapping(fields["payload"], "catalog facet payload"))
    )
    if not isinstance(payload, dict):
        raise ValueError("stored catalog facet payload is invalid")
    return CatalogFacet(
        resource_id=text(fields["resource_id"], "catalog facet resource_id"),
        sync_id=text(fields["sync_id"], "catalog facet sync_id"),
        kind=kind,
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
            "attributes": _frozen_plain(value.attributes),
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
            "truncated": value.truncated,
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
            "truncated",
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
        truncated=boolean(fields["truncated"], "catalog sync truncated"),
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
