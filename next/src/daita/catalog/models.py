"""Provider- and adapter-neutral catalog records.

The catalog owns resource identity and structural truth.  These records keep
mutable discovery timing separate from stable identities and revisions so a
refresh cannot silently turn an observation into a different resource.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
import math
import re
from typing import TypeVar

from .._json import FrozenJsonObject, canonical_json

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RESOURCE_ID = re.compile(r"catalog-resource:sha256:[0-9a-f]{64}\Z")
_RELATIONSHIP_ID = re.compile(r"catalog-relationship:sha256:[0-9a-f]{64}\Z")

_RecordT = TypeVar("_RecordT")


def _required_text(value: str, field_name: str, *, maximum: int = 512) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")


def _optional_text(
    value: str | None,
    field_name: str,
    *,
    maximum: int = 512,
) -> None:
    if value is not None:
        _required_text(value, field_name, maximum=maximum)


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _bounded_float(
    value: float,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(
            f"{field_name} must be a finite number from {minimum} through {maximum}"
        )


def _hash(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical lowercase sha256 hash")


def _record_tuple(
    values: Iterable[_RecordT],
    expected_type: type[_RecordT],
    field_name: str,
) -> tuple[_RecordT, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of records")
    result = tuple(values)
    if any(not isinstance(value, expected_type) for value in result):
        raise TypeError(f"{field_name} must contain {expected_type.__name__} records")
    return result


def _text_tuple(
    values: Iterable[str],
    field_name: str,
    *,
    maximum_items: int,
    allow_empty: bool = True,
    unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} cannot be empty")
    if len(result) > maximum_items:
        raise ValueError(f"{field_name} exceeds {maximum_items} items")
    for value in result:
        _required_text(value, field_name, maximum=256)
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return result


class ResourceKind(str, Enum):
    DATABASE = "database"
    SCHEMA = "schema"
    TABLE = "table"
    VIEW = "view"
    FILE = "file"
    FOLDER = "folder"
    DOCUMENT = "document"
    CLOUD_RESOURCE = "cloud_resource"


class Sensitivity(str, Enum):
    UNKNOWN = "unknown"
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class FacetKind(str, Enum):
    FILE = "file"
    TABULAR = "tabular"


class RelationshipKind(str, Enum):
    CONTAINS = "contains"
    REFERENCES = "references"
    DERIVED_FROM = "derived_from"
    PRODUCES = "produces"
    WRITES_TO = "writes_to"
    READS_FROM = "reads_from"
    OBSERVES = "observes"


class RelationshipProvenance(str, Enum):
    CONNECTOR = "connector"
    DECLARED = "declared"
    INFERRED = "inferred"


class CatalogSyncStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class RelationshipDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


def catalog_resource_id(
    source_id: str,
    kind: ResourceKind,
    native_identity: str,
) -> str:
    """Return the stable source-scoped identity for one native resource."""

    _required_text(source_id, "resource source_id")
    if not isinstance(kind, ResourceKind):
        raise TypeError("resource kind must be a ResourceKind")
    _required_text(native_identity, "resource native_identity")
    digest = _hash(
        {
            "kind": kind.value,
            "native_identity": native_identity,
            "source_id": source_id,
        }
    ).removeprefix("sha256:")
    return f"catalog-resource:sha256:{digest}"


@dataclass(frozen=True, slots=True)
class TabularColumn:
    name: str
    native_type: str
    ordinal: int
    nullable: bool
    primary_key_ordinal: int | None = None
    default_expression: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.name, "column name", maximum=256)
        _required_text(self.native_type, "column native_type", maximum=256)
        _non_negative_int(self.ordinal, "column ordinal")
        if not isinstance(self.nullable, bool):
            raise TypeError("column nullable must be a boolean")
        if self.primary_key_ordinal is not None:
            _positive_int(self.primary_key_ordinal, "column primary_key_ordinal")
        _optional_text(
            self.default_expression,
            "column default_expression",
            maximum=2_048,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "default_expression": self.default_expression,
            "name": self.name,
            "native_type": self.native_type,
            "nullable": self.nullable,
            "ordinal": self.ordinal,
            "primary_key_ordinal": self.primary_key_ordinal,
        }


@dataclass(frozen=True, slots=True)
class TabularIndex:
    name: str
    kind: str
    columns: tuple[str, ...]
    unique: bool
    predicate: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.name, "index name", maximum=256)
        _required_text(self.kind, "index kind", maximum=128)
        columns = _text_tuple(
            self.columns,
            "index columns",
            maximum_items=64,
            allow_empty=False,
        )
        if not isinstance(self.unique, bool):
            raise TypeError("index unique must be a boolean")
        _optional_text(self.predicate, "index predicate", maximum=4_096)
        object.__setattr__(self, "columns", columns)

    def to_payload(self) -> dict[str, object]:
        return {
            "columns": self.columns,
            "kind": self.kind,
            "name": self.name,
            "predicate": self.predicate,
            "unique": self.unique,
        }


@dataclass(frozen=True, slots=True)
class TabularFacet:
    """Typed structural table facts plus a separate nonstructural estimate."""

    columns: tuple[TabularColumn, ...]
    indexes: tuple[TabularIndex, ...] = ()
    row_count_estimate: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        columns = _record_tuple(self.columns, TabularColumn, "tabular columns")
        indexes = _record_tuple(self.indexes, TabularIndex, "tabular indexes")
        _positive_int(self.schema_version, "tabular schema_version")
        if self.row_count_estimate is not None:
            _non_negative_int(self.row_count_estimate, "tabular row_count_estimate")

        column_names = [column.name for column in columns]
        column_ordinals = [column.ordinal for column in columns]
        if len(column_names) != len(set(column_names)):
            raise ValueError("tabular columns cannot contain duplicate names")
        if len(column_ordinals) != len(set(column_ordinals)):
            raise ValueError("tabular columns cannot contain duplicate ordinals")
        primary_ordinals = [
            column.primary_key_ordinal
            for column in columns
            if column.primary_key_ordinal is not None
        ]
        if len(primary_ordinals) != len(set(primary_ordinals)):
            raise ValueError("primary-key ordinals cannot contain duplicates")
        if primary_ordinals and sorted(primary_ordinals) != list(
            range(1, len(primary_ordinals) + 1)
        ):
            raise ValueError("primary-key ordinals must be contiguous from one")

        index_names = [index.name for index in indexes]
        if len(index_names) != len(set(index_names)):
            raise ValueError("tabular indexes cannot contain duplicate names")
        known_columns = set(column_names)
        for index in indexes:
            missing = sorted(set(index.columns) - known_columns)
            if missing:
                raise ValueError(
                    f"index {index.name} references unknown columns: "
                    f"{', '.join(missing)}"
                )

        object.__setattr__(
            self,
            "columns",
            tuple(sorted(columns, key=lambda column: (column.ordinal, column.name))),
        )
        object.__setattr__(
            self,
            "indexes",
            tuple(sorted(indexes, key=lambda index: index.name)),
        )

    def structural_payload(self) -> dict[str, object]:
        """Return facts that change structural resource revision."""

        return {
            "columns": tuple(column.to_payload() for column in self.columns),
            "indexes": tuple(index.to_payload() for index in self.indexes),
        }

    def to_payload(self) -> dict[str, object]:
        """Return the typed facet projection, including nonstructural estimates."""

        return {
            **self.structural_payload(),
            "row_count_estimate": self.row_count_estimate,
        }


@dataclass(frozen=True, slots=True)
class FileFacet:
    """Typed local-file facts with freshness kept outside structural identity."""

    format: str
    media_type: str
    encoding: str
    size_bytes: int
    content_sha256: str
    modified_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.format not in {"csv", "json"}:
            raise ValueError("file format must be 'csv' or 'json'")
        _required_text(self.media_type, "file media_type", maximum=256)
        if self.encoding not in {"utf-8", "utf-8-sig"}:
            raise ValueError("file encoding must be 'utf-8' or 'utf-8-sig'")
        _non_negative_int(self.size_bytes, "file size_bytes")
        _require_hash(self.content_sha256, "file content_sha256")
        _aware(self.modified_at, "file modified_at")
        _positive_int(self.schema_version, "file schema_version")

    def structural_payload(self) -> dict[str, object]:
        """Return facts whose changes create a new structural file revision."""

        return {
            "content_sha256": self.content_sha256,
            "encoding": self.encoding,
            "format": self.format,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
        }

    def to_payload(self) -> dict[str, object]:
        """Return the typed projection, including nonstructural freshness."""

        return {
            **self.structural_payload(),
            "modified_at": self.modified_at.isoformat(),
        }


def catalog_facet_revision(
    kind: FacetKind,
    schema_version: int,
    structural_payload: Mapping[str, object],
) -> str:
    if not isinstance(kind, FacetKind):
        raise TypeError("facet kind must be a FacetKind")
    _positive_int(schema_version, "facet schema_version")
    payload = FrozenJsonObject.from_mapping(structural_payload)
    return _hash(
        {
            "kind": kind.value,
            "payload": payload,
            "schema_version": schema_version,
        }
    )


@dataclass(frozen=True, slots=True)
class CatalogFacet:
    resource_id: str
    sync_id: str
    kind: FacetKind
    schema_version: int
    revision: str
    payload: Mapping[str, object]
    observed_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.resource_id, "facet resource_id")
        _required_text(self.sync_id, "facet sync_id")
        if not isinstance(self.kind, FacetKind):
            raise TypeError("facet kind must be a FacetKind")
        _positive_int(self.schema_version, "facet schema_version")
        _require_hash(self.revision, "facet revision")
        _aware(self.observed_at, "facet observed_at")
        payload = FrozenJsonObject.from_mapping(self.payload)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_tabular(
        cls,
        *,
        resource_id: str,
        sync_id: str,
        observed_at: datetime,
        facet: TabularFacet,
    ) -> CatalogFacet:
        if not isinstance(facet, TabularFacet):
            raise TypeError("facet must be a TabularFacet")
        return cls(
            resource_id=resource_id,
            sync_id=sync_id,
            kind=FacetKind.TABULAR,
            schema_version=facet.schema_version,
            revision=catalog_facet_revision(
                FacetKind.TABULAR,
                facet.schema_version,
                facet.structural_payload(),
            ),
            payload=facet.to_payload(),
            observed_at=observed_at,
        )

    @classmethod
    def from_file(
        cls,
        *,
        resource_id: str,
        sync_id: str,
        observed_at: datetime,
        facet: FileFacet,
    ) -> CatalogFacet:
        if not isinstance(facet, FileFacet):
            raise TypeError("facet must be a FileFacet")
        return cls(
            resource_id=resource_id,
            sync_id=sync_id,
            kind=FacetKind.FILE,
            schema_version=facet.schema_version,
            revision=catalog_facet_revision(
                FacetKind.FILE,
                facet.schema_version,
                facet.structural_payload(),
            ),
            payload=facet.to_payload(),
            observed_at=observed_at,
        )


@dataclass(frozen=True, slots=True)
class RelationshipFieldPair:
    source_field: str
    target_field: str
    ordinal: int

    def __post_init__(self) -> None:
        _required_text(self.source_field, "relationship source_field", maximum=256)
        _required_text(self.target_field, "relationship target_field", maximum=256)
        _non_negative_int(self.ordinal, "relationship field-pair ordinal")

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "source_field": self.source_field,
            "target_field": self.target_field,
        }


def catalog_relationship_id(
    *,
    source_id: str,
    from_resource_id: str,
    to_resource_id: str,
    kind: RelationshipKind,
    provenance: RelationshipProvenance,
    field_pairs: Iterable[RelationshipFieldPair] = (),
) -> str:
    _required_text(source_id, "relationship source_id")
    _required_text(from_resource_id, "relationship from_resource_id")
    _required_text(to_resource_id, "relationship to_resource_id")
    if not isinstance(kind, RelationshipKind):
        raise TypeError("relationship kind must be a RelationshipKind")
    if not isinstance(provenance, RelationshipProvenance):
        raise TypeError("relationship provenance must be RelationshipProvenance")
    pairs = _record_tuple(
        field_pairs,
        RelationshipFieldPair,
        "relationship field_pairs",
    )
    ordered = tuple(sorted(pairs, key=lambda pair: pair.ordinal))
    digest = _hash(
        {
            "field_pairs": tuple(pair.to_payload() for pair in ordered),
            "from_resource_id": from_resource_id,
            "kind": kind.value,
            "provenance": provenance.value,
            "source_id": source_id,
            "to_resource_id": to_resource_id,
        }
    ).removeprefix("sha256:")
    return f"catalog-relationship:sha256:{digest}"


def _relationship_revision(
    *,
    relationship_id: str,
    confidence: float,
    attributes: Mapping[str, object],
) -> str:
    return _hash(
        {
            "attributes": FrozenJsonObject.from_mapping(attributes),
            "confidence": float(confidence),
            "relationship_id": relationship_id,
        }
    )


@dataclass(frozen=True, slots=True)
class CatalogRelationship:
    id: str
    revision: str
    source_id: str
    from_resource_id: str
    to_resource_id: str
    kind: RelationshipKind
    provenance: RelationshipProvenance
    confidence: float
    sync_id: str
    observed_at: datetime
    field_pairs: tuple[RelationshipFieldPair, ...] = ()
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _RELATIONSHIP_ID.fullmatch(self.id) is None:
            raise ValueError("relationship id must be a canonical catalog sha256 id")
        _require_hash(self.revision, "relationship revision")
        _required_text(self.source_id, "relationship source_id")
        _required_text(self.from_resource_id, "relationship from_resource_id")
        _required_text(self.to_resource_id, "relationship to_resource_id")
        if not isinstance(self.kind, RelationshipKind):
            raise TypeError("relationship kind must be a RelationshipKind")
        if not isinstance(self.provenance, RelationshipProvenance):
            raise TypeError("relationship provenance must be RelationshipProvenance")
        _bounded_float(
            self.confidence,
            "relationship confidence",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            self.provenance is RelationshipProvenance.CONNECTOR
            and float(self.confidence) != 1.0
        ):
            raise ValueError("connector relationships require confidence 1.0")
        _required_text(self.sync_id, "relationship sync_id")
        _aware(self.observed_at, "relationship observed_at")

        pairs = _record_tuple(
            self.field_pairs,
            RelationshipFieldPair,
            "relationship field_pairs",
        )
        pairs = tuple(sorted(pairs, key=lambda pair: pair.ordinal))
        if len(pairs) > 64:
            raise ValueError("relationship field_pairs exceed 64 items")
        if len({pair.ordinal for pair in pairs}) != len(pairs):
            raise ValueError("relationship field-pair ordinals cannot repeat")
        if len({(pair.source_field, pair.target_field) for pair in pairs}) != len(
            pairs
        ):
            raise ValueError("relationship field pairs cannot repeat")
        if pairs and [pair.ordinal for pair in pairs] != list(range(len(pairs))):
            raise ValueError("relationship field-pair ordinals must be contiguous")
        if self.kind is RelationshipKind.REFERENCES and not pairs:
            raise ValueError("reference relationships require field pairs")
        if self.kind is not RelationshipKind.REFERENCES and pairs:
            raise ValueError("only reference relationships may contain field pairs")

        attributes = FrozenJsonObject.from_mapping(self.attributes)
        expected_id = catalog_relationship_id(
            source_id=self.source_id,
            from_resource_id=self.from_resource_id,
            to_resource_id=self.to_resource_id,
            kind=self.kind,
            provenance=self.provenance,
            field_pairs=pairs,
        )
        if self.id != expected_id:
            raise ValueError("relationship id does not match its stable identity")
        expected_revision = _relationship_revision(
            relationship_id=self.id,
            confidence=float(self.confidence),
            attributes=attributes,
        )
        if self.revision != expected_revision:
            raise ValueError("relationship revision does not match its structure")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "field_pairs", pairs)
        object.__setattr__(self, "attributes", attributes)

    @classmethod
    def build(
        cls,
        *,
        source_id: str,
        from_resource_id: str,
        to_resource_id: str,
        kind: RelationshipKind,
        provenance: RelationshipProvenance,
        confidence: float,
        sync_id: str,
        observed_at: datetime,
        field_pairs: Iterable[RelationshipFieldPair] = (),
        attributes: Mapping[str, object] = FrozenJsonObject(()),
    ) -> CatalogRelationship:
        pairs = tuple(field_pairs)
        relationship_id = catalog_relationship_id(
            source_id=source_id,
            from_resource_id=from_resource_id,
            to_resource_id=to_resource_id,
            kind=kind,
            provenance=provenance,
            field_pairs=pairs,
        )
        revision = _relationship_revision(
            relationship_id=relationship_id,
            confidence=float(confidence),
            attributes=attributes,
        )
        return cls(
            id=relationship_id,
            revision=revision,
            source_id=source_id,
            from_resource_id=from_resource_id,
            to_resource_id=to_resource_id,
            kind=kind,
            provenance=provenance,
            confidence=confidence,
            sync_id=sync_id,
            observed_at=observed_at,
            field_pairs=pairs,
            attributes=attributes,
        )


def catalog_resource_revision(
    *,
    resource_id: str,
    facet_revisions: Iterable[str],
    relationship_revisions: Iterable[str],
) -> str:
    _required_text(resource_id, "revision resource_id")
    facets = tuple(sorted(facet_revisions))
    relationships = tuple(sorted(relationship_revisions))
    for value in (*facets, *relationships):
        _require_hash(value, "component revision")
    if len(facets) != len(set(facets)):
        raise ValueError("facet revisions cannot contain duplicates")
    if len(relationships) != len(set(relationships)):
        raise ValueError("relationship revisions cannot contain duplicates")
    return _hash(
        {
            "facet_revisions": facets,
            "relationship_revisions": relationships,
            "resource_id": resource_id,
        }
    )


@dataclass(frozen=True, slots=True)
class CatalogResourceRevision:
    resource_id: str
    revision: str
    sync_id: str
    observed_at: datetime
    facet_revisions: tuple[str, ...] = ()
    relationship_revisions: tuple[str, ...] = ()
    source_revision: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.resource_id, "revision resource_id")
        _require_hash(self.revision, "resource revision")
        _required_text(self.sync_id, "revision sync_id")
        _aware(self.observed_at, "revision observed_at")
        facets = tuple(sorted(self.facet_revisions))
        relationships = tuple(sorted(self.relationship_revisions))
        _optional_text(self.source_revision, "source revision", maximum=1_024)
        expected = catalog_resource_revision(
            resource_id=self.resource_id,
            facet_revisions=facets,
            relationship_revisions=relationships,
        )
        if self.revision != expected:
            raise ValueError("resource revision does not match component revisions")
        object.__setattr__(self, "facet_revisions", facets)
        object.__setattr__(self, "relationship_revisions", relationships)

    @classmethod
    def build(
        cls,
        *,
        resource_id: str,
        sync_id: str,
        observed_at: datetime,
        facet_revisions: Iterable[str] = (),
        relationship_revisions: Iterable[str] = (),
        source_revision: str | None = None,
    ) -> CatalogResourceRevision:
        facets = tuple(facet_revisions)
        relationships = tuple(relationship_revisions)
        return cls(
            resource_id=resource_id,
            revision=catalog_resource_revision(
                resource_id=resource_id,
                facet_revisions=facets,
                relationship_revisions=relationships,
            ),
            sync_id=sync_id,
            observed_at=observed_at,
            facet_revisions=facets,
            relationship_revisions=relationships,
            source_revision=source_revision,
        )


@dataclass(frozen=True, slots=True)
class CatalogResource:
    id: str
    agent_id: str
    source_id: str
    native_identity: str
    external_uri: str
    kind: ResourceKind
    name: str
    sensitivity: Sensitivity
    current_revision: str
    current_sync_id: str
    first_observed_at: datetime
    last_observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _RESOURCE_ID.fullmatch(self.id) is None:
            raise ValueError("resource id must be a canonical catalog sha256 id")
        _required_text(self.agent_id, "resource agent_id")
        _required_text(self.source_id, "resource source_id")
        _required_text(self.native_identity, "resource native_identity")
        _required_text(self.external_uri, "resource external_uri", maximum=2_048)
        if not isinstance(self.kind, ResourceKind):
            raise TypeError("resource kind must be a ResourceKind")
        _required_text(self.name, "resource name")
        if not isinstance(self.sensitivity, Sensitivity):
            raise TypeError("resource sensitivity must be a Sensitivity")
        _require_hash(self.current_revision, "resource current_revision")
        _required_text(self.current_sync_id, "resource current_sync_id")
        _aware(self.first_observed_at, "resource first_observed_at")
        _aware(self.last_observed_at, "resource last_observed_at")
        if self.last_observed_at < self.first_observed_at:
            raise ValueError(
                "resource last_observed_at cannot precede first observation"
            )
        expected = catalog_resource_id(
            self.source_id,
            self.kind,
            self.native_identity,
        )
        if self.id != expected:
            raise ValueError("resource id does not match its stable identity")

    @classmethod
    def build(
        cls,
        *,
        agent_id: str,
        source_id: str,
        native_identity: str,
        external_uri: str,
        kind: ResourceKind,
        name: str,
        sensitivity: Sensitivity,
        revision: CatalogResourceRevision,
        first_observed_at: datetime,
        last_observed_at: datetime,
    ) -> CatalogResource:
        if not isinstance(revision, CatalogResourceRevision):
            raise TypeError("revision must be a CatalogResourceRevision")
        resource_id = catalog_resource_id(source_id, kind, native_identity)
        if revision.resource_id != resource_id:
            raise ValueError("resource revision belongs to another resource")
        return cls(
            id=resource_id,
            agent_id=agent_id,
            source_id=source_id,
            native_identity=native_identity,
            external_uri=external_uri,
            kind=kind,
            name=name,
            sensitivity=sensitivity,
            current_revision=revision.revision,
            current_sync_id=revision.sync_id,
            first_observed_at=first_observed_at,
            last_observed_at=last_observed_at,
        )


@dataclass(frozen=True, slots=True)
class CatalogSync:
    id: str
    agent_id: str
    source_id: str
    adapter_id: str
    status: CatalogSyncStatus
    started_at: datetime
    completed_at: datetime | None = None
    source_revision: str | None = None
    resource_count: int = 0
    relationship_count: int = 0
    error_code: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "catalog sync id")
        _required_text(self.agent_id, "catalog sync agent_id")
        _required_text(self.source_id, "catalog sync source_id")
        _required_text(self.adapter_id, "catalog sync adapter_id")
        if not isinstance(self.status, CatalogSyncStatus):
            raise TypeError("catalog sync status must be a CatalogSyncStatus")
        _aware(self.started_at, "catalog sync started_at")
        if self.completed_at is not None:
            _aware(self.completed_at, "catalog sync completed_at")
            if self.completed_at < self.started_at:
                raise ValueError("catalog sync cannot complete before it starts")
        _optional_text(
            self.source_revision, "catalog sync source_revision", maximum=1_024
        )
        _non_negative_int(self.resource_count, "catalog sync resource_count")
        _non_negative_int(
            self.relationship_count,
            "catalog sync relationship_count",
        )
        _optional_text(self.error_code, "catalog sync error_code", maximum=128)

        if self.status is CatalogSyncStatus.RUNNING:
            if self.completed_at is not None or self.error_code is not None:
                raise ValueError("running catalog sync cannot be completed or errored")
            if self.resource_count or self.relationship_count:
                raise ValueError("running catalog sync cannot claim committed counts")
        else:
            if self.completed_at is None:
                raise ValueError("terminal catalog sync requires completed_at")
            if (
                self.status is CatalogSyncStatus.SUCCEEDED
                and self.error_code is not None
            ):
                raise ValueError("successful catalog sync cannot have an error_code")
            if self.status in {
                CatalogSyncStatus.PARTIAL,
                CatalogSyncStatus.FAILED,
            } and (self.error_code is None):
                raise ValueError("partial or failed catalog sync requires error_code")


@dataclass(frozen=True, slots=True)
class SourceCatalogSnapshot:
    """One complete source snapshot suitable for one atomic catalog commit."""

    sync: CatalogSync
    resources: tuple[CatalogResource, ...]
    revisions: tuple[CatalogResourceRevision, ...]
    facets: tuple[CatalogFacet, ...] = ()
    relationships: tuple[CatalogRelationship, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sync, CatalogSync):
            raise TypeError("catalog snapshot sync must be a CatalogSync")
        if self.sync.status is not CatalogSyncStatus.SUCCEEDED:
            raise ValueError("only a successful sync can commit a source snapshot")
        resources = _record_tuple(
            self.resources,
            CatalogResource,
            "catalog snapshot resources",
        )
        revisions = _record_tuple(
            self.revisions,
            CatalogResourceRevision,
            "catalog snapshot revisions",
        )
        facets = _record_tuple(self.facets, CatalogFacet, "catalog snapshot facets")
        relationships = _record_tuple(
            self.relationships,
            CatalogRelationship,
            "catalog snapshot relationships",
        )
        if len(resources) != self.sync.resource_count:
            raise ValueError("catalog sync resource_count does not match snapshot")
        if len(relationships) != self.sync.relationship_count:
            raise ValueError("catalog sync relationship_count does not match snapshot")

        def _unique(records: tuple[object, ...], field_name: str) -> None:
            ids = [getattr(record, "id", None) for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"catalog snapshot has duplicate {field_name}")

        _unique(resources, "resource ids")
        _unique(relationships, "relationship ids")
        resource_by_id = {resource.id: resource for resource in resources}
        revision_by_resource: dict[str, CatalogResourceRevision] = {}
        for revision in revisions:
            if revision.resource_id in revision_by_resource:
                raise ValueError("catalog snapshot has duplicate current revisions")
            revision_by_resource[revision.resource_id] = revision

        for resource in resources:
            if resource.agent_id != self.sync.agent_id:
                raise ValueError("catalog resource belongs to another agent")
            if resource.source_id != self.sync.source_id:
                raise ValueError("catalog resource belongs to another source")
            if resource.current_sync_id != self.sync.id:
                raise ValueError("catalog resource belongs to another sync")
            current_revision = revision_by_resource.get(resource.id)
            if (
                current_revision is None
                or current_revision.revision != resource.current_revision
            ):
                raise ValueError("catalog resource lacks its exact current revision")

        if set(revision_by_resource) != set(resource_by_id):
            raise ValueError("catalog snapshot revisions do not match resources")
        for revision in revisions:
            if revision.sync_id != self.sync.id:
                raise ValueError("catalog revision belongs to another sync")

        facet_revisions_by_resource: dict[str, set[str]] = {
            resource_id: set() for resource_id in resource_by_id
        }
        for facet in facets:
            if facet.resource_id not in resource_by_id:
                raise ValueError("catalog facet references an unknown resource")
            if facet.sync_id != self.sync.id:
                raise ValueError("catalog facet belongs to another sync")
            facet_revisions_by_resource[facet.resource_id].add(facet.revision)

        relationship_revisions_by_resource: dict[str, set[str]] = {
            resource_id: set() for resource_id in resource_by_id
        }
        for relationship in relationships:
            if relationship.source_id != self.sync.source_id:
                raise ValueError("catalog relationship belongs to another source")
            if relationship.sync_id != self.sync.id:
                raise ValueError("catalog relationship belongs to another sync")
            if relationship.from_resource_id not in resource_by_id:
                raise ValueError("catalog relationship has an unknown source endpoint")
            if relationship.to_resource_id not in resource_by_id:
                raise ValueError("catalog relationship has an unknown target endpoint")
            relationship_revisions_by_resource[relationship.from_resource_id].add(
                relationship.revision
            )
            relationship_revisions_by_resource[relationship.to_resource_id].add(
                relationship.revision
            )

        for resource_id, revision in revision_by_resource.items():
            if (
                set(revision.facet_revisions)
                != facet_revisions_by_resource[resource_id]
            ):
                raise ValueError("resource revision does not link its exact facets")
            if (
                set(revision.relationship_revisions)
                != relationship_revisions_by_resource[resource_id]
            ):
                raise ValueError(
                    "resource revision does not link its exact relationships"
                )

        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "revisions", revisions)
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "relationships", relationships)


@dataclass(frozen=True, slots=True)
class CatalogSearchRequest:
    agent_id: str
    query: str
    source_ids: tuple[str, ...] = ()
    resource_kinds: tuple[ResourceKind, ...] = ()
    limit: int = 20

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "catalog search agent_id")
        if not isinstance(self.query, str):
            raise TypeError("catalog search query must be a string")
        if len(self.query) > 1_024:
            raise ValueError("catalog search query exceeds 1024 characters")
        sources = _text_tuple(
            self.source_ids,
            "catalog search source_ids",
            maximum_items=64,
        )
        if isinstance(self.resource_kinds, (str, bytes)):
            raise TypeError("catalog search resource_kinds must be a sequence")
        kinds = tuple(self.resource_kinds)
        if any(not isinstance(kind, ResourceKind) for kind in kinds):
            raise TypeError("catalog search resource_kinds must contain ResourceKind")
        if len(kinds) != len(set(kinds)):
            raise ValueError("catalog search resource_kinds cannot contain duplicates")
        if len(kinds) > len(ResourceKind):
            raise ValueError("catalog search has too many resource kinds")
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 1 <= self.limit <= 50
        ):
            raise ValueError("catalog search limit must be from 1 through 50")
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "resource_kinds", kinds)


@dataclass(frozen=True, slots=True)
class CatalogSearchHit:
    resource_id: str
    source_id: str
    kind: ResourceKind
    name: str
    revision: str
    sensitivity: Sensitivity
    score: float
    matched_fields: tuple[str, ...] = ()
    match_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.resource_id, "catalog search hit resource_id")
        _required_text(self.source_id, "catalog search hit source_id")
        if not isinstance(self.kind, ResourceKind):
            raise TypeError("catalog search hit kind must be ResourceKind")
        _required_text(self.name, "catalog search hit name")
        _require_hash(self.revision, "catalog search hit revision")
        if not isinstance(self.sensitivity, Sensitivity):
            raise TypeError("catalog search hit sensitivity must be Sensitivity")
        _bounded_float(
            self.score,
            "catalog search hit score",
            minimum=0.0,
            maximum=1_000_000.0,
        )
        matched_fields = _text_tuple(
            self.matched_fields,
            "catalog search hit matched_fields",
            maximum_items=32,
        )
        match_reasons = _text_tuple(
            self.match_reasons,
            "catalog search hit match_reasons",
            maximum_items=32,
        )
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "matched_fields", matched_fields)
        object.__setattr__(self, "match_reasons", match_reasons)


@dataclass(frozen=True, slots=True)
class CatalogSearchResult:
    request: CatalogSearchRequest
    hits: tuple[CatalogSearchHit, ...]
    total_matches: int
    truncated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.request, CatalogSearchRequest):
            raise TypeError(
                "catalog search result request must be CatalogSearchRequest"
            )
        hits = _record_tuple(self.hits, CatalogSearchHit, "catalog search hits")
        _non_negative_int(self.total_matches, "catalog search total_matches")
        if not isinstance(self.truncated, bool):
            raise TypeError("catalog search truncated must be a boolean")
        if len(hits) > self.request.limit:
            raise ValueError("catalog search result exceeds request limit")
        if self.total_matches < len(hits):
            raise ValueError("catalog search total_matches cannot be below hit count")
        if self.truncated != (self.total_matches > len(hits)):
            raise ValueError("catalog search truncated disagrees with total_matches")
        resource_ids = [hit.resource_id for hit in hits]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("catalog search hits cannot repeat a resource")
        if self.request.source_ids and any(
            hit.source_id not in self.request.source_ids for hit in hits
        ):
            raise ValueError("catalog search hit is outside requested source scope")
        if self.request.resource_kinds and any(
            hit.kind not in self.request.resource_kinds for hit in hits
        ):
            raise ValueError("catalog search hit is outside requested kind scope")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True, slots=True)
class CatalogTraversalRequest:
    agent_id: str
    from_resource_ids: tuple[str, ...]
    to_resource_ids: tuple[str, ...]
    relationship_kinds: tuple[RelationshipKind, ...] = ()
    max_depth: int = 4
    max_paths: int = 5
    max_nodes: int = 100
    max_edges: int = 200

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "catalog traversal agent_id")
        from_ids = _text_tuple(
            self.from_resource_ids,
            "catalog traversal from_resource_ids",
            maximum_items=16,
            allow_empty=False,
        )
        to_ids = _text_tuple(
            self.to_resource_ids,
            "catalog traversal to_resource_ids",
            maximum_items=16,
            allow_empty=False,
        )
        if isinstance(self.relationship_kinds, (str, bytes)):
            raise TypeError("catalog traversal relationship_kinds must be a sequence")
        kinds = tuple(self.relationship_kinds)
        if any(not isinstance(kind, RelationshipKind) for kind in kinds):
            raise TypeError(
                "catalog traversal relationship_kinds must contain RelationshipKind"
            )
        if len(kinds) != len(set(kinds)):
            raise ValueError("catalog traversal relationship kinds cannot repeat")
        for value, name, maximum in (
            (self.max_depth, "max_depth", 6),
            (self.max_paths, "max_paths", 8),
            (self.max_nodes, "max_nodes", 1_000),
            (self.max_edges, "max_edges", 2_000),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= maximum
            ):
                raise ValueError(
                    f"catalog traversal {name} must be from 1 through {maximum}"
                )
        object.__setattr__(self, "from_resource_ids", from_ids)
        object.__setattr__(self, "to_resource_ids", to_ids)
        object.__setattr__(self, "relationship_kinds", kinds)


@dataclass(frozen=True, slots=True)
class CatalogPathStep:
    relationship_id: str
    from_resource_id: str
    to_resource_id: str
    direction: RelationshipDirection

    def __post_init__(self) -> None:
        _required_text(self.relationship_id, "catalog path relationship_id")
        _required_text(self.from_resource_id, "catalog path from_resource_id")
        _required_text(self.to_resource_id, "catalog path to_resource_id")
        if not isinstance(self.direction, RelationshipDirection):
            raise TypeError("catalog path direction must be RelationshipDirection")


@dataclass(frozen=True, slots=True)
class CatalogPath:
    resource_ids: tuple[str, ...]
    steps: tuple[CatalogPathStep, ...]

    def __post_init__(self) -> None:
        resource_ids = _text_tuple(
            self.resource_ids,
            "catalog path resource_ids",
            maximum_items=7,
            allow_empty=False,
            unique=False,
        )
        steps = _record_tuple(self.steps, CatalogPathStep, "catalog path steps")
        if len(resource_ids) != len(steps) + 1:
            raise ValueError("catalog path resources must bracket every step")
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("catalog path cannot contain a resource cycle")
        for index, step in enumerate(steps):
            if step.from_resource_id != resource_ids[index]:
                raise ValueError("catalog path step source does not match path order")
            if step.to_resource_id != resource_ids[index + 1]:
                raise ValueError("catalog path step target does not match path order")
        relationship_ids = [step.relationship_id for step in steps]
        if len(relationship_ids) != len(set(relationship_ids)):
            raise ValueError("catalog path cannot repeat a relationship")
        object.__setattr__(self, "resource_ids", resource_ids)
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True, slots=True)
class CatalogTraversalResult:
    request: CatalogTraversalRequest
    paths: tuple[CatalogPath, ...]
    reachable: bool
    visited_nodes: int
    visited_edges: int
    truncated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.request, CatalogTraversalRequest):
            raise TypeError(
                "catalog traversal result request must be CatalogTraversalRequest"
            )
        paths = _record_tuple(self.paths, CatalogPath, "catalog traversal paths")
        if not isinstance(self.reachable, bool):
            raise TypeError("catalog traversal reachable must be a boolean")
        if not isinstance(self.truncated, bool):
            raise TypeError("catalog traversal truncated must be a boolean")
        _non_negative_int(self.visited_nodes, "catalog traversal visited_nodes")
        _non_negative_int(self.visited_edges, "catalog traversal visited_edges")
        if len(paths) > self.request.max_paths:
            raise ValueError("catalog traversal result exceeds max_paths")
        if any(len(path.steps) > self.request.max_depth for path in paths):
            raise ValueError("catalog traversal path exceeds max_depth")
        if self.visited_nodes > self.request.max_nodes:
            raise ValueError("catalog traversal result exceeds max_nodes")
        if self.visited_edges > self.request.max_edges:
            raise ValueError("catalog traversal result exceeds max_edges")
        if self.reachable != bool(paths):
            raise ValueError("catalog traversal reachable disagrees with paths")
        object.__setattr__(self, "paths", paths)


__all__ = [
    "CatalogFacet",
    "CatalogPath",
    "CatalogPathStep",
    "CatalogRelationship",
    "CatalogResource",
    "CatalogResourceRevision",
    "CatalogSearchHit",
    "CatalogSearchRequest",
    "CatalogSearchResult",
    "CatalogSync",
    "CatalogSyncStatus",
    "CatalogTraversalRequest",
    "CatalogTraversalResult",
    "FacetKind",
    "FileFacet",
    "RelationshipDirection",
    "RelationshipFieldPair",
    "RelationshipKind",
    "RelationshipProvenance",
    "ResourceKind",
    "Sensitivity",
    "SourceCatalogSnapshot",
    "TabularColumn",
    "TabularFacet",
    "TabularIndex",
    "catalog_facet_revision",
    "catalog_relationship_id",
    "catalog_resource_id",
    "catalog_resource_revision",
]
