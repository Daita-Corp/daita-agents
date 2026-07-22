"""Immutable control-plane records shared by resource adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from hashlib import sha256
import re
from typing import TypeVar

from .._json import FrozenJsonObject, canonical_json
from ..catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    ResourceKind,
    SourceCatalogSnapshot,
    catalog_resource_id,
)

_SOURCE_ID = re.compile(r"source:sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")

_RecordT = TypeVar("_RecordT")


def _text(value: str, field_name: str, *, maximum: int = 512) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def _bounded_int(value: int, field_name: str, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field_name} must be from 1 through {maximum}")


def _records(
    values: Iterable[_RecordT],
    expected_type: type[_RecordT],
    field_name: str,
) -> tuple[_RecordT, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of records")
    records = tuple(values)
    if any(not isinstance(value, expected_type) for value in records):
        raise TypeError(f"{field_name} must contain {expected_type.__name__}")
    return records


def source_registration_id(
    agent_id: str,
    adapter_id: str,
    native_identity: str,
) -> str:
    """Return the stable, agent-scoped identity of an attached source."""

    _text(agent_id, "source agent_id")
    _text(adapter_id, "source adapter_id", maximum=128)
    _text(native_identity, "source native_identity", maximum=2_048)
    encoded = canonical_json(
        {
            "adapter_id": adapter_id,
            "agent_id": agent_id,
            "native_identity": native_identity,
        }
    ).encode("utf-8")
    return f"source:sha256:{sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SourceRegistration:
    id: str
    agent_id: str
    adapter_id: str
    native_identity: str
    display_name: str
    configuration: Mapping[str, object]
    attached_at: datetime
    detached_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or _SOURCE_ID.fullmatch(self.id) is None:
            raise ValueError("source id must be a canonical source sha256 id")
        _text(self.agent_id, "source agent_id")
        _text(self.adapter_id, "source adapter_id", maximum=128)
        _text(self.native_identity, "source native_identity", maximum=2_048)
        _text(self.display_name, "source display_name")
        _aware(self.attached_at, "source attached_at")
        if self.detached_at is not None:
            _aware(self.detached_at, "source detached_at")
            if self.detached_at < self.attached_at:
                raise ValueError("source cannot detach before attachment")
        expected = source_registration_id(
            self.agent_id,
            self.adapter_id,
            self.native_identity,
        )
        if self.id != expected:
            raise ValueError("source id does not match its stable identity")
        object.__setattr__(
            self,
            "configuration",
            FrozenJsonObject.from_mapping(self.configuration),
        )

    @classmethod
    def build(
        cls,
        *,
        agent_id: str,
        adapter_id: str,
        native_identity: str,
        display_name: str,
        configuration: Mapping[str, object],
        attached_at: datetime,
    ) -> SourceRegistration:
        return cls(
            id=source_registration_id(agent_id, adapter_id, native_identity),
            agent_id=agent_id,
            adapter_id=adapter_id,
            native_identity=native_identity,
            display_name=display_name,
            configuration=configuration,
            attached_at=attached_at,
        )

    @property
    def active(self) -> bool:
        return self.detached_at is None

    def detach(self, detached_at: datetime) -> SourceRegistration:
        if self.detached_at is not None:
            raise ValueError("source is already detached")
        return replace(self, detached_at=detached_at)


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    agent_id: str
    source_id: str
    sync_id: str
    requested_at: datetime
    max_resources: int = 1_000
    max_columns_per_resource: int = 512
    max_indexes_per_resource: int = 256
    max_relationships: int = 2_000

    def __post_init__(self) -> None:
        _text(self.agent_id, "discovery agent_id")
        _text(self.source_id, "discovery source_id")
        _text(self.sync_id, "discovery sync_id")
        _aware(self.requested_at, "discovery requested_at")
        _bounded_int(self.max_resources, "discovery max_resources", 10_000)
        _bounded_int(
            self.max_columns_per_resource,
            "discovery max_columns_per_resource",
            4_096,
        )
        _bounded_int(
            self.max_indexes_per_resource,
            "discovery max_indexes_per_resource",
            2_048,
        )
        _bounded_int(
            self.max_relationships,
            "discovery max_relationships",
            20_000,
        )


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    request: DiscoveryRequest
    snapshot: SourceCatalogSnapshot
    completed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.request, DiscoveryRequest):
            raise TypeError("discovery result request must be DiscoveryRequest")
        if not isinstance(self.snapshot, SourceCatalogSnapshot):
            raise TypeError("discovery result snapshot must be SourceCatalogSnapshot")
        _aware(self.completed_at, "discovery completed_at")
        if self.completed_at < self.request.requested_at:
            raise ValueError("discovery cannot complete before it was requested")
        sync = self.snapshot.sync
        if sync.agent_id != self.request.agent_id:
            raise ValueError("discovery snapshot belongs to another agent")
        if sync.source_id != self.request.source_id:
            raise ValueError("discovery snapshot belongs to another source")
        if sync.id != self.request.sync_id:
            raise ValueError("discovery snapshot belongs to another sync")
        if sync.started_at != self.request.requested_at:
            raise ValueError("discovery sync start does not match the request")
        if sync.completed_at != self.completed_at:
            raise ValueError("discovery completion does not match the snapshot")
        if len(self.snapshot.resources) > self.request.max_resources:
            raise ValueError("discovery result exceeds max_resources")
        if len(self.snapshot.relationships) > self.request.max_relationships:
            raise ValueError("discovery result exceeds max_relationships")


@dataclass(frozen=True, slots=True)
class ResourceRef:
    agent_id: str
    source_id: str
    resource_id: str
    native_identity: str
    kind: ResourceKind
    revision: str | None = None

    def __post_init__(self) -> None:
        _text(self.agent_id, "resource reference agent_id")
        _text(self.source_id, "resource reference source_id")
        _text(self.resource_id, "resource reference resource_id")
        _text(
            self.native_identity,
            "resource reference native_identity",
            maximum=2_048,
        )
        if not isinstance(self.kind, ResourceKind):
            raise TypeError("resource reference kind must be ResourceKind")
        expected = catalog_resource_id(
            self.source_id,
            self.kind,
            self.native_identity,
        )
        if self.resource_id != expected:
            raise ValueError("resource reference does not match its stable identity")
        if self.revision is not None and (
            not isinstance(self.revision, str)
            or _SHA256.fullmatch(self.revision) is None
        ):
            raise ValueError("resource reference revision must be a sha256 hash")

    @classmethod
    def from_resource(cls, resource: CatalogResource) -> ResourceRef:
        if not isinstance(resource, CatalogResource):
            raise TypeError("resource must be CatalogResource")
        return cls(
            agent_id=resource.agent_id,
            source_id=resource.source_id,
            resource_id=resource.id,
            native_identity=resource.native_identity,
            kind=resource.kind,
            revision=resource.current_revision,
        )


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    reference: ResourceRef
    resource: CatalogResource
    revision: CatalogResourceRevision
    facets: tuple[CatalogFacet, ...]
    relationships: tuple[CatalogRelationship, ...]
    inspected_at: datetime
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ResourceRef):
            raise TypeError("resource snapshot reference must be ResourceRef")
        if not isinstance(self.resource, CatalogResource):
            raise TypeError("resource snapshot resource must be CatalogResource")
        if not isinstance(self.revision, CatalogResourceRevision):
            raise TypeError(
                "resource snapshot revision must be CatalogResourceRevision"
            )
        facets = _records(self.facets, CatalogFacet, "resource snapshot facets")
        relationships = _records(
            self.relationships,
            CatalogRelationship,
            "resource snapshot relationships",
        )
        _aware(self.inspected_at, "resource snapshot inspected_at")
        if self.source_revision is not None:
            _text(
                self.source_revision,
                "resource snapshot source_revision",
                maximum=1_024,
            )
        if self.resource.agent_id != self.reference.agent_id:
            raise ValueError("resource snapshot belongs to another agent")
        if self.resource.source_id != self.reference.source_id:
            raise ValueError("resource snapshot belongs to another source")
        if self.resource.id != self.reference.resource_id:
            raise ValueError("resource snapshot belongs to another resource")
        if self.resource.native_identity != self.reference.native_identity:
            raise ValueError("resource snapshot has another native identity")
        if self.resource.kind is not self.reference.kind:
            raise ValueError("resource snapshot has another resource kind")
        if self.revision.resource_id != self.resource.id:
            raise ValueError("resource snapshot revision belongs to another resource")
        if self.revision.revision != self.resource.current_revision:
            raise ValueError("resource snapshot revision is not current")
        if self.revision.sync_id != self.resource.current_sync_id:
            raise ValueError("resource snapshot revision belongs to another sync")
        if self.reference.revision is not None and (
            self.reference.revision != self.revision.revision
        ):
            raise ValueError("resource snapshot does not match requested revision")
        if {facet.revision for facet in facets} != set(self.revision.facet_revisions):
            raise ValueError("resource snapshot does not link its exact facets")
        if any(
            facet.resource_id != self.resource.id
            or facet.sync_id != self.resource.current_sync_id
            for facet in facets
        ):
            raise ValueError("resource snapshot facet linkage is invalid")
        if {relationship.revision for relationship in relationships} != set(
            self.revision.relationship_revisions
        ):
            raise ValueError("resource snapshot does not link exact relationships")
        if any(
            relationship.sync_id != self.resource.current_sync_id
            or self.resource.id
            not in {
                relationship.from_resource_id,
                relationship.to_resource_id,
            }
            for relationship in relationships
        ):
            raise ValueError("resource snapshot relationship linkage is invalid")
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "relationships", relationships)


@dataclass(frozen=True, slots=True)
class SourceHealth:
    agent_id: str
    source_id: str
    adapter_id: str
    healthy: bool
    checked_at: datetime
    source_revision: str | None = None
    error_code: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.agent_id, "source health agent_id")
        _text(self.source_id, "source health source_id")
        _text(self.adapter_id, "source health adapter_id", maximum=128)
        if not isinstance(self.healthy, bool):
            raise TypeError("source health healthy must be a boolean")
        _aware(self.checked_at, "source health checked_at")
        if self.source_revision is not None:
            _text(self.source_revision, "source health source_revision", maximum=1_024)
        if self.healthy and self.error_code is not None:
            raise ValueError("healthy source cannot have an error_code")
        if not self.healthy and self.error_code is None:
            raise ValueError("unhealthy source requires error_code")
        if self.error_code is not None and (
            not isinstance(self.error_code, str)
            or _ERROR_CODE.fullmatch(self.error_code) is None
        ):
            raise ValueError("source health error_code is invalid")
        object.__setattr__(
            self,
            "details",
            FrozenJsonObject.from_mapping(self.details),
        )


__all__ = [
    "DiscoveryRequest",
    "DiscoveryResult",
    "ResourceRef",
    "ResourceSnapshot",
    "SourceHealth",
    "SourceRegistration",
    "source_registration_id",
]
