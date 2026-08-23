"""Define the catalog persistence protocol and its lifecycle-specific errors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSnapshotRef,
    CatalogSummary,
    CatalogSync,
    RelationshipKind,
    SourceCatalogSnapshot,
)


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class CatalogStoreError(RuntimeError):
    """Base class for portable catalog repository failures."""


class CatalogResourceNotFoundError(CatalogStoreError):
    def __init__(self, agent_id: str, resource_id: str) -> None:
        _required_text(agent_id, "catalog agent_id")
        _required_text(resource_id, "catalog resource_id")
        self.agent_id = agent_id
        self.resource_id = resource_id
        super().__init__(f"unknown catalog resource for {agent_id}: {resource_id}")


class CatalogRevisionNotFoundError(CatalogStoreError):
    def __init__(self, agent_id: str, resource_id: str, revision: str) -> None:
        _required_text(agent_id, "catalog agent_id")
        _required_text(resource_id, "catalog resource_id")
        _required_text(revision, "catalog revision")
        self.agent_id = agent_id
        self.resource_id = resource_id
        self.revision = revision
        super().__init__(
            f"unknown catalog revision for {agent_id}/{resource_id}: {revision}"
        )


class CatalogSyncConflictError(CatalogStoreError):
    def __init__(self, source_id: str, sync_id: str, reason: str) -> None:
        _required_text(source_id, "catalog source_id")
        _required_text(sync_id, "catalog sync_id")
        _required_text(reason, "catalog sync conflict reason")
        self.source_id = source_id
        self.sync_id = sync_id
        self.reason = reason
        super().__init__(f"catalog sync conflict for {source_id}/{sync_id}: {reason}")


@runtime_checkable
class CatalogStore(Protocol):
    """Persist sync state and complete source snapshots without exposing SQL."""

    async def record_sync(self, sync: CatalogSync) -> CatalogSync: ...

    async def commit_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
    ) -> SourceCatalogSnapshot: ...

    async def list_current_snapshot_refs(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
    ) -> tuple[CatalogSnapshotRef, ...]: ...

    async def load_current_snapshot(
        self,
        ref: CatalogSnapshotRef,
    ) -> SourceCatalogSnapshot | None: ...

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None: ...

    async def summarize_catalog(
        self,
        agent_id: str,
        active_source_ids: tuple[str, ...],
    ) -> CatalogSummary: ...

    async def load_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogResource | None: ...

    async def load_revision(
        self,
        agent_id: str,
        resource_id: str,
        revision: str,
    ) -> CatalogResourceRevision | None: ...

    async def list_resources(
        self,
        agent_id: str,
        source_id: str | None = None,
    ) -> tuple[CatalogResource, ...]: ...

    async def load_facets(
        self,
        agent_id: str,
        resource_id: str,
        revision: str | None = None,
    ) -> tuple[CatalogFacet, ...]: ...

    async def load_incident_relationships(
        self,
        agent_id: str,
        resource_id: str,
        *,
        relationship_kinds: tuple[RelationshipKind, ...] = (),
        limit: int = 50,
    ) -> tuple[CatalogRelationship, ...]: ...


__all__ = [
    "CatalogResourceNotFoundError",
    "CatalogRevisionNotFoundError",
    "CatalogStore",
    "CatalogStoreError",
    "CatalogSyncConflictError",
]
