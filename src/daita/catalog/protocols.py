"""Define the catalog persistence protocol and its lifecycle-specific errors."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import (
    CatalogChange,
    CatalogCommitSummary,
    CatalogEntry,
    CatalogFacet,
    CatalogLink,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchRow,
    CatalogSourceState,
    CatalogSummary,
    CatalogSync,
    RelationshipKind,
    ResourceBatch,
    ResourceKind,
)

if TYPE_CHECKING:
    from ..adapters.models import SourceRegistration


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

    async def commit_catalog(
        self,
        sync: CatalogSync,
        batches: tuple[ResourceBatch, ...],
        *,
        registration: SourceRegistration | None = None,
        full: bool = True,
        probe_tokens: Mapping[str, str] | None = None,
        probed_at: datetime | None = None,
    ) -> CatalogCommitSummary:
        """Commit one pass. ``probe_tokens`` and ``probed_at`` come together: they are
        stored in the same transaction and clear the last error; otherwise the
        stored tokens stay."""
        ...

    async def current_syncs(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> tuple[CatalogSync, ...]: ...

    async def load_source_state(
        self, agent_id: str, source_id: str
    ) -> CatalogSourceState | None: ...

    async def save_source_state(self, state: CatalogSourceState) -> None:
        """Upsert one state row of a known source; no catalog rows change."""
        ...

    async def catalog_detail_counts(
        self, agent_id: str, source_id: str
    ) -> tuple[int, int]:
        """(resources, described) of one source."""
        ...

    async def list_catalog_changes(
        self, agent_id: str, source_id: str, *, limit: int = 50
    ) -> tuple[CatalogChange, ...]: ...

    async def take_unreadable_catalog_sources(
        self, agent_id: str
    ) -> tuple[str, ...]: ...

    async def load_entry(
        self, agent_id: str, resource_id: str
    ) -> CatalogEntry | None: ...

    async def load_entries(
        self, agent_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[CatalogEntry, ...]: ...

    async def load_entries_by_identity(
        self, agent_id: str, source_id: str, native_identities: tuple[str, ...]
    ) -> tuple[CatalogEntry, ...]: ...

    async def list_resource_names(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
        *,
        other_than: ResourceKind | None = None,
        limit: int | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        """(ID, source ID, native name) of the resources of these sources (empty
        means every source), or the ``limit`` smallest IDs, reading columns alone."""
        ...

    async def list_entries(
        self, agent_id: str, source_id: str
    ) -> tuple[CatalogEntry, ...]: ...

    async def search_catalog(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
        *,
        terms: tuple[str, ...] | None,
        anchor_keys: tuple[str, ...] = (),
    ) -> tuple[CatalogSearchRow, ...]: ...

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
        resource_ids: tuple[str, ...],
        *,
        relationship_kinds: tuple[RelationshipKind, ...] = (),
        limit: int = 50,
    ) -> Mapping[str, tuple[CatalogRelationship, ...]]: ...

    async def save_catalog_link(self, link: CatalogLink) -> None: ...

    async def delete_catalog_link(self, agent_id: str, link_id: str) -> bool: ...

    async def list_catalog_links(self, agent_id: str) -> tuple[CatalogLink, ...]: ...


__all__ = [
    "CatalogResourceNotFoundError",
    "CatalogRevisionNotFoundError",
    "CatalogStore",
    "CatalogStoreError",
    "CatalogSyncConflictError",
]
