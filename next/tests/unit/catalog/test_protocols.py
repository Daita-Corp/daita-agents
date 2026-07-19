from __future__ import annotations

from daita.catalog import (
    CatalogResourceNotFoundError,
    CatalogRevisionNotFoundError,
    CatalogStore,
    CatalogSyncConflictError,
)


class _CatalogStoreDouble:
    async def record_sync(self, sync):
        return sync

    async def commit_snapshot(self, snapshot):
        return snapshot

    async def load_sync(self, agent_id, sync_id):
        return None

    async def load_resource(self, agent_id, resource_id):
        return None

    async def load_revision(self, agent_id, resource_id, revision):
        return None

    async def list_resources(self, agent_id, source_id=None):
        return ()

    async def load_facets(self, agent_id, resource_id, revision=None):
        return ()

    async def search(self, request):
        raise NotImplementedError

    async def traverse(self, request):
        raise NotImplementedError


def test_catalog_store_protocol_is_lifecycle_specific() -> None:
    store = _CatalogStoreDouble()

    assert isinstance(store, CatalogStore)
    assert not hasattr(store, "execute")
    assert not hasattr(store, "append_event")


def test_catalog_store_errors_preserve_typed_identity() -> None:
    missing_resource = CatalogResourceNotFoundError("agent-1", "resource-1")
    missing_revision = CatalogRevisionNotFoundError(
        "agent-1",
        "resource-1",
        "sha256:" + "1" * 64,
    )
    conflict = CatalogSyncConflictError("source-1", "sync-1", "already_terminal")

    assert missing_resource.agent_id == "agent-1"
    assert missing_resource.resource_id == "resource-1"
    assert missing_revision.revision == "sha256:" + "1" * 64
    assert conflict.source_id == "source-1"
    assert conflict.sync_id == "sync-1"
    assert conflict.reason == "already_terminal"
