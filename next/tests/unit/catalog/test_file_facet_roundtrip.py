from __future__ import annotations

from datetime import datetime, timezone

from daita.catalog import (
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    FacetKind,
    FileFacet,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    catalog_resource_id,
)
from daita.identity import AgentIdentity
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


async def test_file_facet_round_trips_through_generic_catalog_storage(
    tmp_path,
) -> None:
    database_path = tmp_path / "agent.sqlite3"
    store = await SQLiteOperationStore.open(database_path)
    await store.initialize_identity(
        AgentIdentity(
            id="agent-1",
            display_name="File facet test agent",
            created_at=NOW,
        )
    )
    resource_id = catalog_resource_id(
        "source-1",
        ResourceKind.FILE,
        "exports/customers.csv",
    )
    facet = CatalogFacet.from_file(
        resource_id=resource_id,
        sync_id="sync-1",
        observed_at=NOW,
        facet=FileFacet(
            format="csv",
            media_type="text/csv",
            encoding="utf-8-sig",
            size_bytes=42,
            content_sha256="sha256:" + "a" * 64,
            modified_at=NOW,
        ),
    )
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-1",
        observed_at=NOW,
        facet_revisions=(facet.revision,),
        source_revision="manifest:1",
    )
    resource = CatalogResource.build(
        agent_id="agent-1",
        source_id="source-1",
        native_identity="exports/customers.csv",
        external_uri="local-file://source-1/exports/customers.csv",
        kind=ResourceKind.FILE,
        name="customers.csv",
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )
    snapshot = SourceCatalogSnapshot(
        sync=CatalogSync(
            id="sync-1",
            agent_id="agent-1",
            source_id="source-1",
            adapter_id="local_files",
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=NOW,
            completed_at=NOW,
            source_revision="manifest:1",
            resource_count=1,
        ),
        resources=(resource,),
        revisions=(revision,),
        facets=(facet,),
    )

    try:
        await store.commit_snapshot(snapshot)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(database_path)
    try:
        loaded = await reopened.load_facets("agent-1", resource_id)
        assert loaded == (facet,)
        assert loaded[0].kind is FacetKind.FILE
        assert loaded[0].payload["content_sha256"] == "sha256:" + "a" * 64
        assert loaded[0].payload["modified_at"] == NOW.isoformat()
    finally:
        await reopened.close()
