"""Build small, valid catalog snapshots from the real catalog records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest

from daita.catalog.models import (
    CatalogEntry,
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSourceState,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTier,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceBatch,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    TabularIndex,
    catalog_resource_id,
)

OBSERVED_AT = datetime(2026, 11, 1, 9, 30, tzinfo=UTC)


@dataclass(frozen=True)
class Table:
    native_identity: str
    columns: tuple[str, ...]
    indexed: tuple[str, ...] = ()


def snapshot(
    agent_id: str,
    source_id: str,
    sync_id: str,
    tables: tuple[Table, ...],
    references: tuple[tuple[str, str, str, str], ...] = (),
    *,
    folder: str | None = "main",
) -> SourceCatalogSnapshot:
    """A snapshot of tables plus (from table, from column, to table, to column) links.

    The optional folder is a resource with no facets, like a source root.
    """

    resource_ids = {
        table.native_identity: catalog_resource_id(
            source_id, ResourceKind.TABLE, table.native_identity
        )
        for table in tables
    }
    relationships = tuple(
        CatalogRelationship.build(
            source_id=source_id,
            from_resource_id=resource_ids[source],
            to_resource_id=resource_ids[target],
            kind=RelationshipKind.REFERENCES,
            provenance=RelationshipProvenance.CONNECTOR,
            confidence=1.0,
            sync_id=sync_id,
            observed_at=OBSERVED_AT,
            field_pairs=(RelationshipFieldPair(source_column, target_column, 0),),
        )
        for source, source_column, target, target_column in references
    )
    facets = {
        table.native_identity: CatalogFacet.from_tabular(
            resource_id=resource_ids[table.native_identity],
            sync_id=sync_id,
            observed_at=OBSERVED_AT,
            facet=TabularFacet(
                columns=tuple(
                    TabularColumn(
                        name=name,
                        native_type="TEXT",
                        ordinal=ordinal,
                        nullable=True,
                    )
                    for ordinal, name in enumerate(table.columns)
                ),
                indexes=(
                    (
                        TabularIndex(
                            name=f"{table.native_identity}_index",
                            kind="btree",
                            columns=table.indexed,
                            unique=False,
                        ),
                    )
                    if table.indexed
                    else ()
                ),
            ),
        )
        for table in tables
    }
    resources: list[CatalogResource] = []
    revisions: list[CatalogResourceRevision] = []
    for table in tables:
        resource_id = resource_ids[table.native_identity]
        revision = CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=OBSERVED_AT,
            facet_revisions=(facets[table.native_identity].revision,),
            relationship_revisions=tuple(
                item.revision
                for item in relationships
                if resource_id in (item.from_resource_id, item.to_resource_id)
            ),
        )
        revisions.append(revision)
        resources.append(
            CatalogResource.build(
                agent_id=agent_id,
                source_id=source_id,
                native_identity=table.native_identity,
                external_uri=f"sqlite:///{table.native_identity}",
                kind=ResourceKind.TABLE,
                name=table.native_identity.split(".")[-1],
                sensitivity=Sensitivity.UNKNOWN,
                revision=revision,
                first_observed_at=OBSERVED_AT,
                last_observed_at=OBSERVED_AT,
            )
        )
    if folder is not None:
        folder_id = catalog_resource_id(source_id, ResourceKind.FOLDER, folder)
        revision = CatalogResourceRevision.build(
            resource_id=folder_id, sync_id=sync_id, observed_at=OBSERVED_AT
        )
        revisions.append(revision)
        resources.append(
            CatalogResource.build(
                agent_id=agent_id,
                source_id=source_id,
                native_identity=folder,
                external_uri=f"sqlite:///{folder}",
                kind=ResourceKind.FOLDER,
                name=folder,
                sensitivity=Sensitivity.UNKNOWN,
                revision=revision,
                first_observed_at=OBSERVED_AT,
                last_observed_at=OBSERVED_AT,
            )
        )
    return SourceCatalogSnapshot(
        sync=CatalogSync(
            id=sync_id,
            agent_id=agent_id,
            source_id=source_id,
            adapter_id="sqlite",
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=OBSERVED_AT,
            completed_at=OBSERVED_AT,
            resource_count=len(resources),
            relationship_count=len(relationships),
        ),
        resources=tuple(resources),
        revisions=tuple(revisions),
        facets=tuple(facets.values()),
        relationships=relationships,
    )


def entries(value: SourceCatalogSnapshot) -> tuple[CatalogEntry, ...]:
    """The snapshot's resources as fully described entries."""

    revisions = {item.resource_id: item for item in value.revisions}
    return tuple(
        CatalogEntry(
            resource=resource,
            revision=revisions[resource.id],
            facets=tuple(
                item for item in value.facets if item.resource_id == resource.id
            ),
            detailed=True,
        )
        for resource in value.resources
    )


def batches(value: SourceCatalogSnapshot) -> tuple[ResourceBatch, ...]:
    """The snapshot as one detail batch."""

    return (
        ResourceBatch(
            tier=CatalogTier.DETAIL,
            entries=entries(value),
            relationships=value.relationships,
            truncated=value.sync.truncated,
            source_revision=value.sync.source_revision,
        ),
    )


SHOP = (
    Table("main.customers", ("id", "CustomerName"), ("CustomerName",)),
    Table("main.orders", ("id", "customer_id", "total")),
    Table("main.invoices", ("id", "order_id")),
)
SHOP_REFERENCES = (
    ("main.orders", "customer_id", "main.customers", "id"),
    ("main.invoices", "order_id", "main.orders", "id"),
)


def _listed(entry: CatalogEntry, sync_id: str) -> CatalogEntry:
    """The same resource as a bare listing: no facets, so no description."""

    old = entry.resource
    revision = CatalogResourceRevision.build(
        resource_id=old.id, sync_id=sync_id, observed_at=OBSERVED_AT
    )
    resource = CatalogResource.build(
        agent_id=old.agent_id,
        source_id=old.source_id,
        native_identity=old.native_identity,
        external_uri=old.external_uri,
        kind=old.kind,
        name=old.name,
        sensitivity=old.sensitivity,
        revision=revision,
        first_observed_at=old.first_observed_at,
        last_observed_at=old.last_observed_at,
    )
    return CatalogEntry(resource, revision, (), detailed=False)


async def exercise_keeper_state(store, agent_id: str, source_id: str, register) -> None:
    """The keeper's storage contract, run against any ``SQLStateStore``.

    ``register`` attaches the source (the store refuses state for unknown ones).
    """

    probed = OBSERVED_AT + timedelta(minutes=5)
    with pytest.raises(ValueError):
        await store.save_source_state(CatalogSourceState(agent_id, source_id, "s"))
    await register()

    def listed(value):
        return ResourceBatch(
            tier=CatalogTier.INVENTORY,
            entries=tuple(_listed(e, value.sync.id) for e in entries(value)),
        )

    async def truncated() -> bool:
        (sync,) = await store.current_syncs(agent_id, (source_id,))
        return sync.truncated

    value = snapshot(agent_id, source_id, "sync-1", SHOP, SHOP_REFERENCES)
    total = len(value.resources)
    detail = batches(value)[0]
    # A listing alone leaves the catalog without detail, and says so.
    await store.commit_catalog(value.sync, (listed(value),))
    assert await store.catalog_detail_counts(agent_id, source_id) == (total, 0)
    assert await truncated()
    # Describing some rows keeps it true; the last description clears it.
    some = replace(detail, entries=detail.entries[:2], relationships=())
    await store.commit_catalog(replace(value.sync, id="sync-2"), (listed(value), some))
    assert await store.catalog_detail_counts(agent_id, source_id) == (total, 2)
    assert await truncated()
    await store.commit_catalog(
        replace(value.sync, id="sync-3"),
        (listed(value), detail),
        probe_tokens={"main": "a"},
        probed_at=probed,
    )
    assert await store.catalog_detail_counts(agent_id, source_id) == (total, total)
    assert not await truncated()

    state = await store.load_source_state(agent_id, source_id)
    assert state is not None
    assert (dict(state.probe_tokens), state.last_probe_at) == ({"main": "a"}, probed)

    # A commit without tokens keeps them (and the error); one with tokens clears it.
    await store.save_source_state(replace(state, last_error_code="unreachable"))
    await store.commit_catalog(replace(value.sync, id="sync-4"), batches(value))
    kept = await store.load_source_state(agent_id, source_id)
    assert kept is not None
    assert (dict(kept.probe_tokens), kept.last_error_code) == (
        {"main": "a"},
        "unreachable",
    )
    await store.commit_catalog(
        replace(value.sync, id="sync-5"),
        batches(value),
        probe_tokens={"main": "b"},
        probed_at=probed,
    )
    moved = await store.load_source_state(agent_id, source_id)
    assert moved is not None
    assert (dict(moved.probe_tokens), moved.last_error_code) == ({"main": "b"}, None)
    for options in ({"probe_tokens": {"main": "c"}}, {"probed_at": probed}):
        with pytest.raises(ValueError):
            await store.commit_catalog(
                replace(value.sync, id="sync-6"), batches(value), **options
            )
    assert (await store.load_source_state(agent_id, source_id)) == moved

    # save_source_state changes the state row alone.
    await store.save_source_state(replace(moved, keep_current=False))
    saved = await store.load_source_state(agent_id, source_id)
    assert saved is not None and not saved.keep_current
    assert await store.catalog_detail_counts(agent_id, source_id) == (total, total)
    assert await store.catalog_detail_counts(agent_id, "source:none") == (0, 0)
