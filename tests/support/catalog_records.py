"""Build small, valid catalog snapshots from the real catalog records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from daita.catalog.models import (
    CatalogEntry,
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
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
