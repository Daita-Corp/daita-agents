from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from daita._json import FrozenJsonObject
from daita.catalog import (
    CatalogFacet,
    CatalogPath,
    CatalogPathStep,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchHit,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    CatalogTraversalResult,
    RelationshipDirection,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    TabularIndex,
    catalog_resource_id,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _column(
    name: str,
    ordinal: int,
    *,
    native_type: str = "INTEGER",
    primary_key_ordinal: int | None = None,
) -> TabularColumn:
    return TabularColumn(
        name=name,
        native_type=native_type,
        ordinal=ordinal,
        nullable=primary_key_ordinal is None,
        primary_key_ordinal=primary_key_ordinal,
    )


def _facet(
    resource_id: str,
    *,
    sync_id: str = "sync-1",
    columns: tuple[TabularColumn, ...] | None = None,
    row_count_estimate: int | None = 10,
) -> CatalogFacet:
    facet_columns = columns or (
        _column("id", 0, primary_key_ordinal=1),
        _column("name", 1, native_type="TEXT"),
    )
    indexed_column = max(facet_columns, key=lambda column: column.ordinal).name
    return CatalogFacet.from_tabular(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet=TabularFacet(
            columns=facet_columns,
            indexes=(
                TabularIndex(
                    name=f"idx_{resource_id[-8:]}_{indexed_column}",
                    kind="btree",
                    columns=(indexed_column,),
                    unique=False,
                ),
            ),
            row_count_estimate=row_count_estimate,
        ),
    )


def _resource(
    *,
    source_id: str,
    native_identity: str,
    revision: CatalogResourceRevision,
    name: str | None = None,
) -> CatalogResource:
    return CatalogResource.build(
        agent_id="agent-1",
        source_id=source_id,
        native_identity=native_identity,
        external_uri=f"sqlite:///data.db#{native_identity}",
        kind=ResourceKind.TABLE,
        name=name or native_identity.rsplit(".", 1)[-1],
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )


def _snapshot() -> SourceCatalogSnapshot:
    source_id = "source-1"
    sync_id = "sync-1"
    orders_id = catalog_resource_id(source_id, ResourceKind.TABLE, "main.orders")
    customers_id = catalog_resource_id(
        source_id,
        ResourceKind.TABLE,
        "main.customers",
    )
    orders_facet = _facet(orders_id)
    customers_facet = _facet(customers_id)
    relationship = CatalogRelationship.build(
        source_id=source_id,
        from_resource_id=orders_id,
        to_resource_id=customers_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.CONNECTOR,
        confidence=1.0,
        sync_id=sync_id,
        observed_at=NOW,
        field_pairs=(
            RelationshipFieldPair(
                source_field="customer_id",
                target_field="id",
                ordinal=0,
            ),
        ),
        attributes={"constraint": "orders_customer_fk"},
    )
    orders_revision = CatalogResourceRevision.build(
        resource_id=orders_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet_revisions=(orders_facet.revision,),
        relationship_revisions=(relationship.revision,),
        source_revision="schema_version:3",
    )
    customers_revision = CatalogResourceRevision.build(
        resource_id=customers_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet_revisions=(customers_facet.revision,),
        relationship_revisions=(relationship.revision,),
        source_revision="schema_version:3",
    )
    resources = (
        _resource(
            source_id=source_id,
            native_identity="main.orders",
            revision=orders_revision,
        ),
        _resource(
            source_id=source_id,
            native_identity="main.customers",
            revision=customers_revision,
        ),
    )
    sync = CatalogSync(
        id=sync_id,
        agent_id="agent-1",
        source_id=source_id,
        adapter_id="sqlite",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
        source_revision="schema_version:3",
        resource_count=2,
        relationship_count=1,
    )
    return SourceCatalogSnapshot(
        sync=sync,
        resources=resources,
        revisions=(orders_revision, customers_revision),
        facets=(orders_facet, customers_facet),
        relationships=(relationship,),
    )


def test_resource_identity_is_stable_and_source_scoped() -> None:
    first_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    second_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    other_source_id = catalog_resource_id(
        "source-2",
        ResourceKind.TABLE,
        "main.orders",
    )

    assert first_id == second_id
    assert first_id.startswith("catalog-resource:sha256:")
    assert first_id != other_source_id


def test_resource_identity_ignores_mutable_display_and_observation_facts() -> None:
    resource_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-1",
        observed_at=NOW,
    )
    first = _resource(
        source_id="source-1",
        native_identity="main.orders",
        revision=revision,
        name="Orders",
    )
    second = replace(
        first,
        name="Renamed display label",
        external_uri="sqlite:///moved.db#main.orders",
        last_observed_at=NOW + timedelta(days=1),
    )

    assert first.id == second.id


def test_tabular_facet_is_order_stable_and_row_estimate_is_nonstructural() -> None:
    resource_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    columns = (
        _column("id", 0, primary_key_ordinal=1),
        _column("status", 1, native_type="TEXT"),
    )
    first = _facet(resource_id, columns=columns, row_count_estimate=10)
    second = _facet(
        resource_id,
        columns=tuple(reversed(columns)),
        row_count_estimate=999,
    )

    assert first.revision == second.revision
    column_payloads = first.payload["columns"]
    assert isinstance(column_payloads, tuple)
    column_names: list[object] = []
    for column_payload in column_payloads:
        assert isinstance(column_payload, FrozenJsonObject)
        column_names.append(column_payload["name"])
    assert column_names == ["id", "status"]
    assert first.payload["row_count_estimate"] == 10


@pytest.mark.parametrize(
    "changed_columns",
    [
        (
            _column("id", 0, native_type="TEXT", primary_key_ordinal=1),
            _column("name", 1, native_type="TEXT"),
        ),
        (
            _column("id", 0, primary_key_ordinal=1),
            _column("name", 1, native_type="BLOB"),
        ),
    ],
)
def test_structural_revision_changes_with_column_contract(
    changed_columns: tuple[TabularColumn, ...],
) -> None:
    resource_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    baseline = _facet(resource_id)
    changed = _facet(resource_id, columns=changed_columns)

    assert baseline.revision != changed.revision


def test_tabular_facet_rejects_duplicate_ordinals_and_bad_index_links() -> None:
    with pytest.raises(ValueError, match="duplicate ordinals"):
        TabularFacet(
            columns=(
                _column("id", 0),
                _column("name", 0, native_type="TEXT"),
            )
        )

    with pytest.raises(ValueError, match="unknown columns"):
        TabularFacet(
            columns=(_column("id", 0),),
            indexes=(
                TabularIndex(
                    name="idx_missing",
                    kind="btree",
                    columns=("missing",),
                    unique=False,
                ),
            ),
        )


def test_composite_relationship_identity_is_order_stable_and_preserves_ordinals() -> (
    None
):
    orders_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    customers_id = catalog_resource_id(
        "source-1",
        ResourceKind.TABLE,
        "main.customers",
    )
    pairs = (
        RelationshipFieldPair("customer_region", "region", 1),
        RelationshipFieldPair("customer_id", "id", 0),
    )
    first = CatalogRelationship.build(
        source_id="source-1",
        from_resource_id=orders_id,
        to_resource_id=customers_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.CONNECTOR,
        confidence=1.0,
        sync_id="sync-1",
        observed_at=NOW,
        field_pairs=pairs,
    )
    second = CatalogRelationship.build(
        source_id="source-1",
        from_resource_id=orders_id,
        to_resource_id=customers_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.CONNECTOR,
        confidence=1.0,
        sync_id="sync-2",
        observed_at=NOW + timedelta(days=1),
        field_pairs=tuple(reversed(pairs)),
    )

    assert first.id == second.id
    assert first.revision == second.revision
    assert [pair.ordinal for pair in first.field_pairs] == [0, 1]


def test_relationship_provenance_is_part_of_identity_and_connector_is_authoritative() -> (
    None
):
    orders_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    customers_id = catalog_resource_id(
        "source-1",
        ResourceKind.TABLE,
        "main.customers",
    )
    pair = RelationshipFieldPair("customer_id", "id", 0)
    connector = CatalogRelationship.build(
        source_id="source-1",
        from_resource_id=orders_id,
        to_resource_id=customers_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.CONNECTOR,
        confidence=1.0,
        sync_id="sync-1",
        observed_at=NOW,
        field_pairs=(pair,),
    )
    inferred = CatalogRelationship.build(
        source_id="source-1",
        from_resource_id=orders_id,
        to_resource_id=customers_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.INFERRED,
        confidence=0.75,
        sync_id="sync-1",
        observed_at=NOW,
        field_pairs=(pair,),
    )

    assert connector.id != inferred.id
    with pytest.raises(ValueError, match="confidence 1.0"):
        replace(connector, confidence=0.9)


def test_relationship_and_facet_payloads_are_recursively_immutable() -> None:
    resource_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    facet = _facet(resource_id)
    attributes = {"discovery": {"constraint_ids": [1, 2]}}
    relationship = CatalogRelationship.build(
        source_id="source-1",
        from_resource_id=resource_id,
        to_resource_id=resource_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.INFERRED,
        confidence=0.5,
        sync_id="sync-1",
        observed_at=NOW,
        field_pairs=(RelationshipFieldPair("parent_id", "id", 0),),
        attributes=attributes,
    )
    attributes["discovery"]["constraint_ids"].append(3)

    assert isinstance(facet.payload, FrozenJsonObject)
    assert isinstance(relationship.attributes, FrozenJsonObject)
    assert relationship.attributes.to_dict() == {
        "discovery": {"constraint_ids": [1, 2]}
    }


def test_resource_revision_changes_when_incident_relationship_changes() -> None:
    resource_id = catalog_resource_id("source-1", ResourceKind.TABLE, "main.orders")
    facet = _facet(resource_id)
    first = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-1",
        observed_at=NOW,
        facet_revisions=(facet.revision,),
    )
    second = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-2",
        observed_at=NOW + timedelta(seconds=1),
        facet_revisions=(facet.revision,),
        relationship_revisions=("sha256:" + "1" * 64,),
    )

    assert first.revision != second.revision


def test_catalog_sync_enforces_status_chronology_and_error_contract() -> None:
    running = CatalogSync(
        id="sync-running",
        agent_id="agent-1",
        source_id="source-1",
        adapter_id="sqlite",
        status=CatalogSyncStatus.RUNNING,
        started_at=NOW,
    )
    assert running.completed_at is None

    with pytest.raises(ValueError, match="requires completed_at"):
        replace(running, status=CatalogSyncStatus.SUCCEEDED)
    with pytest.raises(ValueError, match="requires error_code"):
        replace(
            running,
            status=CatalogSyncStatus.FAILED,
            completed_at=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="before it starts"):
        replace(
            running,
            status=CatalogSyncStatus.FAILED,
            completed_at=NOW - timedelta(seconds=1),
            error_code="discovery_failed",
        )


def test_source_snapshot_validates_all_resource_facet_relationship_links() -> None:
    snapshot = _snapshot()

    assert len(snapshot.resources) == 2
    assert snapshot.sync.resource_count == 2
    assert (
        snapshot.relationships[0].revision
        in snapshot.revisions[0].relationship_revisions
    )

    with pytest.raises(ValueError, match="exact facets"):
        replace(snapshot, facets=snapshot.facets[:1])
    with pytest.raises(ValueError, match="resource_count"):
        replace(snapshot, sync=replace(snapshot.sync, resource_count=1))


def test_search_request_and_result_enforce_scope_limit_and_truncation() -> None:
    snapshot = _snapshot()
    request = CatalogSearchRequest(
        agent_id="agent-1",
        query='orders "quoted" OR *',
        source_ids=("source-1",),
        resource_kinds=(ResourceKind.TABLE,),
        limit=1,
    )
    hit = CatalogSearchHit(
        resource_id=snapshot.resources[0].id,
        source_id="source-1",
        kind=ResourceKind.TABLE,
        name="orders",
        revision=snapshot.resources[0].current_revision,
        sensitivity=Sensitivity.INTERNAL,
        score=6.0,
        matched_fields=("customer_id",),
        match_reasons=("exact asset",),
    )
    result = CatalogSearchResult(
        request=request,
        hits=(hit,),
        total_matches=2,
        truncated=True,
    )

    assert result.hits == (hit,)
    with pytest.raises(ValueError, match="truncated disagrees"):
        replace(result, truncated=False)
    with pytest.raises(ValueError, match="outside requested source"):
        replace(result, hits=(replace(hit, source_id="source-2"),))
    with pytest.raises(ValueError, match="from 1 through 50"):
        replace(request, limit=51)


def test_traversal_records_enforce_path_linkage_cycles_and_hard_bounds() -> None:
    request = CatalogTraversalRequest(
        agent_id="agent-1",
        from_resource_ids=("orders",),
        to_resource_ids=("customers",),
        relationship_kinds=(RelationshipKind.REFERENCES,),
        max_depth=2,
        max_paths=1,
        max_nodes=4,
        max_edges=4,
    )
    step = CatalogPathStep(
        relationship_id="relationship-1",
        from_resource_id="orders",
        to_resource_id="customers",
        direction=RelationshipDirection.FORWARD,
    )
    path = CatalogPath(resource_ids=("orders", "customers"), steps=(step,))
    result = CatalogTraversalResult(
        request=request,
        paths=(path,),
        reachable=True,
        visited_nodes=2,
        visited_edges=1,
        truncated=False,
    )

    assert result.reachable is True
    with pytest.raises(ValueError, match="resource cycle"):
        CatalogPath(
            resource_ids=("orders", "customers", "orders"),
            steps=(
                step,
                CatalogPathStep(
                    relationship_id="relationship-2",
                    from_resource_id="customers",
                    to_resource_id="orders",
                    direction=RelationshipDirection.REVERSE,
                ),
            ),
        )
    with pytest.raises(ValueError, match="exceeds max_nodes"):
        replace(result, visited_nodes=5)
    with pytest.raises(ValueError, match="reachable disagrees"):
        replace(result, paths=(), reachable=True)
