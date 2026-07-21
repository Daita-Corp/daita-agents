from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from daita.catalog import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchRequest,
    CatalogSync,
    CatalogSyncConflictError,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.identity import AgentIdentity
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _facet(resource_id: str, sync_id: str, column: str) -> CatalogFacet:
    return CatalogFacet.from_tabular(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet=TabularFacet(
            columns=(
                TabularColumn(
                    name="id",
                    native_type="INTEGER",
                    ordinal=0,
                    nullable=False,
                    primary_key_ordinal=1,
                ),
                TabularColumn(
                    name=column,
                    native_type="TEXT",
                    ordinal=1,
                    nullable=True,
                ),
            ),
        ),
    )


def _snapshot(
    *,
    sync_id: str = "sync-1",
    native_names: tuple[str, ...] = ("main.orders", "main.customers"),
    completed_offset: int = 1,
) -> SourceCatalogSnapshot:
    source_id = "source-1"
    resource_ids = tuple(
        catalog_resource_id(source_id, ResourceKind.TABLE, native_name)
        for native_name in native_names
    )
    facets = tuple(
        _facet(
            resource_id,
            sync_id,
            "customer_id" if native_name.endswith("orders") else "display_name",
        )
        for resource_id, native_name in zip(resource_ids, native_names, strict=True)
    )
    relationships: tuple[CatalogRelationship, ...] = ()
    if len(resource_ids) == 2:
        relationships = (
            CatalogRelationship.build(
                source_id=source_id,
                from_resource_id=resource_ids[0],
                to_resource_id=resource_ids[1],
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
            ),
        )
    revisions = tuple(
        CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=NOW,
            facet_revisions=(facet.revision,),
            relationship_revisions=tuple(
                relationship.revision
                for relationship in relationships
                if resource_id
                in {
                    relationship.from_resource_id,
                    relationship.to_resource_id,
                }
            ),
            source_revision=f"schema:{sync_id}",
        )
        for resource_id, facet in zip(resource_ids, facets, strict=True)
    )
    resources = tuple(
        CatalogResource.build(
            agent_id="agent-1",
            source_id=source_id,
            native_identity=native_name,
            external_uri=f"sqlite:///data.db#{native_name}",
            kind=ResourceKind.TABLE,
            name=native_name.rsplit(".", 1)[-1],
            sensitivity=Sensitivity.INTERNAL,
            revision=revision,
            first_observed_at=NOW,
            last_observed_at=NOW + timedelta(seconds=completed_offset),
        )
        for native_name, revision in zip(native_names, revisions, strict=True)
    )
    sync = CatalogSync(
        id=sync_id,
        agent_id="agent-1",
        source_id=source_id,
        adapter_id="sqlite",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW + timedelta(seconds=completed_offset - 1),
        completed_at=NOW + timedelta(seconds=completed_offset),
        source_revision=f"schema:{sync_id}",
        resource_count=len(resources),
        relationship_count=len(relationships),
    )
    return SourceCatalogSnapshot(
        sync=sync,
        resources=resources,
        revisions=revisions,
        facets=facets,
        relationships=relationships,
    )


async def _open_store(path) -> SQLiteOperationStore:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(
            id="agent-1",
            display_name="Catalog test agent",
            created_at=NOW,
        )
    )
    return store


async def test_catalog_snapshot_reopens_searches_and_traverses_current_state(
    tmp_path,
) -> None:
    path = tmp_path / "agent.sqlite3"
    snapshot = _snapshot()
    store = await _open_store(path)
    running = replace(
        snapshot.sync,
        status=CatalogSyncStatus.RUNNING,
        completed_at=None,
        resource_count=0,
        relationship_count=0,
        source_revision=None,
    )

    assert await store.record_sync(running) == running
    committed = await store.commit_snapshot(snapshot)
    assert all(
        resource.first_observed_at == resource.last_observed_at
        for resource in committed.resources
    )
    assert await store.commit_snapshot(snapshot) == committed
    assert await store.load_sync("agent-1", "sync-1") == snapshot.sync
    assert await store.list_resources("agent-1", "source-1") == tuple(
        sorted(
            committed.resources, key=lambda item: (item.kind.value, item.name, item.id)
        )
    )

    orders = next(
        resource for resource in committed.resources if resource.name == "orders"
    )
    customers = next(
        resource for resource in committed.resources if resource.name == "customers"
    )
    assert await store.load_resource("agent-1", orders.id) == orders
    assert await store.load_revision(
        "agent-1", orders.id, orders.current_revision
    ) == next(
        revision for revision in snapshot.revisions if revision.resource_id == orders.id
    )
    assert await store.load_facets("agent-1", orders.id) == tuple(
        facet for facet in snapshot.facets if facet.resource_id == orders.id
    )
    relationship = snapshot.relationships[0]
    assert await store.load_incident_relationships(
        "agent-1",
        orders.id,
        relationship_kinds=(RelationshipKind.REFERENCES,),
        limit=1,
    ) == (relationship,)
    assert await store.load_relationships(
        "agent-1",
        (relationship.id,),
    ) == (relationship,)
    assert await store.load_relationships("other-agent", (relationship.id,)) == ()

    hostile = await store.search(
        CatalogSearchRequest(
            agent_id="agent-1",
            query='orders "quoted" OR *',
            source_ids=("source-1",),
            limit=1,
        )
    )
    assert hostile.hits[0].resource_id == orders.id
    structural = await store.search(
        CatalogSearchRequest(agent_id="agent-1", query="customer_id", limit=10)
    )
    assert orders.id in {hit.resource_id for hit in structural.hits}
    assert "structure" in next(
        hit.matched_fields for hit in structural.hits if hit.resource_id == orders.id
    )

    traversal = await store.traverse(
        CatalogTraversalRequest(
            agent_id="agent-1",
            from_resource_ids=(orders.id,),
            to_resource_ids=(customers.id,),
            relationship_kinds=(RelationshipKind.REFERENCES,),
            max_depth=2,
            max_paths=2,
            max_nodes=4,
            max_edges=4,
        )
    )
    assert traversal.reachable is True
    assert traversal.paths[0].resource_ids == (orders.id, customers.id)
    assert traversal.visited_nodes == 2
    assert traversal.visited_edges == 1

    await store.close()
    reopened = await SQLiteOperationStore.open(path)
    try:
        assert await reopened.load_resource("agent-1", orders.id) == orders
        assert await reopened.load_facets("agent-1", orders.id, orders.current_revision)
        assert await reopened.load_relationships("agent-1", (relationship.id,)) == (
            relationship,
        )
    finally:
        await reopened.close()


async def test_complete_snapshot_replaces_projection_but_retains_revision_history(
    tmp_path,
) -> None:
    store = await _open_store(tmp_path / "agent.sqlite3")
    first = _snapshot()
    second = _snapshot(
        sync_id="sync-2",
        native_names=("main.invoices",),
        completed_offset=3,
    )
    removed = first.resources[0]
    removed_revision = first.revisions[0]
    try:
        await store.commit_snapshot(first)
        second_committed = await store.commit_snapshot(second)

        assert (
            await store.list_resources("agent-1", "source-1")
            == second_committed.resources
        )
        assert await store.load_resource("agent-1", removed.id) is None
        assert (
            await store.load_revision(
                "agent-1",
                removed.id,
                removed_revision.revision,
            )
            == removed_revision
        )
        removed_search = await store.search(
            CatalogSearchRequest(agent_id="agent-1", query="orders")
        )
        assert removed_search.hits == ()
        assert removed_search.total_matches == 0
        assert removed_search.truncated is False
        assert (
            await store.load_relationships(
                "agent-1",
                tuple(relationship.id for relationship in first.relationships),
            )
            == ()
        )
        assert (
            await store.load_incident_relationships(
                "agent-1",
                removed.id,
            )
            == ()
        )
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("query", "expected_names"),
    (
        ("customer", {"customer", "customers", "orders"}),
        ("invoice", {"invoice", "invoices"}),
    ),
)
async def test_catalog_search_ranks_exact_tokens_before_safe_prefix_matches(
    tmp_path,
    query: str,
    expected_names: set[str],
) -> None:
    store = await _open_store(tmp_path / f"{query}.sqlite3")
    snapshot = _snapshot(
        native_names=(
            "main.customer",
            "main.customers",
            "main.invoice",
            "main.invoices",
            "main.orders",
        )
    )
    try:
        await store.commit_snapshot(snapshot)

        result = await store.search(
            CatalogSearchRequest(
                agent_id="agent-1",
                query=query,
                source_ids=("source-1",),
                resource_kinds=(ResourceKind.TABLE,),
                limit=10,
            )
        )

        assert {hit.name for hit in result.hits} == expected_names
        assert result.hits[0].name == query
        assert result.hits[0].match_reasons == ("lexical_exact",)
        reasons = tuple(hit.match_reasons for hit in result.hits)
        first_prefix = reasons.index(("lexical_prefix",))
        assert all(reason == ("lexical_exact",) for reason in reasons[:first_prefix])
        assert all(reason == ("lexical_prefix",) for reason in reasons[first_prefix:])
        assert (
            await store.search(
                CatalogSearchRequest(
                    agent_id="agent-1",
                    query=query,
                    source_ids=("other-source",),
                    limit=10,
                )
            )
        ).hits == ()
        assert (
            await store.search(
                CatalogSearchRequest(
                    agent_id="agent-1",
                    query=query,
                    resource_kinds=(ResourceKind.FILE,),
                    limit=10,
                )
            )
        ).hits == ()
    finally:
        await store.close()


@pytest.mark.parametrize(
    "query",
    (
        '" orders OR customers * NEAR(',
        "a an id",
        "café 客户",
        " ".join(f"term{index}" for index in range(40)),
    ),
    ids=("operators-and-quotes", "short", "unicode", "maximum-terms"),
)
async def test_catalog_search_expansion_is_literal_safe_and_bounded(
    tmp_path,
    query: str,
) -> None:
    store = await _open_store(tmp_path / "safe-search.sqlite3")
    try:
        await store.commit_snapshot(_snapshot())

        result = await store.search(
            CatalogSearchRequest(
                agent_id="agent-1",
                query=query,
                source_ids=("source-1",),
                resource_kinds=(ResourceKind.TABLE,),
                limit=1,
            )
        )

        assert len(result.hits) <= 1
        assert result.total_matches >= len(result.hits)
        assert all(hit.source_id == "source-1" for hit in result.hits)
        assert all(hit.kind is ResourceKind.TABLE for hit in result.hits)
    finally:
        await store.close()


async def test_current_relationship_reads_validate_bounds_before_storage_io(
    tmp_path,
) -> None:
    store = await _open_store(tmp_path / "relationship-bounds.sqlite3")
    try:
        with pytest.raises(ValueError, match="2001"):
            await store.load_incident_relationships(
                "agent-1",
                "resource-1",
                limit=2_002,
            )
        with pytest.raises(ValueError, match="64"):
            await store.load_relationships(
                "agent-1",
                tuple(f"relationship-{index}" for index in range(65)),
            )
        with pytest.raises(ValueError, match="duplicates"):
            await store.load_relationships(
                "agent-1",
                ("relationship-1", "relationship-1"),
            )
    finally:
        await store.close()


async def test_snapshot_failure_rolls_back_sync_history_and_current_projection(
    tmp_path,
) -> None:
    store = await _open_store(tmp_path / "agent.sqlite3")
    first = _snapshot()
    failing = _snapshot(
        sync_id="sync-fail",
        native_names=("main.fail",),
        completed_offset=5,
    )
    try:
        first_committed = await store.commit_snapshot(first)
        store._connection.execute(  # noqa: SLF001 - transaction fault injection
            "CREATE TRIGGER catalog_test_fail BEFORE INSERT ON catalog_resources "
            "WHEN NEW.name = 'fail' BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            await store.commit_snapshot(failing)
        store._connection.execute("DROP TRIGGER catalog_test_fail")  # noqa: SLF001

        assert await store.load_sync("agent-1", "sync-fail") is None
        assert await store.list_resources("agent-1", "source-1") == tuple(
            sorted(
                first_committed.resources,
                key=lambda item: (item.kind.value, item.name, item.id),
            )
        )
    finally:
        await store.close()


async def test_sync_conflicts_and_traversal_bounds_are_fail_closed(tmp_path) -> None:
    store = await _open_store(tmp_path / "agent.sqlite3")
    snapshot = _snapshot()
    running = replace(
        snapshot.sync,
        status=CatalogSyncStatus.RUNNING,
        completed_at=None,
        source_revision=None,
        resource_count=0,
        relationship_count=0,
    )
    try:
        with pytest.raises(CatalogSyncConflictError, match="requires_snapshot"):
            await store.record_sync(snapshot.sync)
        await store.record_sync(running)
        with pytest.raises(CatalogSyncConflictError, match="agent_identity_mismatch"):
            await store.record_sync(replace(running, id="wrong", agent_id="agent-2"))
        await store.commit_snapshot(snapshot)

        orders, customers = (
            next(resource for resource in snapshot.resources if resource.name == name)
            for name in ("orders", "customers")
        )
        bounded = await store.traverse(
            CatalogTraversalRequest(
                agent_id="agent-1",
                from_resource_ids=(orders.id,),
                to_resource_ids=(customers.id,),
                max_depth=1,
                max_paths=1,
                max_nodes=1,
                max_edges=1,
            )
        )
        assert bounded.reachable is False
        assert bounded.truncated is True
        assert bounded.visited_nodes == 1
        assert bounded.visited_edges == 1
    finally:
        await store.close()
