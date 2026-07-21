from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any, cast

import pytest

from daita._json import FrozenJsonObject
from daita.adapters import DiscoveryRequest, LocalDirectorySource, SourceRegistration
from daita.capabilities import CapabilityRegistry, EvidenceCandidate, ExecutionRequest
from daita.catalog import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_INSPECT_EVIDENCE_KIND,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_SEARCH_EVIDENCE_KIND,
    CATALOG_TRAVERSE_CAPABILITY_ID,
    CATALOG_TRAVERSE_EVIDENCE_KIND,
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceNotFoundError,
    CatalogResourceRevision,
    CatalogSearchRequest,
    CatalogService,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    FacetKind,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_declarations,
    catalog_resource_id,
)
from daita.domains.data import CatalogDataView, ResourceSchema
from daita.identity import AgentIdentity
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
AGENT_ID = "agent-1"
REGISTRATION = SourceRegistration.build(
    agent_id=AGENT_ID,
    adapter_id="sqlite",
    native_identity="catalog-service-test",
    display_name="Catalog service test source",
    configuration={
        "secret_reference": "secret:catalog-test",
        "truthy_text": "true",
        "write_access": True,
    },
    attached_at=NOW,
)
SOURCE_ID = REGISTRATION.id
RESOURCE_ID = catalog_resource_id(SOURCE_ID, ResourceKind.TABLE, "main.orders")


def _json_object(value: object) -> FrozenJsonObject:
    assert isinstance(value, FrozenJsonObject)
    return value


def _json_array(value: object) -> tuple[Any, ...]:
    assert isinstance(value, tuple)
    return cast(tuple[Any, ...], value)


def _json_string(value: object) -> str:
    assert isinstance(value, str)
    return value


def _snapshot(
    sync_id: str,
    *,
    columns: tuple[str, ...],
    offset: int,
) -> SourceCatalogSnapshot:
    observed_at = NOW + timedelta(seconds=offset)
    facet = CatalogFacet.from_tabular(
        resource_id=RESOURCE_ID,
        sync_id=sync_id,
        observed_at=observed_at,
        facet=TabularFacet(
            columns=tuple(
                TabularColumn(
                    name=name,
                    native_type="INTEGER" if index == 0 else "TEXT",
                    ordinal=index,
                    nullable=index != 0,
                    primary_key_ordinal=1 if index == 0 else None,
                )
                for index, name in enumerate(columns)
            ),
        ),
    )
    revision = CatalogResourceRevision.build(
        resource_id=RESOURCE_ID,
        sync_id=sync_id,
        observed_at=observed_at,
        facet_revisions=(facet.revision,),
        source_revision=f"schema:{sync_id}",
    )
    resource = CatalogResource.build(
        agent_id=AGENT_ID,
        source_id=SOURCE_ID,
        native_identity="main.orders",
        external_uri="sqlite:///data.db#main.orders",
        kind=ResourceKind.TABLE,
        name="orders",
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=observed_at,
    )
    sync = CatalogSync(
        id=sync_id,
        agent_id=AGENT_ID,
        source_id=SOURCE_ID,
        adapter_id="sqlite",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=observed_at - timedelta(seconds=1),
        completed_at=observed_at,
        source_revision=f"schema:{sync_id}",
        resource_count=1,
        relationship_count=0,
    )
    return SourceCatalogSnapshot(
        sync=sync,
        resources=(resource,),
        revisions=(revision,),
        facets=(facet,),
    )


def _relationship_snapshot() -> SourceCatalogSnapshot:
    sync_id = "sync-relationship"
    observed_at = NOW + timedelta(seconds=10)
    orders_id = catalog_resource_id(
        SOURCE_ID,
        ResourceKind.TABLE,
        "main.order_lines",
    )
    customers_id = catalog_resource_id(
        SOURCE_ID,
        ResourceKind.TABLE,
        "main.customer_regions",
    )
    facets = (
        CatalogFacet.from_tabular(
            resource_id=orders_id,
            sync_id=sync_id,
            observed_at=observed_at,
            facet=TabularFacet(
                columns=(
                    TabularColumn("customer_id", "INTEGER", 0, False),
                    TabularColumn("customer_region", "TEXT", 1, False),
                )
            ),
        ),
        CatalogFacet.from_tabular(
            resource_id=customers_id,
            sync_id=sync_id,
            observed_at=observed_at,
            facet=TabularFacet(
                columns=(
                    TabularColumn("id", "INTEGER", 0, False),
                    TabularColumn("region", "TEXT", 1, False),
                )
            ),
        ),
    )
    relationship = CatalogRelationship.build(
        source_id=SOURCE_ID,
        from_resource_id=orders_id,
        to_resource_id=customers_id,
        kind=RelationshipKind.REFERENCES,
        provenance=RelationshipProvenance.CONNECTOR,
        confidence=1.0,
        sync_id=sync_id,
        observed_at=observed_at,
        field_pairs=(
            RelationshipFieldPair("customer_id", "id", 0),
            RelationshipFieldPair("customer_region", "region", 1),
        ),
    )
    revisions = tuple(
        CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=observed_at,
            facet_revisions=(facet.revision,),
            relationship_revisions=(relationship.revision,),
        )
        for resource_id, facet in zip(
            (orders_id, customers_id),
            facets,
            strict=True,
        )
    )
    resources = tuple(
        CatalogResource.build(
            agent_id=AGENT_ID,
            source_id=SOURCE_ID,
            native_identity=native_identity,
            external_uri=f"sqlite:///data.db#{native_identity}",
            kind=ResourceKind.TABLE,
            name=name,
            sensitivity=Sensitivity.INTERNAL,
            revision=revision,
            first_observed_at=observed_at,
            last_observed_at=observed_at,
        )
        for native_identity, name, revision in zip(
            ("main.order_lines", "main.customer_regions"),
            ("order_lines", "customer_regions"),
            revisions,
            strict=True,
        )
    )
    return SourceCatalogSnapshot(
        sync=CatalogSync(
            id=sync_id,
            agent_id=AGENT_ID,
            source_id=SOURCE_ID,
            adapter_id="sqlite",
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=observed_at - timedelta(seconds=1),
            completed_at=observed_at,
            resource_count=2,
            relationship_count=1,
        ),
        resources=resources,
        revisions=revisions,
        facets=facets,
        relationships=(relationship,),
    )


async def _local_catalog_stack(
    database_path: Path,
    root: Path,
    *,
    omit_modified_at: bool = False,
) -> tuple[SQLiteOperationStore, CatalogService, SourceCatalogSnapshot]:
    adapter = await LocalDirectorySource(root).open(
        agent_id=AGENT_ID,
        attached_at=NOW,
        clock=lambda: NOW + timedelta(seconds=20),
    )
    try:
        result = await adapter.discover(
            DiscoveryRequest(
                agent_id=AGENT_ID,
                source_id=adapter.registration.id,
                sync_id="sync-local",
                requested_at=NOW,
            )
        )
    finally:
        await adapter.close()
    snapshot = result.snapshot
    if omit_modified_at:
        removed = False
        projected_facets = []
        for facet in snapshot.facets:
            if not removed and facet.kind is FacetKind.FILE:
                projected_facets.append(
                    replace(
                        facet,
                        payload={
                            key: value
                            for key, value in facet.payload.items()
                            if key != "modified_at"
                        },
                    )
                )
                removed = True
            else:
                projected_facets.append(facet)
        snapshot = replace(snapshot, facets=tuple(projected_facets))
    store = await SQLiteOperationStore.open(database_path)
    await store.initialize_identity(
        AgentIdentity(
            id=AGENT_ID,
            display_name="Local catalog service test agent",
            created_at=NOW,
        )
    )
    await store.register_source(adapter.registration)
    snapshot = await store.commit_snapshot(snapshot)
    return store, CatalogService(store, store), snapshot


async def _catalog_stack(
    path: Path,
) -> tuple[SQLiteOperationStore, CatalogService, CatalogDataView]:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(
            id=AGENT_ID,
            display_name="Catalog service test agent",
            created_at=NOW,
        )
    )
    await store.register_source(REGISTRATION)
    await store.commit_snapshot(
        _snapshot("sync-1", columns=("id", "legacy_name"), offset=1)
    )
    await store.commit_snapshot(
        _snapshot("sync-2", columns=("id", "customer_name"), offset=3)
    )
    service = CatalogService(store, store)
    return store, service, CatalogDataView(store, service, store)


async def test_service_and_data_view_preserve_scope_current_schema_and_trust(
    tmp_path: Path,
) -> None:
    store, service, view = await _catalog_stack(tmp_path / "agent.sqlite3")
    try:
        result = await service.search(
            CatalogSearchRequest(
                agent_id=AGENT_ID,
                query="orders",
                source_ids=(SOURCE_ID,),
            )
        )
        assert tuple(hit.resource_id for hit in result.hits) == (RESOURCE_ID,)
        assert await service.search(
            CatalogSearchRequest(agent_id="other-agent", query="orders")
        ) == type(result)(
            request=CatalogSearchRequest(agent_id="other-agent", query="orders"),
            hits=(),
            total_matches=0,
            truncated=False,
        )
        assert await service.search(
            CatalogSearchRequest(
                agent_id=AGENT_ID,
                query="orders",
                source_ids=("other-source",),
            )
        ) == type(result)(
            request=CatalogSearchRequest(
                agent_id=AGENT_ID,
                query="orders",
                source_ids=("other-source",),
            ),
            hits=(),
            total_matches=0,
            truncated=False,
        )

        inspection = await service.inspect_resource(AGENT_ID, RESOURCE_ID)
        assert isinstance(inspection, FrozenJsonObject)
        inspection_payload = inspection.to_dict()
        assert inspection_payload["trust_classification"] == "untrusted_external_data"
        resource_payload = inspection["resource"]
        assert isinstance(resource_payload, FrozenJsonObject)
        assert resource_payload["source_id"] == SOURCE_ID
        facets_payload = inspection["facets"]
        assert isinstance(facets_payload, tuple)
        facet_record = facets_payload[0]
        assert isinstance(facet_record, FrozenJsonObject)
        facet_payload = facet_record["payload"]
        assert isinstance(facet_payload, FrozenJsonObject)
        columns_payload = facet_payload["columns"]
        assert isinstance(columns_payload, tuple)
        column_names: list[object] = []
        for column_payload in columns_payload:
            assert isinstance(column_payload, FrozenJsonObject)
            column_names.append(column_payload["name"])
        assert column_names == ["id", "customer_name"]
        assert "legacy_name" not in str(facet_payload)
        with pytest.raises(CatalogResourceNotFoundError):
            await service.inspect_resource("other-agent", RESOURCE_ID)

        assert await view.resource_schemas(AGENT_ID, SOURCE_ID) == (
            ResourceSchema(
                resource_id=RESOURCE_ID,
                source_id=SOURCE_ID,
                name="orders",
                columns=("id", "customer_name"),
                aliases=("main.orders",),
            ),
        )
        assert await view.resource_schemas(AGENT_ID, "other-source") == ()
        assert await view.resource_schemas("other-agent", SOURCE_ID) == ()
        context = await view.catalog_context(AGENT_ID, "orders", limit=1)
        assert context["trust_classification"] == "untrusted_external_data"
        context_resources = context["resources"]
        assert isinstance(context_resources, tuple)
        context_resource = context_resources[0]
        assert isinstance(context_resource, FrozenJsonObject)
        assert context_resource["resource_id"] == RESOURCE_ID
    finally:
        await store.close()


async def test_inspection_projects_current_edges_and_traversal_v2_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "relationships.sqlite3"
    snapshot = _relationship_snapshot()
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(
            id=AGENT_ID,
            display_name="Relationship catalog test agent",
            created_at=NOW,
        )
    )
    await store.register_source(REGISTRATION)
    await store.commit_snapshot(snapshot)
    service = CatalogService(store, store)
    orders, customers = snapshot.resources
    relationship = snapshot.relationships[0]
    try:
        inspection = await service.inspect_resource(AGENT_ID, orders.id)

        assert inspection["incident_relationships_truncated"] is False
        projected_relationships = inspection["incident_relationships"]
        assert isinstance(projected_relationships, tuple)
        projected_relationship = projected_relationships[0]
        assert isinstance(projected_relationship, FrozenJsonObject)
        assert projected_relationship["relationship_id"] == relationship.id
        assert projected_relationship["revision"] == relationship.revision
        assert projected_relationship["direction"] == "forward"
        assert projected_relationship["from_resource_revision"] == (
            orders.current_revision
        )
        assert projected_relationship["to_resource_revision"] == (
            customers.current_revision
        )
        field_pairs = tuple(
            _json_object(item)
            for item in _json_array(projected_relationship["field_pairs"])
        )
        assert tuple(
            (
                pair["source_field"],
                pair["target_field"],
                pair["ordinal"],
            )
            for pair in field_pairs
        ) == (
            ("customer_id", "id", 0),
            ("customer_region", "region", 1),
        )
        neighbors = _json_array(inspection["neighbors"])
        first_neighbor = _json_object(neighbors[0])
        assert first_neighbor["resource_id"] == customers.id
        assert first_neighbor["revision"] == customers.current_revision

        declarations = catalog_declarations(AGENT_ID, service)
        registry = CapabilityRegistry(
            capabilities=declarations.capabilities,
            executors=declarations.executors,
            tool_views=declarations.tool_views,
        )
        capability, executor = registry.resolve_execution(
            CATALOG_TRAVERSE_CAPABILITY_ID
        )
        candidate = await executor.execute(
            _execution_request(
                capability.id,
                executor.executor_id,
                {
                    "from_resource_ids": [orders.id],
                    "to_resource_ids": [customers.id],
                    "relationship_kinds": ["references"],
                    "max_depth": 2,
                    "max_paths": 2,
                    "max_nodes": 4,
                    "max_edges": 4,
                },
            )
        )

        assert candidate.kind == CATALOG_TRAVERSE_EVIDENCE_KIND
        assert candidate.schema_version == 2
        assert registry.validate_evidence(capability.id, candidate) is candidate
        assert candidate.payload["reachable"] is True
        assert candidate.payload["visited_nodes"] == 2
        assert candidate.payload["visited_edges"] == 1
        paths = _json_array(candidate.payload["paths"])
        steps = _json_array(_json_object(paths[0])["steps"])
        first_step = _json_object(steps[0])
        assert first_step["relationship_id"] == relationship.id
        assert first_step["field_pairs"] == projected_relationship["field_pairs"]
        request_payload = _json_object(candidate.payload["request"])
        assert request_payload["relationship_kinds"] == ("references",)

        with pytest.raises(ValueError, match="max_depth"):
            await executor.execute(
                _execution_request(
                    capability.id,
                    executor.executor_id,
                    {
                        "from_resource_ids": [orders.id],
                        "to_resource_ids": [customers.id],
                        "max_depth": 7,
                    },
                )
            )
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        reopened_service = CatalogService(reopened, reopened)
        traversal = await reopened_service.traverse(
            CatalogTraversalRequest(
                agent_id=AGENT_ID,
                from_resource_ids=(orders.id,),
                to_resource_ids=(customers.id,),
                relationship_kinds=(RelationshipKind.REFERENCES,),
            )
        )
        assert traversal["reachable"] is True
        reopened_paths = _json_array(traversal["paths"])
        reopened_relationships = _json_array(_json_object(reopened_paths[0])["steps"])
        assert _json_object(reopened_relationships[0])["revision"] == (
            relationship.revision
        )
        await reopened.commit_snapshot(
            _snapshot(
                "sync-replaced",
                columns=("id", "current_name"),
                offset=30,
            )
        )
        with pytest.raises(CatalogResourceNotFoundError):
            await reopened_service.traverse(
                CatalogTraversalRequest(
                    agent_id=AGENT_ID,
                    from_resource_ids=(orders.id,),
                    to_resource_ids=(customers.id,),
                )
            )
        assert (
            await reopened.load_relationships(
                AGENT_ID,
                (relationship.id,),
            )
            == ()
        )
    finally:
        await reopened.close()


async def test_folder_inspection_uses_connector_mtime_and_discloses_equal_maxima(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    first = root / "a-export.csv"
    tied = root / "m-export.csv"
    lexically_last = root / "z-export.csv"
    for index, path in enumerate((first, tied, lexically_last), start=1):
        path.write_text(f"id,name\n{index},name-{index}\n", encoding="utf-8")
    newest_timestamp = NOW.timestamp() - 60
    older_timestamp = NOW.timestamp() - 300
    os.utime(first, (newest_timestamp, newest_timestamp))
    os.utime(tied, (newest_timestamp, newest_timestamp))
    os.utime(lexically_last, (older_timestamp, older_timestamp))
    store, service, snapshot = await _local_catalog_stack(
        tmp_path / "local.sqlite3",
        root,
    )
    folder = next(
        resource
        for resource in snapshot.resources
        if resource.kind is ResourceKind.FOLDER
    )
    files = tuple(
        resource
        for resource in snapshot.resources
        if resource.kind is ResourceKind.FILE
    )
    try:
        folder_inspection = await service.inspect_resource(AGENT_ID, folder.id)
        assert folder_inspection["incident_relationships_truncated"] is False
        assert len(_json_array(folder_inspection["incident_relationships"])) == 3
        folder_selection = _json_object(folder_inspection["selection_facts"])
        hierarchy = _json_object(folder_selection["hierarchy"])
        assert hierarchy["basis"] == "contains"
        assert hierarchy["authority"] == "catalog_relationship"
        assert hierarchy["child_count"] == 3
        assert hierarchy["truncated"] is False
        children = tuple(
            _json_object(child) for child in _json_array(hierarchy["children"])
        )
        assert {child["resource_id"] for child in children} == {
            resource.id for resource in files
        }

        freshness_by_name: dict[str, FrozenJsonObject] = {}
        for resource in files:
            inspection = await service.inspect_resource(AGENT_ID, resource.id)
            selection = _json_object(inspection["selection_facts"])
            freshness = _json_object(selection["freshness"])
            assert freshness["basis"] == "file.modified_at"
            assert freshness["authority"] == "connector_metadata"
            assert freshness["available"] is True
            assert freshness["sync_id"] == resource.current_sync_id
            assert _json_string(freshness["facet_revision"]).startswith("sha256:")
            assert set(freshness) == {
                "authority",
                "available",
                "basis",
                "facet_revision",
                "observed_at",
                "sync_id",
                "value",
            }
            freshness_by_name[resource.name] = freshness

        greatest = max(
            _json_string(fact["value"]) for fact in freshness_by_name.values()
        )
        greatest_names = {
            name
            for name, fact in freshness_by_name.items()
            if _json_string(fact["value"]) == greatest
        }
        assert greatest_names == {"a-export.csv", "m-export.csv"}
        assert _json_string(freshness_by_name["z-export.csv"]["value"]) < greatest
        newest_child_file = _json_object(folder_selection["newest_child_file"])
        assert newest_child_file["status"] == "ambiguous"
        assert newest_child_file["selected_resource_id"] is None
        assert newest_child_file["ambiguity_reasons"] == ("equal_greatest_values",)
        assert set(_json_array(newest_child_file["tied_resource_ids"])) == {
            resource.id
            for resource in files
            if resource.name in {"a-export.csv", "m-export.csv"}
        }

        target = next(resource for resource in files if resource.name == "a-export.csv")
        traversal = await service.traverse(
            CatalogTraversalRequest(
                agent_id=AGENT_ID,
                from_resource_ids=(folder.id,),
                to_resource_ids=(target.id,),
                relationship_kinds=(RelationshipKind.CONTAINS,),
                max_depth=1,
                max_paths=1,
                max_nodes=4,
                max_edges=4,
            )
        )
        assert traversal["reachable"] is True
        paths = _json_array(traversal["paths"])
        steps = _json_array(_json_object(paths[0])["steps"])
        step = _json_object(steps[0])
        assert step["kind"] == "contains"
        assert step["path_from_resource_id"] == folder.id
        assert step["path_to_resource_id"] == target.id
    finally:
        await store.close()


async def test_file_inspection_marks_missing_connector_freshness_ineligible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-freshness"
    root.mkdir()
    (root / "incomplete.csv").write_text("id\n1\n", encoding="utf-8")
    store, service, snapshot = await _local_catalog_stack(
        tmp_path / "missing.sqlite3",
        root,
        omit_modified_at=True,
    )
    resource = next(
        item for item in snapshot.resources if item.kind is ResourceKind.FILE
    )
    try:
        inspection = await service.inspect_resource(AGENT_ID, resource.id)
        selection = _json_object(inspection["selection_facts"])
        freshness = _json_object(selection["freshness"])

        assert freshness["available"] is False
        assert freshness["value"] is None
        assert freshness["basis"] == "file.modified_at"
        assert freshness["facet_revision"] is not None
        assert freshness["sync_id"] == resource.current_sync_id
        assert freshness["observed_at"] is not None
        folder = next(
            item for item in snapshot.resources if item.kind is ResourceKind.FOLDER
        )
        folder_inspection = await service.inspect_resource(AGENT_ID, folder.id)
        folder_selection = _json_object(folder_inspection["selection_facts"])
        newest_child_file = _json_object(folder_selection["newest_child_file"])
        assert newest_child_file["status"] == "ambiguous"
        assert newest_child_file["selected_resource_id"] is None
        assert newest_child_file["ambiguity_reasons"] == ("missing_freshness",)
        assert newest_child_file["missing_freshness_resource_ids"] == (resource.id,)
    finally:
        await store.close()


async def test_folder_inspection_fetches_limit_plus_one_for_true_truncation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bounded-children"
    root.mkdir()
    for index in range(51):
        (root / f"export-{index:02d}.csv").write_text(
            f"id\n{index}\n",
            encoding="utf-8",
        )
    store, service, snapshot = await _local_catalog_stack(
        tmp_path / "bounded.sqlite3",
        root,
    )
    folder = next(
        resource
        for resource in snapshot.resources
        if resource.kind is ResourceKind.FOLDER
    )
    try:
        inspection = await service.inspect_resource(AGENT_ID, folder.id)

        assert inspection["incident_relationships_truncated"] is True
        assert len(_json_array(inspection["incident_relationships"])) == 50
        assert len(_json_array(inspection["neighbors"])) == 50
        selection = _json_object(inspection["selection_facts"])
        hierarchy = _json_object(selection["hierarchy"])
        assert hierarchy["child_count"] == 50
        assert hierarchy["truncated"] is True
        assert len(_json_array(hierarchy["children"])) == 50
        newest_child_file = _json_object(selection["newest_child_file"])
        assert newest_child_file["status"] == "ambiguous"
        assert "candidate_set_truncated" in _json_array(
            newest_child_file["ambiguity_reasons"]
        )
    finally:
        await store.close()


async def test_folder_newest_selection_uses_mtime_not_filename_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "newest-by-mtime"
    root.mkdir()
    selected_path = root / "a-export.csv"
    lexically_last = root / "z-export.csv"
    selected_path.write_text("id\n1\n", encoding="utf-8")
    lexically_last.write_text("id\n2\n", encoding="utf-8")
    selected_mtime = NOW.timestamp() - 60
    older_mtime = NOW.timestamp() - 300
    os.utime(selected_path, (selected_mtime, selected_mtime))
    os.utime(lexically_last, (older_mtime, older_mtime))
    store, service, snapshot = await _local_catalog_stack(
        tmp_path / "newest.sqlite3",
        root,
    )
    folder = next(
        resource
        for resource in snapshot.resources
        if resource.kind is ResourceKind.FOLDER
    )
    selected = next(
        resource
        for resource in snapshot.resources
        if resource.name == selected_path.name
    )
    try:
        inspection = await service.inspect_resource(AGENT_ID, folder.id)
        selection = _json_object(inspection["selection_facts"])
        newest_child_file = _json_object(selection["newest_child_file"])

        assert newest_child_file["status"] == "selected"
        assert newest_child_file["selected_resource_id"] == selected.id
        assert newest_child_file["ambiguity_reasons"] == ()
        assert newest_child_file["missing_freshness_resource_ids"] == ()
    finally:
        await store.close()


async def test_source_routing_and_model_catalog_views_exclude_detached_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "detached.sqlite3"
    store, service, view = await _catalog_stack(path)
    try:
        facts = await view.source_routing_facts(
            AGENT_ID,
            ("missing", "truthy_text", "write_access"),
        )
        assert tuple(fact.to_dict() for fact in facts) == (
            {
                "adapter_id": "sqlite",
                "configuration_flags": {
                    "missing": False,
                    "truthy_text": False,
                    "write_access": True,
                },
                "source_id": SOURCE_ID,
            },
        )
        assert "secret_reference" not in str(facts)

        await store.detach_source(
            AGENT_ID,
            SOURCE_ID,
            NOW + timedelta(minutes=1),
        )
        assert await view.source_routing_facts(AGENT_ID, ()) == ()
        result = await service.search(
            CatalogSearchRequest(agent_id=AGENT_ID, query="orders")
        )
        assert result.hits == ()
        assert result.total_matches == 0
        assert result.truncated is False
        with pytest.raises(CatalogResourceNotFoundError):
            await service.inspect_resource(AGENT_ID, RESOURCE_ID)
        with pytest.raises(CatalogResourceNotFoundError):
            await service.traverse(
                CatalogTraversalRequest(
                    agent_id=AGENT_ID,
                    from_resource_ids=(RESOURCE_ID,),
                    to_resource_ids=(RESOURCE_ID,),
                )
            )
        context = await service.catalog_context(AGENT_ID, "orders", limit=12)
        assert context["resources"] == ()
        assert context["total_matches"] == 0
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        reopened_service = CatalogService(reopened, reopened)
        reopened_view = CatalogDataView(reopened, reopened_service, reopened)
        assert await reopened_view.source_routing_facts(AGENT_ID, ()) == ()
        assert (
            await reopened_service.search(
                CatalogSearchRequest(agent_id=AGENT_ID, query="orders")
            )
        ).hits == ()
        with pytest.raises(CatalogResourceNotFoundError):
            await reopened_service.traverse(
                CatalogTraversalRequest(
                    agent_id=AGENT_ID,
                    from_resource_ids=(RESOURCE_ID,),
                    to_resource_ids=(RESOURCE_ID,),
                )
            )
    finally:
        await reopened.close()


def _execution_request(
    capability_id: str,
    executor_id: str,
    arguments: dict[str, object],
) -> ExecutionRequest:
    return ExecutionRequest(
        operation_id="operation-1",
        task_id=f"task-{capability_id}",
        turn_id="turn-1",
        capability_id=capability_id,
        executor_id=executor_id,
        attempt=1,
        fencing_token=1,
        arguments=arguments,
    )


async def test_catalog_executors_return_schema_valid_untrusted_evidence(
    tmp_path: Path,
) -> None:
    store, service, _ = await _catalog_stack(tmp_path / "agent.sqlite3")
    declarations = catalog_declarations(AGENT_ID, service)
    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=declarations.executors,
        tool_views=declarations.tool_views,
    )
    try:
        search_capability, search_executor = registry.resolve_execution(
            CATALOG_SEARCH_CAPABILITY_ID
        )
        search_candidate = await search_executor.execute(
            _execution_request(
                search_capability.id,
                search_executor.executor_id,
                {"query": "orders", "source_id": SOURCE_ID, "limit": 5},
            )
        )
        assert isinstance(search_candidate, EvidenceCandidate)
        assert search_candidate.kind == CATALOG_SEARCH_EVIDENCE_KIND
        assert (
            registry.validate_evidence(
                CATALOG_SEARCH_CAPABILITY_ID,
                search_candidate,
            )
            is search_candidate
        )
        assert search_candidate.payload["trust_classification"] == (
            "untrusted_external_data"
        )
        search_hits = search_candidate.payload["hits"]
        assert isinstance(search_hits, tuple)
        search_hit = search_hits[0]
        assert isinstance(search_hit, FrozenJsonObject)
        assert search_hit["resource_id"] == RESOURCE_ID
        assert search_candidate.artifact is None

        scoped_empty = await search_executor.execute(
            _execution_request(
                search_capability.id,
                search_executor.executor_id,
                {"query": "orders", "source_id": "other-source"},
            )
        )
        assert scoped_empty.payload["hits"] == ()

        other_agent_declarations = catalog_declarations("other-agent", service)
        other_search = other_agent_declarations.executors[0]
        other_search_candidate = await other_search.execute(
            _execution_request(
                CATALOG_SEARCH_CAPABILITY_ID,
                other_search.executor_id,
                {"query": "orders"},
            )
        )
        assert other_search_candidate.payload["hits"] == ()

        inspect_capability, inspect_executor = registry.resolve_execution(
            CATALOG_INSPECT_CAPABILITY_ID
        )
        inspect_candidate = await inspect_executor.execute(
            _execution_request(
                inspect_capability.id,
                inspect_executor.executor_id,
                {"resource_id": RESOURCE_ID},
            )
        )
        assert inspect_candidate.kind == CATALOG_INSPECT_EVIDENCE_KIND
        assert (
            registry.validate_evidence(
                CATALOG_INSPECT_CAPABILITY_ID,
                inspect_candidate,
            )
            is inspect_candidate
        )
        assert inspect_candidate.payload["trust_classification"] == (
            "untrusted_external_data"
        )
        inspected_resource = inspect_candidate.payload["resource"]
        assert isinstance(inspected_resource, FrozenJsonObject)
        assert inspected_resource["source_id"] == SOURCE_ID
        inspected_facets = inspect_candidate.payload["facets"]
        assert isinstance(inspected_facets, tuple)
        inspected_facet = inspected_facets[0]
        assert isinstance(inspected_facet, FrozenJsonObject)
        inspected_payload = inspected_facet["payload"]
        assert isinstance(inspected_payload, FrozenJsonObject)
        inspected_columns = inspected_payload["columns"]
        assert isinstance(inspected_columns, tuple)
        inspected_column = inspected_columns[1]
        assert isinstance(inspected_column, FrozenJsonObject)
        assert inspected_column["name"] == "customer_name"
        other_inspect = other_agent_declarations.executors[1]
        with pytest.raises(CatalogResourceNotFoundError):
            await other_inspect.execute(
                _execution_request(
                    CATALOG_INSPECT_CAPABILITY_ID,
                    other_inspect.executor_id,
                    {"resource_id": RESOURCE_ID},
                )
            )
    finally:
        await store.close()
