from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import CapabilityRegistry, EvidenceCandidate, ExecutionRequest
from daita.catalog import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_INSPECT_EVIDENCE_KIND,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_SEARCH_EVIDENCE_KIND,
    CatalogFacet,
    CatalogResource,
    CatalogResourceNotFoundError,
    CatalogResourceRevision,
    CatalogSearchRequest,
    CatalogService,
    CatalogSync,
    CatalogSyncStatus,
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
SOURCE_ID = "source-1"
RESOURCE_ID = catalog_resource_id(SOURCE_ID, ResourceKind.TABLE, "main.orders")


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
    await store.commit_snapshot(
        _snapshot("sync-1", columns=("id", "legacy_name"), offset=1)
    )
    await store.commit_snapshot(
        _snapshot("sync-2", columns=("id", "customer_name"), offset=3)
    )
    service = CatalogService(store)
    return store, service, CatalogDataView(store, service)


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
