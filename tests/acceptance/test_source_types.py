"""Source types registered at composition attach, catalog, query and degrade."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from daita import Agent, AgentConfig, SQLiteSource
from daita.adapters.models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from daita.adapters.protocols import SourceClosedError, StaleResourceError
from daita.adapters.registry import (
    SourceBackendContext,
    SourceOperation,
    SourceType,
    SourceTypeRegistry,
)
from daita.catalog.models import (
    CatalogFacet,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from daita.domains.data.capabilities import SqlReadResult
from daita.domains.data.results import project_result_rows
from daita.hosting.errors import AgentHomeError
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import Transcript
from daita.security import EmptySecretProvider
from tests.support.source_contract import assert_source_type_contract
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
MEMORY_ADAPTER_ID = "memory-table"


def _snapshot(registration: SourceRegistration, sync_id: str) -> SourceCatalogSnapshot:
    resource_id = catalog_resource_id(registration.id, ResourceKind.TABLE, "regions")
    facet = CatalogFacet.from_tabular(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet=TabularFacet(
            columns=(
                TabularColumn(
                    name="region", native_type="text", ordinal=0, nullable=False
                ),
                TabularColumn(
                    name="revenue", native_type="integer", ordinal=1, nullable=False
                ),
            )
        ),
    )
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet_revisions=(facet.revision,),
        source_revision="memory:v1",
    )
    resource = CatalogResource.build(
        agent_id=registration.agent_id,
        source_id=registration.id,
        native_identity="regions",
        external_uri="memory://regions",
        kind=ResourceKind.TABLE,
        name="regions",
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )
    return SourceCatalogSnapshot(
        sync=CatalogSync(
            id=sync_id,
            agent_id=registration.agent_id,
            source_id=registration.id,
            adapter_id=MEMORY_ADAPTER_ID,
            status=CatalogSyncStatus.SUCCEEDED,
            started_at=NOW,
            completed_at=NOW,
            source_revision="memory:v1",
            resource_count=1,
        ),
        resources=(resource,),
        revisions=(revision,),
        facets=(facet,),
    )


class _MemoryAdapter:
    """One in-memory table; no SQL, no external I/O."""

    def __init__(self, registration: SourceRegistration) -> None:
        self._registration = registration
        self._latest: SourceCatalogSnapshot | None = None
        self._closed = False

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        if self._closed:
            raise SourceClosedError(self._registration.id)
        self._latest = _snapshot(self._registration, request.sync_id)
        return DiscoveryResult(request=request, snapshot=self._latest, completed_at=NOW)

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        if self._closed:
            raise SourceClosedError(self._registration.id)
        assert self._latest is not None
        current = self._latest.resources[0]
        if (
            resource.revision is not None
            and resource.revision != current.current_revision
        ):
            raise StaleResourceError(self._registration.id, resource.resource_id)
        return ResourceSnapshot(
            reference=resource,
            resource=current,
            revision=self._latest.revisions[0],
            facets=self._latest.facets,
            relationships=(),
            inspected_at=NOW,
            source_revision="memory:v1",
        )

    async def health(self) -> SourceHealth:
        return SourceHealth(
            agent_id=self._registration.agent_id,
            source_id=self._registration.id,
            adapter_id=MEMORY_ADAPTER_ID,
            healthy=not self._closed,
            checked_at=NOW,
        )

    async def close(self) -> None:
        self._closed = True


@dataclass(frozen=True, slots=True)
class _MemorySource:
    name: str

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> _MemoryAdapter:
        del clock
        return _MemoryAdapter(
            SourceRegistration.build(
                agent_id=agent_id,
                adapter_id=MEMORY_ADAPTER_ID,
                native_identity=f"memory:{self.name}",
                display_name=self.name,
                configuration={"name": self.name},
                attached_at=attached_at,
            )
        )


def _memory_from_registration(
    registration: SourceRegistration, *, secret_provider: object
) -> _MemorySource:
    del secret_provider
    return _MemorySource(str(registration.configuration["name"]))


class _StubReadBackend:
    """Answer one SQL read from fixed rows, recording what was routed here."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    async def execute_read(self, **request: object) -> SqlReadResult:
        self.calls.append(request)
        revisions = request["expected_resource_revisions"]
        assert isinstance(revisions, tuple)
        return SqlReadResult(
            source_id=str(request["source_id"]),
            canonical_sql=str(request["sql"]),
            sql_fingerprint="sha256:" + "1" * 64,
            resource_ids=tuple(resource_id for resource_id, _ in revisions),
            resource_revisions=revisions,
            source_revision="memory:v1",
            columns=("region", "revenue"),
            projection=project_result_rows(
                ({"region": "north", "revenue": 15},), max_rows=100, max_bytes=16_384
            ),
        )


MEMORY_TYPE = SourceType(
    adapter_id=MEMORY_ADAPTER_ID,
    label="In-memory table",
    from_registration=_memory_from_registration,
)


def _sql_memory_type(backend: _StubReadBackend) -> SourceType:
    def backends(context: SourceBackendContext) -> Mapping[SourceOperation, object]:
        del context
        return {SourceOperation.SQL_READ: backend}

    return SourceType(
        adapter_id=MEMORY_ADAPTER_ID,
        label="In-memory SQL table",
        from_registration=_memory_from_registration,
        sql_dialect="sqlite",
        operations=frozenset({SourceOperation.SQL_READ}),
        create_backends=backends,
    )


def _tool(call_id: str, name: str, arguments: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _profile(provider: ToolboxAwareMockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )


def _results(transcript: Transcript) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in transcript.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _sqlite(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("INSERT INTO customers VALUES (1, 'Ada')")
    return path


async def _run(
    agent: Agent,
    provider: ToolboxAwareMockModelProvider,
    script: tuple[ModelResponse, ...],
) -> dict[str, ToolResultBlock]:
    provider.replace_script(script)
    result = await agent.run("Use the connected data.")
    return _results(await agent.transcript(result.run_id))


@pytest.mark.unit
def test_registry_rejects_a_duplicate_adapter_id() -> None:
    registry = SourceTypeRegistry((MEMORY_TYPE,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(MEMORY_TYPE)
    with pytest.raises(ValueError, match="already registered"):
        SourceTypeRegistry((MEMORY_TYPE, MEMORY_TYPE))
    with pytest.raises(ValueError, match="already registered"):
        SourceTypeRegistry((registry.require("sqlite"),))


@pytest.mark.contract
async def test_in_memory_source_type_satisfies_the_source_contract() -> None:
    await assert_source_type_contract(
        MEMORY_TYPE, _MemorySource("contract"), secret_provider=EmptySecretProvider()
    )


@pytest.mark.acceptance
async def test_registered_type_without_sql_attaches_catalogs_reopens_and_detaches(
    tmp_path: Path,
) -> None:
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:memory-type")
    config = AgentConfig(source_types=(MEMORY_TYPE,))
    agent = await Agent.create(
        "memory-type",
        root=tmp_path,
        config=config,
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(_MemorySource("Regions"))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        results = await _run(
            agent,
            provider,
            (
                _tool("search", "catalog_search", {"query": "regions revenue"}),
                _tool("describe", "catalog_describe", {"resource_ids": [resource.id]}),
                _stop("The regions table has region and revenue."),
            ),
        )
        assert not results["search"].is_error
        assert resource.id in repr(results["search"].output)
        assert not results["describe"].is_error
        assert "revenue" in repr(results["describe"].output)
        assert "data_query" not in {tool.name for tool in provider.requests[0].tools}
    finally:
        await agent.close()

    agent = await Agent.open(
        "memory-type",
        root=tmp_path,
        config=config,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert await agent.list_sources() == (source,)
        assert await agent.unavailable_sources() == ()
        assert await agent.refresh_source(source.id) == source
        detached = await agent.detach(source.id)
        assert detached.id == source.id and not detached.active
    finally:
        await agent.close()


@pytest.mark.acceptance
async def test_registered_sql_type_gets_data_query_routed_to_its_backend(
    tmp_path: Path,
) -> None:
    backend = _StubReadBackend()
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:memory-sql")
    agent = await Agent.create(
        "memory-sql",
        root=tmp_path,
        config=AgentConfig(source_types=(_sql_memory_type(backend),)),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(_MemorySource("Regions"))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        results = await _run(
            agent,
            provider,
            (
                _tool(
                    "query",
                    "data_query",
                    {
                        "source_id": source.id,
                        "resource_ids": [resource.id],
                        "sql": "SELECT region, revenue FROM regions",
                    },
                ),
                _stop("North revenue is 15."),
            ),
        )
        assert "data_query" in {tool.name for tool in provider.requests[0].tools}
        assert not results["query"].is_error
        assert "north" in repr(results["query"].output)
        assert [call["source_id"] for call in backend.calls] == [source.id]
    finally:
        await agent.close()


@pytest.mark.acceptance
async def test_unregistered_type_in_a_stored_home_degrades_without_breaking_sqlite(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "unknown-type",
        root=tmp_path,
        config=AgentConfig(source_types=(MEMORY_TYPE,)),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        memory = await agent.attach(_MemorySource("Regions"))
        customers = await agent.attach(
            SQLiteSource(_sqlite(tmp_path / "customers.sqlite"), name="Customers")
        )
    finally:
        await agent.close()

    provider = ToolboxAwareMockModelProvider((), provider_id="mock:unknown-type")
    agent = await Agent.open(
        "unknown-type",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        assert {item.id for item in await agent.list_sources()} == {
            memory.id,
            customers.id,
        }
        assert await agent.unavailable_sources() == (memory,)
        with pytest.raises(AgentHomeError, match="not registered"):
            await agent.refresh_source(memory.id)
        assert (await agent.refresh_source(customers.id)).id == customers.id

        (table,) = await agent.list_catalog_resources(source_id=customers.id)
        (memory_table,) = await agent.list_catalog_resources(source_id=memory.id)
        results = await _run(
            agent,
            provider,
            (
                _tool(
                    "customers",
                    "data_query",
                    {
                        "source_id": customers.id,
                        "resource_ids": [table.id],
                        "sql": "SELECT name FROM customers",
                    },
                ),
                _tool(
                    "regions",
                    "data_query",
                    {
                        "source_id": memory.id,
                        "resource_ids": [memory_table.id],
                        "sql": "SELECT region FROM regions",
                    },
                ),
                _stop("Ada is the only customer."),
            ),
        )
        assert not results["customers"].is_error
        assert "Ada" in repr(results["customers"].output)
        assert results["regions"].is_error
        error = results["regions"].output["error"]
        assert isinstance(error, Mapping) and error["code"] == "source_scope_violation"
    finally:
        await agent.close()
