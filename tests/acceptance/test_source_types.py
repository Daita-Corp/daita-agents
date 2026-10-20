"""Source types registered at composition attach, catalog, query and degrade."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, AgentConfig, SQLiteSource
from daita.adapters.registry import (
    SourceBackendContext,
    SourceOperation,
    SourceType,
    SourceTypeRegistry,
)
from daita.domains.data.capabilities import (
    SqlReadResult,
)
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
from tests.support.memory_source import (
    MEMORY_ADAPTER_ID,
    MEMORY_TYPE,
    MemorySource,
    memory_from_registration,
)
from tests.support.source_contract import (
    assert_source_type_contract,
    assert_tabular_read_contract,
)
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


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


def _sql_memory_type(backend: _StubReadBackend) -> SourceType:
    def backends(context: SourceBackendContext) -> Mapping[SourceOperation, object]:
        del context
        return {SourceOperation.SQL_READ: backend}

    return SourceType(
        adapter_id=MEMORY_ADAPTER_ID,
        label="In-memory SQL table",
        from_registration=memory_from_registration,
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
    source_scope_ids: tuple[str, ...] = (),
) -> dict[str, ToolResultBlock]:
    provider.replace_script(script)
    result = await agent.run(
        "Use the connected data.", source_scope_ids=source_scope_ids
    )
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
        MEMORY_TYPE, MemorySource("contract"), secret_provider=EmptySecretProvider()
    )


@pytest.mark.contract
async def test_in_memory_tabular_reads_satisfy_the_tabular_read_contract(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "memory-tabular-contract",
        root=tmp_path,
        config=AgentConfig(source_types=(MEMORY_TYPE,)),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        source = await agent.attach(MemorySource("Regions"))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        await assert_tabular_read_contract(
            agent,
            source_id=source.id,
            resource_id=resource.id,
            columns=("region", "revenue"),
            key_column="region",
            range_column="revenue",
        )
    finally:
        await agent.close()


@pytest.mark.acceptance
async def test_type_with_only_tabular_reads_joins_a_sqlite_table_but_gets_no_sql(
    tmp_path: Path,
) -> None:
    database = tmp_path / "managers.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE managers (region TEXT, manager TEXT)")
        connection.executemany(
            "INSERT INTO managers VALUES (?, ?)",
            (("north", "Ada"), ("south", "Grace"), ("east", "Edsger")),
        )
    provider = ToolboxAwareMockModelProvider((), provider_id="mock:memory-join")
    agent = await Agent.create(
        "memory-join",
        root=tmp_path,
        config=AgentConfig(source_types=(MEMORY_TYPE,)),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        regions = await agent.attach(MemorySource("Regions"))
        managers = await agent.attach(SQLiteSource(database, name="Managers"))
        (regions_table,) = await agent.list_catalog_resources(source_id=regions.id)
        (managers_table,) = await agent.list_catalog_resources(source_id=managers.id)
        results = await _run(
            agent,
            provider,
            (
                _tool(
                    "single",
                    "data_query",
                    {
                        "source_id": regions.id,
                        "resource_ids": [regions_table.id],
                        "sql": "SELECT region, revenue FROM regions",
                    },
                ),
                _tool(
                    "join",
                    "data_query",
                    {
                        "inputs": [
                            {
                                "alias": "r",
                                "source_id": regions.id,
                                "resource_id": regions_table.id,
                                "columns": ["region", "revenue"],
                                "filters": [
                                    {"column": "revenue", "operator": "gte", "value": 5}
                                ],
                            },
                            {
                                "alias": "m",
                                "source_id": managers.id,
                                "resource_ids": [managers_table.id],
                                "sql": "SELECT region, manager FROM managers",
                            },
                        ],
                        "sql": "SELECT m.manager, r.revenue FROM r "
                        "JOIN m ON m.region = r.region ORDER BY m.manager",
                    },
                ),
                _stop("Ada and Edsger lead the 15 regions; Grace leads 7."),
            ),
        )
    finally:
        await agent.close()

    single = results["single"].output["error"]
    assert isinstance(single, Mapping)
    assert single["code"] == "sql_source_adapter_unsupported"
    assert not results["join"].is_error, results["join"].output
    data = cast(Mapping[str, object], results["join"].output["data"])
    assert [
        dict(row) for row in cast(tuple[Mapping[str, object], ...], data["rows"])
    ] == [
        {"manager": "Ada", "revenue": 15},
        {"manager": "Edsger", "revenue": 15},
        {"manager": "Grace", "revenue": 7},
    ]
    regions_input = cast(tuple[Mapping[str, object], ...], data["inputs"])[0]
    assert regions_input["source_id"] == regions.id
    assert regions_input["columns"] == ("region", "revenue")
    assert [dict(item) for item in regions_input["resource_revisions"]] == [  # type: ignore[attr-defined]
        {"resource_id": regions_table.id, "revision": regions_table.current_revision}
    ]


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
        source = await agent.attach(MemorySource("Regions"))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        results = await _run(
            agent,
            provider,
            (
                _tool("search", "catalog_search", {"query": "regions revenue"}),
                _tool("describe", "catalog_describe", {"resource_ids": [resource.id]}),
                _stop("The regions table has region and revenue."),
            ),
            # Only the SQL-less source: the live session folder answers SQL.
            source_scope_ids=(source.id,),
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
        source = await agent.attach(MemorySource("Regions"))
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
        memory = await agent.attach(MemorySource("Regions"))
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
