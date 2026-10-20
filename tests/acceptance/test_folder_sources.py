"""Folder sources join databases, detect file changes and never outlive a session."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, FolderSource, SQLiteSource
from daita.adapters.folder import FolderSourceError
from daita.adapters.registry import SourceOperation, UnknownSourceTypeError
from daita.catalog import CatalogTraversalRequest
from daita.catalog.models import ResourceKind
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import Transcript
from tests.catalog._schema_support import _mapping_sequence, _schema
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance


def _call(call_id: str, name: str, arguments: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _result(transcript: Transcript, call_id: str) -> ToolResultBlock:
    return next(
        block
        for message in transcript.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.call_id == call_id
    )


async def _agent(
    tmp_path: Path, name: str
) -> tuple[Agent, ToolboxAwareMockModelProvider]:
    provider = ToolboxAwareMockModelProvider((), provider_id=f"mock:{name}")
    state = tmp_path / "state"
    agent = await Agent.create(
        name,
        root=state,
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        workspace=workspace_for(state),
    )
    return agent, provider


async def test_customer_file_links_to_database_orders_and_joins_in_one_query(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "files"
    folder.mkdir()
    customers_file = folder / "customers.csv"
    customers_file.write_text(
        "id,region\n1,north\n2,south\n3,north\n", encoding="utf-8"
    )
    database = tmp_path / "orders.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, "
            "total INTEGER)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            ((1, 1, 100), (2, 2, 40), (3, 3, 25), (4, 1, 15)),
        )
    agent, provider = await _agent(tmp_path, "folder-join")
    try:
        files = await agent.attach(FolderSource(folder, name="Customer files"))
        orders_source = await agent.attach(SQLiteSource(database))
        (customers,) = [
            item
            for item in await agent.list_catalog_resources(source_id=files.id)
            if item.kind is ResourceKind.FILE
        ]
        (orders,) = await agent.list_catalog_resources(source_id=orders_source.id)
        link = await agent.declare_catalog_link(
            orders.id, customers.id, (("customer_id", "id"),)
        )

        traversal = await agent._embedded._data_view.traverse(
            CatalogTraversalRequest(
                agent_id=agent.id,
                from_resource_ids=(orders.id,),
                to_resource_ids=(customers.id,),
            )
        )
        assert traversal["reachable"] is True
        schema = await _schema(agent, resource_ids=(orders.id, customers.id), limit=4)
        (join_path,) = _mapping_sequence(schema["paths"])
        assert join_path["relationship_ids"] == (link.id,)

        combined = {
            "inputs": [
                {
                    "alias": "o",
                    "source_id": orders_source.id,
                    "resource_ids": [orders.id],
                    "sql": "SELECT customer_id, SUM(total) AS revenue "
                    "FROM orders GROUP BY customer_id",
                },
                {
                    "alias": "c",
                    "source_id": files.id,
                    "resource_ids": [customers.id],
                    "sql": "SELECT id, region FROM data",
                },
            ],
            "sql": "SELECT c.region, SUM(o.revenue) AS revenue FROM o "
            "JOIN c ON c.id = o.customer_id GROUP BY c.region ORDER BY c.region",
        }
        provider.replace_script(
            (_call("revenue", "data_query", combined), _stop("North 140, south 40."))
        )
        joined = await agent.run("What is revenue by customer region?")
        data = cast(
            Mapping[str, object],
            _result(await agent.transcript(joined.run_id), "revenue").output["data"],
        )
        assert [
            dict(row) for row in cast(tuple[Mapping[str, object], ...], data["rows"])
        ] == [{"region": "north", "revenue": 140}, {"region": "south", "revenue": 40}]

        # A changed file no longer matches its cataloged revision.
        customers_file.write_text(
            "id,region\n1,west\n2,south\n3,north\n", encoding="utf-8"
        )
        provider.replace_script(
            (_call("again", "data_query", combined), _stop("The file changed."))
        )
        changed = await agent.run("Recompute revenue by customer region.")
        stale = _result(await agent.transcript(changed.run_id), "again")
        assert stale.is_error
        assert cast(Mapping[str, object], stale.output["error"])["code"] == (
            "query_binding_stale"
        )

        await agent.refresh_source(files.id)
        (refreshed,) = [
            item
            for item in await agent.list_catalog_resources(source_id=files.id)
            if item.kind is ResourceKind.FILE
        ]
        assert refreshed.current_revision != customers.current_revision
        backend = agent._embedded._source_backends[SourceOperation.SQL_READ]["folder"]
        with pytest.raises(FolderSourceError) as pinned:
            await backend.execute_read(
                agent_id=agent.id,
                source_id=files.id,
                sql="SELECT id, region FROM data",
                parameters=(),
                max_rows=100,
                max_bytes=16_384,
                expected_resource_revisions=(
                    (customers.id, customers.current_revision),
                ),
            )
        assert pinned.value.error_code == "query_binding_stale"
    finally:
        await agent.close()


async def test_session_folder_source_is_never_stored_across_reopen(
    tmp_path: Path,
) -> None:
    agent, provider = await _agent(tmp_path, "session-folder")
    session_source_id = agent.session_source_id
    assert session_source_id is not None
    (agent.workspace.root / "notes.txt").write_text("session note", encoding="utf-8")
    provider.replace_script(
        (
            _call(
                "read",
                "file_read",
                {"source_id": session_source_id, "path": "notes.txt"},
            ),
            _stop("read"),
        )
    )
    try:
        result = await agent.run("Read my notes.")
        read = _result(await agent.transcript(result.run_id), "read")
        assert not read.is_error
        assert await agent.list_sources() == ()
    finally:
        await agent.close()

    reopened = await Agent.open(
        "session-folder",
        root=tmp_path / "state",
        workspace=workspace_for(tmp_path / "state"),
    )
    try:
        assert await reopened.list_sources() == ()
        assert await reopened._embedded._store.list_sources(reopened.id) == ()
        # The same workspace yields the same live identity, never a stored one.
        assert reopened.session_source_id == session_source_id
    finally:
        await reopened.close()


async def test_hosted_composition_has_no_folder_type_or_session_source(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("hosted-folder", root=tmp_path, hosted=True)
    try:
        assert agent.session_source_id is None
        assert agent._embedded._source_types.get("folder") is None
        assert not {"file_search", "file_read"} & set(
            agent._embedded._capabilities.tool_names
        )
        folder = tmp_path.parent / f"{tmp_path.name}-files"
        folder.mkdir()
        with pytest.raises(UnknownSourceTypeError):
            await agent.attach(FolderSource(folder))
    finally:
        await agent.close()
