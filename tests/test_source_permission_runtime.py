from __future__ import annotations

from _workspace_support import workspace_for

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from _capability_runtime_support import execute_projected

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.catalog import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SCHEMA_CAPABILITY_ID,
    CATALOG_SEARCH_CAPABILITY_ID,
    CATALOG_TRAVERSE_CAPABILITY_ID,
    CatalogSchemaRequest,
    CatalogSearchRequest,
    CatalogTraversalRequest,
)
from daita.catalog.capabilities import CatalogProjection, catalog_declarations
from daita.catalog.models import CatalogResource
from daita.catalog.protocols import CatalogResourceNotFoundError
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from daita.storage.sqlite_codecs import encode_source_read_scope
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope


def _database(path: Path, *, future: bool = False) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE parent (id INTEGER PRIMARY KEY, label TEXT);
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER REFERENCES parent(id),
                label TEXT
            );
            CREATE TABLE public_tail (id INTEGER PRIMARY KEY, label TEXT);
            """)
        if future:
            connection.execute(
                "CREATE TABLE future_resource (id INTEGER PRIMARY KEY, label TEXT)"
            )


async def _agent_with_scope(
    tmp_path: Path,
    *,
    mode: SourceReadMode,
    selected_names: tuple[str, ...] = (),
) -> tuple[Agent, Path, str, dict[str, CatalogResource]]:
    database = tmp_path / f"{mode.value}.sqlite"
    _database(database)
    agent = await Agent.create(
        f"permission-runtime-{mode.value}",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
    )
    registration = await agent.attach(SQLiteSource(database))
    resources = {
        resource.name: resource
        for resource in await agent._embedded._store.list_resources(
            agent.id,
            registration.id,
        )
    }
    scope = (
        SourceReadScope(
            agent_id=agent.id,
            source_id=registration.id,
            mode=mode,
            resource_ids=tuple(resources[name].id for name in selected_names),
        )
        if mode is SourceReadMode.SELECTED
        else SourceReadScope(
            agent_id=agent.id,
            source_id=registration.id,
            mode=mode,
        )
    )
    await agent._embedded._store.replace_source_permission_scopes(scope, ())
    return agent, database, registration.id, resources


async def test_read_scope_all_selected_none_and_refresh_semantics(
    tmp_path: Path,
) -> None:
    all_agent, all_path, all_source_id, all_resources = await _agent_with_scope(
        tmp_path,
        mode=SourceReadMode.ALL,
    )
    selected_agent, selected_path, selected_source_id, selected_resources = (
        await _agent_with_scope(
            tmp_path,
            mode=SourceReadMode.SELECTED,
            selected_names=("parent",),
        )
    )
    none_agent, _none_path, none_source_id, _none_resources = await _agent_with_scope(
        tmp_path,
        mode=SourceReadMode.NONE,
    )
    try:
        assert await all_agent._embedded._data_view.readable_resource_ids(
            all_agent.id,
            (all_source_id,),
        ) == frozenset(resource.id for resource in all_resources.values())
        assert await selected_agent._embedded._data_view.readable_resource_ids(
            selected_agent.id,
            (selected_source_id,),
        ) == frozenset({selected_resources["parent"].id})
        assert not await none_agent._embedded._data_view.readable_resource_ids(
            none_agent.id,
            (none_source_id,),
        )

        with sqlite3.connect(all_path) as connection:
            connection.execute(
                "CREATE TABLE future_resource (id INTEGER PRIMARY KEY, label TEXT)"
            )
        with sqlite3.connect(selected_path) as connection:
            connection.execute(
                "CREATE TABLE future_resource (id INTEGER PRIMARY KEY, label TEXT)"
            )
        await all_agent.refresh_source(all_source_id)
        await selected_agent.refresh_source(selected_source_id)
        all_current = await all_agent._embedded._store.list_resources(
            all_agent.id,
            all_source_id,
        )
        selected_current = await selected_agent._embedded._store.list_resources(
            selected_agent.id,
            selected_source_id,
        )
        all_readable = await all_agent._embedded._data_view.readable_resource_ids(
            all_agent.id,
            (all_source_id,),
        )
        selected_readable = (
            await selected_agent._embedded._data_view.readable_resource_ids(
                selected_agent.id,
                (selected_source_id,),
            )
        )
        assert (
            next(item.id for item in all_current if item.name == "future_resource")
            in all_readable
        )
        assert (
            next(item.id for item in selected_current if item.name == "future_resource")
            not in selected_readable
        )
        assert len(all_current) == 4
        assert len(selected_current) == 4
    finally:
        await all_agent.close()
        await selected_agent.close()
        await none_agent.close()


async def test_model_catalog_surfaces_filter_before_limits_totals_and_graphs(
    tmp_path: Path,
) -> None:
    agent, _path, source_id, resources = await _agent_with_scope(
        tmp_path,
        mode=SourceReadMode.SELECTED,
        selected_names=("parent", "public_tail"),
    )
    try:
        view = agent._embedded._data_view
        context = await view.catalog_context(
            agent.id,
            "table",
            limit=1,
            source_ids=(source_id,),
        )
        context_resources = cast(tuple[FrozenJsonObject, ...], context["resources"])
        assert context["total_matches"] == 2
        assert context["returned_count"] == 1
        assert context["truncated"] is True
        assert context_resources[0]["resource_id"] in {
            resources["parent"].id,
            resources["public_tail"].id,
        }

        search = await view.search(
            CatalogSearchRequest(agent_id=agent.id, query="table", limit=1)
        )
        assert search.total_matches == 2
        assert search.returned_count == 1
        assert search.truncated is True
        assert len(search.hits) == 1

        with pytest.raises(CatalogResourceNotFoundError):
            await view.catalog_context(
                agent.id,
                "parent",
                resource_ids=(resources["child"].id,),
                limit=12,
            )

        schema = await view.schema_slice(
            CatalogSchemaRequest(
                agent_id=agent.id,
                query="table",
                limit=10,
                include_relationships=True,
            )
        )
        schema_resources = cast(tuple[FrozenJsonObject, ...], schema["resources"])
        assert {item["resource_id"] for item in schema_resources} == {
            resources["parent"].id,
            resources["public_tail"].id,
        }
        assert schema["total_matches"] == 2
        assert schema["relationships"] == ()

        inspected = await view.inspect_resource(agent.id, resources["parent"].id)
        assert inspected["incident_relationships"] == ()
        assert inspected["neighbors"] == ()

        traversed = await view.traverse(
            CatalogTraversalRequest(
                agent_id=agent.id,
                from_resource_ids=(resources["parent"].id,),
                to_resource_ids=(resources["public_tail"].id,),
            )
        )
        assert traversed["reachable"] is False
        assert traversed["visited_edges"] == 0

        with pytest.raises(Exception):
            await view.inspect_resource(agent.id, resources["child"].id)
        with pytest.raises(Exception):
            await view.traverse(
                CatalogTraversalRequest(
                    agent_id=agent.id,
                    from_resource_ids=(resources["parent"].id,),
                    to_resource_ids=(resources["child"].id,),
                )
            )
    finally:
        await agent.close()


async def test_initial_model_context_contains_only_readable_resources(
    tmp_path: Path,
) -> None:
    agent, _path, source_id, resources = await _agent_with_scope(
        tmp_path,
        mode=SourceReadMode.SELECTED,
        selected_names=("parent",),
    )
    try:
        context = await agent._embedded._data_view.catalog_context(
            agent.id,
            "tables parent child",
            limit=12,
            source_ids=(source_id,),
        )
        context_resources = cast(tuple[FrozenJsonObject, ...], context["resources"])
        assert tuple(item["resource_id"] for item in context_resources) == (
            resources["parent"].id,
        )
        assert resources["child"].id not in str(context)
    finally:
        await agent.close()


async def test_runtime_denies_guessed_and_multi_resource_reads_before_io(
    tmp_path: Path,
) -> None:
    agent, _path, source_id, resources = await _agent_with_scope(
        tmp_path,
        mode=SourceReadMode.SELECTED,
        selected_names=("parent",),
    )

    runtime = agent._embedded._capability_runtime
    run = RunInput(
        id="permission-runtime-run",
        agent_id=agent.id,
        message="read data",
        created_at=agent._embedded.identity.created_at,
    )
    try:
        calls = (
            ToolCall(
                id="denied-name",
                name="data_query_sqlite",
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT * FROM child",
                },
            ),
            ToolCall(
                id="missing-name",
                name="data_query_sqlite",
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT * FROM guessed_secret",
                },
            ),
            ToolCall(
                id="mixed-read",
                name="data_query_sqlite",
                arguments={
                    "source_id": source_id,
                    "sql": "SELECT * FROM parent JOIN child ON parent.id = child.parent_id",
                },
            ),
        )
        results = await execute_projected(
            runtime,
            run,
            calls,
        )
        errors = tuple(
            cast(Mapping[str, object], result.output["error"]) for result in results
        )
        assert tuple(error["code"] for error in errors) == (
            "resource_read_not_allowed",
            "resource_read_not_allowed",
            "resource_read_not_allowed",
        )
        assert all(not error["details"] for error in errors)
        assert resources["child"].id not in str(
            tuple(result.output for result in results)
        )
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "mutation",
    ("missing", "corrupt", "foreign", "undecodable"),
)
async def test_invalid_scope_state_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    agent, _path, source_id, resources = await _agent_with_scope(
        tmp_path,
        mode=SourceReadMode.ALL,
    )
    state_path = agent.home / "state.db"
    try:
        with sqlite3.connect(state_path) as connection:
            if mutation == "missing":
                connection.execute(
                    "DELETE FROM source_read_scopes WHERE agent_id = ? AND source_id = ?",
                    (agent.id, source_id),
                )
            elif mutation == "corrupt":
                connection.execute(
                    "UPDATE source_read_scopes SET data = ? WHERE agent_id = ? AND source_id = ?",
                    ("{", agent.id, source_id),
                )
            elif mutation == "undecodable":
                connection.execute(
                    "UPDATE source_read_scopes SET data = ? WHERE agent_id = ? AND source_id = ?",
                    (
                        '{"type":"SourceReadScope","fields":{"version":99,"mode":"all","resource_ids":[]}}',
                        agent.id,
                        source_id,
                    ),
                )
            else:
                foreign_path = tmp_path / "foreign.sqlite"
                _database(foreign_path)
                other = await agent.attach(SQLiteSource(foreign_path, name="Foreign"))
                other_resources = await agent._embedded._store.list_resources(
                    agent.id,
                    other.id,
                )
                foreign_scope = SourceReadScope(
                    agent_id=agent.id,
                    source_id=source_id,
                    mode=SourceReadMode.SELECTED,
                    resource_ids=(other_resources[0].id,),
                )
                connection.execute(
                    "UPDATE source_read_scopes SET data = ? WHERE agent_id = ? AND source_id = ?",
                    (encode_source_read_scope(foreign_scope), agent.id, source_id),
                )
        with pytest.raises(RuntimeError):
            await agent._embedded._data_view.readable_resource_ids(
                agent.id,
                (source_id,),
            )
        assert resources
    finally:
        await agent.close()


def test_model_catalog_registry_exposes_no_permission_mutation_capability() -> None:
    class Projection:
        pass

    declarations = catalog_declarations(
        "agent-permissions",
        cast(CatalogProjection, Projection()),
    )
    assert {capability.id for capability in declarations.capabilities} == {
        CATALOG_SEARCH_CAPABILITY_ID,
        CATALOG_SCHEMA_CAPABILITY_ID,
        CATALOG_INSPECT_CAPABILITY_ID,
        CATALOG_TRAVERSE_CAPABILITY_ID,
    }
    assert not any(
        "permission" in capability.id for capability in declarations.capabilities
    )
