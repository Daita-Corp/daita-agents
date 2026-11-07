"""A SQLite table has one identity per name, and upgrades carry what refers to it."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, SQLiteSource
from daita.adapters import sqlite as sqlite_adapter
from daita.catalog.models import ResourceKind, catalog_resource_id
from daita.llm.models import ToolCall
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from daita.semantics import (
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticSubject,
)
from daita.storage.home_migrations import revision_0004
from daita.storage.sqlite_codecs import (
    decode_catalog_relationship,
    encode_catalog_snapshot,
)
from tests.support.capability_runtime import execute_projected
from tests.support.catalog_records import Table, entries, snapshot
from tests.support.home_revisions import restore_catalog_snapshots
from tests.support.workspace import workspace_for

_NOW = datetime(2026, 11, 2, tzinfo=UTC)


async def _agent(root: Path, *, create: bool) -> Agent:
    provider = MockModelProvider(())
    options = {
        "root": root,
        "model": provider,
        "model_profile": provider.model_profile,
        "workspace": workspace_for(root),
    }
    if create:
        return await Agent.create("folding", **options)  # type: ignore[arg-type]
    return await Agent.open("folding", **options)  # type: ignore[arg-type]


def _database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE Companies (id INTEGER PRIMARY KEY, Name TEXT)")
        connection.execute(
            "CREATE TABLE orders (id INTEGER, company INTEGER REFERENCES Companies(id))"
        )
        connection.execute("INSERT INTO Companies VALUES (1, 'Acme')")
    return path


async def _old_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: Path
) -> tuple[str, str]:
    """A home whose catalog hashes declared case, as released homes do."""

    with monkeypatch.context() as patch:
        patch.setattr(sqlite_adapter, "_native_identity", lambda name: f"main.{name}")
        agent = await _agent(tmp_path, create=True)
        try:
            source = await agent.attach(SQLiteSource(database))
            resources = await agent.list_catalog_resources(source_id=source.id)
            companies = next(item for item in resources if item.name == "Companies")
            assert companies.native_identity == "main.Companies"
            preview = await agent.preview_source_permissions(
                source_id=source.id,
                read_mode="selected",
                read_resource_ids=(companies.id,),
                relational_write_scopes={},
            )
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
            await agent._embedded._store.save_semantic_annotation(
                agent.id,
                SemanticAnnotation(
                    id="annotation-companies",
                    agent_id=agent.id,
                    subject=SemanticSubject(
                        source_ids=(source.id,),
                        resource_ids=(companies.id,),
                        fields=(SemanticFieldReference(companies.id, "Name"),),
                    ),
                    kind=SemanticKind.METRIC_DEFINITION,
                    statement="Name is the legal company name.",
                    evidence=(
                        SemanticEvidence(
                            SemanticEvidenceKind.USER_ASSERTION,
                            "run-seed",
                            message_position=0,
                        ),
                    ),
                    catalog_revisions=(
                        ResourceRevisionBinding(
                            companies.id, companies.current_revision
                        ),
                    ),
                    created_at=_NOW,
                    confirmed_at=_NOW,
                    confirmed_by="local-user",
                ),
            )
            return agent.id, source.id
        finally:
            await agent.close()


def _fold(root: Path) -> None:
    home = next(root.rglob("state.db"))
    with sqlite3.connect(home) as connection:
        # The fold reads the snapshots table an older home still has.
        restore_catalog_snapshots(connection, truncated_flag=True)
        revision_0004._fold_sqlite_identities(connection)
        revision_0004._convert_catalog_snapshots(connection)
        connection.commit()


async def _query(agent: Agent, source_id: str, sql: str) -> Mapping[str, object]:
    loop = agent._embedded._loop
    assert loop is not None
    outcome = await execute_projected(
        cast(object, loop._tools),
        RunInput(
            id="folding-run",
            agent_id=agent.id,
            message="read",
            created_at=_NOW,
            conversation_id="folding",
            source_scope_ids=(source_id,),
        ),
        (
            ToolCall(
                id="read",
                name="data_query",
                arguments={"source_id": source_id, "sql": sql},
            ),
        ),
    )
    return cast(Mapping[str, object], outcome.ordered_results[0].output)


async def test_an_upgraded_home_keeps_scopes_annotations_and_its_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = _database(tmp_path / "shop.sqlite")
    _agent_id, source_id = await _old_home(tmp_path, monkeypatch, database)

    _fold(tmp_path)

    agent = await _agent(tmp_path, create=False)
    try:
        resources = await agent.list_catalog_resources(source_id=source_id)
        companies = next(item for item in resources if item.name == "Companies")
        assert companies.native_identity == "main.companies"
        assert companies.id == catalog_resource_id(
            source_id, ResourceKind.TABLE, "main.companies"
        )
        scope = await agent._embedded._store.load_source_read_scope(agent.id, source_id)
        assert scope is not None and scope.resource_ids == (companies.id,)

        # The stored SELECTED scope still admits its table, however it is spelled,
        # and nothing else.
        for sql in ("SELECT * FROM Companies", "SELECT * FROM COMPANIES"):
            assert "error" not in await _query(agent, source_id, sql)
        refused = await _query(agent, source_id, "SELECT * FROM orders")
        error = cast(Mapping[str, object], refused["error"])
        assert error["code"] == "resource_read_not_allowed"

        (view,) = await agent._embedded._semantic_views()
        assert view.annotation.subject.resource_ids == (companies.id,)
        assert view.annotation.subject.fields[0].resource_id == companies.id
        assert view.state is not SemanticAnnotationState.STALE

        # Rediscovery computes the same hashes, so the chain was rebuilt correctly.
        await agent.refresh_source(source_id)
        refreshed = await agent.list_catalog_resources(source_id=source_id)
        assert {item.id: item.current_revision for item in refreshed} == {
            item.id: item.current_revision for item in resources
        }
        (view,) = await agent._embedded._semantic_views()
        assert view.state is not SemanticAnnotationState.STALE
    finally:
        await agent.close()


async def test_a_lower_case_catalog_is_left_exactly_as_it_was(tmp_path: Path) -> None:
    database = tmp_path / "plain.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER)")
    agent = await _agent(tmp_path, create=True)
    try:
        await agent.attach(SQLiteSource(database))
    finally:
        await agent.close()
    home = next(tmp_path.rglob("state.db"))
    query = "SELECT id, revision, native_identity FROM catalog_resources ORDER BY id"
    with sqlite3.connect(home) as connection:
        before = connection.execute(query).fetchall()
    assert before

    _fold(tmp_path)

    with sqlite3.connect(home) as connection:
        assert connection.execute(query).fetchall() == before


def test_tables_differing_only_by_case_fail_closed() -> None:
    def resource(native: str) -> dict[str, object]:
        return {
            "fields": {
                "id": "catalog-resource:sha256:" + native,
                "kind": {"value": "table"},
                "native_identity": native,
            }
        }

    fields = {
        "resources": [resource("main.Orders"), resource("main.orders")],
        "facets": [],
        "relationships": [],
        "revisions": [],
    }

    with pytest.raises(ValueError, match="differing only by case"):
        revision_0004._folded_snapshot("source", json.loads(json.dumps(fields)))


def test_folded_identities_land_in_the_per_resource_rows() -> None:
    mixed = snapshot(
        "agent",
        "source:shop",
        "sync-1",
        (
            Table("main.Customers", ("id", "Name")),
            Table("main.Orders", ("id", "customer_id")),
        ),
        (("main.Orders", "customer_id", "main.Customers", "id"),),
        folder=None,
    )
    lower = snapshot(
        "agent",
        "source:shop",
        "sync-1",
        (
            Table("main.customers", ("id", "Name")),
            Table("main.orders", ("id", "customer_id")),
        ),
        (("main.orders", "customer_id", "main.customers", "id"),),
        folder=None,
    )
    stored = json.loads(encode_catalog_snapshot(mixed))
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE snapshots (agent_id TEXT, source_id TEXT, sync_id TEXT, data TEXT);"
        "CREATE TABLE sources (agent_id TEXT, id TEXT, data TEXT);"
        "CREATE TABLE source_read_scopes (agent_id TEXT, source_id TEXT, data TEXT);"
        "CREATE TABLE semantic_annotations (agent_id TEXT, id TEXT, data TEXT);"
    )
    connection.execute(
        "INSERT INTO snapshots VALUES ('agent', 'source:shop', 'sync-1', ?)",
        (json.dumps(stored),),
    )
    connection.execute(
        "INSERT INTO sources VALUES ('agent', 'source:shop', ?)",
        ('{"fields":{"adapter_id":"sqlite"}}',),
    )

    revision_0004._fold_sqlite_identities(connection)
    revision_0004._convert_catalog_snapshots(connection)

    assert {
        row[0]: (row[1], row[2])
        for row in connection.execute(
            "SELECT id, revision, native_identity FROM catalog_resources"
        )
    } == {
        item.resource.id: (
            item.resource.current_revision,
            item.resource.native_identity,
        )
        for item in entries(lower)
    }
    assert {
        row[0]: decode_catalog_relationship(row[1])
        for row in connection.execute("SELECT id, data FROM catalog_relationships")
    } == {item.id: item for item in lower.relationships}
    assert {
        row[0] for row in connection.execute("SELECT name_key FROM catalog_search")
    } == {"customers", "orders"}


def test_a_snapshot_that_does_not_decode_stops_the_upgrade() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE snapshots (agent_id TEXT, source_id TEXT, sync_id TEXT, data TEXT)"
    )
    connection.execute(
        "INSERT INTO snapshots VALUES ('agent', 'source:shop', 'sync-1', '{}')"
    )

    with pytest.raises(ValueError):
        revision_0004._convert_catalog_snapshots(connection)
