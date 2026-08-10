from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from daita import Agent, PostgreSQLSource
from daita.adapters import postgresql as postgresql_module
from daita.adapters.models import DiscoveryRequest, SourceRegistration
from daita.catalog.models import (
    CatalogFacet,
    ResourceKind,
    TabularColumn,
    TabularFacet,
)
from daita.hosting import embedded as embedded_module
from daita.hosting.embedded import AgentHomeError
from daita.loop.models import RunInput
from daita.security import EmptySecretProvider

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


def _configuration(*, write_access: bool | None = False) -> dict[str, object]:
    configuration: dict[str, object] = {
        "database": "warehouse",
        "host": "db.example.test",
        "port": 5432,
        "schemas": ("public",),
        "ssl_mode": "require",
        "username": "reader",
    }
    if write_access is not None:
        configuration["write_access"] = write_access
    return configuration


def _registration(
    agent_id: str,
    *,
    adapter_id: str = "postgresql",
    native_identity: str = "postgresql:test",
    write_access: bool | None = False,
) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id=adapter_id,
        native_identity=native_identity,
        display_name="Warehouse",
        configuration=(
            _configuration(write_access=write_access)
            if adapter_id == "postgresql"
            else {"path": "/tmp/read-only.sqlite"}
        ),
        attached_at=NOW,
    )


def test_postgresql_write_access_defaults_false_and_reconstructs_fail_closed():
    source = PostgreSQLSource(
        host="db.example.test",
        database="warehouse",
        username="reader",
        secret_provider=EmptySecretProvider(),
    )
    assert source.write_access is False
    assert postgresql_module._source_configuration(source)["write_access"] is False

    missing = _registration("agent-source", write_access=None)
    reconstructed = embedded_module._source_from_registration(
        missing,
        secret_provider=EmptySecretProvider(),
    )
    assert isinstance(reconstructed, PostgreSQLSource)
    assert reconstructed.write_access is False

    invalid = replace(
        missing,
        configuration={**dict(missing.configuration), "write_access": "yes"},
    )
    with pytest.raises(AgentHomeError, match="configuration is invalid"):
        embedded_module._source_from_registration(
            invalid,
            secret_provider=EmptySecretProvider(),
        )

    with pytest.raises(TypeError, match="write_access must be a boolean"):
        PostgreSQLSource(
            host="db.example.test",
            database="warehouse",
            username="reader",
            write_access=1,  # type: ignore[arg-type]
            secret_provider=EmptySecretProvider(),
        )


async def test_user_owned_source_write_access_toggle_uses_shared_lock_and_only_changes_flag(
    tmp_path,
):
    agent = await Agent.create("source-write-access", root=tmp_path, clock=lambda: NOW)
    registration = _registration(agent.id)
    other = _registration(agent.id, native_identity="postgresql:other")
    await agent._embedded._store.register_source(registration)
    await agent._embedded._store.register_source(other)

    lock = agent._embedded._mutation_lock
    await lock.acquire()
    task = asyncio.create_task(agent.set_source_write_access(registration.id, True))
    await asyncio.sleep(0)
    assert not task.done()
    lock.release()
    try:
        enabled = await task
        assert enabled == replace(
            registration,
            configuration={**dict(registration.configuration), "write_access": True},
        )
        assert (await agent._embedded._store.load_source(agent.id, other.id)) == other

        disabled = await agent.set_source_write_access(registration.id, False)
        assert disabled == registration
        assert await agent.list_sources() == tuple(
            sorted((registration, other), key=lambda item: item.id)
        )
    finally:
        if lock.locked():
            lock.release()
        await agent.close()


async def test_source_write_access_toggle_rejects_non_exact_or_unowned_targets(
    tmp_path,
):
    agent = await Agent.create(
        "source-write-access-reject", root=tmp_path, clock=lambda: NOW
    )
    postgres = _registration(agent.id)
    sqlite = _registration(agent.id, adapter_id="sqlite", native_identity="sqlite:test")
    foreign = _registration("another-agent")
    detached = _registration(agent.id, native_identity="postgresql:detached")
    for registration in (postgres, sqlite, foreign, detached):
        await agent._embedded._store.register_source(registration)
    await agent._embedded._store.detach_source(agent.id, detached.id, NOW)
    try:
        with pytest.raises(TypeError, match="enabled must be a boolean"):
            await agent.set_source_write_access(postgres.id, 1)  # type: ignore[arg-type]
        for source_id in ("missing", sqlite.id, foreign.id, detached.id):
            with pytest.raises(ValueError, match="active PostgreSQL source"):
                await agent.set_source_write_access(source_id, True)
        assert (
            await agent._embedded._store.load_source(agent.id, postgres.id) == postgres
        )
    finally:
        await agent.close()


async def test_enabled_source_write_access_round_trips_through_agent_reopen(tmp_path):
    agent = await Agent.create(
        "source-write-access-reopen", root=tmp_path, clock=lambda: NOW
    )
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    enabled = await agent.set_source_write_access(registration.id, True)
    await agent.close()

    reopened = await Agent.open(
        "source-write-access-reopen", root=tmp_path, clock=lambda: NOW
    )
    try:
        assert await reopened.list_sources() == (enabled,)
        reconstructed = embedded_module._source_from_registration(
            enabled,
            secret_provider=EmptySecretProvider(),
        )
        assert isinstance(reconstructed, PostgreSQLSource)
        assert reconstructed.write_access is True
    finally:
        await reopened.close()


def test_postgresql_discovery_projects_write_relevant_column_facts():
    column = postgresql_module._column(
        {
            "column_name": "account_id",
            "native_type": "bigint",
            "type_schema": "pg_catalog",
            "type_name": "int8",
            "ordinal": 2,
            "nullable": False,
            "default_expression": None,
            "primary_key_ordinal": 1,
            "is_identity": True,
            "is_generated": False,
            "is_updatable": False,
        }
    )
    assert column == TabularColumn(
        name="account_id",
        native_type="bigint",
        native_type_namespace="pg_catalog",
        native_type_name="int8",
        ordinal=1,
        nullable=False,
        primary_key_ordinal=1,
        identity=True,
        generated=False,
        updatable=False,
    )
    assert "attidentity" in postgresql_module._COLUMNS_SQL
    assert "attgenerated" in postgresql_module._COLUMNS_SQL
    assert "is_updatable" in postgresql_module._COLUMNS_SQL


@pytest.mark.parametrize(
    "changed",
    (
        {"nullable": True},
        {"native_type_namespace": "extension"},
        {"native_type_name": "custom_int8"},
        {"identity": True},
        {"generated": True},
        {"updatable": False},
        {"primary_key_ordinal": 1},
    ),
)
def test_each_write_relevant_column_fact_changes_structural_revision(changed):
    base = TabularColumn(
        name="account_id",
        native_type="bigint",
        native_type_namespace="pg_catalog",
        native_type_name="int8",
        ordinal=0,
        nullable=False,
        identity=False,
        generated=False,
        updatable=True,
    )

    def revision(column: TabularColumn) -> str:
        return CatalogFacet.from_tabular(
            resource_id="catalog-resource:sha256:" + "a" * 64,
            sync_id="catalog-sync-write-facts",
            observed_at=NOW,
            facet=TabularFacet(columns=(column,)),
        ).revision

    assert revision(replace(base, **changed)) != revision(base)


async def test_catalog_projection_round_trips_ordered_keys_and_column_write_facts(
    tmp_path,
):
    agent = await Agent.create("catalog-write-facts", root=tmp_path, clock=lambda: NOW)
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    columns = (
        TabularColumn(
            name="tenant_id",
            native_type="bigint",
            native_type_namespace="pg_catalog",
            native_type_name="int8",
            ordinal=0,
            nullable=False,
            primary_key_ordinal=2,
            updatable=True,
        ),
        TabularColumn(
            name="account_id",
            native_type="bigint",
            native_type_namespace="pg_catalog",
            native_type_name="int8",
            ordinal=1,
            nullable=False,
            primary_key_ordinal=1,
            identity=True,
            updatable=False,
        ),
        TabularColumn(
            name="computed_status",
            native_type="text",
            native_type_namespace="pg_catalog",
            native_type_name="text",
            ordinal=2,
            nullable=True,
            generated=True,
            updatable=False,
        ),
    )
    structure = postgresql_module.PostgreSQLStructure(
        tables=(
            postgresql_module._TableStructure(
                schema="public",
                name="accounts",
                kind=ResourceKind.TABLE,
                columns=columns,
                indexes=(),
            ),
        ),
        relationships=(),
        source_revision="catalog:sha256:" + "5" * 64,
    )
    request = DiscoveryRequest(
        agent_id=agent.id,
        source_id=registration.id,
        sync_id="catalog-sync-write-facts",
        requested_at=NOW,
    )
    snapshot = postgresql_module._catalog_snapshot(
        registration, request, structure, NOW
    )
    await agent._embedded._store.commit_snapshot(snapshot)

    try:
        schema = (
            await agent._embedded._data_view.resource_schemas(agent.id, registration.id)
        )[0]
        assert schema.primary_key_columns == ("account_id", "tenant_id")
        assert schema.column_nullability == (
            ("tenant_id", False),
            ("account_id", False),
            ("computed_status", True),
        )
        assert schema.column_type_provenance == (
            ("tenant_id", "pg_catalog", "int8"),
            ("account_id", "pg_catalog", "int8"),
            ("computed_status", "pg_catalog", "text"),
        )
        assert schema.identity_columns == ("account_id",)
        assert schema.generated_columns == ("computed_status",)
        assert schema.updatable_columns == ("tenant_id",)
        agent_id = agent.id
        source_id = registration.id
    finally:
        await agent.close()

    reopened = await Agent.open("catalog-write-facts", root=tmp_path, clock=lambda: NOW)
    try:
        assert (
            await reopened._embedded._data_view.resource_schemas(agent_id, source_id)
        )[0] == schema
    finally:
        await reopened.close()


async def test_write_access_projects_only_the_postgresql_preview_and_update_tools(
    tmp_path,
):
    agent = await Agent.create("write-access-no-tool", root=tmp_path, clock=lambda: NOW)
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    run = RunInput(
        id="run-no-write-tool",
        agent_id=agent.id,
        message="test projection",
        created_at=NOW,
    )
    preview = "data_preview_postgresql_update"
    update = "data_update_postgresql"
    forbidden = {
        "data_preview_sqlite_update",
        "data_update_sqlite",
    }
    try:
        before = {
            definition.name
            for definition in await agent._embedded._data_tool_runtime.definitions(run)
        }
        await agent.set_source_write_access(registration.id, True)
        after = {
            definition.name
            for definition in await agent._embedded._data_tool_runtime.definitions(run)
        }
        assert {preview, update}.isdisjoint(before)
        assert forbidden.isdisjoint(after)
        assert {preview, update}.issubset(after)
        assert after == before | {preview, update}
    finally:
        await agent.close()
