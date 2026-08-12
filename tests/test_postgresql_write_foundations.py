from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from daita import Agent, PostgreSQLSource
from daita.adapters import postgresql as postgresql_module
from daita.adapters.models import DiscoveryRequest, DiscoveryResult, SourceRegistration
from daita.capabilities import ExtensionDeclarations
from daita.catalog.models import (
    CatalogFacet,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
)
from daita.hosting import embedded as embedded_module
from daita.hosting.embedded import AgentHomeError
from daita.loop.models import RunInput
from daita.security import EmptySecretProvider
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope

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


def test_postgresql_write_access_projection_never_changes_connection_identity():
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

    enabled_projection = _registration("agent-enabled", write_access=True)
    enabled_connection = embedded_module._source_from_registration(
        enabled_projection,
        secret_provider=EmptySecretProvider(),
    )
    assert isinstance(enabled_connection, PostgreSQLSource)
    assert enabled_connection.write_access is False

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


async def test_source_wide_write_enable_is_no_longer_persistable(tmp_path):
    agent = await Agent.create("source-write-access", root=tmp_path, clock=lambda: NOW)
    registration = _registration(agent.id)
    other = _registration(agent.id, native_identity="postgresql:other")
    await agent._embedded._store.register_source(registration)
    await agent._embedded._store.register_source(other)

    try:
        with pytest.raises(ValueError, match="exact table scopes"):
            await agent.set_source_write_access(registration.id, True)
        assert (await agent._embedded._store.load_source(agent.id, other.id)) == other
        with sqlite3.connect(agent.home / "state.db") as connection:
            source_data = connection.execute(
                "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                (agent.id, registration.id),
            ).fetchone()[0]
            assert "write_access" not in source_data
            assert connection.execute(
                "SELECT COUNT(*) FROM postgresql_update_scopes"
            ).fetchone() == (0,)

        disabled = await agent.set_source_write_access(registration.id, False)
        assert disabled == registration
        assert await agent.list_sources() == tuple(
            sorted((registration, other), key=lambda item: item.id)
        )
        with sqlite3.connect(agent.home / "state.db") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM postgresql_update_scopes"
            ).fetchone() == (0,)
    finally:
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
        with pytest.raises(ValueError, match="exact table scopes"):
            await agent.set_source_write_access(postgres.id, True)
        assert (
            await agent._embedded._store.load_source(agent.id, postgres.id) == postgres
        )
    finally:
        await agent.close()


async def test_default_read_scope_round_trips_through_agent_reopen(tmp_path):
    agent = await Agent.create(
        "source-write-access-reopen", root=tmp_path, clock=lambda: NOW
    )
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()

    reopened = await Agent.open(
        "source-write-access-reopen", root=tmp_path, clock=lambda: NOW
    )
    try:
        assert await reopened.list_sources() == (registration,)
        reconstructed = embedded_module._source_from_registration(
            registration,
            secret_provider=EmptySecretProvider(),
        )
        assert isinstance(reconstructed, PostgreSQLSource)
        assert reconstructed.write_access is False
        with sqlite3.connect(reopened.home / "state.db") as connection:
            source_data = connection.execute("SELECT data FROM sources").fetchone()[0]
            assert "write_access" not in source_data
            assert connection.execute(
                "SELECT source_id FROM source_read_scopes"
            ).fetchone() == (registration.id,)
            assert connection.execute(
                "SELECT COUNT(*) FROM postgresql_update_scopes"
            ).fetchone() == (0,)
    finally:
        await reopened.close()


async def test_detach_revokes_scopes_and_storage_reattachment_starts_read_only(
    tmp_path,
):
    agent = await Agent.create("source-write-detach", root=tmp_path, clock=lambda: NOW)
    registration = _registration(agent.id)
    registered = await agent._embedded._store.register_source(registration)
    detached = await agent.detach(registration.id)
    assert detached.active is False
    assert detached.configuration["write_access"] is False
    with sqlite3.connect(agent.home / "state.db") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_read_scopes"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM postgresql_update_scopes"
        ).fetchone() == (0,)

    sync = CatalogSync(
        id="catalog-sync-reattach",
        agent_id=agent.id,
        source_id=registration.id,
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision="catalog:sha256:" + "a" * 64,
    )
    await agent._embedded._store.commit_snapshot(
        SourceCatalogSnapshot(sync=sync, resources=(), revisions=()),
        registration=registered,
    )
    reattached = await agent._embedded._store.load_source(agent.id, registration.id)
    try:
        assert reattached is not None
        assert reattached.active is True
        assert reattached.configuration["write_access"] is False
        with sqlite3.connect(agent.home / "state.db") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM source_read_scopes"
            ).fetchone() == (1,)
            assert connection.execute(
                "SELECT COUNT(*) FROM postgresql_update_scopes"
            ).fetchone() == (0,)
    finally:
        await agent.close()


@pytest.mark.parametrize("read_mode", (SourceReadMode.ALL, SourceReadMode.NONE))
async def test_postgresql_refresh_preserves_read_scope_outside_connection_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    read_mode: SourceReadMode,
):
    agent = await Agent.create(
        f"postgresql-refresh-{read_mode.value}",
        root=tmp_path,
        clock=lambda: NOW,
    )
    registration = _registration(agent.id)
    registered = await agent._embedded._store.register_source(registration)
    scope = SourceReadScope(
        agent_id=agent.id,
        source_id=registration.id,
        mode=read_mode,
    )
    await agent._embedded._store.replace_source_permission_scopes(scope, ())
    expected = registered
    connection_registration = replace(
        registered,
        configuration={**dict(registered.configuration), "write_access": False},
    )
    closed = False

    class _Adapter:
        @property
        def registration(self) -> SourceRegistration:
            return connection_registration

        def declarations(self) -> ExtensionDeclarations:
            return ExtensionDeclarations()

        async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
            sync = CatalogSync(
                id=request.sync_id,
                agent_id=request.agent_id,
                source_id=request.source_id,
                adapter_id="postgresql",
                status=CatalogSyncStatus.SUCCEEDED,
                started_at=request.requested_at,
                completed_at=NOW,
                source_revision="catalog:sha256:" + "a" * 64,
            )
            return DiscoveryResult(
                request=request,
                snapshot=SourceCatalogSnapshot(
                    sync=sync,
                    resources=(),
                    revisions=(),
                ),
                completed_at=NOW,
            )

        async def inspect(self, resource):
            raise AssertionError("refresh must not inspect individual resources")

        async def health(self):
            raise AssertionError("refresh must not run a separate health check")

        async def close(self) -> None:
            nonlocal closed
            closed = True

    async def open_source(
        source: PostgreSQLSource,
        *,
        agent_id: str,
        attached_at: datetime,
        clock,
    ) -> _Adapter:
        del clock
        assert source.write_access is False
        assert agent_id == agent.id
        assert attached_at == registration.attached_at
        return _Adapter()

    monkeypatch.setattr(PostgreSQLSource, "open", open_source)
    try:
        refreshed = await agent.refresh_source(registration.id)
        assert refreshed == expected
        assert (
            await agent._embedded._store.load_source(agent.id, registration.id)
            == expected
        )
        with sqlite3.connect(agent.home / "state.db") as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM postgresql_update_scopes"
            ).fetchone() == (0,)
            assert (
                "write_access"
                not in connection.execute("SELECT data FROM sources").fetchone()[0]
            )
        assert (
            await agent._embedded._store.load_source_read_scope(
                agent.id,
                registration.id,
            )
            == scope
        )
        assert closed is True
    finally:
        await agent.close()


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


async def test_source_wide_write_enable_does_not_change_projected_tools(
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
        with pytest.raises(ValueError, match="exact table scopes"):
            await agent.set_source_write_access(registration.id, True)
        after = {
            definition.name
            for definition in await agent._embedded._data_tool_runtime.definitions(run)
        }
        await agent.set_source_write_access(registration.id, False)
        disabled = {
            definition.name
            for definition in await agent._embedded._data_tool_runtime.definitions(run)
        }
        assert {preview, update}.isdisjoint(before)
        assert forbidden.isdisjoint(after)
        assert after == before
        assert disabled == before
    finally:
        await agent.close()
