from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

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
from daita.security import EmptySecretProvider
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _configuration() -> dict[str, object]:
    return {
        "database": "warehouse",
        "host": "db.example.test",
        "port": 5432,
        "schemas": ("public",),
        "ssl_mode": "require",
        "username": "reader",
    }


def _registration(
    agent_id: str,
    *,
    adapter_id: str = "postgresql",
    native_identity: str = "postgresql:test",
) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id=adapter_id,
        native_identity=native_identity,
        display_name="Warehouse",
        configuration=(
            _configuration()
            if adapter_id == "postgresql"
            else {"path": "/tmp/read-only.sqlite"}
        ),
        attached_at=NOW,
    )


async def test_default_read_scope_round_trips_through_agent_reopen(tmp_path):
    agent = await Agent.create(
        "source-permissions-reopen", root=tmp_path, clock=lambda: NOW
    )
    registration = _registration(agent.id)
    await agent._embedded._store.register_source(registration)
    await agent.close()

    reopened = await Agent.open(
        "source-permissions-reopen", root=tmp_path, clock=lambda: NOW
    )
    try:
        assert await reopened.list_sources() == (registration,)
        reconstructed = embedded_module._source_from_registration(
            registration,
            secret_provider=EmptySecretProvider(),
        )
        assert isinstance(reconstructed, PostgreSQLSource)
        with sqlite3.connect(reopened.home / "state.db") as connection:
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
    connection_registration = registered
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
