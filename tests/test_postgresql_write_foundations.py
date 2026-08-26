from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from daita import Agent, PostgreSQLSource
from daita.adapters import postgresql as postgresql_module
from daita.adapters.models import DiscoveryRequest, DiscoveryResult, SourceRegistration
from daita.catalog.models import (
    CatalogFacet,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
)
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_CAPABILITY_ID,
    POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
    postgresql_update_capability_declarations,
    postgresql_update_preview_capability_declarations,
)
from daita.domains.data.context import _system_prompt
from daita.hosting import embedded as embedded_module
from daita.security import EmptySecretProvider, SecretReference
from daita.storage.sqlite_records import SourceReadMode, SourceReadScope

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def test_model_contract_makes_update_tool_call_the_only_approval_trigger() -> None:
    preview = postgresql_update_preview_capability_declarations()
    update = postgresql_update_capability_declarations()

    preview_description = preview.tool_views[0].description
    update_description = update.tool_views[0].description
    assert "pass the exact successful preview immediately" in preview_description
    assert "preview alone does not request approval" in preview_description
    assert "Calling this tool opens the approval interaction" in update_description
    assert "approved structured PostgreSQL update" not in update_description

    prompt = _system_prompt(
        {},
        capability_ids=frozenset(
            {
                POSTGRESQL_UPDATE_PREVIEW_CAPABILITY_ID,
                POSTGRESQL_UPDATE_CAPABILITY_ID,
            }
        ),
        tool_manifest=(),
        has_on_demand_tools=True,
        memory_text="",
        user_profile="",
        skill_index=None,
        semantic_text="",
        candidate_text="",
        artifact_destinations=(),
        final=False,
    )
    assert "a successful preview is not a terminal answer" in prompt
    assert "in the same run, call data_update_postgresql" in prompt
    assert "is what requests runtime approval and opens the approval card" in prompt
    assert "preview alone does neither" in prompt
    assert "Never claim that an approval card is displayed" in prompt
    assert (
        "Stop after preview only when the user explicitly requested preview" in prompt
    )


class _Keychain:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def resolve(self, reference: SecretReference) -> str:
        return self.values[reference.name]

    async def set(self, reference: SecretReference, value: str) -> None:
        self.values[reference.name] = value

    async def delete(self, reference: SecretReference) -> None:
        self.values.pop(reference.name, None)


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


async def test_committed_source_edit_deletes_only_the_previous_owned_credential(
    tmp_path,
) -> None:
    keychain = _Keychain()
    agent = await Agent.create(
        "postgresql-edit-credential-cleanup",
        root=tmp_path,
        keychain=keychain,
        clock=lambda: NOW,
    )
    old_reference = await agent.store_postgresql_password("old-secret")
    new_reference = await agent.store_postgresql_password("new-secret")
    current = replace(
        _registration(agent.id, native_identity="postgresql:reader"),
        configuration={
            **_configuration(),
            "credential_ref": old_reference.to_uri(),
        },
    )
    initial_sync = CatalogSync(
        id="sync-initial",
        agent_id=agent.id,
        source_id=current.id,
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision="catalog:initial",
    )
    await agent._embedded._store.commit_snapshot(
        SourceCatalogSnapshot(
            sync=initial_sync,
            resources=(),
            revisions=(),
        ),
        registration=current,
    )
    edited = replace(
        _registration(agent.id, native_identity="postgresql:writer"),
        configuration={
            **_configuration(),
            "username": "writer",
            "credential_ref": new_reference.to_uri(),
        },
    )
    closed = False

    class _Adapter:
        @property
        def registration(self) -> SourceRegistration:
            return edited

        async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
            sync = CatalogSync(
                id=request.sync_id,
                agent_id=request.agent_id,
                source_id=request.source_id,
                adapter_id="postgresql",
                status=CatalogSyncStatus.SUCCEEDED,
                started_at=request.requested_at,
                completed_at=NOW,
                source_revision="catalog:edited",
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
            raise AssertionError(resource)

        async def health(self):
            raise AssertionError("edit must not run a separate health check")

        async def close(self) -> None:
            nonlocal closed
            closed = True

    class _Source:
        async def open(self, *, agent_id, attached_at, clock):
            del clock
            assert agent_id == agent.id
            assert attached_at == NOW
            return _Adapter()

    async def confirm(_preview) -> bool:
        return True

    try:
        result = await agent.edit_source(
            current.id,
            _Source(),
            confirmation_handler=confirm,
        )

        assert result is not None
        assert result.source == edited
        assert result.previous_credential_deleted
        assert old_reference.name not in keychain.values
        assert keychain.values[new_reference.name] == "new-secret"
        assert closed
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
