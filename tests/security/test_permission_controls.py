from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

import pytest

from daita import Agent, SQLiteSource
from daita.adapters.models import SourceRegistration
from daita.adapters.postgresql_write import PostgreSQLWriteBackend, RelationalWriteError
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
from daita.catalog.protocols import CatalogStoreError
from daita.hosting.errors import AgentHomeError
from daita.hosting.sources import _source_edit_read_scope
from daita.storage.sqlite_codecs import encode_relational_write_scope
from daita.storage.sqlite_records import RelationalWriteScope, SourceReadMode
from tests.support.relational_writes import update_constraints
from tests.support.workspace import workspace_for

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def _registration(agent_id: str) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity="postgresql:permission-control",
        display_name="Permission warehouse",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "reader",
        },
        attached_at=NOW,
    )


def _snapshot(
    registration: SourceRegistration,
    *,
    sync_id: str,
    tables: tuple[tuple[str, int], ...],
) -> tuple[SourceCatalogSnapshot, dict[str, CatalogResource]]:
    resources: list[CatalogResource] = []
    revisions: list[CatalogResourceRevision] = []
    facets: list[CatalogFacet] = []
    by_name: dict[str, CatalogResource] = {}
    for table_name, assignment_count in tables:
        native_identity = f"public.{table_name}"
        resource_id = catalog_resource_id(
            registration.id,
            ResourceKind.TABLE,
            native_identity,
        )
        columns = (
            TabularColumn(
                name="id",
                native_type="bigint",
                native_type_namespace="pg_catalog",
                native_type_name="int8",
                ordinal=0,
                nullable=False,
                primary_key_ordinal=1,
                updatable=True,
            ),
            *tuple(
                TabularColumn(
                    name=f"value_{index:02d}",
                    native_type="text",
                    native_type_namespace="pg_catalog",
                    native_type_name="text",
                    ordinal=index,
                    nullable=True,
                    updatable=True,
                )
                for index in range(1, assignment_count + 1)
            ),
        )
        facet = CatalogFacet.from_tabular(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=NOW,
            facet=TabularFacet(columns=columns),
        )
        revision = CatalogResourceRevision.build(
            resource_id=resource_id,
            sync_id=sync_id,
            observed_at=NOW,
            facet_revisions=(facet.revision,),
            source_revision=f"catalog:{sync_id}",
        )
        resource = CatalogResource.build(
            agent_id=registration.agent_id,
            source_id=registration.id,
            native_identity=native_identity,
            external_uri=f"postgresql://permission/{native_identity}",
            kind=ResourceKind.TABLE,
            name=table_name,
            sensitivity=Sensitivity.INTERNAL,
            revision=revision,
            first_observed_at=NOW,
            last_observed_at=NOW,
        )
        resources.append(resource)
        revisions.append(revision)
        facets.append(facet)
        by_name[table_name] = resource
    sync = CatalogSync(
        id=sync_id,
        agent_id=registration.agent_id,
        source_id=registration.id,
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision=f"catalog:{sync_id}",
        resource_count=len(resources),
    )
    return (
        SourceCatalogSnapshot(
            sync=sync,
            resources=tuple(resources),
            revisions=tuple(revisions),
            facets=tuple(facets),
        ),
        by_name,
    )


def _live_columns(monkeypatch: pytest.MonkeyPatch, count: int = 64) -> None:
    """Enabling writes reads the table live; answer for the catalog's fake tables."""

    async def inspect(self, **_arguments: object) -> dict[str, tuple[str, ...]]:
        return {"columns": tuple(f"value_{i:02d}" for i in range(count))}

    monkeypatch.setattr(PostgreSQLWriteBackend, "inspect_write_table", inspect)


async def _agent_with_catalog(tmp_path, tables=("orders", "customers")):
    agent = await Agent.create(
        "permission-control",
        root=tmp_path,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    registration = _registration(agent.id)
    snapshot, resources = _snapshot(
        registration,
        sync_id="sync-one",
        tables=tuple((name, 2) for name in tables),
    )
    await agent._embedded._store.commit_snapshot(
        snapshot,
        registration=registration,
    )
    return agent, registration, resources


async def test_inspect_is_exact_complete_catalog_and_secret_free(tmp_path):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        inspection = await agent.inspect_source_permissions(source.id)
        assert inspection.state.read_scope.mode is SourceReadMode.ALL
        assert inspection.state.relational_write_scopes == ()
        assert {item.resource_id for item in inspection.resources} == {
            resource.id for resource in resources.values()
        }
        assert all(
            item.eligible_assignment_columns == ("id", "value_01", "value_02")
            for item in inspection.resources
        )
        safe = repr(asdict(inspection))
        assert "credential" not in safe
        assert "configuration" not in safe
        assert "db.example.test" not in safe
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("mode", "selected_names"),
    (
        (SourceReadMode.ALL, ()),
        (SourceReadMode.SELECTED, ("orders",)),
        (SourceReadMode.SELECTED, ("orders", "customers")),
        (SourceReadMode.NONE, ()),
    ),
)
async def test_preview_apply_read_modes(tmp_path, mode, selected_names):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        selected_ids = tuple(resources[name].id for name in selected_names)
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode=mode,
            read_resource_ids=selected_ids,
            relational_write_scopes={},
        )
        assert preview.before.read_scope.mode is SourceReadMode.ALL
        assert preview.after.read_scope.mode is mode
        assert set(preview.after.read_scope.resource_ids) == set(selected_ids)
        applied = await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )
        assert applied.state == preview.after
        assert (
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
        ).state == preview.after
    finally:
        await agent.close()


async def test_write_adds_read_uses_exact_columns_and_read_narrowing_revokes(
    tmp_path, monkeypatch
):
    _live_columns(monkeypatch)
    agent, source, resources = await _agent_with_catalog(tmp_path)
    orders = resources["orders"]
    try:
        write_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="none",
            read_resource_ids=(),
            relational_write_scopes={
                orders.native_identity: update_constraints(("value_01", "value_02"))
            },
        )
        assert write_preview.after.read_scope.mode is SourceReadMode.SELECTED
        assert write_preview.after.read_scope.resource_ids == (orders.id,)
        assert write_preview.automatic_read_additions == (orders.id,)
        assert write_preview.after.relational_write_scopes[
            0
        ].allowed_update_columns == ("value_01", "value_02")
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=write_preview.confirmation_fingerprint,
        )

        revoke_preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="none",
            read_resource_ids=(),
            relational_write_scopes={},
        )
        assert revoke_preview.dependent_write_revocations == (orders.id,)
        assert revoke_preview.after.relational_write_scopes == ()
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=revoke_preview.confirmation_fingerprint,
        )
    finally:
        await agent.close()


async def test_write_selected_many_all_current_and_future_table_exclusion(
    tmp_path, monkeypatch
):
    _live_columns(monkeypatch)
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        mapping = {
            resource.native_identity: update_constraints(("value_01", "value_02"))
            for resource in resources.values()
        }
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="all",
            read_resource_ids=(),
            relational_write_scopes=mapping,
        )
        assert len(preview.after.relational_write_scopes) == 2
        await agent.apply_source_permissions(
            source_id=source.id,
            confirmation_fingerprint=preview.confirmation_fingerprint,
        )

        refreshed, refreshed_resources = _snapshot(
            source,
            sync_id="sync-two",
            tables=(("orders", 2), ("customers", 2), ("future", 2)),
        )
        await agent._embedded._store.commit_snapshot(
            refreshed,
            registration=source,
        )
        inspection = await agent.inspect_source_permissions(source.id)
        assert refreshed_resources["future"].id not in {
            scope.resource_id for scope in inspection.state.relational_write_scopes
        }
    finally:
        await agent.close()


async def test_wide_column_choices_support_exact_binding_and_stale_preview(
    tmp_path, monkeypatch
):
    _live_columns(monkeypatch)
    agent = await Agent.create(
        "permission-bound",
        root=tmp_path,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    source = _registration(agent.id)
    snapshot, resources = _snapshot(
        source,
        sync_id="sync-wide",
        tables=(("wide", 33),),
    )
    await agent._embedded._store.commit_snapshot(snapshot, registration=source)
    wide = resources["wide"]
    all_columns = tuple(f"value_{index:02d}" for index in range(1, 34))
    try:
        inspection = await agent.inspect_source_permissions(source.id)
        assert inspection.resources[0].eligible_assignment_columns == (
            "id",
            *all_columns,
        )
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="all",
            read_resource_ids=(),
            relational_write_scopes={
                wide.native_identity: update_constraints(all_columns)
            },
        )
        assert (
            preview.after.relational_write_scopes[0].allowed_update_columns
            == all_columns
        )

        refreshed, _ = _snapshot(
            source,
            sync_id="sync-wide-refreshed",
            tables=(("wide", 33),),
        )
        await agent._embedded._store.commit_snapshot(refreshed, registration=source)
        with pytest.raises(ValueError, match="catalog changed"):
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint=preview.confirmation_fingerprint,
            )
        assert (
            await agent.inspect_source_permissions(source.id)
        ).state.relational_write_scopes == ()
    finally:
        await agent.close()


async def test_unknown_or_wrong_confirmation_never_changes_state(tmp_path):
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        preview = await agent.preview_source_permissions(
            source_id=source.id,
            read_mode="selected",
            read_resource_ids=(resources["orders"].id,),
            relational_write_scopes={},
        )
        with pytest.raises(ValueError, match="unknown or expired"):
            await agent.apply_source_permissions(
                source_id=source.id,
                confirmation_fingerprint="sha256:" + "0" * 64,
            )
        assert (
            await agent.inspect_source_permissions(source.id)
        ).state == preview.before
    finally:
        await agent.close()


async def test_hosting_rejects_write_scopes_for_a_type_without_native_writes(
    tmp_path,
):
    database = tmp_path / "customers.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
    root = tmp_path / "state"
    agent = await Agent.create(
        "sqlite-write-scope",
        root=root,
        clock=lambda: NOW,
        workspace=workspace_for(tmp_path),
    )
    try:
        home = agent.home
        source = await agent.attach(SQLiteSource(database))
        (table,) = await agent.list_catalog_resources(source_id=source.id)
        with pytest.raises(ValueError, match="supports native writes"):
            await agent.preview_source_permissions(
                source_id=source.id,
                read_mode="all",
                read_resource_ids=(),
                relational_write_scopes={"main.t": update_constraints(("name",))},
            )
    finally:
        await agent.close()

    # Storage keeps no source-type knowledge, so forge a stored scope directly.
    forged = RelationalWriteScope(
        agent_id=source.agent_id,
        source_id=source.id,
        resource_id=table.id,
        table=table.native_identity,
        allowed_operations=("update",),
        allowed_insert_columns=(),
        allowed_update_columns=("name",),
        max_rows=10,
        authorization_fingerprint="sha256:" + "1" * 64,
    )
    with sqlite3.connect(home / "state.db") as connection:
        connection.execute(
            """INSERT INTO relational_write_scopes(
                   agent_id, source_id, resource_id, authorization_fingerprint, data
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                forged.agent_id,
                forged.source_id,
                forged.resource_id,
                forged.authorization_fingerprint,
                encode_relational_write_scope(forged),
            ),
        )

    with pytest.raises(AgentHomeError, match="foreign to its source type"):
        await Agent.open(
            "sqlite-write-scope",
            root=root,
            clock=lambda: NOW,
            workspace=workspace_for(tmp_path),
        )


def _write_scope(*operations: str, columns=("value_01",), max_rows=10) -> dict:
    return {
        "allowed_operations": operations,
        "allowed_insert_columns": columns if "insert" in operations else (),
        "allowed_update_columns": columns if "update" in operations else (),
        "max_rows": max_rows,
    }


async def _apply(agent, source, *, mode=SourceReadMode.ALL, read_ids=(), scopes):
    preview = await agent.preview_source_permissions(
        source_id=source.id,
        read_mode=mode,
        read_resource_ids=read_ids,
        relational_write_scopes=scopes,
    )
    return await agent.apply_source_permissions(
        source_id=source.id,
        confirmation_fingerprint=preview.confirmation_fingerprint,
    )


def _source_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def inspect(self, **_arguments: object) -> dict[str, tuple[str, ...]]:
        raise RelationalWriteError("postgresql_connect_failed", "The source is down.")

    monkeypatch.setattr(PostgreSQLWriteBackend, "inspect_write_table", inspect)


async def test_revoking_or_narrowing_needs_no_live_read(tmp_path, monkeypatch):
    """Taking access away must work with the source down or a kept table dropped."""

    _live_columns(monkeypatch)
    agent, source, _resources = await _agent_with_catalog(tmp_path)
    kept = _write_scope("insert", "update", columns=("value_01", "value_02"))
    try:
        await _apply(
            agent,
            source,
            scopes={"public.orders": kept, "public.dropped": kept},
        )
        _source_unreachable(monkeypatch)

        inspection = await _apply(agent, source, scopes={"public.dropped": kept})
        assert [s.table for s in inspection.state.relational_write_scopes] == [
            "public.dropped"
        ]

        narrower = _write_scope("update", columns=("value_01",), max_rows=5)
        inspection = await _apply(agent, source, scopes={"public.dropped": narrower})
        (scope,) = inspection.state.relational_write_scopes
        assert scope.allowed_operations == ("update",)
        assert scope.max_rows == 5

        inspection = await _apply(agent, source, scopes={})
        assert inspection.state.relational_write_scopes == ()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "wider",
    (
        _write_scope("update", "delete", columns=("value_01",)),
        _write_scope("update", columns=("value_01", "value_02")),
        _write_scope("update", columns=("value_01",), max_rows=11),
    ),
)
async def test_widening_a_scope_still_reads_the_table_live(
    tmp_path, monkeypatch, wider
):
    _live_columns(monkeypatch)
    agent, source, _resources = await _agent_with_catalog(tmp_path)
    try:
        await _apply(agent, source, scopes={"public.orders": _write_scope("update")})
        _source_unreachable(monkeypatch)
        with pytest.raises(RelationalWriteError):
            await agent.preview_source_permissions(
                source_id=source.id,
                read_mode=SourceReadMode.ALL,
                read_resource_ids=(),
                relational_write_scopes={"public.orders": wider},
            )
    finally:
        await agent.close()


@pytest.mark.parametrize("mode", (SourceReadMode.SELECTED, SourceReadMode.NONE))
async def test_a_table_the_catalog_has_not_seen_enables_under_narrow_reads(
    tmp_path, monkeypatch, mode
):
    """The automatic read permission is for the write's table, not a catalog row."""

    _live_columns(monkeypatch)
    agent, source, resources = await _agent_with_catalog(tmp_path)
    selected = (resources["orders"].id,) if mode is SourceReadMode.SELECTED else ()
    scopes = {"public.fresh": _write_scope("update")}
    try:
        inspection = await _apply(
            agent, source, mode=mode, read_ids=selected, scopes=scopes
        )
        (scope,) = inspection.state.relational_write_scopes
        assert scope.table == "public.fresh"
        assert inspection.state.read_scope.mode is SourceReadMode.SELECTED
        assert scope.resource_id in inspection.state.read_scope.resource_ids

        # Previewing what is now stored must not trip on that read permission.
        again = await _apply(
            agent,
            source,
            mode=SourceReadMode.SELECTED,
            read_ids=inspection.state.read_scope.resource_ids,
            scopes=scopes,
        )
        assert again.state == inspection.state
    finally:
        await agent.close()


async def test_a_read_nobody_has_permission_for_is_still_refused(tmp_path):
    agent, source, _resources = await _agent_with_catalog(tmp_path)
    try:
        with pytest.raises(ValueError, match="not current resources"):
            await agent.preview_source_permissions(
                source_id=source.id,
                read_mode=SourceReadMode.SELECTED,
                read_resource_ids=(
                    catalog_resource_id(source.id, ResourceKind.TABLE, "public.other"),
                ),
                relational_write_scopes={},
            )
    finally:
        await agent.close()


async def test_a_stored_scope_the_catalog_has_not_seen_is_offered_for_removal(
    tmp_path, monkeypatch
):
    _live_columns(monkeypatch)
    agent, source, _resources = await _agent_with_catalog(tmp_path)
    scope = _write_scope("insert", "update", columns=("value_01",))
    try:
        await _apply(agent, source, scopes={"public.fresh": scope})
        inspection = await agent.inspect_source_permissions(source.id)
        offered = {item.display_name: item for item in inspection.resources}
        assert offered["public.fresh"].resource_kind == "table"
        assert offered["public.fresh"].eligible_insert_columns == ("value_01",)
        assert offered["public.fresh"].eligible_assignment_columns == ("value_01",)
        # Never listed twice, or under an ID.
        assert len(offered) == len(inspection.resources)
        assert "public.orders" in offered
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "broken", ("list_current_snapshot_refs", "list_resources", "resource_schemas")
)
async def test_a_failing_catalog_read_does_not_stop_a_revoke(
    tmp_path, monkeypatch, broken
):
    """The stored permissions are enough to take access away."""

    _live_columns(monkeypatch)
    agent, source, _resources = await _agent_with_catalog(tmp_path)
    try:
        await _apply(agent, source, scopes={"public.orders": _write_scope("update")})

        async def fails(*_args: object, **_kwargs: object) -> object:
            raise CatalogStoreError("the catalog could not be read")

        embedded = agent._embedded
        owner = embedded._data_view if broken == "resource_schemas" else embedded._store
        monkeypatch.setattr(owner, broken, fails)
        inspection = await _apply(agent, source, scopes={})
        assert inspection.state.relational_write_scopes == ()
    finally:
        await agent.close()


async def test_a_source_edit_names_no_read_permission_by_its_id(tmp_path, monkeypatch):
    _live_columns(monkeypatch)
    agent, source, resources = await _agent_with_catalog(tmp_path)
    try:
        inspection = await _apply(
            agent,
            source,
            mode=SourceReadMode.NONE,
            scopes={"public.fresh": _write_scope("update")},
        )
        (scope,) = inspection.state.relational_write_scopes
        _read, omitted = _source_edit_read_scope(
            current_read_scope=inspection.state.read_scope,
            current_resources=tuple(resources.values()),
            replacement=_snapshot(
                _registration(agent.id), sync_id="sync-two", tables=(("orders", 2),)
            )[0],
        )
        assert omitted
        assert not any(scope.resource_id in item for item in omitted)
    finally:
        await agent.close()
