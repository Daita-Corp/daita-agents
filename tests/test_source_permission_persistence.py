from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from daita.adapters.models import SourceRegistration
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
from daita.catalog.service import CatalogService
from daita.domains.data.catalog import CatalogDataView
from daita.llm.models import ModelSensitivity
from daita.storage import sqlite as sqlite_module
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_migrations import migration_rows
from daita.storage.sqlite_records import (
    PostgreSQLUpdateScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    postgresql_update_authorization_fingerprint,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def _registration(
    agent_id: str = "agent-permissions",
    *,
    native_identity: str = "postgresql:permission-tests",
) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id=agent_id,
        adapter_id="postgresql",
        native_identity=native_identity,
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
    amount_type: str = "numeric",
    amount_updatable: bool = True,
    row_count_estimate: int | None = 10,
    sensitivity: Sensitivity = Sensitivity.INTERNAL,
) -> tuple[SourceCatalogSnapshot, CatalogResource, CatalogFacet]:
    resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "public.orders",
    )
    facet = CatalogFacet.from_tabular(
        resource_id=resource_id,
        sync_id=sync_id,
        observed_at=NOW,
        facet=TabularFacet(
            columns=(
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
                TabularColumn(
                    name="amount",
                    native_type=amount_type,
                    native_type_namespace="pg_catalog",
                    native_type_name=(
                        "numeric" if amount_type == "numeric" else "text"
                    ),
                    ordinal=1,
                    nullable=False,
                    updatable=amount_updatable,
                ),
            ),
            row_count_estimate=row_count_estimate,
        ),
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
        native_identity="public.orders",
        external_uri=f"postgresql://{registration.id}/public/orders",
        kind=ResourceKind.TABLE,
        name="orders",
        sensitivity=sensitivity,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )
    sync = CatalogSync(
        id=sync_id,
        agent_id=registration.agent_id,
        source_id=registration.id,
        adapter_id="postgresql",
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision=f"catalog:{sync_id}",
        resource_count=1,
        relationship_count=0,
    )
    return (
        SourceCatalogSnapshot(
            sync=sync,
            resources=(resource,),
            revisions=(revision,),
            facets=(facet,),
        ),
        resource,
        facet,
    )


def _scope(
    registration: SourceRegistration,
    resource: CatalogResource,
    facet: CatalogFacet,
) -> PostgreSQLUpdateScope:
    return PostgreSQLUpdateScope(
        agent_id=registration.agent_id,
        source_id=registration.id,
        resource_id=resource.id,
        allowed_assignment_columns=("amount",),
        authorization_fingerprint=postgresql_update_authorization_fingerprint(
            source=registration,
            resource=resource,
            facet=facet,
            allowed_assignment_columns=("amount",),
        ),
    )


async def test_fresh_schema_has_only_scoped_permission_tables(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert "source_read_scopes" in tables
        assert "postgresql_update_scopes" in tables
        assert (
            tuple(
                connection.execute(
                    "SELECT ordinal, migration_id, checksum "
                    "FROM state_migrations ORDER BY ordinal"
                )
            )
            == migration_rows()
        )


async def test_admitted_resource_scope_derives_sensitivity_and_fails_closed(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "sensitivity.db")
    registration = _registration()
    public_snapshot, resource, _ = _snapshot(
        registration,
        sync_id="sync-public",
        sensitivity=Sensitivity.PUBLIC,
    )
    await store.commit_snapshot(public_snapshot, registration=registration)
    view = CatalogDataView(store, CatalogService(store, store), store)
    try:
        assert (
            await view.admitted_model_sensitivity(
                registration.agent_id,
                (registration.id,),
            )
            is ModelSensitivity.PUBLIC
        )

        confidential_snapshot, _, _ = _snapshot(
            registration,
            sync_id="sync-confidential",
            sensitivity=Sensitivity.CONFIDENTIAL,
        )
        await store.commit_snapshot(confidential_snapshot, registration=registration)
        assert (
            await view.admitted_model_sensitivity(
                registration.agent_id,
                (registration.id,),
            )
            is ModelSensitivity.CONFIDENTIAL
        )

        unknown_snapshot, _, _ = _snapshot(
            registration,
            sync_id="sync-unknown",
            sensitivity=Sensitivity.UNKNOWN,
        )
        await store.commit_snapshot(unknown_snapshot, registration=registration)
        assert (
            await view.admitted_model_sensitivity(
                registration.agent_id,
                (registration.id,),
            )
            is None
        )

        stale_resource_id = catalog_resource_id(
            registration.id,
            ResourceKind.TABLE,
            "public.missing",
        )
        await store.replace_source_permission_scopes(
            SourceReadScope(
                agent_id=registration.agent_id,
                source_id=registration.id,
                mode=SourceReadMode.SELECTED,
                resource_ids=(stale_resource_id,),
            ),
            (),
        )
        assert (
            await view.admitted_model_sensitivity(
                registration.agent_id,
                (registration.id,),
            )
            is None
        )
        with pytest.raises(SourcePermissionStateError):
            await view.readable_resource_ids(
                registration.agent_id,
                (registration.id,),
            )
        assert resource.id != stale_resource_id
    finally:
        await store.close()


async def test_attach_refresh_detach_and_reopen_preserve_narrow_scopes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    registration = _registration()
    first, resource, facet = _snapshot(registration, sync_id="sync-first")
    await store.commit_snapshot(first, registration=registration)

    assert await store.load_source_read_scope(
        registration.agent_id, registration.id
    ) == SourceReadScope.allow_all(
        agent_id=registration.agent_id,
        source_id=registration.id,
    )
    selected = SourceReadScope(
        agent_id=registration.agent_id,
        source_id=registration.id,
        mode=SourceReadMode.SELECTED,
        resource_ids=(resource.id,),
    )
    update_scope = _scope(registration, resource, facet)
    await store.replace_source_permission_scopes(selected, (update_scope,))

    second, _, _ = _snapshot(
        registration,
        sync_id="sync-second",
        row_count_estimate=999,
        sensitivity=Sensitivity.CONFIDENTIAL,
    )
    await store.commit_snapshot(second, registration=registration)
    assert (
        await store.load_source_read_scope(registration.agent_id, registration.id)
        == selected
    )
    assert await store.list_postgresql_update_scopes(
        registration.agent_id, registration.id
    ) == (update_scope,)
    await store.close()

    reopened = await SQLiteStateStore.open(path)
    assert (
        await reopened.load_source_read_scope(registration.agent_id, registration.id)
        == selected
    )
    assert await reopened.list_postgresql_update_scopes(
        registration.agent_id, registration.id
    ) == (update_scope,)
    detached = await reopened.detach_source(
        registration.agent_id,
        registration.id,
        NOW + timedelta(seconds=1),
    )
    assert not detached.active
    assert (
        await reopened.load_source_read_scope(registration.agent_id, registration.id)
        is None
    )
    assert (
        await reopened.list_postgresql_update_scopes(
            registration.agent_id, registration.id
        )
        == ()
    )
    await reopened.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_read_scopes"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM postgresql_update_scopes"
        ).fetchone() == (0,)


async def test_source_edit_atomically_hands_off_catalog_and_scopes(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    current = _registration(native_identity="postgresql:current-reader")
    current_snapshot, current_resource, current_facet = _snapshot(
        current,
        sync_id="sync-current",
    )
    await store.commit_snapshot(current_snapshot, registration=current)
    await store.replace_source_permission_scopes(
        SourceReadScope(
            agent_id=current.agent_id,
            source_id=current.id,
            mode=SourceReadMode.SELECTED,
            resource_ids=(current_resource.id,),
        ),
        (_scope(current, current_resource, current_facet),),
    )
    edited = _registration(native_identity="postgresql:edited-writer")
    edited_snapshot, edited_resource, _ = _snapshot(
        edited,
        sync_id="sync-edited",
    )
    edited_read_scope = SourceReadScope(
        agent_id=edited.agent_id,
        source_id=edited.id,
        mode=SourceReadMode.SELECTED,
        resource_ids=(edited_resource.id,),
    )

    await store.commit_source_edit(
        edited_snapshot,
        registration=edited,
        replaced_source_id=current.id,
        replaced_at=NOW + timedelta(minutes=1),
        read_scope=edited_read_scope,
    )

    detached = await store.load_source(current.agent_id, current.id)
    assert detached is not None and not detached.active
    assert await store.load_source_read_scope(current.agent_id, current.id) is None
    assert await store.list_postgresql_update_scopes(current.agent_id, current.id) == ()
    assert await store.load_source(edited.agent_id, edited.id) == edited
    assert await store.load_source_read_scope(edited.agent_id, edited.id) == (
        edited_read_scope
    )
    assert await store.list_postgresql_update_scopes(edited.agent_id, edited.id) == ()
    assert await store.load_active_source_id(edited.agent_id) == edited.id
    refs = await store.list_current_snapshot_refs(edited.agent_id, (edited.id,))
    assert len(refs) == 1 and refs[0].sync_id == "sync-edited"
    await store.close()


async def test_same_identity_source_edit_updates_connection_and_clears_updates(
    tmp_path: Path,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    current = _registration()
    current_snapshot, resource, facet = _snapshot(
        current,
        sync_id="sync-current",
    )
    await store.commit_snapshot(current_snapshot, registration=current)
    await store.replace_source_permission_scopes(
        SourceReadScope.allow_all(agent_id=current.agent_id, source_id=current.id),
        (_scope(current, resource, facet),),
    )
    edited = replace(
        current,
        display_name="Edited warehouse",
        configuration={
            **dict(current.configuration),
            "credential_ref": "keychain://agent-permissions%3Apostgresql%3Anew",
        },
        attached_at=NOW + timedelta(minutes=1),
    )
    edited_snapshot, _, _ = _snapshot(edited, sync_id="sync-edited")
    read_scope = SourceReadScope.allow_all(
        agent_id=edited.agent_id,
        source_id=edited.id,
    )

    await store.commit_source_edit(
        edited_snapshot,
        registration=edited,
        replaced_source_id=current.id,
        replaced_at=NOW + timedelta(minutes=2),
        read_scope=read_scope,
    )

    assert await store.list_sources(edited.agent_id) == (edited,)
    assert await store.load_source_read_scope(edited.agent_id, edited.id) == read_scope
    assert await store.list_postgresql_update_scopes(edited.agent_id, edited.id) == ()
    assert await store.load_active_source_id(edited.agent_id) == edited.id
    await store.close()


@pytest.mark.parametrize("damage", ("missing", "corrupt", "foreign"))
async def test_missing_corrupt_or_foreign_read_scope_fails_closed(
    tmp_path: Path,
    damage: str,
) -> None:
    path = tmp_path / f"{damage}.db"
    store = await SQLiteStateStore.open(path)
    registration = _registration(f"agent-{damage}")
    snapshot, _, _ = _snapshot(registration, sync_id=f"sync-{damage}")
    await store.commit_snapshot(snapshot, registration=registration)
    foreign_resource_id = "catalog-resource:sha256:" + "f" * 64
    if damage == "foreign":
        foreign = _registration(
            registration.agent_id,
            native_identity="postgresql:foreign-permission-tests",
        )
        foreign_snapshot, foreign_resource, _ = _snapshot(
            foreign,
            sync_id="sync-foreign-source",
        )
        await store.commit_snapshot(foreign_snapshot, registration=foreign)
        foreign_resource_id = foreign_resource.id
    with sqlite3.connect(path) as connection:
        if damage == "missing":
            connection.execute(
                "DELETE FROM source_read_scopes WHERE agent_id = ? AND source_id = ?",
                (registration.agent_id, registration.id),
            )
        elif damage == "corrupt":
            connection.execute(
                "UPDATE source_read_scopes SET data = ? "
                "WHERE agent_id = ? AND source_id = ?",
                ('{"not":"a scope"}', registration.agent_id, registration.id),
            )
        else:
            payload = json.loads(
                connection.execute(
                    "SELECT data FROM source_read_scopes "
                    "WHERE agent_id = ? AND source_id = ?",
                    (registration.agent_id, registration.id),
                ).fetchone()[0]
            )
            payload["fields"]["mode"] = "selected"
            payload["fields"]["resource_ids"] = [foreign_resource_id]
            connection.execute(
                "UPDATE source_read_scopes SET data = ? "
                "WHERE agent_id = ? AND source_id = ?",
                (
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    registration.agent_id,
                    registration.id,
                ),
            )

    with pytest.raises(SourcePermissionStateError):
        await store.load_source(registration.agent_id, registration.id)
    await store.close()


def test_authorization_fingerprint_stales_only_on_relevant_facts() -> None:
    registration = _registration()
    _, resource, facet = _snapshot(registration, sync_id="sync-original")
    original = postgresql_update_authorization_fingerprint(
        source=registration,
        resource=resource,
        facet=facet,
        allowed_assignment_columns=("amount",),
    )
    _, metadata_resource, metadata_facet = _snapshot(
        registration,
        sync_id="sync-metadata",
        row_count_estimate=999_999,
        sensitivity=Sensitivity.RESTRICTED,
    )
    assert (
        postgresql_update_authorization_fingerprint(
            source=registration,
            resource=metadata_resource,
            facet=metadata_facet,
            allowed_assignment_columns=("amount",),
        )
        == original
    )
    _, type_resource, type_facet = _snapshot(
        registration,
        sync_id="sync-type",
        amount_type="text",
    )
    assert (
        postgresql_update_authorization_fingerprint(
            source=registration,
            resource=type_resource,
            facet=type_facet,
            allowed_assignment_columns=("amount",),
        )
        != original
    )
    _, blocked_resource, blocked_facet = _snapshot(
        registration,
        sync_id="sync-blocked",
        amount_updatable=False,
    )
    with pytest.raises(ValueError, match="not eligible"):
        postgresql_update_authorization_fingerprint(
            source=registration,
            resource=blocked_resource,
            facet=blocked_facet,
            allowed_assignment_columns=("amount",),
        )


async def test_invalid_scope_replacement_rolls_back_both_families(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    registration = _registration()
    snapshot, resource, facet = _snapshot(registration, sync_id="sync-rollback")
    await store.commit_snapshot(snapshot, registration=registration)
    selected = SourceReadScope(
        agent_id=registration.agent_id,
        source_id=registration.id,
        mode=SourceReadMode.SELECTED,
        resource_ids=(resource.id,),
    )
    update_scope = _scope(registration, resource, facet)
    await store.replace_source_permission_scopes(selected, (update_scope,))

    stale = replace(update_scope, authorization_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="fingerprint is stale"):
        await store.replace_source_permission_scopes(
            SourceReadScope.allow_all(
                agent_id=registration.agent_id,
                source_id=registration.id,
            ),
            (stale,),
        )
    assert (
        await store.load_source_read_scope(registration.agent_id, registration.id)
        == selected
    )
    assert await store.list_postgresql_update_scopes(
        registration.agent_id, registration.id
    ) == (update_scope,)
    await store.close()


async def test_cancel_before_scope_transaction_start_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    registration = _registration()
    snapshot, resource, facet = _snapshot(registration, sync_id="sync-cancel")
    await store.commit_snapshot(snapshot, registration=registration)
    original = await store.load_source_read_scope(
        registration.agent_id, registration.id
    )
    entered = threading.Event()
    release = threading.Event()

    class _ControlledGate(sqlite_module._CatalogCommitGate):
        def start(self, connection: sqlite3.Connection) -> bool:
            entered.set()
            assert release.wait(timeout=5)
            return super().start(connection)

    monkeypatch.setattr(sqlite_module, "_CatalogCommitGate", _ControlledGate)
    selected = SourceReadScope(
        agent_id=registration.agent_id,
        source_id=registration.id,
        mode=SourceReadMode.SELECTED,
        resource_ids=(resource.id,),
    )
    task = asyncio.create_task(
        store.replace_source_permission_scopes(
            selected,
            (_scope(registration, resource, facet),),
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (
        await store.load_source_read_scope(registration.agent_id, registration.id)
        == original
    )
    assert (
        await store.list_postgresql_update_scopes(
            registration.agent_id, registration.id
        )
        == ()
    )


async def test_cancel_before_source_edit_transaction_keeps_current_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    current = _registration(native_identity="postgresql:current")
    current_snapshot, _, _ = _snapshot(current, sync_id="sync-current")
    await store.commit_snapshot(current_snapshot, registration=current)
    edited = _registration(native_identity="postgresql:edited")
    edited_snapshot, _, _ = _snapshot(edited, sync_id="sync-edited")
    entered = threading.Event()
    release = threading.Event()

    class _ControlledGate(sqlite_module._CatalogCommitGate):
        def start(self, connection: sqlite3.Connection) -> bool:
            entered.set()
            assert release.wait(timeout=5)
            return super().start(connection)

    monkeypatch.setattr(sqlite_module, "_CatalogCommitGate", _ControlledGate)
    task = asyncio.create_task(
        store.commit_source_edit(
            edited_snapshot,
            registration=edited,
            replaced_source_id=current.id,
            replaced_at=NOW + timedelta(minutes=1),
            read_scope=SourceReadScope.allow_all(
                agent_id=edited.agent_id,
                source_id=edited.id,
            ),
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await store.load_source(current.agent_id, current.id) == current
    assert await store.load_source(edited.agent_id, edited.id) is None
    assert await store.load_active_source_id(current.agent_id) == current.id
    assert await store.load_source_read_scope(
        current.agent_id,
        current.id,
    ) == SourceReadScope.allow_all(agent_id=current.agent_id, source_id=current.id)
    await store.close()
    await store.close()
