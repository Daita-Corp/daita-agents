from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
from daita.storage.sqlite import SQLiteStateStore
from daita.storage import sqlite as sqlite_module
from daita.storage.sqlite_migrations import migration_rows
from daita.storage.sqlite_records import (
    PostgreSQLUpdateScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    postgresql_update_authorization_fingerprint,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


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
        assert "postgresql_write_admissions" not in tables
        assert (
            tuple(
                connection.execute(
                    "SELECT ordinal, migration_id, checksum "
                    "FROM state_migrations ORDER BY ordinal"
                )
            )
            == migration_rows()
        )


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
    await store.close()
