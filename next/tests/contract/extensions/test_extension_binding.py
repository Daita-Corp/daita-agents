from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pytest

from daita.extensions import (
    ExtensionBinding,
    ExtensionBindingConflictError,
    ExtensionKind,
)
from daita.identity import AgentIdentity
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteCorruptionError, SQLiteOperationStore

NOW = datetime(2026, 7, 20, 20, 0, tzinfo=timezone.utc)


def _binding(*, version: str = "1.0.0") -> ExtensionBinding:
    return ExtensionBinding(
        id="example",
        version=version,
        kind=ExtensionKind.CAPABILITY_PROVIDER,
        declaration_fingerprint="sha256:" + "a" * 64,
        manifest_fingerprint="sha256:" + "b" * 64,
    )


async def test_sqlite_extension_set_is_normalized_immutable_and_reopen_exact(
    tmp_path,
) -> None:
    path = tmp_path / "extensions.db"
    identity = AgentIdentity(
        id="agent-extensions",
        display_name="Extensions",
        created_at=NOW,
    )
    bindings = (_binding(),)
    store = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    await store.initialize_identity(identity)

    assert await store.load_extension_bindings(identity.id) is None
    assert await store.bind_extension_bindings(identity.id, bindings) == bindings
    assert await store.bind_extension_bindings(identity.id, bindings) == bindings
    with pytest.raises(ExtensionBindingConflictError, match="another extension set"):
        await store.bind_extension_bindings(
            identity.id,
            (_binding(version="2.0.0"),),
        )
    await store.close()

    reopened = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    assert await reopened.load_extension_bindings(identity.id) == bindings
    await reopened.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        assert connection.execute(
            "SELECT extension_count FROM agent_extension_sets"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT position, extension_id, version, kind, "
            "declaration_fingerprint, manifest_fingerprint "
            "FROM agent_extensions"
        ).fetchone() == (
            0,
            "example",
            "1.0.0",
            "capability_provider",
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE agent_extensions SET version = '2.0.0'")


async def test_migration_seventeen_invents_no_legacy_extension_binding(
    tmp_path,
) -> None:
    path = tmp_path / "legacy-v16.db"
    identity = AgentIdentity(
        id="agent-legacy-extensions",
        display_name="Legacy extensions",
        created_at=NOW,
    )
    legacy = await sqlite_owner._open_with_migrations(
        path,
        migrations=sqlite_owner._MIGRATIONS[:16],
        busy_timeout_ms=5_000,
        backup_path=None,
        clock=lambda: NOW,
    )
    await legacy.initialize_identity(identity)
    await legacy.close()

    store = await SQLiteOperationStore.open(
        path,
        backup_path=tmp_path / "legacy-v16.backup.db",
        clock=lambda: NOW,
    )
    try:
        assert await store.load_extension_bindings(identity.id) is None
    finally:
        await store.close()


async def test_extension_binding_fingerprint_tamper_fails_closed(tmp_path) -> None:
    path = tmp_path / "tampered-extensions.db"
    identity = AgentIdentity(
        id="agent-tampered-extensions",
        display_name="Tampered extensions",
        created_at=NOW,
    )
    store = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    await store.initialize_identity(identity)
    await store.bind_extension_bindings(identity.id, (_binding(),))

    store._connection.execute("DROP TRIGGER agent_extension_sets_reject_update")
    store._connection.execute(
        "UPDATE agent_extension_sets SET fingerprint = ?",
        ("sha256:" + "c" * 64,),
    )
    store._connection.commit()

    with pytest.raises(SQLiteCorruptionError, match="extension set"):
        await store.load_extension_bindings(identity.id)
    await store.close()
