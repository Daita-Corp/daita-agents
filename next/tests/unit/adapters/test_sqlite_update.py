from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace
import sqlite3
from typing import cast

import pytest

from daita._json import canonical_json
from daita.adapters.models import SourceRegistration
from daita.adapters.sqlite_update import SQLiteUpdateBackend, SQLiteUpdateError
from daita.domains.data import (
    ResourceSchema,
    SQLiteUpdateKnownNoEffectError,
)
from daita.domains.data.controller import (
    SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
    SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
)
from daita.operations.models import TaskStatus
from daita.operations.store import OperationStore

NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
RESOURCE_REVISION = "sha256:" + "a" * 64


class Sources:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(self, registration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id: str, source_id: str):
        if agent_id == self.registration.agent_id and source_id == self.registration.id:
            return self.registration
        return None

    async def list_sources(self, agent_id: str):
        return (self.registration,) if agent_id == self.registration.agent_id else ()

    async def detach_source(self, agent_id, source_id, detached_at):
        self.registration = self.registration.detach(detached_at)
        return self.registration


class Catalog:
    def __init__(
        self,
        source_id: str,
        source_revision: str,
        resource: ResourceSchema,
    ) -> None:
        self.source_id = source_id
        self.source_revision = source_revision
        self.resource = resource

    async def resource_schemas(self, agent_id: str, source_id: str):
        if agent_id != "agent-1" or source_id != self.source_id:
            return ()
        return (self.resource,)


class Operations:
    def __init__(self) -> None:
        self.versioned: SimpleNamespace | None = None

    async def load(self, operation_id: str) -> SimpleNamespace:
        if self.versioned is None or operation_id != "operation-1":
            raise KeyError(operation_id)
        return self.versioned


def _database(path: Path, *, trigger: bool = False) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE orders (id TEXT PRIMARY KEY, status TEXT NOT NULL);
            INSERT INTO orders (id, status) VALUES ('order-1', 'pending');
            """)
        if trigger:
            connection.execute("""
                CREATE TRIGGER orders_audit AFTER UPDATE ON orders
                BEGIN
                    SELECT 1;
                END
                """)


def _backend(
    path: Path,
    *,
    write_access: bool = True,
    table_name: str = "orders",
    columns: tuple[str, ...] = ("id", "status"),
    column_declared_types: tuple[tuple[str, str], ...] = (
        ("id", "TEXT"),
        ("status", "TEXT"),
    ),
    unique_key_columns: tuple[str, ...] = ("id",),
    protected_paths: tuple[Path, ...] = (),
) -> tuple[SQLiteUpdateBackend, SourceRegistration, Operations]:
    configuration: dict[str, object] = {"path": str(path)}
    if write_access:
        configuration["write_access"] = True
    registration = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity=str(path),
        display_name="Orders",
        configuration=configuration,
        attached_at=NOW,
    )
    with sqlite3.connect(path) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    assert isinstance(schema_version, int)
    operations = Operations()
    resource = ResourceSchema(
        resource_id="resource-orders",
        source_id=registration.id,
        name=table_name,
        aliases=(f"main.{table_name}",),
        columns=columns,
        revision=RESOURCE_REVISION,
        source_revision=f"schema_version:{schema_version}",
        resource_kind="table",
        sensitivity_class="internal",
        writable=True,
        unique_key_columns=unique_key_columns,
        column_declared_types=column_declared_types,
    )
    return (
        SQLiteUpdateBackend(
            Sources(registration),
            Catalog(
                registration.id,
                f"schema_version:{schema_version}",
                resource,
            ),
            cast(OperationStore, operations),
            protected_paths=protected_paths,
        ),
        registration,
        operations,
    )


async def _preview(
    backend: SQLiteUpdateBackend,
    source_id: str,
    *,
    key_column: str = "id",
    key_value: str = "order-1",
    target_column: str = "status",
    expected_value: str = "pending",
    new_value: str = "complete",
):
    return await backend.execute_update_impact(
        agent_id="agent-1",
        source_id=source_id,
        resource_id="resource-orders",
        key_column=key_column,
        key_value=key_value,
        target_column=target_column,
        expected_value=expected_value,
        new_value=new_value,
        maximum_rows=1,
    )


def _accept_impact(operations: Operations, result) -> None:
    payload = result.evidence_payload()
    content_hash = (
        "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    )
    evidence = SimpleNamespace(
        id="evidence-impact-1",
        operation_id="operation-1",
        task_id="task-impact-1",
        kind=SQLITE_UPDATE_IMPACT_EVIDENCE_KIND,
        accepted=True,
        payload=payload,
        content_hash=content_hash,
        blob_id=None,
    )
    task = SimpleNamespace(
        id="task-impact-1",
        status=TaskStatus.SUCCEEDED,
        capability_id=SQLITE_UPDATE_IMPACT_CAPABILITY_ID,
        evidence_ids=(evidence.id,),
    )
    operations.versioned = SimpleNamespace(
        snapshot=SimpleNamespace(
            operation=SimpleNamespace(agent_id="agent-1"),
            evidence=(evidence,),
            tasks=(task,),
        )
    )


async def test_preview_then_same_operation_evidence_applies_exactly_one_update(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration, operations = _backend(path)

    impact = await _preview(backend, registration.id)
    assert impact.matched_rows == 1
    assert impact.eligible_rows == 1
    assert impact.maximum_rows == 1
    _accept_impact(operations, impact)

    result = await backend.execute_update(
        operation_id="operation-1",
        agent_id="agent-1",
        source_id=registration.id,
        resource_id="resource-orders",
        key_column="id",
        key_value="order-1",
        target_column="status",
        expected_value="pending",
        new_value="complete",
        impact_evidence_id="evidence-impact-1",
        maximum_rows=1,
    )

    assert result.affected_rows == 1
    assert result.impact_evidence_id == "evidence-impact-1"
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status, typeof(status) FROM orders WHERE id = 'order-1'"
        ).fetchone() == ("complete", "text")


async def test_update_rejects_stale_impact_without_running_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration, operations = _backend(path)
    impact = await _preview(backend, registration.id)
    _accept_impact(operations, impact)
    assert operations.versioned is not None
    operations.versioned.snapshot.evidence[0].payload = {
        **impact.evidence_payload().to_dict(),
        "eligible_rows": 0,
    }
    ran_update = False

    async def forbidden(*args, **kwargs):
        nonlocal ran_update
        ran_update = True
        raise AssertionError("update connector must not run")

    monkeypatch.setattr("daita.adapters.sqlite_update._run_update", forbidden)

    with pytest.raises(SQLiteUpdateKnownNoEffectError) as caught:
        await backend.execute_update(
            operation_id="operation-1",
            agent_id="agent-1",
            source_id=registration.id,
            resource_id="resource-orders",
            key_column="id",
            key_value="order-1",
            target_column="status",
            expected_value="pending",
            new_value="complete",
            impact_evidence_id="evidence-impact-1",
            maximum_rows=1,
        )

    assert caught.value.code == "impact_evidence_stale"
    assert ran_update is False


async def test_preview_requires_explicit_source_write_access(tmp_path: Path) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration, _ = _backend(path, write_access=False)

    with pytest.raises(SQLiteUpdateError) as caught:
        await _preview(backend, registration.id)

    assert caught.value.code == "source_write_access_required"


async def test_preview_fails_closed_for_table_with_triggers(tmp_path: Path) -> None:
    path = tmp_path / "orders.db"
    _database(path, trigger=True)
    backend, registration, _ = _backend(path)

    with pytest.raises(SQLiteUpdateError) as caught:
        await _preview(backend, registration.id)

    assert caught.value.code == "table_triggers_not_supported"


@pytest.mark.parametrize("use_hardlink", (False, True))
async def test_protected_agent_state_inode_is_rejected_for_preview_and_write(
    tmp_path: Path,
    use_hardlink: bool,
) -> None:
    protected_path = tmp_path / "state.db"
    _database(protected_path)
    source_path = protected_path
    if use_hardlink:
        source_path = tmp_path / "state-alias.db"
        os.link(protected_path, source_path)
    backend, registration, _ = _backend(
        source_path,
        protected_paths=(protected_path,),
    )

    with pytest.raises(SQLiteUpdateError) as preview_error:
        await _preview(backend, registration.id)
    assert preview_error.value.code == "protected_agent_state_source"

    with pytest.raises(SQLiteUpdateKnownNoEffectError) as update_error:
        await backend.execute_update(
            operation_id="operation-1",
            agent_id="agent-1",
            source_id=registration.id,
            resource_id="resource-orders",
            key_column="id",
            key_value="order-1",
            target_column="status",
            expected_value="pending",
            new_value="complete",
            impact_evidence_id="evidence-impact-1",
            maximum_rows=1,
        )
    assert update_error.value.code == "protected_agent_state_source"
    with sqlite3.connect(protected_path) as connection:
        assert connection.execute(
            "SELECT status FROM orders WHERE id = 'order-1'"
        ).fetchone() == ("pending",)


async def test_unicode_identifier_collision_cannot_authorize_foreign_key_cascade(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unicode-identifiers.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE "ß" (
                id TEXT PRIMARY KEY,
                status TEXT UNIQUE NOT NULL
            );
            CREATE TABLE "ss" (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL REFERENCES "ß"(status) ON UPDATE CASCADE
            );
            INSERT INTO "ß" (id, status) VALUES ('order-1', 'pending');
            INSERT INTO "ss" (id, status) VALUES ('child-1', 'pending');
            """)
    backend, registration, operations = _backend(
        path,
        table_name="ß",
        unique_key_columns=("id", "status"),
    )
    impact = await _preview(backend, registration.id)
    _accept_impact(operations, impact)

    with pytest.raises(SQLiteUpdateKnownNoEffectError) as caught:
        await backend.execute_update(
            operation_id="operation-1",
            agent_id="agent-1",
            source_id=registration.id,
            resource_id="resource-orders",
            key_column="id",
            key_value="order-1",
            target_column="status",
            expected_value="pending",
            new_value="complete",
            impact_evidence_id="evidence-impact-1",
            maximum_rows=1,
        )

    assert caught.value.code == "sqlite_update_failed"
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT status FROM "ß"').fetchone() == ("pending",)
        assert connection.execute('SELECT status FROM "ss"').fetchone() == ("pending",)


@pytest.mark.parametrize(
    ("resource_name", "columns", "declared_types", "key_column", "target_column"),
    (
        (
            "boxes",
            ("id", "min_x", "max_x"),
            (("id", "TEXT"), ("min_x", "TEXT"), ("max_x", "TEXT")),
            "id",
            "min_x",
        ),
        (
            "boxes_rowid",
            ("rowid", "nodeno"),
            (("rowid", "TEXT"), ("nodeno", "TEXT")),
            "rowid",
            "nodeno",
        ),
        (
            "orders_view",
            ("id", "status"),
            (("id", "TEXT"), ("status", "TEXT")),
            "id",
            "status",
        ),
    ),
)
async def test_live_scope_rejects_virtual_shadow_and_view_objects(
    tmp_path: Path,
    resource_name: str,
    columns: tuple[str, ...],
    declared_types: tuple[tuple[str, str], ...],
    key_column: str,
    target_column: str,
) -> None:
    path = tmp_path / f"{resource_name}.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE VIRTUAL TABLE boxes USING rtree(id, min_x, max_x);
            CREATE TABLE orders (id TEXT PRIMARY KEY, status TEXT NOT NULL);
            CREATE VIEW orders_view AS SELECT id, status FROM orders;
            """)
    backend, registration, _ = _backend(
        path,
        table_name=resource_name,
        columns=columns,
        column_declared_types=declared_types,
        unique_key_columns=(key_column,),
    )

    with pytest.raises(SQLiteUpdateError) as caught:
        await _preview(
            backend,
            registration.id,
            key_column=key_column,
            key_value="1",
            target_column=target_column,
            expected_value="1",
            new_value="2",
        )

    assert caught.value.code == "resource_not_plain_table"


async def test_same_schema_version_file_replacement_fails_catalog_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.db"
    replacement = tmp_path / "replacement.db"
    _database(path)
    backend, registration, _ = _backend(path)
    with sqlite3.connect(path) as connection:
        expected_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    with sqlite3.connect(replacement) as connection:
        connection.executescript("""
            CREATE TABLE orders (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                note TEXT
            );
            INSERT INTO orders (id, status) VALUES ('order-1', 'pending');
            """)
        connection.execute(f"PRAGMA schema_version = {expected_version}")
    os.replace(replacement, path)

    with pytest.raises(SQLiteUpdateError) as caught:
        await _preview(backend, registration.id)

    assert caught.value.code == "catalog_resource_stale"


async def test_path_replacement_during_connect_is_rejected_by_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "orders.db"
    replacement = tmp_path / "replacement.db"
    _database(path)
    _database(replacement)
    backend, registration, _ = _backend(path)
    real_connect = sqlite3.connect
    swapped = False

    def swapping_connect(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(replacement, path)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(
        "daita.adapters.sqlite_update.sqlite3.connect", swapping_connect
    )

    with pytest.raises(SQLiteUpdateError) as caught:
        await _preview(backend, registration.id)

    assert caught.value.code == "source_path_changed"


async def test_live_integer_affinity_is_rejected_even_if_catalog_claims_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "integer-status.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE orders (id TEXT PRIMARY KEY, status INTEGER NOT NULL);
            INSERT INTO orders (id, status) VALUES ('order-1', 1);
            """)
    backend, registration, _ = _backend(path)

    with pytest.raises(SQLiteUpdateError) as caught:
        await _preview(
            backend,
            registration.id,
            expected_value="1",
            new_value="01",
        )

    assert caught.value.code == "target_column_affinity_not_supported"


async def test_commit_loss_remains_ambiguous_after_durable_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration, operations = _backend(path)
    impact = await _preview(backend, registration.id)
    _accept_impact(operations, impact)

    from daita.adapters import sqlite_update as sqlite_update_module

    real_connect_source = sqlite_update_module._connect_source

    class CommitLossConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def __getattr__(self, name: str):
            return getattr(self._connection, name)

        def execute(self, statement: str, *args, **kwargs):
            result = self._connection.execute(statement, *args, **kwargs)
            if statement == "COMMIT":
                raise sqlite3.OperationalError("connection lost after commit")
            return result

    def ambiguous_connect(*args, **kwargs):
        return CommitLossConnection(real_connect_source(*args, **kwargs))

    monkeypatch.setattr(sqlite_update_module, "_connect_source", ambiguous_connect)

    with pytest.raises(SQLiteUpdateError) as caught:
        await backend.execute_update(
            operation_id="operation-1",
            agent_id="agent-1",
            source_id=registration.id,
            resource_id="resource-orders",
            key_column="id",
            key_value="order-1",
            target_column="status",
            expected_value="pending",
            new_value="complete",
            impact_evidence_id="evidence-impact-1",
            maximum_rows=1,
        )

    assert not isinstance(caught.value, SQLiteUpdateKnownNoEffectError)
    assert caught.value.code == "sqlite_update_failed"
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status FROM orders WHERE id = 'order-1'"
        ).fetchone() == ("complete",)
