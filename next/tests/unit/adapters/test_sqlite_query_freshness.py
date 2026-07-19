from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from daita.adapters.models import SourceRegistration
from daita.adapters.sqlite_query import SQLiteQueryBackend, SQLiteQueryError
from daita.capabilities import CapabilityRegistry, ExecutionRequest
from daita.domains.data.capabilities import SQLiteReadResult, sqlite_query_declarations
from daita.domains.data.sql import ResourceSchema

NOW = datetime(2026, 7, 18, 22, 0, tzinfo=timezone.utc)
RESOURCE_REVISION = "sha256:" + "a" * 64


class Sources:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(
        self,
        registration: SourceRegistration,
    ) -> SourceRegistration:
        self.registration = registration
        return registration

    async def load_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration | None:
        if agent_id == self.registration.agent_id and source_id == self.registration.id:
            return self.registration
        return None

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        return (self.registration,) if agent_id == self.registration.agent_id else ()

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ) -> SourceRegistration:
        assert agent_id == self.registration.agent_id
        assert source_id == self.registration.id
        self.registration = self.registration.detach(detached_at)
        return self.registration


class Catalog:
    def __init__(
        self,
        source_id: str,
        source_revision: str | None,
        *,
        resource_revision: str | None = RESOURCE_REVISION,
    ) -> None:
        self.source_id = source_id
        self.source_revision = source_revision
        self.resource_revision = resource_revision

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        if agent_id != "agent-1" or source_id != self.source_id:
            return ()
        return (
            ResourceSchema(
                resource_id="resource-orders",
                source_id=source_id,
                name="orders",
                aliases=("main.orders",),
                columns=("id", "status"),
                revision=self.resource_revision,
                source_revision=self.source_revision,
            ),
        )


def _database(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT NOT NULL);
            INSERT INTO orders (status) VALUES ('complete'), ('pending');
            """)
        version = connection.execute("PRAGMA schema_version").fetchone()[0]
    assert isinstance(version, int)
    return f"schema_version:{version}"


def _backend(
    path: Path,
    source_revision: str | None,
    *,
    resource_revision: str | None = RESOURCE_REVISION,
) -> tuple[SQLiteQueryBackend, SourceRegistration]:
    registration = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity=str(path),
        display_name="Orders",
        configuration={"path": str(path)},
        attached_at=NOW,
    )
    return (
        SQLiteQueryBackend(
            Sources(registration),
            Catalog(
                registration.id,
                source_revision,
                resource_revision=resource_revision,
            ),
        ),
        registration,
    )


async def _read(
    backend: SQLiteQueryBackend,
    registration: SourceRegistration,
) -> SQLiteReadResult:
    return await backend.execute_read(
        agent_id="agent-1",
        source_id=registration.id,
        sql="SELECT id, status FROM orders ORDER BY id",
        parameters=(),
        max_rows=10,
        max_bytes=4_096,
    )


async def test_current_catalog_revision_executes_on_the_matching_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "current.db"
    source_revision = _database(path)
    backend, registration = _backend(path, source_revision)

    result = await _read(backend, registration)

    assert result.source_revision == source_revision
    assert result.resource_revisions == (("resource-orders", RESOURCE_REVISION),)
    assert result.projection.returned_rows == 2


async def test_schema_mutation_rejects_the_stale_catalog_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stale.db"
    source_revision = _database(path)
    backend, registration = _backend(path, source_revision)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE orders ADD COLUMN note TEXT")

    with pytest.raises(SQLiteQueryError) as caught:
        await _read(backend, registration)

    assert caught.value.code == "catalog_source_stale"


async def test_query_revision_provenance_satisfies_the_evidence_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.db"
    source_revision = _database(path)
    backend, registration = _backend(path, source_revision)
    declarations = sqlite_query_declarations("agent-1", backend)
    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=declarations.executors,
        tool_views=declarations.tool_views,
    )
    capability = declarations.capabilities[0]
    executor = declarations.executors[0]

    candidate = await executor.execute(
        ExecutionRequest(
            operation_id="operation-1",
            task_id="task-1",
            turn_id="turn-1",
            capability_id=capability.id,
            executor_id=capability.executor_id,
            attempt=1,
            fencing_token=1,
            arguments={
                "source_id": registration.id,
                "sql": "SELECT id FROM orders ORDER BY id",
            },
        )
    )
    accepted = registry.validate_evidence(capability.id, candidate)

    assert accepted.payload["source_revision"] == source_revision
    revisions = accepted.payload["resource_revisions"]
    assert isinstance(revisions, tuple)
    assert revisions[0]["resource_id"] == "resource-orders"
    assert revisions[0]["revision"] == RESOURCE_REVISION


async def test_stale_catalog_fails_before_the_user_statement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "before-user-sql.db"
    source_revision = _database(path)
    backend, registration = _backend(path, source_revision)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE new_resource (id INTEGER PRIMARY KEY)")
    user_sql_executed = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal user_sql_executed
        user_sql_executed = True
        raise AssertionError("stale catalog must fail before user SQL")

    monkeypatch.setattr(
        "daita.adapters.sqlite_query._execute_user_query",
        forbidden,
    )

    with pytest.raises(SQLiteQueryError) as caught:
        await _read(backend, registration)

    assert caught.value.code == "catalog_source_stale"
    assert user_sql_executed is False


@pytest.mark.parametrize(
    ("source_revision", "resource_revision", "expected_code"),
    [
        (None, RESOURCE_REVISION, "catalog_provenance_missing"),
        ("schema:1", RESOURCE_REVISION, "catalog_source_revision_invalid"),
        ("schema_version:01", RESOURCE_REVISION, "catalog_source_revision_invalid"),
        ("schema_version:1", None, "catalog_provenance_missing"),
    ],
)
async def test_missing_or_malformed_provenance_fails_before_source_query_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_revision: str | None,
    resource_revision: str | None,
    expected_code: str,
) -> None:
    path = tmp_path / f"invalid-{expected_code}.db"
    _database(path)
    backend, registration = _backend(
        path,
        source_revision,
        resource_revision=resource_revision,
    )
    source_query_started = False

    async def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal source_query_started
        source_query_started = True
        raise AssertionError("invalid provenance must fail before source query I/O")

    monkeypatch.setattr("daita.adapters.sqlite_query._run_query", forbidden)

    with pytest.raises(SQLiteQueryError) as caught:
        await _read(backend, registration)

    assert caught.value.code == expected_code
    assert source_query_started is False
