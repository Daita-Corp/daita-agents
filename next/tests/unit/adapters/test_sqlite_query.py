from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from daita.adapters.models import SourceRegistration
from daita.adapters.sqlite_query import SQLiteQueryBackend, SQLiteQueryError
from daita.domains.data import ResourceSchema

NOW = datetime(2026, 7, 18, 21, 0, tzinfo=timezone.utc)
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
    def __init__(self, source_id: str, source_revision: str) -> None:
        self.source_id = source_id
        self.source_revision = source_revision

    async def resource_schemas(self, agent_id: str, source_id: str):
        if agent_id != "agent-1" or source_id != self.source_id:
            return ()
        return (
            ResourceSchema(
                resource_id="resource-orders",
                source_id=source_id,
                name="orders",
                aliases=("main.orders",),
                columns=("id", "status"),
                revision=RESOURCE_REVISION,
                source_revision=self.source_revision,
            ),
        )


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT NOT NULL);
            INSERT INTO orders (status) VALUES ('complete'), ('pending'), ('complete');
            """)


def _backend(path: Path) -> tuple[SQLiteQueryBackend, SourceRegistration]:
    registration = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity=str(path),
        display_name="Orders",
        configuration={"path": str(path)},
        attached_at=NOW,
    )
    with sqlite3.connect(path) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    assert isinstance(schema_version, int)
    return (
        SQLiteQueryBackend(
            Sources(registration),
            Catalog(registration.id, f"schema_version:{schema_version}"),
        ),
        registration,
    )


async def test_query_backend_revalidates_then_returns_bounded_untrusted_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration = _backend(path)

    result = await backend.execute_read(
        agent_id="agent-1",
        source_id=registration.id,
        sql="SELECT id, status FROM orders ORDER BY id",
        parameters=(),
        max_rows=2,
        max_bytes=1_000,
    )

    assert result.columns == ("id", "status")
    assert result.resource_ids == ("resource-orders",)
    assert result.projection.returned_rows == 2
    assert result.projection.truncated is True
    assert result.projection.truncation_reasons == ("row_limit",)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


async def test_invalid_sql_is_rejected_before_connector_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration = _backend(path)
    opened = False

    async def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("connector I/O must not run")

    monkeypatch.setattr("daita.adapters.sqlite_query._run_query", forbidden)

    with pytest.raises(SQLiteQueryError) as caught:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            sql="DELETE FROM orders WHERE id = 1",
            parameters=(),
            max_rows=10,
            max_bytes=1_000,
        )

    assert caught.value.code == "query_revalidation_failed"
    assert opened is False


async def test_query_backend_fails_closed_for_another_agent_or_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.db"
    _database(path)
    backend, registration = _backend(path)

    with pytest.raises(SQLiteQueryError) as caught:
        await backend.execute_read(
            agent_id="agent-other",
            source_id=registration.id,
            sql="SELECT id FROM orders",
            parameters=(),
            max_rows=10,
            max_bytes=1_000,
        )

    assert caught.value.code == "source_not_available"
