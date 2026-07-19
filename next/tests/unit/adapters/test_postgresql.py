from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio
import traceback
from typing import Any

import pytest

from daita.adapters import DiscoveryRequest, PostgreSQLSource, PostgreSQLSourceError
from daita.adapters.postgresql import _close_postgresql_connection
from daita.catalog import RelationshipKind, ResourceKind
from daita.security import SecretReference
from daita import Agent

NOW = datetime(2026, 7, 19, 10, 0, tzinfo=timezone.utc)


async def test_postgresql_cleanup_is_bounded_and_terminates_stalled_connection() -> (
    None
):
    class StalledConnection:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.terminated = False

        async def close(self) -> None:
            self.started.set()
            await asyncio.Event().wait()

        def terminate(self) -> None:
            self.terminated = True

    connection = StalledConnection()

    completed = await _close_postgresql_connection(
        connection,
        timeout_seconds=0.01,
    )

    assert completed is False
    assert connection.started.is_set()
    assert connection.terminated is True


async def test_repeated_cancellation_terminates_postgresql_cleanup() -> None:
    class StalledConnection:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.terminated = False

        async def close(self) -> None:
            self.started.set()
            await asyncio.Event().wait()

        def terminate(self) -> None:
            self.terminated = True

    connection = StalledConnection()
    cleanup = asyncio.create_task(
        _close_postgresql_connection(connection, timeout_seconds=1.0)
    )
    await connection.started.wait()
    cleanup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert connection.terminated is True


class Secrets:
    def __init__(self) -> None:
        self.references: list[SecretReference] = []

    async def resolve(self, reference: SecretReference) -> str:
        self.references.append(reference)
        return "private-password"


class Transaction:
    def __init__(self) -> None:
        self.started = False
        self.committed = False
        self.rolled_back = False

    async def start(self) -> None:
        self.started = True

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class Connection:
    def __init__(self) -> None:
        self.closed = False
        self.tx = Transaction()
        self.settings: list[tuple[str, tuple[object, ...]]] = []

    def transaction(self, **kwargs: object) -> Transaction:
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return self.tx

    async def execute(self, query: str, *arguments: object) -> None:
        self.settings.append((query, arguments))

    async def fetch(self, query: str, *arguments: object):
        if "daita:postgresql.resources" in query:
            return [
                {
                    "schema_name": "public",
                    "resource_name": "customers",
                    "resource_kind": "table",
                },
                {
                    "schema_name": "public",
                    "resource_name": "orders",
                    "resource_kind": "table",
                },
                {
                    "schema_name": "public",
                    "resource_name": "unsafe_extension_values",
                    "resource_kind": "table",
                },
            ]
        if "daita:postgresql.columns" in query:
            table = arguments[1]
            if table == "customers":
                return [
                    {
                        "column_name": "id",
                        "native_type": "integer",
                        "type_schema": "pg_catalog",
                        "type_name": "int4",
                        "ordinal": 1,
                        "nullable": False,
                        "default_expression": None,
                        "primary_key_ordinal": 1,
                    },
                    {
                        "column_name": "name",
                        "native_type": "text",
                        "type_schema": "pg_catalog",
                        "type_name": "text",
                        "ordinal": 2,
                        "nullable": False,
                        "default_expression": None,
                        "primary_key_ordinal": None,
                    },
                ]
            if table == "unsafe_extension_values":
                return [
                    {
                        "column_name": "payload",
                        "native_type": "secret_type",
                        "type_schema": "public",
                        "type_name": "secret_type",
                        "ordinal": 1,
                        "nullable": False,
                        "default_expression": None,
                        "primary_key_ordinal": None,
                    }
                ]
            return [
                {
                    "column_name": "id",
                    "native_type": "integer",
                    "type_schema": "pg_catalog",
                    "type_name": "int4",
                    "ordinal": 1,
                    "nullable": False,
                    "default_expression": None,
                    "primary_key_ordinal": 1,
                },
                {
                    "column_name": "customer_id",
                    "native_type": "integer",
                    "type_schema": "pg_catalog",
                    "type_name": "int4",
                    "ordinal": 2,
                    "nullable": False,
                    "default_expression": None,
                    "primary_key_ordinal": None,
                },
                {
                    "column_name": "status",
                    "native_type": "text",
                    "type_schema": "pg_catalog",
                    "type_name": "text",
                    "ordinal": 3,
                    "nullable": True,
                    "default_expression": None,
                    "primary_key_ordinal": None,
                },
            ]
        if "daita:postgresql.indexes" in query:
            return []
        if "daita:postgresql.relationships" in query:
            return [
                {
                    "constraint_name": "orders_customer_fk",
                    "source_schema": "public",
                    "source_table": "orders",
                    "target_schema": "public",
                    "target_table": "customers",
                    "source_columns": ["customer_id"],
                    "target_columns": ["id"],
                    "match_type": "SIMPLE",
                    "on_update": "NO ACTION",
                    "on_delete": "RESTRICT",
                }
            ]
        raise AssertionError(query)

    async def fetchval(self, query: str) -> int:
        assert query == "SELECT 1"
        return 1

    async def close(self) -> None:
        self.closed = True


class Asyncpg:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.kwargs: dict[str, object] | None = None

    async def connect(self, **kwargs: object) -> Connection:
        self.kwargs = kwargs
        return self.connection


async def test_postgresql_source_persists_only_secret_reference_and_discovers_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection()
    asyncpg = Asyncpg(connection)
    secrets = Secrets()
    monkeypatch.setattr("daita.adapters.postgresql._load_asyncpg", lambda: asyncpg)
    source = PostgreSQLSource(
        host="db.internal",
        port=5432,
        database="warehouse",
        username="agent_reader",
        credential=SecretReference.environment("DAITA_POSTGRES_PASSWORD"),
        schemas=("public",),
        ssl_mode="require",
        name="Warehouse",
        secret_provider=secrets,
    )

    adapter = await source.open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW + timedelta(seconds=1),
    )
    request = DiscoveryRequest(
        agent_id="agent-1",
        source_id=adapter.registration.id,
        sync_id="sync-1",
        requested_at=NOW,
    )
    result = await adapter.discover(request)

    assert connection.settings == [
        (
            "SELECT set_config('search_path', $1, true)",
            ("pg_catalog",),
        )
    ]

    assert adapter.registration.adapter_id == "postgresql"
    assert (
        adapter.registration.configuration["credential_ref"]
        == "env:DAITA_POSTGRES_PASSWORD"
    )
    assert "private-password" not in repr(adapter.registration.configuration)
    assert "dsn" not in adapter.registration.configuration
    assert secrets.references == [
        SecretReference.environment("DAITA_POSTGRES_PASSWORD")
    ]
    assert asyncpg.kwargs == {
        "host": "db.internal",
        "port": 5432,
        "database": "warehouse",
        "user": "agent_reader",
        "password": "private-password",
        "ssl": "require",
        "timeout": 10.0,
        "command_timeout": 10.0,
    }
    assert {resource.name: resource.kind for resource in result.snapshot.resources} == {
        "customers": ResourceKind.TABLE,
        "orders": ResourceKind.TABLE,
    }
    source_revision = result.snapshot.sync.source_revision
    assert isinstance(source_revision, str)
    assert source_revision.startswith("catalog:sha256:")
    relationship = result.snapshot.relationships[0]
    assert relationship.kind is RelationshipKind.REFERENCES
    assert [
        (pair.source_field, pair.target_field) for pair in relationship.field_pairs
    ] == [("customer_id", "id")]
    assert connection.tx.started is True
    assert connection.tx.committed is True

    await adapter.close()
    assert connection.closed is True


async def test_postgresql_connector_failure_drops_untrusted_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedAsyncpg:
        async def connect(self, **kwargs: object) -> None:
            raise RuntimeError("postgresql://reader:secret@db.internal/warehouse")

    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: FailedAsyncpg(),
    )
    source = PostgreSQLSource(
        host="db.internal",
        database="warehouse",
        username="reader",
    )

    with pytest.raises(PostgreSQLSourceError) as caught:
        await source.open(agent_id="agent-1", attached_at=NOW, clock=lambda: NOW)

    rendered = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert caught.value.code == "postgresql_connect_failed"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret" not in rendered


async def test_postgresql_discovery_failure_drops_server_and_rollback_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedTransaction(Transaction):
        async def rollback(self) -> None:
            raise RuntimeError("rollback included private server detail")

    class FailedConnection(Connection):
        def __init__(self) -> None:
            super().__init__()
            self.tx = FailedTransaction()

        async def fetch(self, query: str, *arguments: object):
            raise RuntimeError("server echoed secret table identifier")

    connection = FailedConnection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )
    source = PostgreSQLSource(
        host="db.internal",
        database="warehouse",
        username="reader",
    )
    adapter = await source.open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW,
    )
    request = DiscoveryRequest(
        agent_id="agent-1",
        source_id=adapter.registration.id,
        sync_id="sync-failed",
        requested_at=NOW,
    )

    with pytest.raises(PostgreSQLSourceError) as caught:
        await adapter.discover(request)

    rendered = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert caught.value.code == "postgresql_discovery_failed"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret" not in rendered
    assert "private" not in rendered


def test_postgresql_source_rejects_raw_credentials() -> None:
    unsafe_constructor: Any = PostgreSQLSource
    with pytest.raises(TypeError):
        unsafe_constructor(
            host="db.internal",
            database="warehouse",
            username="reader",
            password="forbidden",
        )
    with pytest.raises(ValueError, match="host"):
        PostgreSQLSource(
            host="postgresql://reader:forbidden@db.internal/warehouse",
            database="warehouse",
            username="reader",
        )


async def test_public_agent_attaches_postgresql_through_generic_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )
    secrets = Secrets()
    agent = await Agent.create(
        "postgres-agent",
        root=tmp_path,
        secret_provider=secrets,
    )
    try:
        registration = await agent.attach(
            PostgreSQLSource(
                host="db.internal",
                database="warehouse",
                username="reader",
                schemas=("public",),
                ssl_mode="require",
                secret_provider=secrets,
            )
        )
    finally:
        await agent.close()

    assert registration.adapter_id == "postgresql"
    assert connection.closed is True

    reopened = await Agent.open(
        "postgres-agent",
        root=tmp_path,
        secret_provider=secrets,
    )
    await reopened.close()


async def test_missing_asyncpg_has_exact_postgresql_extra_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str):
        assert name == "asyncpg"
        raise ImportError("blocked for test")

    monkeypatch.setattr("daita.adapters.postgresql.import_module", missing)
    source = PostgreSQLSource(
        host="db.internal",
        database="warehouse",
        username="reader",
    )

    with pytest.raises(ImportError) as caught:
        await source.open(agent_id="agent-1", attached_at=NOW, clock=lambda: NOW)
    assert str(caught.value) == (
        "asyncpg is required. Install with: " "pip install 'daita-agents[postgresql]'"
    )


async def test_secret_provider_failure_does_not_cross_diagnostic_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSecrets:
        async def resolve(self, reference: SecretReference) -> str:
            raise RuntimeError("private-password")

    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(Connection()),
    )
    source = PostgreSQLSource(
        host="db.internal",
        database="warehouse",
        username="reader",
        credential=SecretReference.environment("DAITA_POSTGRES_PASSWORD"),
        secret_provider=FailingSecrets(),
    )

    with pytest.raises(PostgreSQLSourceError) as caught:
        await source.open(agent_id="agent-1", attached_at=NOW, clock=lambda: NOW)

    assert getattr(caught.value, "code", None) == "postgresql_credential_unavailable"
    assert "private-password" not in str(caught.value)
    assert caught.value.__context__ is None
