from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from ipaddress import ip_address, ip_network
from uuid import UUID
import asyncio
import traceback
from typing import Protocol

import pytest

from daita._json import FrozenJsonObject
from daita.adapters.models import SourceRegistration
from daita.adapters.postgresql import PostgreSQLStructure
from daita.adapters.postgresql_query import (
    PostgreSQLQueryBackend,
    PostgreSQLQueryError,
    _json_value,
    _unique_columns,
)
from daita.domains.data import ResourceSchema
from daita.security import SecretReference

NOW = datetime(2026, 7, 19, 11, 0, tzinfo=timezone.utc)
SOURCE_REVISION = "catalog:sha256:" + "b" * 64
RESOURCE_REVISION = "sha256:" + "a" * 64


def test_postgresql_native_values_have_exact_canonical_projections() -> None:
    bit_string_type = type(
        "BitString",
        (),
        {
            "__module__": "asyncpg.pgproto.types",
            "as_string": lambda self: "101001",
        },
    )

    assert _json_value(ip_address("2001:db8::1")) == "2001:db8::1"
    assert _json_value(ip_network("10.0.0.0/24")) == "10.0.0.0/24"
    assert _json_value(timedelta(days=-2, seconds=3, microseconds=4)) == {
        "type": "interval",
        "days": -2,
        "seconds": 3,
        "microseconds": 4,
    }
    assert _json_value(bit_string_type()) == {
        "type": "bit_string",
        "value": "101001",
    }


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
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    async def resource_schemas(self, agent_id: str, source_id: str):
        if agent_id != "agent-1" or source_id != self.source_id:
            return ()
        return (
            ResourceSchema(
                resource_id="resource-orders",
                source_id=source_id,
                name="orders",
                aliases=("public.orders",),
                columns=(
                    "id",
                    "amount",
                    "created_at",
                    "external_id",
                    "payload",
                    "status",
                ),
                resource_kind="table",
                column_declared_types=(
                    ("id", "integer"),
                    ("amount", "numeric"),
                    ("created_at", "timestamp with time zone"),
                    ("external_id", "uuid"),
                    ("payload", "bytea"),
                    ("status", "text"),
                ),
                revision=RESOURCE_REVISION,
                source_revision=SOURCE_REVISION,
            ),
        )


class Secrets:
    async def resolve(self, reference: SecretReference) -> str:
        assert reference == SecretReference.environment("DAITA_POSTGRES_PASSWORD")
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


class Attribute:
    def __init__(self, name: str) -> None:
        self.name = name


class PreparedStatement(Protocol):
    def get_attributes(self) -> tuple[Attribute, ...]: ...

    def cursor(self, *arguments: object, **kwargs: object) -> Cursor: ...


class Cursor:
    def __init__(self, rows: tuple[object, ...]) -> None:
        self.rows = rows

    def __aiter__(self):
        async def values():
            for row in self.rows:
                yield row

        return values()


class Statement:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self.rows = rows
        self.cursor_arguments: tuple[object, ...] | None = None

    def get_attributes(self):
        return tuple(Attribute(name) for name in self.rows[0])

    def cursor(self, *arguments: object, **kwargs: object) -> Cursor:
        assert kwargs["prefetch"] == 3
        self.cursor_arguments = arguments
        return Cursor(self.rows)


class AsyncpgRecordLike:
    """Match asyncpg.Record's API without satisfying Mapping or Sequence."""

    def __init__(self, *values: object) -> None:
        self._values = values

    def values(self) -> tuple[object, ...]:
        return self._values

    def __len__(self) -> int:
        return len(self._values)


class RecordStatement:
    def __init__(self, names: tuple[str, ...], rows: tuple[object, ...]) -> None:
        self.names = names
        self.rows = rows

    def get_attributes(self):
        return tuple(Attribute(name) for name in self.names)

    def cursor(self, *arguments: object, **kwargs: object) -> Cursor:
        return Cursor(self.rows)


class Connection:
    def __init__(self) -> None:
        self.tx = Transaction()
        self.closed = False
        self.settings: list[tuple[str, tuple[object, ...]]] = []
        self.prepared_sql: str | None = None
        self.statement = Statement(
            (
                {
                    "id": 1,
                    "amount": Decimal("9007199254740993.000000000000000001"),
                    "created_at": NOW,
                    "external_id": UUID("12345678-1234-5678-1234-567812345678"),
                    "payload": b"abc",
                },
                {
                    "id": 2,
                    "amount": Decimal("2.00"),
                    "created_at": NOW,
                    "external_id": None,
                    "payload": None,
                },
                {
                    "id": 3,
                    "amount": Decimal("3.00"),
                    "created_at": NOW,
                    "external_id": None,
                    "payload": None,
                },
            )
        )

    def transaction(self, **kwargs: object) -> Transaction:
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return self.tx

    async def execute(self, sql: str, *arguments: object) -> None:
        self.settings.append((sql, arguments))

    async def prepare(self, sql: str, **kwargs: object) -> PreparedStatement:
        assert kwargs["timeout"] == 5.0
        self.prepared_sql = sql
        if "daita:postgresql.bounded_result" in sql:
            bounded_rows = tuple(
                {**row, "__daita_within_result_limit": True}
                for row in self.statement.rows
            )
            self.statement = Statement(bounded_rows)
        return self.statement

    async def close(self) -> None:
        self.closed = True


class Asyncpg:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    async def connect(self, **kwargs: object) -> Connection:
        return self.connection


def _registration() -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="postgresql",
        native_identity="postgresql:test",
        display_name="Warehouse",
        configuration={
            "host": "db.internal",
            "port": 5432,
            "database": "warehouse",
            "username": "reader",
            "credential_ref": "env:DAITA_POSTGRES_PASSWORD",
            "schemas": ("public",),
            "ssl_mode": "require",
        },
        attached_at=NOW,
    )


async def test_postgresql_query_revalidates_revision_then_returns_bounded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()
    connection = Connection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )

    async def current(*args, **kwargs):
        return PostgreSQLStructure((), (), SOURCE_REVISION)

    monkeypatch.setattr("daita.adapters.postgresql_query._load_structure", current)
    backend = PostgreSQLQueryBackend(
        Sources(registration),
        Catalog(registration.id),
        Secrets(),
        statement_timeout_seconds=5.0,
    )

    result = await backend.execute_read(
        agent_id="agent-1",
        source_id=registration.id,
        sql="SELECT id, amount, created_at, external_id, payload FROM public.orders WHERE id > $1 ORDER BY id",
        parameters=(0,),
        max_rows=2,
        max_bytes=10_000,
    )

    assert result.resource_revisions == (("resource-orders", RESOURCE_REVISION),)
    assert result.source_revision == SOURCE_REVISION
    assert result.projection.returned_rows == 2
    assert result.projection.truncated is True
    first = result.projection.rows[0]
    amount = first["amount"]
    assert isinstance(amount, FrozenJsonObject)
    assert amount.to_dict() == {
        "type": "decimal",
        "value": "9007199254740993.000000000000000001",
    }
    assert first["created_at"] == NOW.isoformat()
    assert first["external_id"] == "12345678-1234-5678-1234-567812345678"
    payload = first["payload"]
    assert isinstance(payload, FrozenJsonObject)
    assert payload["encoding"] == "base64"
    assert connection.prepared_sql is not None
    assert "daita:postgresql.bounded_result" in connection.prepared_sql
    assert "pg_catalog.pg_column_size" in connection.prepared_sql
    assert "pg_catalog.octet_length(pg_catalog.to_jsonb" in connection.prepared_sql
    assert "<= 10000" in connection.prepared_sql
    assert "LIMIT 3" in connection.prepared_sql
    assert "FROM ONLY public.orders" in connection.prepared_sql
    assert (
        "SELECT set_config('search_path', $1, true)",
        ("pg_catalog",),
    ) in connection.settings
    assert connection.statement.cursor_arguments == (0,)
    assert connection.tx.started is True
    assert connection.tx.committed is True
    assert connection.closed is True


async def test_invalid_postgresql_sql_is_rejected_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()
    opened = False

    async def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("connection must not open")

    monkeypatch.setattr("daita.adapters.postgresql_query._connect", forbidden)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )

    with pytest.raises(PostgreSQLQueryError) as caught:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            sql="DELETE FROM public.orders WHERE id = $1",
            parameters=(1,),
            max_rows=10,
            max_bytes=1_000,
        )

    assert caught.value.code == "query_revalidation_failed"
    assert opened is False


async def test_missing_query_dependency_preserves_actionable_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()

    async def missing(*args: object, **kwargs: object) -> None:
        raise ImportError(
            "asyncpg is required. Install with: "
            "pip install 'daita-agents[postgresql]'"
        )

    monkeypatch.setattr("daita.adapters.postgresql_query._connect", missing)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )

    with pytest.raises(ImportError) as caught:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            sql="SELECT id FROM public.orders",
            parameters=(),
            max_rows=10,
            max_bytes=1_000,
        )

    assert str(caught.value) == (
        "asyncpg is required. Install with: " "pip install 'daita-agents[postgresql]'"
    )


def test_postgresql_result_columns_preserve_case_distinct_names() -> None:
    assert _unique_columns(("id", "Id", "id")) == ("id", "Id", "id__2")
    assert _unique_columns(("id", "id", "id__2")) == (
        "id",
        "id__3",
        "id__2",
    )


async def test_stale_postgresql_structure_never_prepares_user_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = _registration()
    connection = Connection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )

    async def stale(*args, **kwargs):
        return PostgreSQLStructure((), (), "catalog:sha256:" + "0" * 64)

    monkeypatch.setattr("daita.adapters.postgresql_query._load_structure", stale)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )

    with pytest.raises(PostgreSQLQueryError) as caught:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            sql="SELECT id FROM public.orders",
            parameters=(),
            max_rows=10,
            max_bytes=1_000,
        )

    assert caught.value.code == "catalog_source_stale"
    assert connection.prepared_sql is None
    assert connection.tx.rolled_back is True
    assert connection.closed is True


async def test_postgresql_query_cancellation_rolls_back_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingConnection(Connection):
        def __init__(self) -> None:
            super().__init__()
            self.preparing = asyncio.Event()

        async def prepare(self, sql: str, **kwargs: object) -> Statement:
            self.preparing.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    registration = _registration()
    connection = BlockingConnection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )

    async def current(*args, **kwargs):
        return PostgreSQLStructure((), (), SOURCE_REVISION)

    monkeypatch.setattr("daita.adapters.postgresql_query._load_structure", current)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )
    task = asyncio.create_task(
        backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            sql="SELECT id FROM public.orders",
            parameters=(),
            max_rows=10,
            max_bytes=1_000,
        )
    )
    await connection.preparing.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert connection.tx.rolled_back is True
    assert connection.closed is True


async def test_postgresql_query_failure_drops_untrusted_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedConnection(Connection):
        async def prepare(self, sql: str, **kwargs: object) -> Statement:
            raise RuntimeError("secret bound value and private SQL")

    registration = _registration()
    connection = FailedConnection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )

    async def current(*args, **kwargs):
        return PostgreSQLStructure((), (), SOURCE_REVISION)

    monkeypatch.setattr("daita.adapters.postgresql_query._load_structure", current)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )

    with pytest.raises(PostgreSQLQueryError) as caught:
        await backend.execute_read(
            agent_id="agent-1",
            source_id=registration.id,
            sql="SELECT id FROM public.orders",
            parameters=(),
            max_rows=10,
            max_bytes=1_000,
        )

    rendered = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert caught.value.code == "postgresql_query_failed"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret" not in rendered


async def test_postgresql_query_accepts_asyncpg_record_api_without_mapping_abc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordConnection(Connection):
        def __init__(self) -> None:
            super().__init__()
            self.record_statement = RecordStatement(
                ("id", "status"),
                (AsyncpgRecordLike(7, "open"),),
            )

        async def prepare(self, sql: str, **kwargs: object) -> RecordStatement:
            if "daita:postgresql.bounded_result" in sql:
                self.record_statement = RecordStatement(
                    ("id", "status", "__daita_within_result_limit"),
                    (AsyncpgRecordLike(7, "open", True),),
                )
            return self.record_statement

    registration = _registration()
    connection = RecordConnection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )

    async def current(*args, **kwargs):
        return PostgreSQLStructure((), (), SOURCE_REVISION)

    monkeypatch.setattr("daita.adapters.postgresql_query._load_structure", current)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )

    result = await backend.execute_read(
        agent_id="agent-1",
        source_id=registration.id,
        sql="SELECT id, status FROM public.orders",
        parameters=(),
        max_rows=10,
        max_bytes=1_000,
    )

    assert result.projection.rows[0].to_dict() == {"id": 7, "status": "open"}


async def test_postgresql_truncates_oversized_value_at_server_projection_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedConnection(Connection):
        def __init__(self) -> None:
            super().__init__()
            self.statement = Statement(({"id": 1},))

        async def prepare(self, sql: str, **kwargs: object) -> Statement:
            self.prepared_sql = sql
            if "daita:postgresql.bounded_result" in sql:
                self.statement = Statement(
                    (
                        {
                            "__daita_column_0": 1,
                            "__daita_within_result_limit": True,
                        },
                        {
                            "__daita_column_0": None,
                            "__daita_within_result_limit": False,
                        },
                    )
                )
            return self.statement

    registration = _registration()
    connection = OversizedConnection()
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: Asyncpg(connection),
    )

    async def current(*args, **kwargs):
        return PostgreSQLStructure((), (), SOURCE_REVISION)

    monkeypatch.setattr("daita.adapters.postgresql_query._load_structure", current)
    backend = PostgreSQLQueryBackend(
        Sources(registration), Catalog(registration.id), Secrets()
    )

    result = await backend.execute_read(
        agent_id="agent-1",
        source_id=registration.id,
        sql="SELECT id FROM public.orders",
        parameters=(),
        max_rows=2,
        max_bytes=1_000,
    )

    assert tuple(row.to_dict() for row in result.projection.rows) == ({"id": 1},)
    assert result.projection.total_rows == 2
    assert result.projection.truncated is True
    assert result.projection.truncation_reasons == ("byte_limit",)
    assert connection.prepared_sql is not None
    assert "pg_catalog.pg_column_size" in connection.prepared_sql
    assert "pg_catalog.octet_length(pg_catalog.to_jsonb" in connection.prepared_sql
    assert connection.tx.committed is True
    assert connection.closed is True
