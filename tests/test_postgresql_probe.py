from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from _workspace_support import workspace_for

from daita import Agent
from daita.adapters import (
    PostgreSQLProbeResult,
    PostgreSQLSource,
    PostgreSQLSourceError,
)
from daita.security import EmptySecretProvider, SecretReference


class _Secrets:
    def __init__(self, value: str = "database-password") -> None:
        self.value = value
        self.references: list[SecretReference] = []

    async def resolve(self, reference: SecretReference) -> str:
        self.references.append(reference)
        return self.value


class _Connection:
    def __init__(
        self,
        rows: Sequence[object] = (),
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.rows = rows
        self.failure = failure
        self.fetches: list[tuple[str, tuple[object, ...]]] = []
        self.close_calls = 0
        self.terminate_calls = 0

    async def fetch(self, sql: str, *arguments: object):
        self.fetches.append((sql, arguments))
        if self.failure is not None:
            raise self.failure
        return self.rows

    async def close(self) -> None:
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1


class _RecordLike:
    """asyncpg.Record-shaped row without Mapping registration."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def items(self):
        return self._values.items()


class _Asyncpg:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.connects: list[dict[str, object]] = []

    async def connect(self, **kwargs: object) -> _Connection:
        self.connects.append(dict(kwargs))
        return self.connection


def _source(secret_provider: Any, *, username: str = "reader") -> PostgreSQLSource:
    return PostgreSQLSource(
        host="db.example.test",
        port=5432,
        database="warehouse",
        username=username,
        credential=SecretReference.keychain("agent:postgresql:credential"),
        schemas=("placeholder",),
        ssl_mode="require",
        secret_provider=secret_provider,
    )


async def test_postgresql_probe_uses_fixed_bounded_sql_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _Connection(
        (
            {"schema_name": "analytics", "has_base_tables": True},
            {"schema_name": "empty", "has_base_tables": False},
            {"schema_name": "information_schema", "has_base_tables": True},
            {"schema_name": "pg_catalog", "has_base_tables": True},
        )
    )
    driver = _Asyncpg(connection)
    monkeypatch.setattr("daita.adapters.postgresql._load_asyncpg", lambda: driver)
    secrets = _Secrets()

    result = await _source(secrets).probe()

    assert result == PostgreSQLProbeResult.build(
        (("analytics", True), ("empty", False))
    )
    assert secrets.references == [
        SecretReference.keychain("agent:postgresql:credential")
    ]
    assert len(driver.connects) == 1
    assert driver.connects[0]["password"] == "database-password"
    assert driver.connects[0]["timeout"] == 10.0
    assert driver.connects[0]["command_timeout"] == 10.0
    assert len(connection.fetches) == 1
    sql, arguments = connection.fetches[0]
    assert "daita:postgresql.schema_probe" in sql
    assert "information_schema" in sql
    assert "pg\\_%" in sql
    assert arguments == (101,)
    assert connection.close_calls == 1
    assert connection.terminate_calls == 0


async def test_postgresql_probe_accepts_supabase_pooler_role_names(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _Connection(())
    driver = _Asyncpg(connection)
    monkeypatch.setattr("daita.adapters.postgresql._load_asyncpg", lambda: driver)

    await _source(_Secrets(), username="postgres.project_ref").probe()

    assert driver.connects[0]["user"] == "postgres.project_ref"


async def test_postgresql_probe_accepts_asyncpg_record_shaped_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    row = _RecordLike(schema_name="analytics", has_base_tables=True)
    assert not isinstance(row, Mapping)
    connection = _Connection((row,))
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: _Asyncpg(connection),
    )

    result = await _source(_Secrets()).probe()

    assert result == PostgreSQLProbeResult.build((("analytics", True),))
    assert connection.close_calls == 1


async def test_postgresql_probe_reports_schema_presentation_truncation(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _Connection(
        tuple(
            _RecordLike(
                schema_name=f"schema_{index:03d}",
                has_base_tables=index % 2 == 0,
            )
            for index in range(101)
        )
    )
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: _Asyncpg(connection),
    )

    result = await _source(_Secrets()).probe()

    assert len(result.schemas) == 100
    assert result.truncated is True
    assert result.schemas[-1].name == "schema_099"


async def test_postgresql_probe_rejects_out_of_contract_results_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
):
    rows = tuple(
        {"schema_name": f"schema_{index}", "has_base_tables": False}
        for index in range(102)
    )
    connection = _Connection(rows)
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: _Asyncpg(connection),
    )

    with pytest.raises(PostgreSQLSourceError) as raised:
        await _source(_Secrets()).probe()

    assert raised.value.code == "postgresql_probe_result_invalid"
    assert connection.close_calls == 1


async def test_postgresql_probe_failure_is_normalized_without_server_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
):
    diagnostic = "password=DO-NOT-EXPOSE host=attacker.example"
    connection = _Connection(failure=RuntimeError(diagnostic))
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: _Asyncpg(connection),
    )

    with pytest.raises(PostgreSQLSourceError) as raised:
        await _source(_Secrets()).probe()

    assert raised.value.code == "postgresql_probe_failed"
    assert diagnostic not in str(raised.value)
    assert diagnostic not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert connection.close_calls == 1


async def test_postgresql_probe_cancellation_closes_the_connection(
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _Connection(failure=asyncio.CancelledError())
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: _Asyncpg(connection),
    )

    with pytest.raises(asyncio.CancelledError):
        await _source(_Secrets()).probe()

    assert connection.close_calls == 1


async def test_postgresql_probe_missing_secret_is_normalized_before_connect(
    monkeypatch: pytest.MonkeyPatch,
):
    driver = _Asyncpg(_Connection())
    monkeypatch.setattr("daita.adapters.postgresql._load_asyncpg", lambda: driver)

    with pytest.raises(PostgreSQLSourceError) as raised:
        await _source(EmptySecretProvider()).probe()

    assert raised.value.code == "postgresql_credential_unavailable"
    assert driver.connects == []


async def test_agent_postgresql_probe_persists_no_source_or_catalog_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    connection = _Connection(({"schema_name": "analytics", "has_base_tables": True},))
    monkeypatch.setattr(
        "daita.adapters.postgresql._load_asyncpg",
        lambda: _Asyncpg(connection),
    )
    secrets = _Secrets()
    agent = await Agent.create(
        "probe-only",
        root=tmp_path,
        secret_provider=secrets,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.probe_postgresql(
            host="db.example.test",
            port=5432,
            database="warehouse",
            username="reader",
            credential=SecretReference.keychain("agent:postgresql:probe-only"),
            ssl_mode="require",
        )

        assert tuple(schema.name for schema in result.schemas) == ("analytics",)
        assert await agent.list_sources() == ()
        assert await agent.list_catalog_resources() == ()
    finally:
        await agent.close()
