"""Real least-privileged PostgreSQL catalog and bounded-read acceptance.

This module never provisions or mutates a database.  It runs only against an
explicit test-owned table and reader role supplied through the named
environment variables below.  Missing configuration is a skipped, not passed,
live row.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import pytest

from daita import Agent, PostgreSQLSource
from daita.adapters.postgresql_query import PostgreSQLQueryBackend
from daita.catalog import CatalogSearchRequest, CatalogService
from daita.domains.data import CatalogDataView
from daita.security import EnvironmentSecretProvider, SecretReference
from daita.storage.sqlite import SQLiteOperationStore

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,62}\Z")
_REQUIRED_ENVIRONMENT = (
    "DAITA_LIVE_POSTGRES_HOST",
    "DAITA_LIVE_POSTGRES_PORT",
    "DAITA_LIVE_POSTGRES_DATABASE",
    "DAITA_LIVE_POSTGRES_READER_USER",
    "DAITA_LIVE_POSTGRES_READER_PASSWORD",
    "DAITA_LIVE_POSTGRES_SCHEMA",
    "DAITA_LIVE_POSTGRES_TABLE",
    "DAITA_LIVE_POSTGRES_EXPECTED_ROW_COUNT",
)


def _configuration() -> dict[str, str]:
    if os.environ.get("DAITA_RUN_LIVE_POSTGRES") != "1":
        pytest.skip("requires DAITA_RUN_LIVE_POSTGRES=1")
    missing = tuple(name for name in _REQUIRED_ENVIRONMENT if not os.environ.get(name))
    if missing:
        pytest.skip(f"requires environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in _REQUIRED_ENVIRONMENT}


def _identifier(configuration: dict[str, str], name: str) -> str:
    value = configuration[name]
    if _IDENTIFIER.fullmatch(value) is None:
        pytest.fail(f"{name} must name one safe PostgreSQL identifier")
    return value


@pytest.mark.integration
@pytest.mark.requires_db
async def test_live_postgresql_catalog_and_bounded_read_are_least_privileged(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    schema = _identifier(configuration, "DAITA_LIVE_POSTGRES_SCHEMA")
    table = _identifier(configuration, "DAITA_LIVE_POSTGRES_TABLE")
    username = _identifier(configuration, "DAITA_LIVE_POSTGRES_READER_USER")
    database = _identifier(configuration, "DAITA_LIVE_POSTGRES_DATABASE")
    try:
        port = int(configuration["DAITA_LIVE_POSTGRES_PORT"])
        expected_count = int(configuration["DAITA_LIVE_POSTGRES_EXPECTED_ROW_COUNT"])
    except ValueError:
        pytest.fail("PostgreSQL port and expected row count must be integers")
    if not 1 <= port <= 65_535 or expected_count < 0:
        pytest.fail("PostgreSQL port or expected row count is outside its safe range")

    secret_provider = EnvironmentSecretProvider()
    root = tmp_path / "agent-state"
    agent = await Agent.create(
        "postgresql-live-reader",
        root=root,
        secret_provider=secret_provider,
    )
    try:
        registration = await agent.attach(
            PostgreSQLSource(
                host=configuration["DAITA_LIVE_POSTGRES_HOST"],
                port=port,
                database=database,
                username=username,
                credential=SecretReference.environment(
                    "DAITA_LIVE_POSTGRES_READER_PASSWORD"
                ),
                schemas=(schema,),
                ssl_mode=os.environ.get(
                    "DAITA_LIVE_POSTGRES_SSL_MODE",
                    "disable",
                ),
                name="Phase 9 live PostgreSQL reader",
                secret_provider=secret_provider,
            )
        )
        agent_id = agent.id
        state_path = agent.home / "state.db"
    finally:
        await agent.close()

    store = await SQLiteOperationStore.open(state_path)
    try:
        resources = await store.list_resources(agent_id, registration.id)
        if not any(
            resource.native_identity == f"{schema}.{table}" for resource in resources
        ):
            pytest.fail("the configured test-owned base table was not cataloged")

        catalog_service = CatalogService(store)
        search = await catalog_service.search(
            CatalogSearchRequest(
                agent_id=agent_id,
                query=table,
                source_ids=(registration.id,),
                limit=10,
            )
        )
        if not any(hit.name == table for hit in search.hits):
            pytest.fail("the configured test-owned base table was not searchable")

        backend = PostgreSQLQueryBackend(
            store,
            CatalogDataView(store, catalog_service),
            secret_provider,
        )
        result = await backend.execute_read(
            agent_id=agent_id,
            source_id=registration.id,
            sql=f"SELECT COUNT(*) AS row_count FROM {schema}.{table}",
            parameters=(),
            max_rows=2,
            max_bytes=4_096,
        )
    finally:
        await store.close()

    assert result.projection.returned_rows == 1
    assert result.projection.truncated is False
    assert result.projection.rows[0]["row_count"] == expected_count

    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("requires asyncpg from the postgresql extra")
    connection = await asyncpg.connect(
        host=configuration["DAITA_LIVE_POSTGRES_HOST"],
        port=port,
        database=database,
        user=username,
        password=configuration["DAITA_LIVE_POSTGRES_READER_PASSWORD"],
        ssl=(
            False
            if os.environ.get("DAITA_LIVE_POSTGRES_SSL_MODE", "disable") == "disable"
            else os.environ["DAITA_LIVE_POSTGRES_SSL_MODE"]
        ),
        timeout=10,
        command_timeout=10,
    )
    try:
        privileges = await connection.fetchrow(
            """
            SELECT
                NOT rolsuper AND NOT rolcreaterole AND NOT rolcreatedb
                    AND NOT rolreplication AS ordinary_login,
                has_schema_privilege(current_user, $1, 'USAGE') AS schema_usage,
                has_schema_privilege(current_user, $1, 'CREATE') AS schema_create,
                has_table_privilege(current_user, $2, 'SELECT') AS table_select,
                has_table_privilege(current_user, $2, 'UPDATE') AS table_update,
                has_table_privilege(current_user, $2, 'INSERT') AS table_insert,
                has_table_privilege(current_user, $2, 'DELETE') AS table_delete
            FROM pg_roles
            WHERE rolname = current_user
            """,
            schema,
            f"{schema}.{table}",
        )
    finally:
        await connection.close()

    assert privileges is not None
    assert privileges["ordinary_login"] is True
    assert privileges["schema_usage"] is True
    assert privileges["schema_create"] is False
    assert privileges["table_select"] is True
    assert privileges["table_update"] is False
    assert privileges["table_insert"] is False
    assert privileges["table_delete"] is False

    credential = configuration["DAITA_LIVE_POSTGRES_READER_PASSWORD"].encode()
    for path in root.rglob("*"):
        if path.is_file() and credential in path.read_bytes():
            pytest.fail("agent state persisted PostgreSQL credential material")
