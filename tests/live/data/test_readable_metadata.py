"""Metadata SQL on a real PostgreSQL shows only what the connecting role can read.

The connection check reads the same server for what that role can change.

Needs the disposable postgres-large fixture server
(tests/fixtures/postgres-large/compose.yaml). Every case runs inside one
transaction on its ``postgres`` database that always rolls back, so it creates
its schemas, tables and roles there and leaves nothing behind:

    DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 DAITA_LARGE_POSTGRES_ADMIN_PASSWORD=... \
    pytest tests/live/data/test_readable_metadata.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from daita.adapters import postgresql_query
from daita.adapters.models import SourceRegistration
from daita.adapters.postgresql import (
    _HEALTH_SQL,
    _RESOURCES_SQL,
    _SCHEMA_PROBE_SQL,
    COLUMNS_SQL,
    _health_details,
    _load_structure,
)
from daita.adapters.postgresql_query import (
    _LIVE_RELATIONS,
    _LIVE_SCHEMA,
    _LIVE_SEARCH,
    _describe_relation,
)

pytestmark = [
    pytest.mark.requires_db,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_POSTGRES_LARGE_FIXTURE") != "1"
        or not os.environ.get("DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"),
        reason=(
            "Start tests/fixtures/postgres-large/compose.yaml, then set "
            "DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 and "
            "DAITA_LARGE_POSTGRES_ADMIN_PASSWORD."
        ),
    ),
]

_SCHEMA = "daita_meta_main"
_OTHER_SCHEMA = "daita_meta_other"
_ROLE = "daita_meta_reader"


@pytest.fixture
async def server() -> AsyncIterator[Any]:
    import asyncpg  # type: ignore[import-untyped]

    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433")),
        database="postgres",
        user="postgres",
        password=os.environ["DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"],
        ssl=False,
        timeout=5,
        command_timeout=30,
    )
    transaction = connection.transaction()
    await transaction.start()
    try:
        assert await connection.fetchval("SELECT current_database()") == "postgres"
        await connection.execute(
            f"CREATE SCHEMA {_SCHEMA}; CREATE SCHEMA {_OTHER_SCHEMA};"
            f"CREATE ROLE {_ROLE}; GRANT USAGE ON SCHEMA {_SCHEMA}, {_OTHER_SCHEMA} "
            f"TO {_ROLE}"
        )
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


def _registration() -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent",
        adapter_id="postgresql",
        native_identity="postgres-large/postgres",
        display_name="Readable metadata",
        configuration={"schemas": (_SCHEMA,)},
        attached_at=datetime(2026, 11, 2, tzinfo=UTC),
    )


async def _listed(server: Any) -> dict[str, list[str]]:
    """Table names each listing shows the role, read as that role."""

    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    try:
        structure = await _load_structure(
            server,
            _registration(),
            max_resources=100,
            max_columns=100,
            max_indexes=100,
            max_relationships=100,
        )
        schemas = [_SCHEMA]
        return {
            "discovery": [table.name for table in structure.tables],
            "live schema": [
                row[1]
                for row in await server.fetch(_LIVE_SCHEMA, schemas, _SCHEMA, 100)
            ],
            "live search": [
                row[1]
                for row in await server.fetch(
                    _LIVE_SEARCH, schemas, ["events"], ["%events%"], 100
                )
            ],
            "resources": [
                row["resource_name"]
                for row in await server.fetch(_RESOURCES_SQL, schemas, 100)
            ],
        }
    finally:
        await server.execute("RESET ROLE")


@pytest.mark.parametrize(
    ("granted", "columns", "indexes"),
    (
        # Part of the key: no key at all, yet the unique key on c is whole.
        ("a, c", ["a", "c"], ["t_c_key"]),
        # Only the second key column: nothing to raise on, nothing to show.
        ("b", ["b"], []),
    ),
)
async def test_a_composite_key_the_role_cannot_fully_read_shows_no_key(
    server: Any, granted: str, columns: list[str], indexes: list[str]
) -> None:
    await server.execute(
        f"CREATE TABLE {_SCHEMA}.t (a int, b int, c int, PRIMARY KEY (a, b), UNIQUE (c));"
        f"GRANT SELECT ({granted}) ON {_SCHEMA}.t TO {_ROLE}"
    )
    await server.execute(f"SET LOCAL ROLE {_ROLE}")

    # The server itself withholds the key, whoever reads the rows.
    rows = await server.fetch(COLUMNS_SQL, _SCHEMA, "t", 100)
    assert [row["primary_key_ordinal"] for row in rows] == [None] * len(columns)
    structure = await _load_structure(
        server,
        _registration(),
        max_resources=100,
        max_columns=100,
        max_indexes=100,
        max_relationships=100,
    )

    (table,) = structure.tables
    assert [column.name for column in table.columns] == columns
    assert [column.primary_key_ordinal for column in table.columns] == [None] * len(
        columns
    )
    assert [index.name for index in table.indexes] == indexes


async def test_describe_never_shows_a_key_its_column_cap_splits(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await server.execute(
        f"CREATE TABLE {_SCHEMA}.wide (a int, b int, c int, d int, PRIMARY KEY (a, d));"
        f"GRANT SELECT ON {_SCHEMA}.wide TO {_ROLE}"
    )
    monkeypatch.setattr(postgresql_query, "DESCRIBE_MAX_COLUMNS", 2)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")

    (relation,) = await server.fetch(_LIVE_RELATIONS, [_SCHEMA], ["wide"], "wide")
    described: Mapping[str, object] = await _describe_relation(
        server, relation, (_SCHEMA,)
    )

    assert described["columns_truncated"] is True
    assert described["primary_key"] == []


@pytest.mark.parametrize(
    ("parent", "grants", "listed"),
    (
        # A child the role reads alone is its own table.
        (_SCHEMA, ("events_2026",), ["events_2026"]),
        # A parent the role reads too holds its children.
        (_SCHEMA, ("events", "events_2026"), ["events"]),
        # A parent in a schema the source does not list holds nothing.
        (_OTHER_SCHEMA, ("events", "events_2026"), ["events_2026"]),
    ),
)
async def test_a_partition_is_listed_unless_a_readable_parent_in_the_source_holds_it(
    server: Any, parent: str, grants: tuple[str, ...], listed: list[str]
) -> None:
    await server.execute(
        f"CREATE TABLE {parent}.events (id int, day date) PARTITION BY RANGE (day);"
        f"CREATE TABLE {_SCHEMA}.events_2026 PARTITION OF {parent}.events "
        "FOR VALUES FROM ('2026-01-01') TO ('2027-01-01')"
    )
    for name in grants:
        schema = parent if name == "events" else _SCHEMA
        await server.execute(f"GRANT SELECT ON {schema}.{name} TO {_ROLE}")

    shown = await _listed(server)

    assert shown == {
        "discovery": listed,
        "live schema": listed,
        "live search": listed,
        "resources": listed,
    }
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    probe = {
        row["schema_name"]: row["has_base_tables"]
        for row in await server.fetch(_SCHEMA_PROBE_SQL, 1_000)
    }
    assert probe[_SCHEMA] is True


async def _checked_as(server: Any, role: str | None) -> dict[str, object]:
    """The connection check's answer for ``role``, or the connecting user."""

    if role is not None:
        await server.execute(f"SET LOCAL ROLE {role}")
    try:
        return _health_details(await server.fetchrow(_HEALTH_SQL, [_SCHEMA]))
    finally:
        await server.execute("RESET ROLE")


@pytest.mark.parametrize(
    ("grant", "read_only", "warnings"),
    (
        ("GRANT SELECT ON {schema}.t TO {role}", True, 0),
        # One writable table, by table grant or by a single writable column.
        ("GRANT SELECT, INSERT ON {schema}.t TO {role}", False, 1),
        ("GRANT SELECT, UPDATE (v) ON {schema}.t TO {role}", False, 1),
        (
            "GRANT SELECT ON {schema}.t TO {role}; GRANT CREATE ON SCHEMA {schema} TO {role}",
            False,
            1,
        ),
        # Writes in a schema the source does not list are not this source's.
        (
            "GRANT SELECT ON {schema}.t TO {role}; GRANT INSERT ON {other}.o TO {role}",
            True,
            0,
        ),
    ),
    ids=("select-only", "insert", "column-update", "create-on-schema", "other-schema"),
)
async def test_the_connection_check_sees_what_a_role_can_change(
    server: Any, grant: str, read_only: bool, warnings: int
) -> None:
    await server.execute(
        f"CREATE TABLE {_SCHEMA}.t (id int, v int);"
        f"CREATE TABLE {_OTHER_SCHEMA}.o (id int)"
    )
    await server.execute(grant.format(schema=_SCHEMA, other=_OTHER_SCHEMA, role=_ROLE))

    details = await _checked_as(server, _ROLE)

    assert details["role"] == _ROLE
    assert details["read_only"] is read_only
    assert len(details["warnings"]) == warnings  # type: ignore[arg-type]


async def test_the_connection_check_warns_that_the_superuser_can_do_anything(
    server: Any,
) -> None:
    details = await _checked_as(server, None)

    assert details["role"] == "postgres"
    assert details["read_only"] is False
    warnings = cast(tuple[str, ...], details["warnings"])
    assert any("superuser" in text for text in warnings)
    # It holds every built-in role implicitly; the superuser warning covers that.
    assert not any("built-in roles" in text for text in warnings)


async def test_describe_after_ddl_shows_the_current_columns(server: Any) -> None:
    await server.execute(
        f"CREATE TABLE {_SCHEMA}.shifting (id int, old_name text);"
        f"GRANT SELECT ON {_SCHEMA}.shifting TO {_ROLE}"
    )

    async def described() -> list[str]:
        await server.execute(f"SET LOCAL ROLE {_ROLE}")
        try:
            (relation,) = await server.fetch(
                _LIVE_RELATIONS, [_SCHEMA], ["shifting"], "shifting"
            )
            result = await _describe_relation(server, relation, (_SCHEMA,))
            columns = cast(list[Mapping[str, str]], result["columns"])
            return [column["name"] for column in columns]
        finally:
            await server.execute("RESET ROLE")

    assert await described() == ["id", "old_name"]
    # New columns need the grant too: the table grant covers them.
    await server.execute(f"ALTER TABLE {_SCHEMA}.shifting ADD COLUMN added int")
    assert await described() == ["id", "old_name", "added"]
    await server.execute(f"ALTER TABLE {_SCHEMA}.shifting DROP COLUMN old_name")
    assert await described() == ["id", "added"]
