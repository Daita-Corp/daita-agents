"""A PostgreSQL change probe moves exactly when a catalog read would record a change.

The probe is one statement per check. It must move for every structural thing a
revision stores and for every grant, revoke and role change that alters what the
connecting role can read, and it must stay still for what a revision does not
store (rows, statistics, table comments, enum labels).

Needs the disposable postgres-large fixture server
(tests/fixtures/postgres-large/compose.yaml). Every case runs inside one
transaction on its ``postgres`` database that always rolls back, so it creates
its schemas, tables and roles there and leaves nothing behind:

    DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 DAITA_LARGE_POSTGRES_ADMIN_PASSWORD=... \
    pytest tests/live/data/test_change_probe.py -s
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from daita.adapters.models import DiscoveryRequest, SourceProbe, SourceRegistration
from daita.adapters.postgresql import _batches, _load_structure, probe_tokens
from tests.support.source_contract import (
    ProbeMutation,
    assert_change_detection_contract,
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

_MAIN = "daita_probe_main"
_AUX = "daita_probe_aux"
_WIDE = "daita_probe_wide"
_ROLE = "daita_probe_reader"
_GRANTOR = "daita_probe_grantor"
_NOW = datetime(2026, 11, 7, tzinfo=UTC)


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
        timeout=10,
        command_timeout=60,
    )
    # The same isolation lets discovery's own transaction nest as a savepoint.
    transaction = connection.transaction(isolation="repeatable_read")
    await transaction.start()
    try:
        assert await connection.fetchval("SELECT current_database()") == "postgres"
        await connection.execute(
            f"CREATE SCHEMA {_MAIN}; CREATE SCHEMA {_AUX}; CREATE SCHEMA {_WIDE};"
            f"CREATE ROLE {_ROLE}; CREATE ROLE {_GRANTOR};"
            f"GRANT USAGE ON SCHEMA {_MAIN}, {_AUX}, {_WIDE} TO {_ROLE}, {_GRANTOR}"
        )
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


def _registration(*schemas: str) -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent",
        adapter_id="postgresql",
        native_identity="postgres-large/postgres",
        display_name="Change probe",
        configuration={"schemas": schemas},
        attached_at=_NOW,
    )


@dataclass
class _Counting:
    """Passes everything to the connection and records each tagged statement."""

    inner: Any
    tags: list[str] = field(default_factory=list)

    async def fetch(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if match := re.search(r"/\* (daita:postgresql\.\w+) \*/", sql):
            self.tags.append(match[1])
        return await self.inner.fetch(sql, *args, **kwargs)


class _Watched:
    """One role's view of the schemas, and an administrator who changes them."""

    def __init__(self, server: Any, role: str, *schemas: str) -> None:
        self.server, self.role, self.schemas = server, role, schemas
        self.registration = _registration(*schemas)

    async def probe(self) -> SourceProbe:
        await self.server.execute(f"SET LOCAL ROLE {self.role}")
        try:
            tokens = await probe_tokens(self.server, self.schemas)
        finally:
            await self.server.execute("RESET ROLE")
        return SourceProbe(checked_at=_NOW, tokens=tokens)

    async def revisions(self) -> dict[str, dict[str, str]]:
        """What a catalog read as this role records now: schema -> table -> revision."""

        await self.server.execute(f"SET LOCAL ROLE {self.role}")
        try:
            structure = await _load_structure(
                self.server,
                self.registration,
                max_resources=1_000,
                max_columns=512,
                max_indexes=256,
                max_relationships=2_000,
            )
        finally:
            await self.server.execute("RESET ROLE")
        request = DiscoveryRequest(
            agent_id="agent",
            source_id=self.registration.id,
            sync_id="sync",
            requested_at=_NOW,
        )
        _, detail = _batches(request, structure)
        result: dict[str, dict[str, str]] = {}
        for entry in detail.entries:
            schema, _, _ = entry.resource.native_identity.partition(".")
            result.setdefault(schema, {})[
                entry.resource.native_identity
            ] = entry.revision.revision
        return result

    def change(self, *statements: str) -> Callable[[], Any]:
        async def apply() -> None:
            for statement in statements:
                await self.server.execute(statement)

        return apply


async def _build_structure(server: Any) -> None:
    await server.execute(f"""
        CREATE TYPE {_MAIN}.mood AS ENUM ('sad', 'ok');
        CREATE TABLE {_MAIN}.people (id int, name text, mood {_MAIN}.mood);
        CREATE TABLE {_MAIN}.orders (id int PRIMARY KEY, person_id int);
        CREATE TABLE {_AUX}.items (id int PRIMARY KEY, order_id int);
        CREATE TABLE {_MAIN}.events (id int, day date) PARTITION BY RANGE (day);
        CREATE TABLE {_MAIN}.events_2026 PARTITION OF {_MAIN}.events
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
        CREATE TABLE {_MAIN}.events_2027 (id int, day date);
        CREATE VIEW {_MAIN}.people_view AS SELECT id FROM {_MAIN}.people;
        GRANT SELECT ON ALL TABLES IN SCHEMA {_MAIN}, {_AUX} TO {_ROLE}
        """)


async def test_probe_moves_exactly_with_what_a_catalog_read_records(
    server: Any,
) -> None:
    await _build_structure(server)
    watched = _Watched(server, _ROLE, _MAIN, _AUX)
    people, orders = f"{_MAIN}.people", f"{_MAIN}.orders"
    main, both = frozenset({_MAIN}), frozenset({_MAIN, _AUX})

    def move(
        name: str, *statements: str, moves: frozenset[str] = main
    ) -> ProbeMutation:
        return ProbeMutation(name, watched.change(*statements), moves)

    def still(name: str, *statements: str) -> ProbeMutation:
        return ProbeMutation(name, watched.change(*statements))

    await assert_change_detection_contract(
        watched.probe,
        [
            still("rows inserted", f"INSERT INTO {people} VALUES (1, 'a', 'ok')"),
            still("statistics refreshed", f"ANALYZE {people}"),
            still("table comment", f"COMMENT ON TABLE {people} IS 'who we know'"),
            still("enum label added", f"ALTER TYPE {_MAIN}.mood ADD VALUE 'happy'"),
            still("table nobody may read", f"CREATE TABLE {_MAIN}.hidden (id int)"),
            move(
                "table added",
                f"CREATE TABLE {_MAIN}.extra (id int)",
                f"GRANT SELECT ON {_MAIN}.extra TO {_ROLE}",
            ),
            move("table renamed", f"ALTER TABLE {_MAIN}.extra RENAME TO more"),
            move("table dropped", f"DROP TABLE {_MAIN}.more"),
            move("column added", f"ALTER TABLE {people} ADD COLUMN age int"),
            move("column renamed", f"ALTER TABLE {people} RENAME COLUMN age TO years"),
            move(
                "column type changed",
                f"ALTER TABLE {people} ALTER name TYPE varchar(20)",
            ),
            move(
                "column made required", f"ALTER TABLE {people} ALTER name SET NOT NULL"
            ),
            move(
                "column default set", f"ALTER TABLE {people} ALTER years SET DEFAULT 7"
            ),
            move("column comment", f"COMMENT ON COLUMN {people}.name IS 'full name'"),
            move("column dropped", f"ALTER TABLE {people} DROP COLUMN years"),
            move("index added", f"CREATE INDEX people_name ON {people} (name)"),
            move("index dropped", "DROP INDEX " + f"{_MAIN}.people_name"),
            move("primary key added", f"ALTER TABLE {people} ADD PRIMARY KEY (id)"),
            move(
                "foreign key added",
                f"ALTER TABLE {orders} ADD CONSTRAINT o_p "
                f"FOREIGN KEY (person_id) REFERENCES {people} (id)",
            ),
            move("foreign key dropped", f"ALTER TABLE {orders} DROP CONSTRAINT o_p"),
            move(
                "primary key dropped",
                f"ALTER TABLE {people} DROP CONSTRAINT people_pkey",
            ),
            move(
                "foreign key across schemas added",
                f"ALTER TABLE {_AUX}.items ADD CONSTRAINT i_o "
                f"FOREIGN KEY (order_id) REFERENCES {orders} (id)",
                moves=both,
            ),
            move(
                "foreign key across schemas dropped",
                f"ALTER TABLE {_AUX}.items DROP CONSTRAINT i_o",
                moves=both,
            ),
            move(
                "view created",
                f"CREATE VIEW {_MAIN}.names AS SELECT id FROM {people}",
                f"GRANT SELECT ON {_MAIN}.names TO {_ROLE}",
            ),
            move(
                "view replaced with another column",
                f"CREATE OR REPLACE VIEW {_MAIN}.names AS SELECT id, name FROM {people}",
            ),
            still(
                "view replaced with the same columns",
                f"CREATE OR REPLACE VIEW {_MAIN}.names AS "
                f"SELECT id, name FROM {people} WHERE id > 0",
            ),
            move(
                "partition attached",
                f"ALTER TABLE {_MAIN}.events ATTACH PARTITION {_MAIN}.events_2027 "
                "FOR VALUES FROM ('2027-01-01') TO ('2028-01-01')",
            ),
            move(
                "partition detached",
                f"ALTER TABLE {_MAIN}.events DETACH PARTITION {_MAIN}.events_2027",
            ),
        ],
        revisions=watched.revisions,
    )


async def test_probe_moves_with_grants_revokes_and_role_membership(
    server: Any,
) -> None:
    """Measured as the probing role: what it may read is what the probe hashes."""

    await server.execute(f"""
        CREATE TABLE {_MAIN}.a (id int PRIMARY KEY, secret text, note text);
        CREATE TABLE {_MAIN}.b (id int);
        GRANT SELECT ON {_MAIN}.a TO {_ROLE}
        """)
    watched = _Watched(server, _ROLE, _MAIN)
    main = frozenset({_MAIN})

    def move(name: str, *statements: str) -> ProbeMutation:
        return ProbeMutation(name, watched.change(*statements), main)

    await assert_change_detection_contract(
        watched.probe,
        [
            move("grant select", f"GRANT SELECT ON {_MAIN}.b TO {_ROLE}"),
            move("revoke select", f"REVOKE SELECT ON {_MAIN}.b FROM {_ROLE}"),
            move(
                "revoke one column of a table grant",
                f"REVOKE SELECT ON {_MAIN}.a FROM {_ROLE}",
                f"GRANT SELECT (id, note) ON {_MAIN}.a TO {_ROLE}",
            ),
            move("grant a column", f"GRANT SELECT (secret) ON {_MAIN}.a TO {_ROLE}"),
            move(
                "revoke a column", f"REVOKE SELECT (secret) ON {_MAIN}.a FROM {_ROLE}"
            ),
            move(
                "role membership granted",
                f"GRANT SELECT ON {_MAIN}.b TO {_GRANTOR}",
                f"GRANT {_GRANTOR} TO {_ROLE}",
            ),
            move("role membership revoked", f"REVOKE {_GRANTOR} FROM {_ROLE}"),
            move(
                "schema usage revoked", f"REVOKE USAGE ON SCHEMA {_MAIN} FROM {_ROLE}"
            ),
            move("schema usage granted", f"GRANT USAGE ON SCHEMA {_MAIN} TO {_ROLE}"),
        ],
        revisions=watched.revisions,
    )


async def _build_wide_tables(server: Any, count: int) -> None:
    """``count`` tables of 30 columns, a key, a second index and some comments."""

    types = ("int", "bigint", "bool", "date", "timestamptz", f"{_WIDE}.state")
    columns = ", ".join(f"c{i:02} {types[i % len(types)]}" for i in range(1, 30))
    await server.execute(
        f"CREATE TYPE {_WIDE}.state AS ENUM ('new', 'open', 'done');"
        f"GRANT USAGE ON TYPE {_WIDE}.state TO {_ROLE}"
    )
    await server.execute(
        f"""
        DO $$
        BEGIN
            FOR i IN 1..{count} LOOP
                EXECUTE format(
                    'CREATE TABLE {_WIDE}.%I (id bigint PRIMARY KEY, {columns})',
                    'wide_' || lpad(i::text, 4, '0'));
                EXECUTE format('CREATE INDEX ON {_WIDE}.%I (c02)',
                    'wide_' || lpad(i::text, 4, '0'));
                IF i % 3 = 0 THEN
                    EXECUTE format('COMMENT ON COLUMN {_WIDE}.%I.c01 IS %L',
                        'wide_' || lpad(i::text, 4, '0'), 'a comment');
                END IF;
            END LOOP;
        END $$
        """,
        timeout=600,
    )
    await server.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {_WIDE} TO {_ROLE}")


async def test_probing_fifteen_hundred_tables_is_one_quick_statement(
    server: Any,
) -> None:
    await _build_wide_tables(server, 1_500)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    counting = _Counting(server)

    seconds = []
    for _ in range(3):
        started = time.perf_counter()
        tokens = await probe_tokens(counting, (_WIDE,))
        seconds.append(time.perf_counter() - started)

    print(
        "\nprobe of 1,500 tables:",
        ", ".join(f"{value * 1_000:.0f} ms" for value in seconds),
    )
    assert list(tokens) == [_WIDE]
    assert counting.tags == ["daita:postgresql.change_probe"] * 3


async def test_fill_describes_what_lacks_detail_and_keeps_the_rest(
    server: Any,
) -> None:
    await _build_wide_tables(server, 6)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    registration = _registration(_WIDE)
    request = DiscoveryRequest(
        agent_id="agent", source_id=registration.id, sync_id="s", requested_at=_NOW
    )

    async def read(**options: Any) -> Any:
        structure = await _load_structure(
            server,
            registration,
            max_columns=512,
            max_indexes=256,
            max_relationships=2_000,
            **options,
        )
        return structure, _batches(request, structure)[1].entries

    first, kept = await read(max_resources=2)
    assert len(first.tables) == 2 and first.truncated
    middle, kept = await read(max_resources=2, previous=kept, fill=True)
    assert len(middle.tables) == 4 and middle.truncated
    last, kept = await read(max_resources=2, previous=kept, fill=True)
    assert len(last.tables) == 6 and not last.truncated
