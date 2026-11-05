"""Attaching a PostgreSQL source reads each table's detail in a few bulk statements.

The bulk statements must say exactly what the per-table statements say, however
many tables there are and whatever the connecting role can read. Attaching a
thousand tables must stay quick over a slow link, and a cut by time must leave
the connection usable.

Needs the disposable postgres-large fixture server
(tests/fixtures/postgres-large/compose.yaml). Every case runs inside one
transaction on its ``postgres`` database that always rolls back, so it creates
its schemas, tables and roles there and leaves nothing behind. The timed cases
reach the server through a local proxy that delays every trip:

    DAITA_RUN_POSTGRES_LARGE_FIXTURE=1 DAITA_LARGE_POSTGRES_ADMIN_PASSWORD=... \
    pytest tests/live/data/test_bulk_discovery.py -s
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from daita.adapters import postgresql
from daita.adapters.models import DiscoveryRequest, SourceRegistration
from daita.adapters.postgresql import (
    _BULK_COLUMNS_SQL,
    _BULK_INDEXES_SQL,
    _HEALTH_SQL,
    _RESOURCES_SQL,
    COLUMNS_SQL,
    INDEXES_SQL,
    PostgreSQLResourceAdapter,
    _index,
    _load_structure,
    _TableStructure,
    columns_from_rows,
)
from daita.catalog.models import ResourceKind, TabularFacet

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

_MAIN = "daita_bulk_main"
_AUX = "daita_bulk_aux"
_OFF = "daita_bulk_off"
_WIDE = "daita_bulk_wide"
_ROLE = "daita_bulk_reader"
_NOW = datetime(2026, 11, 4, tzinfo=UTC)
_ACCEPT_SECONDS = 5.0
_ONE_WAY = 0.025


class _LatencyProxy:
    """A TCP proxy to the fixture server that delays each trip by ``delay``.

    Each direction keeps byte order through a queue and a sender that waits until
    a chunk is due, so the delay is per trip and not per chunk.
    """

    def __init__(self) -> None:
        self.delay = 0.0
        self.chunks = 0
        self.port = 0
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for writer in self._writers:
            writer.close()
        if self._server is not None:
            await self._server.wait_closed()

    async def _accept(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            "127.0.0.1", int(os.environ.get("DAITA_LARGE_POSTGRES_PORT", "55433"))
        )
        self._writers += [client_writer, upstream_writer]
        for source, sink in (
            (client_reader, upstream_writer),
            (upstream_reader, client_writer),
        ):
            queue: asyncio.Queue[tuple[float, bytes] | None] = asyncio.Queue()
            self._spawn(self._receive(source, queue))
            self._spawn(self._send(queue, sink))

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _receive(
        self,
        source: asyncio.StreamReader,
        queue: asyncio.Queue[tuple[float, bytes] | None],
    ) -> None:
        loop = asyncio.get_running_loop()
        while chunk := await source.read(65_536):
            self.chunks += 1
            queue.put_nowait((loop.time() + self.delay, chunk))
        queue.put_nowait(None)

    async def _send(
        self,
        queue: asyncio.Queue[tuple[float, bytes] | None],
        sink: asyncio.StreamWriter,
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            while (item := await queue.get()) is not None:
                due, chunk = item
                if (wait := due - loop.time()) > 0:
                    await asyncio.sleep(wait)
                sink.write(chunk)
                await sink.drain()
        except ConnectionError:
            pass
        finally:
            sink.close()


@pytest.fixture
async def proxy() -> AsyncIterator[_LatencyProxy]:
    latency = _LatencyProxy()
    await latency.start()
    try:
        yield latency
    finally:
        await latency.stop()


async def _connect(proxy: _LatencyProxy) -> Any:
    import asyncpg  # type: ignore[import-untyped]

    return await asyncpg.connect(
        host="127.0.0.1",
        port=proxy.port,
        database="postgres",
        user="postgres",
        password=os.environ["DAITA_LARGE_POSTGRES_ADMIN_PASSWORD"],
        ssl=False,
        timeout=10,
        command_timeout=30,
    )


@pytest.fixture
async def server(proxy: _LatencyProxy) -> AsyncIterator[Any]:
    connection = await _connect(proxy)
    # The same isolation lets discovery's own transaction nest as a savepoint.
    transaction = connection.transaction(isolation="repeatable_read")
    await transaction.start()
    try:
        assert await connection.fetchval("SELECT current_database()") == "postgres"
        await connection.execute(
            f"CREATE SCHEMA {_MAIN}; CREATE SCHEMA {_AUX}; CREATE SCHEMA {_OFF};"
            f"CREATE SCHEMA {_WIDE}; CREATE ROLE {_ROLE};"
            f"GRANT USAGE ON SCHEMA {_MAIN}, {_AUX}, {_OFF}, {_WIDE} TO {_ROLE}"
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
        display_name="Bulk discovery",
        configuration={"schemas": schemas},
        attached_at=_NOW,
    )


def _request(registration: SourceRegistration, max_resources: int) -> DiscoveryRequest:
    return DiscoveryRequest(
        agent_id=registration.agent_id,
        source_id=registration.id,
        sync_id="sync",
        requested_at=_NOW,
        max_resources=max_resources,
    )


@dataclass
class _Counting:
    """Passes everything to the connection and records each tagged statement."""

    inner: Any
    tags: list[str] = field(default_factory=list)

    def _count(self, sql: str) -> None:
        if match := re.search(r"/\* (daita:postgresql\.\w+) \*/", sql):
            self.tags.append(match[1])

    async def fetch(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self._count(sql)
        return await self.inner.fetch(sql, *args, **kwargs)

    async def fetchrow(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self._count(sql)
        return await self.inner.fetchrow(sql, *args, **kwargs)

    async def fetchval(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self._count(sql)
        return await self.inner.fetchval(sql, *args, **kwargs)

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self._count(sql)
        return await self.inner.execute(sql, *args, **kwargs)

    def cursor(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self._count(sql)
        return self.inner.cursor(sql, *args, **kwargs)

    def transaction(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.transaction(*args, **kwargs)


# --- the identity schema -----------------------------------------------------


async def _build_identity_schema(server: Any) -> None:
    await server.execute(f"""
        CREATE TYPE {_MAIN}.mood AS ENUM ('sad', 'ok', 'happy');
        CREATE DOMAIN {_MAIN}.mood_d AS {_MAIN}.mood;
        CREATE TABLE {_MAIN}.people (
            id int, kind int,
            mood {_MAIN}.mood NOT NULL, feeling {_MAIN}.mood_d,
            moods {_MAIN}.mood[], noted timestamptz DEFAULT now(),
            PRIMARY KEY (id, kind)
        );
        COMMENT ON TABLE {_MAIN}.people IS 'People we know';
        COMMENT ON COLUMN {_MAIN}.people.mood IS 'How they feel';
        COMMENT ON COLUMN {_MAIN}.people.feeling IS 'How they feel, later';

        CREATE TABLE {_MAIN}.pairs (a int, b int, c int, PRIMARY KEY (a, b), UNIQUE (c));
        CREATE TABLE {_MAIN}.pairs_whole (a int, b int, PRIMARY KEY (a, b));

        CREATE TABLE {_MAIN}.events (id int, day date) PARTITION BY RANGE (day);
        CREATE TABLE {_MAIN}.events_2026 PARTITION OF {_MAIN}.events
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
        CREATE TABLE {_MAIN}.logs (id int, day date) PARTITION BY RANGE (day);
        CREATE TABLE {_MAIN}.logs_2026 PARTITION OF {_MAIN}.logs
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
        CREATE TABLE {_OFF}.audit (id int, day date) PARTITION BY RANGE (day);
        CREATE TABLE {_MAIN}.audit_2026 PARTITION OF {_OFF}.audit
            FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

        CREATE VIEW {_MAIN}.people_view AS SELECT id, mood FROM {_MAIN}.people;
        CREATE MATERIALIZED VIEW {_MAIN}.people_mat AS SELECT id, kind FROM {_MAIN}.people;

        CREATE TABLE {_MAIN}.shared (id int);
        CREATE TABLE {_AUX}.shared (id int, extra int);
        CREATE INDEX shared_idx ON {_MAIN}.shared (id);
        CREATE INDEX shared_idx ON {_AUX}.shared (id, extra);

        CREATE TABLE {_MAIN}.indexed (a int, b int, c int, d int);
        CREATE INDEX ix_expr ON {_MAIN}.indexed ((a + b));
        CREATE INDEX ix_part ON {_MAIN}.indexed (a) WHERE b > 0;
        CREATE INDEX ix_incl ON {_MAIN}.indexed (a) INCLUDE (c);
        CREATE UNIQUE INDEX ix_uniq ON {_MAIN}.indexed (b, c);

        CREATE TABLE {_MAIN}.empty ();
        CREATE TABLE {_MAIN}.empty_expr ();
        CREATE INDEX ix_constant ON {_MAIN}.empty_expr ((1));

        CREATE TABLE {_MAIN}.secret (x int);

        CREATE TABLE {_MAIN}."Customers" (id int PRIMARY KEY, "Name" int, name int);
        CREATE INDEX "Customers_Name_Idx" ON {_MAIN}."Customers" ("Name");
        CREATE TABLE {_MAIN}.customers (id int PRIMARY KEY, "Name" int, name int);
        CREATE INDEX customers_name_idx ON {_MAIN}.customers (name);
        CREATE TABLE {_AUX}.orders (
            id int PRIMARY KEY,
            customer_id int REFERENCES {_MAIN}.customers (id)
        );

        GRANT SELECT ON ALL TABLES IN SCHEMA {_MAIN}, {_AUX}, {_OFF} TO {_ROLE};
        REVOKE SELECT ON {_MAIN}.secret, {_MAIN}.pairs, {_MAIN}.logs FROM {_ROLE};
        GRANT SELECT (a, c) ON {_MAIN}.pairs TO {_ROLE}
        """)


_ROLES = pytest.mark.parametrize("role", (None, _ROLE), ids=("superuser", "reader"))
_BUDGETS = pytest.mark.parametrize(
    ("max_resources", "max_columns", "max_indexes"),
    (
        (1_000, 512, 256),
        (1_000, 2, 256),
        (1_000, 512, 1),
        (3, 512, 256),
    ),
    ids=("defaults", "two-columns", "one-index", "three-tables"),
)


async def _reference(
    server: Any,
    schemas: tuple[str, ...],
    max_resources: int,
    max_columns: int,
    max_indexes: int,
) -> tuple[tuple[_TableStructure, ...], bool]:
    """What the per-table statements say: one columns and one indexes read each."""

    resources = await server.fetch(_RESOURCES_SQL, list(schemas), max_resources + 1)
    truncated = len(resources) > max_resources
    tables = []
    for row in resources[:max_resources]:
        schema, name = row["schema_name"], row["resource_name"]
        column_rows = await server.fetch(COLUMNS_SQL, schema, name, max_columns + 1)
        index_rows = await server.fetch(INDEXES_SQL, schema, name, max_indexes + 1)
        columns = columns_from_rows(column_rows, max_columns)
        known = {column.name for column in columns}
        truncated |= len(column_rows) > max_columns or len(index_rows) > max_indexes
        tables.append(
            _TableStructure(
                schema=schema,
                name=name,
                kind=(
                    ResourceKind.VIEW
                    if row["resource_kind"] == "view"
                    else ResourceKind.TABLE
                ),
                columns=columns,
                indexes=tuple(
                    index
                    for index_row in index_rows[:max_indexes]
                    if (index := _index(index_row, known)) is not None
                ),
            )
        )
    return tuple(sorted(tables, key=lambda item: (item.schema, item.name))), truncated


async def _structure(
    connection: Any,
    registration: SourceRegistration,
    max_resources: int = 1_000,
    max_columns: int = 512,
    max_indexes: int = 256,
) -> Any:
    return await _load_structure(
        connection,
        registration,
        max_resources=max_resources,
        max_columns=max_columns,
        max_indexes=max_indexes,
        max_relationships=2_000,
    )


@_ROLES
@_BUDGETS
async def test_bulk_structure_equals_the_per_table_reads(
    server: Any,
    role: str | None,
    max_resources: int,
    max_columns: int,
    max_indexes: int,
) -> None:
    await _build_identity_schema(server)
    if role is not None:
        await server.execute(f"SET LOCAL ROLE {role}")
    registration = _registration(_MAIN, _AUX)

    expected, expected_truncated = await _reference(
        server, (_MAIN, _AUX), max_resources, max_columns, max_indexes
    )
    structure = await _structure(
        server, registration, max_resources, max_columns, max_indexes
    )

    assert expected
    assert structure.tables == expected
    assert structure.truncated is expected_truncated
    if max_resources == 1_000:
        assert {"Customers", "customers", "shared", "empty", "empty_expr"} <= {
            table.name for table in structure.tables
        }


async def _bulk_rows(
    server: Any, sql: str, wanted: list[tuple[str, str]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    cursor = await server.cursor(
        sql, [s for s, _ in wanted], [n for _, n in wanted], 10_000
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in await cursor.fetch(100_000):
        grouped[(row["schema_name"], row["resource_name"])].append(dict(row))
    return grouped


def _without(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k not in keys} for row in rows]


@_ROLES
async def test_bulk_rows_equal_the_single_table_rows(
    server: Any, role: str | None
) -> None:
    await _build_identity_schema(server)
    if role is not None:
        await server.execute(f"SET LOCAL ROLE {role}")
    wanted = [
        (row["schema_name"], row["resource_name"])
        for row in await server.fetch(_RESOURCES_SQL, [_MAIN, _AUX], 1_000)
    ]

    columns = await _bulk_rows(server, _BULK_COLUMNS_SQL, wanted)
    indexes = await _bulk_rows(server, _BULK_INDEXES_SQL, wanted)

    for schema, name in wanted:
        single_columns = [
            dict(row) for row in await server.fetch(COLUMNS_SQL, schema, name, 10_000)
        ]
        single_indexes = [
            dict(row) for row in await server.fetch(INDEXES_SQL, schema, name, 10_000)
        ]
        assert (
            _without(
                columns[(schema, name)], "schema_name", "resource_name", "column_rank"
            )
            == single_columns
        ), (schema, name)
        assert (
            _without(
                indexes[(schema, name)], "schema_name", "resource_name", "index_rank"
            )
            == single_indexes
        ), (schema, name)

    people = {row["column_name"]: row for row in columns[(_MAIN, "people")]}
    assert people["mood"]["enum_values"] == ["sad", "ok", "happy"]
    assert people["feeling"]["enum_values"] == ["sad", "ok", "happy"]
    assert people["mood"]["comment"] == "How they feel"
    assert people["feeling"]["comment"] == "How they feel, later"
    assert [people[c]["primary_key_ordinal"] for c in ("id", "kind")] == [1, 2]
    pairs = {row["column_name"]: row for row in columns[(_MAIN, "pairs")]}
    if role is None:
        assert [pairs[c]["primary_key_ordinal"] for c in ("a", "b")] == [1, 2]
    else:
        assert set(pairs) == {"a", "c"}
        assert {row["primary_key_ordinal"] for row in pairs.values()} == {None}
        assert [row["index_name"] for row in indexes[(_MAIN, "pairs")]] == [
            "pairs_c_key"
        ]
    for schema in (_MAIN, _AUX):
        assert "shared_idx" in {
            row["index_name"] for row in indexes[(schema, "shared")]
        }


# --- how many statements --------------------------------------------------------


async def _build_wide_tables(server: Any, count: int) -> None:
    """``count`` tables of 30 columns, a key, a second index and some comments.

    Types without a TOAST table keep each table at three locks, so a thousand
    tables fit in one transaction.
    """

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


@pytest.mark.parametrize("count", (1, 100))
async def test_a_structure_read_takes_a_handful_of_statements(
    server: Any, count: int
) -> None:
    await _build_wide_tables(server, count)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    counting = _Counting(server)

    structure = await _structure(counting, _registration(_WIDE))

    assert len(structure.tables) == count
    assert len(counting.tags) <= 6, counting.tags


# --- latency ---------------------------------------------------------------------


def _tables(result: Any) -> dict[str, TabularFacet]:
    snapshot = result.snapshot
    names = {item.id: item.name for item in snapshot.resources}
    return {
        names[facet.resource_id]: TabularFacet.from_payload(facet.payload)
        for facet in snapshot.facets
    }


async def _attach(
    server: Any, proxy: _LatencyProxy, max_resources: int
) -> tuple[Any, _Counting, dict[str, float], int]:
    """Time the connect, the connection check and discovery over a 50 ms link."""

    registration = _registration(_WIDE)
    counting = _Counting(server)
    adapter = PostgreSQLResourceAdapter(
        registration=registration, connection=counting, clock=lambda: _NOW
    )
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    proxy.delay = _ONE_WAY
    chunks_before = proxy.chunks
    try:
        started = time.perf_counter()
        extra = await _connect(proxy)
        await extra.close()
        connected = time.perf_counter()
        await counting.fetchrow(_HEALTH_SQL, [_WIDE])
        checked = time.perf_counter()
        result = await adapter.discover(_request(registration, max_resources))
        finished = time.perf_counter()
    finally:
        proxy.delay = 0.0
    timings = {
        "connect": connected - started,
        "health": checked - connected,
        "discover": finished - checked,
        "total": finished - started,
    }
    return result, counting, timings, proxy.chunks - chunks_before


def _show(label: str, timings: dict[str, float], chunks: int, tags: int) -> None:
    print(
        f"\n{label}: "
        + ", ".join(f"{name} {seconds:.2f}s" for name, seconds in timings.items())
        + f", {chunks} chunks through the proxy, {tags} tagged statements"
    )


async def test_attaching_a_thousand_tables_takes_under_five_seconds_at_50ms(
    server: Any, proxy: _LatencyProxy
) -> None:
    await _build_wide_tables(server, 1_000)

    result, counting, timings, chunks = await _attach(server, proxy, 1_000)

    _show("1,000 tables", timings, chunks, len(counting.tags))
    tables = _tables(result)
    assert timings["total"] < _ACCEPT_SECONDS
    assert result.snapshot.sync.truncated is False
    assert len(result.snapshot.resources) == 1_000
    assert {len(facet.columns) for facet in tables.values()} == {30}
    assert len([t for t in counting.tags if t != "daita:postgresql.health"]) <= 6


async def test_a_larger_source_is_cut_by_the_table_budget_not_by_time(
    server: Any, proxy: _LatencyProxy
) -> None:
    await _build_wide_tables(server, 1_500)

    result, counting, timings, chunks = await _attach(server, proxy, 1_000)
    _show("1,500 tables, default budget", timings, chunks, len(counting.tags))

    names = [item.name for item in result.snapshot.resources]
    assert names == [f"wide_{i:04}" for i in range(1, 1_001)]
    assert result.snapshot.sync.truncated is True
    tables = _tables(result)
    assert {len(facet.columns) for facet in tables.values()} == {30}
    assert {len(facet.indexes) for facet in tables.values()} == {2}
    assert timings["discover"] < 30

    full, _, full_timings, full_chunks = await _attach(server, proxy, 1_500)
    _show("1,500 tables, budget 1,500", full_timings, full_chunks, 0)
    assert len(full.snapshot.resources) == 1_500
    assert full.snapshot.sync.truncated is False


async def test_a_discovery_cut_by_time_leaves_the_transaction_usable(
    server: Any, proxy: _LatencyProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _build_wide_tables(server, 20)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    registration = _registration(_WIDE)
    expected, _ = await _reference(server, (_WIDE,), 1_000, 512, 256)

    monkeypatch.setattr(postgresql, "_DISCOVERY_SECONDS", 0.04)
    proxy.delay = _ONE_WAY
    try:
        structure = await _structure(server, registration)
        still_works = await server.fetchval("SELECT 1")
    finally:
        proxy.delay = 0.0

    assert structure.truncated is True
    assert set(structure.tables) <= set(expected)
    assert still_works == 1
    assert await server.fetchval("SELECT 1") == 1


async def test_a_discovery_cut_midway_keeps_whole_tables_in_listing_order(
    server: Any, proxy: _LatencyProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _build_wide_tables(server, 400)
    await server.execute(f"SET LOCAL ROLE {_ROLE}")
    registration = _registration(_WIDE)
    expected, _ = await _reference(server, (_WIDE,), 1_000, 512, 256)

    monkeypatch.setattr(postgresql, "_BULK_FETCH_ROWS", 100)
    monkeypatch.setattr(postgresql, "_DISCOVERY_SECONDS", 4.0)
    proxy.delay = _ONE_WAY
    try:
        structure = await _structure(server, registration)
        still_works = await server.fetchval("SELECT 1")
    finally:
        proxy.delay = 0.0

    print(f"\nkept {len(structure.tables)} of 400 tables")
    assert structure.truncated is True
    assert 0 < len(structure.tables) < 400
    assert structure.tables == expected[: len(structure.tables)]
    assert {len(t.columns) for t in structure.tables} == {30}
    assert {len(t.indexes) for t in structure.tables} == {2}
    assert still_works == 1
