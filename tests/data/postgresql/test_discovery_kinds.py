"""Discovery keeps what it is given, and what a limit cuts is marked truncated."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daita.adapters import postgresql as postgresql_module
from daita.adapters.models import (
    DiscoveryRequest,
    SourceRegistration,
    source_registration_id,
)
from daita.adapters.postgresql import PostgreSQLResourceAdapter, _load_structure
from daita.catalog.models import (
    ResourceKind,
    SourceCatalogSnapshot,
    TabularFacet,
)
from daita.storage.sqlite import SQLiteStateStore

_NOW = datetime(2026, 11, 2, tzinfo=UTC)

Row = Mapping[str, object]
_STATUS = {
    "column_name": "status",
    "native_type": "sales.order_status",
    "type_schema": "sales",
    "type_name": "order_status",
    "ordinal": 1,
    "nullable": False,
    "default_expression": None,
    "collation": None,
    "is_identity": False,
    "is_generated": False,
    "is_updatable": True,
    "primary_key_ordinal": None,
    "comment": "Where the order is.",
    "enum_values": ["open", "shipped"],
}


def _column(name: str, ordinal: int, key: tuple[int, int] | None = None) -> Row:
    """One column row; ``key`` is its primary-key position and the key's size."""

    return {
        **_STATUS,
        "column_name": name,
        "native_type": "integer",
        "type_schema": "pg_catalog",
        "type_name": "int4",
        "ordinal": ordinal,
        "comment": None,
        "enum_values": None,
        "primary_key_ordinal": None if key is None else key[0],
        "primary_key_size": None if key is None else key[1],
    }


def _index(name: str, *columns: str) -> Row:
    return {
        "index_name": name,
        "index_kind": "btree",
        "columns": list(columns),
        "simple_columns": True,
        "is_unique": False,
        "write_conflict_supported": False,
        "predicate": None,
    }


def _foreign_key(name: str, source: str, column: str, target: str = "orders") -> Row:
    return {
        "constraint_name": name,
        "source_schema": "sales",
        "source_table": source,
        "target_schema": "sales",
        "target_table": target,
        "source_columns": [column],
        "target_columns": [column],
        "match_type": "SIMPLE",
        "on_update": "NO ACTION",
        "on_delete": "NO ACTION",
    }


class _Cursor:
    """The rows a bulk statement streams, handed out a batch per round trip."""

    def __init__(self, connection: _Connection, tag: str, rows: list[Row]) -> None:
        self._connection = connection
        self._tag = tag
        self._rows = rows
        self._fetches = 0

    async def fetch(self, n: int, *, timeout: float | None = None) -> list[Row]:
        stalled = self._connection.stall == (self._tag, self._fetches)
        self._fetches += 1
        await self._connection.round_trip(1, timeout, stall=stalled)
        batch, self._rows = self._rows[:n], self._rows[n:]
        return batch


class _Connection:
    """Answers discovery's tagged queries from fixed rows, honoring each LIMIT.

    Every driver round trip waits ``latency`` and is counted in ``round_trips``;
    a wait longer than its ``timeout`` raises ``TimeoutError`` as asyncpg does.
    """

    def __init__(
        self,
        tables: Sequence[tuple[str, str, str]] = (
            ("sales", "orders", "table"),
            ("sales", "open_orders", "view"),
        ),
        *,
        columns: Mapping[tuple[str, str], Sequence[Row]] | None = None,
        indexes: Mapping[tuple[str, str], Sequence[Row]] | None = None,
        relationships: Sequence[Row] = (),
        latency: float = 0.0,
        stall: tuple[str, int] | None = None,
    ) -> None:
        self.tables = list(tables)
        self.columns = dict(columns or {})
        self.indexes = dict(indexes or {})
        self.relationships = list(relationships)
        self.latency = latency
        # (statement tag, 0-based cursor fetch) that waits far past any budget.
        self.stall = stall
        self.reads: list[tuple[str, str]] = []
        self.statements: list[str] = []
        self.round_trips = 0
        self._prepared: set[str] = set()

    async def round_trip(
        self, count: int, timeout: float | None = None, *, stall: bool = False
    ) -> None:
        self.round_trips += count
        wait = self.latency * count + (60.0 if stall else 0.0)
        await asyncio.wait_for(asyncio.sleep(wait), timeout)

    def _tag(self, sql: str) -> str:
        match = re.search(r"daita:postgresql\.(\w+)", sql)
        assert match, sql
        self.statements.append(match.group(1))
        return match.group(1)

    async def fetch(self, sql: str, *args: object) -> list[Row]:
        limit = args[-1]
        assert isinstance(limit, int)
        tag = self._tag(sql)
        # The first run of a statement text is prepared, then executed.
        await self.round_trip(1 if sql in self._prepared else 2)
        self._prepared.add(sql)
        rows: Sequence[Row]
        if tag == "resources":
            rows = [
                {"schema_name": schema, "resource_name": name, "resource_kind": kind}
                for schema, name, kind in sorted(self.tables)
            ]
        elif tag == "relationships":
            rows = self.relationships
        else:
            rows = []
        return list(rows[:limit])

    async def cursor(
        self,
        sql: str,
        schemas: Sequence[str],
        names: Sequence[str],
        limit: int,
        timeout: float | None = None,
    ) -> _Cursor:
        tag = self._tag(sql)
        await self.round_trip(2, timeout)
        wanted = list(zip(schemas, names, strict=True))
        if tag == "bulk_columns":
            self.reads.extend(wanted)
            source, default = self.columns, [_STATUS]
        else:
            source, default = self.indexes, []
        return _Cursor(
            self,
            tag,
            [
                {**row, "schema_name": schema, "resource_name": name}
                for schema, name in wanted
                for row in list(source.get((schema, name), default))[:limit]
            ],
        )

    # Discovery runs in one transaction; this connection is its own. A bulk
    # read inside it is a savepoint.
    def transaction(self, **_: object) -> _Connection:
        return self

    async def __aenter__(self) -> None:
        await self.round_trip(1)

    async def __aexit__(self, *_: object) -> None:
        await self.round_trip(1)

    async def start(self) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def execute(self, *_: object) -> None: ...


async def _discover(
    connection: _Connection,
    *,
    registration: SourceRegistration | None = None,
    sync_id: str = "sync",
    **request: Any,
) -> SourceCatalogSnapshot:
    registration = registration or _registration()
    adapter = PostgreSQLResourceAdapter(
        registration=registration, connection=connection, clock=lambda: _NOW
    )
    result = await adapter.discover(
        DiscoveryRequest(
            agent_id=registration.agent_id,
            source_id=registration.id,
            sync_id=sync_id,
            requested_at=_NOW,
            **request,
        )
    )
    return result.snapshot


def _tabular(snapshot: SourceCatalogSnapshot, name: str) -> TabularFacet:
    (resource,) = [item for item in snapshot.resources if item.name == name]
    (facet,) = [item for item in snapshot.facets if item.resource_id == resource.id]
    return TabularFacet.from_payload(facet.payload)


async def test_views_and_custom_types_are_kept_with_their_comments() -> None:
    registration = _registration()
    structure = await _load_structure(
        _Connection(),
        registration,
        max_resources=10,
        max_columns=10,
        max_indexes=10,
        max_relationships=10,
    )
    tables = {table.name: table for table in structure.tables}
    assert tables["orders"].kind is ResourceKind.TABLE
    assert tables["open_orders"].kind is ResourceKind.VIEW
    (column,) = tables["orders"].columns
    assert column.native_type == "sales.order_status"
    assert column.comment == "Where the order is."


@pytest.mark.parametrize("count", (1, 50, 1_000, 1_500))
async def test_discovery_sends_a_few_statements_however_many_tables_there_are(
    count: int,
) -> None:
    connection = _Connection(
        [("sales", f"t{index:04}", "table") for index in range(count)]
    )

    await _discover(connection)

    assert len(connection.statements) <= 6
    assert not {"columns", "indexes"} & set(connection.statements)


@pytest.mark.parametrize(("count", "truncated"), ((1_000, False), (1_500, True)))
async def test_attaching_beyond_the_resource_limit_keeps_the_first_thousand(
    tmp_path: Path, count: int, truncated: bool
) -> None:
    registration = _registration()
    tables = [("sales", f"t{index:04}", "table") for index in range(count)]
    wide = [_column(f"c{position}", position) for position in range(1, 21)]
    connection = _Connection(
        tables, columns={("sales", name): wide for _, name, _ in tables}, latency=0.05
    )

    snapshot = await _discover(connection, registration=registration)

    assert snapshot.sync.truncated is truncated
    assert [item.name for item in snapshot.resources] == [
        f"t{index:04}" for index in range(1_000)
    ]
    assert all(
        len(_tabular(snapshot, item.name).columns) == 20 for item in snapshot.resources
    )
    if count == 1_000:
        assert connection.round_trips * 0.05 < 5
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        await store.commit_snapshot(snapshot, registration=registration)
        stored = await store.load_sync(registration.agent_id, snapshot.sync.id)
    finally:
        await store.close()
    assert stored is not None and stored.truncated is truncated


@pytest.mark.parametrize(
    ("columns", "shown", "key"),
    (
        # The key is whole, so it stays.
        ([_column("a", 1, (1, 1)), _column("b", 2), _column("c", 3)], 2, ("a",)),
        # The second key column is past the cut.
        ([_column("a", 1, (1, 2)), _column("b", 2), _column("c", 3, (2, 2))], 2, ()),
        # Only the second key column is left, which has no valid position alone.
        ([_column("a", 1), _column("b", 2, (2, 2)), _column("c", 3, (1, 2))], 2, ()),
    ),
)
async def test_a_wide_table_keeps_its_first_columns_and_a_whole_primary_key_or_none(
    columns: list[Row], shown: int, key: tuple[str, ...]
) -> None:
    connection = _Connection(
        [("sales", "orders", "table")],
        columns={("sales", "orders"): columns},
        indexes={("sales", "orders"): [_index("by_a", "a"), _index("by_c", "c")]},
        relationships=[
            _foreign_key("a_fk", "orders", "a"),
            _foreign_key("c_fk", "orders", "c"),
        ],
    )

    snapshot = await _discover(connection, max_columns_per_resource=shown)

    orders = _tabular(snapshot, "orders")
    assert [column.name for column in orders.columns] == ["a", "b"]
    assert [
        column.name for column in orders.columns if column.primary_key_ordinal
    ] == list(key)
    # Neither the index nor the key on the cut column points outside the table.
    assert [index.name for index in orders.indexes] == ["by_a"]
    assert snapshot.sync.truncated is True
    assert [
        pair.source_field
        for relationship in snapshot.relationships
        for pair in relationship.field_pairs
    ] == ["a"]


async def test_more_indexes_and_relationships_than_the_limits_keep_the_first() -> None:
    connection = _Connection(
        [("sales", name, "table") for name in ("orders", "x", "y", "z")],
        indexes={
            ("sales", "orders"): [_index(name, "status") for name in ("a", "b", "c")]
        },
        relationships=[_foreign_key(f"{name}_fk", name, "status") for name in "xyz"],
    )

    within = await _discover(connection, sync_id="within")
    assert within.sync.truncated is False

    snapshot = await _discover(
        connection, max_indexes_per_resource=2, max_relationships=2
    )
    assert snapshot.sync.truncated is True
    assert [index.name for index in _tabular(snapshot, "orders").indexes] == ["a", "b"]
    names = {item.id: item.name for item in snapshot.resources}
    assert {names[item.from_resource_id] for item in snapshot.relationships} == {
        "x",
        "y",
    }


@pytest.mark.parametrize(
    ("stall", "kept"),
    (
        # The first batch of 10 ends in t09, which is the table in progress.
        # A stalled column read still leaves indexes their share of the budget.
        (("bulk_columns", 1), 9),
        (("bulk_indexes", 1), 9),
        (("bulk_columns", 0), 0),
        (("bulk_indexes", 0), 0),
    ),
)
async def test_a_source_slower_than_the_budget_keeps_the_tables_it_read(
    monkeypatch: pytest.MonkeyPatch, stall: tuple[str, int], kept: int
) -> None:
    monkeypatch.setattr(postgresql_module, "_DISCOVERY_SECONDS", 1.0)
    monkeypatch.setattr(postgresql_module, "_BULK_FETCH_ROWS", 10)
    names = [f"t{index:02}" for index in range(20)]
    connection = _Connection(
        [("sales", name, "table") for name in names],
        indexes={("sales", name): [_index("by_status", "status")] for name in names},
        relationships=[_foreign_key("fk", "t00", "status", "t01")],
        stall=stall,
    )

    snapshot = await _discover(connection)

    assert snapshot.sync.truncated is True
    assert [item.name for item in snapshot.resources] == names[:kept]
    for name in names[:kept]:
        facet = _tabular(snapshot, name)
        assert [column.name for column in facet.columns] == ["status"]
        assert [index.name for index in facet.indexes] == ["by_status"]
    assert len(snapshot.relationships) == (1 if kept >= 2 else 0)
    assert "relationships" in connection.statements
    assert ("estimates" in connection.statements) is (kept > 0)
    if stall == ("bulk_columns", 0):
        assert "bulk_indexes" not in connection.statements


@pytest.mark.parametrize(
    "index_rows",
    (
        [],
        [{**_index("expression", "id"), "simple_columns": False}],
        [_index("by_id", "id")],
    ),
)
async def test_tables_without_columns_or_indexes_stay_listed_beside_the_others(
    index_rows: list[Row],
) -> None:
    connection = _Connection(
        [("sales", name, "table") for name in ("a_bare", "b_plain", "c_indexed")],
        columns={
            ("sales", "a_bare"): [],
            ("sales", "b_plain"): [_column("id", 1)],
            ("sales", "c_indexed"): [_column("id", 1), _column("owner", 2)],
        },
        indexes={
            ("sales", "a_bare"): index_rows,
            ("sales", "c_indexed"): [_index("by_owner", "owner")],
        },
    )

    snapshot = await _discover(connection)

    bare = _tabular(snapshot, "a_bare")
    assert bare.columns == () and bare.indexes == ()
    plain = _tabular(snapshot, "b_plain")
    assert [column.name for column in plain.columns] == ["id"]
    assert plain.indexes == ()
    indexed = _tabular(snapshot, "c_indexed")
    assert [column.name for column in indexed.columns] == ["id", "owner"]
    assert [index.name for index in indexed.indexes] == ["by_owner"]
    assert snapshot.sync.truncated is False


async def test_refreshing_one_schema_rereads_only_that_schema() -> None:
    registration = _registration(("sales", "ops"))
    connection = _Connection(
        [
            ("sales", "orders", "table"),
            ("ops", "jobs", "table"),
            ("ops", "audit", "table"),
        ],
        columns={("ops", "jobs"): [_column("id", 1, (1, 1))]},
    )
    first = await _discover(connection, registration=registration, sync_id="first")

    # The server moves on: ops.jobs changes, ops.audit goes, ops.fresh arrives.
    connection.tables = [
        ("sales", "orders", "table"),
        ("ops", "jobs", "table"),
        ("ops", "fresh", "table"),
    ]
    connection.columns = {
        ("sales", "orders"): [_column("total", 1)],
        ("ops", "jobs"): [_column("id", 1, (1, 1)), _column("owner", 2)],
    }
    connection.reads.clear()
    second = await _discover(
        connection,
        registration=registration,
        sync_id="second",
        refresh_schemas=("sales",),
        previous=first,
    )

    assert sorted(connection.reads) == [("ops", "fresh"), ("sales", "orders")]
    assert second.sync.truncated is False
    revisions = {item.name: item.current_revision for item in second.resources}
    before = {item.name: item.current_revision for item in first.resources}
    assert set(revisions) == {"orders", "jobs", "fresh"}
    assert revisions["jobs"] == before["jobs"]
    assert revisions["orders"] != before["orders"]
    assert [column.name for column in _tabular(second, "jobs").columns] == ["id"]
    assert [column.name for column in _tabular(second, "orders").columns] == ["total"]


async def test_a_reused_table_at_the_limit_keeps_the_snapshot_truncated() -> None:
    registration = _registration(("sales", "ops"))
    connection = _Connection(
        [("ops", "jobs", "table")],
        columns={("ops", "jobs"): [_column("id", 1), _column("owner", 2)]},
    )
    first = await _discover(
        connection, registration=registration, max_columns_per_resource=2
    )
    assert first.sync.truncated is False

    connection.statements.clear()
    second = await _discover(
        connection,
        registration=registration,
        sync_id="second",
        refresh_schemas=("sales",),
        previous=first,
        max_columns_per_resource=2,
    )

    assert second.sync.truncated is True
    assert connection.reads == [("ops", "jobs")]
    # Every listed table kept its facet, so nothing is read again.
    assert not {"bulk_columns", "bulk_indexes"} & set(connection.statements)


def _registration(schemas: tuple[str, ...] = ("sales",)) -> SourceRegistration:
    from daita.llm.models import ModelSensitivity

    return SourceRegistration(
        id=source_registration_id("agent", "postgresql", "db.example.test/sales"),
        agent_id="agent",
        adapter_id="postgresql",
        native_identity="db.example.test/sales",
        display_name="Sales",
        configuration={
            "host": "db.example.test",
            "port": 5432,
            "database": "sales",
            "user": "reader",
            "schemas": schemas,
        },
        attached_at=datetime(2026, 11, 2, tzinfo=UTC),
        presentation_sensitivity=ModelSensitivity.INTERNAL,
    )
