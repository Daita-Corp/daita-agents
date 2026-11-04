"""Discovery keeps what it is given, and what a limit cuts is marked truncated."""

from __future__ import annotations

import asyncio
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


class _Connection:
    """Answers discovery's tagged queries from fixed rows, honoring each LIMIT."""

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
        column_seconds: float = 0.0,
    ) -> None:
        self.tables = list(tables)
        self.columns = dict(columns or {})
        self.indexes = dict(indexes or {})
        self.relationships = list(relationships)
        self.column_seconds = column_seconds
        self.reads: list[tuple[str, str]] = []

    async def fetch(self, sql: str, *args: object) -> list[Row]:
        limit = args[-1]
        assert isinstance(limit, int)
        if "daita:postgresql.resources" in sql:
            rows: Sequence[Row] = [
                {"schema_name": schema, "resource_name": name, "resource_kind": kind}
                for schema, name, kind in sorted(self.tables)
            ]
        elif "daita:postgresql.columns" in sql:
            key = (str(args[0]), str(args[1]))
            self.reads.append(key)
            await asyncio.sleep(self.column_seconds)
            rows = self.columns.get(key, [_STATUS])
        elif "daita:postgresql.indexes" in sql:
            rows = self.indexes.get((str(args[0]), str(args[1])), [])
        elif "daita:postgresql.relationships" in sql:
            rows = self.relationships
        else:
            rows = []
        return list(rows[:limit])

    # Discovery runs in one transaction; this connection is its own.
    def transaction(self, **_: object) -> _Connection:
        return self

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


@pytest.mark.parametrize(("count", "truncated"), ((1_000, False), (1_500, True)))
async def test_attaching_beyond_the_resource_limit_keeps_the_first_thousand(
    tmp_path: Path, count: int, truncated: bool
) -> None:
    registration = _registration()
    connection = _Connection(
        [("sales", f"t{index:04}", "table") for index in range(count)]
    )

    snapshot = await _discover(connection, registration=registration)

    assert snapshot.sync.truncated is truncated
    assert [item.name for item in snapshot.resources] == [
        f"t{index:04}" for index in range(1_000)
    ]
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


async def test_a_source_slower_than_the_budget_keeps_the_tables_it_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(postgresql_module, "_DISCOVERY_SECONDS", 0.05)
    connection = _Connection(
        [("sales", f"t{index}", "table") for index in range(5)], column_seconds=0.1
    )

    snapshot = await _discover(connection)

    assert snapshot.sync.truncated is True
    assert [item.name for item in snapshot.resources] == ["t0"]


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
