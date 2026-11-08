"""A SQLite source says when its catalog is out of date, and keeps what it has."""

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from daita import SQLiteSource
from daita.adapters.models import DiscoveryRequest
from daita.adapters.registry import SourceOperation, SourceTypeRegistry
from daita.adapters.sqlite_query import SQLiteQueryBackend
from tests.support.source_contract import (
    ProbeMutation,
    StaticSources,
    assert_change_detection_contract,
    reconcile_pass,
    scoped_revisions,
)

_NOW = datetime(2026, 11, 7, tzinfo=UTC)
_AGENT = "agent"


async def _open(database: Path):
    adapter = await SQLiteSource(database).open(
        agent_id=_AGENT, attached_at=_NOW, clock=lambda: _NOW
    )
    backend = SQLiteQueryBackend(StaticSources(adapter.registration))
    return adapter, backend


def _request(adapter, **options) -> DiscoveryRequest:
    return DiscoveryRequest(
        agent_id=_AGENT,
        source_id=adapter.registration.id,
        sync_id="sync",
        requested_at=_NOW,
        **options,
    )


def _run(database: Path, *statements: str):
    async def apply() -> None:
        with sqlite3.connect(database) as connection:
            for statement in statements:
                connection.execute(statement)

    return apply


@pytest.mark.contract
async def test_sqlite_probe_moves_with_the_catalog(tmp_path: Path) -> None:
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, total REAL)")
    assert (
        SourceOperation.CHANGE_PROBE
        in SourceTypeRegistry().require("sqlite").operations
    )
    adapter, backend = await _open(database)
    main = frozenset({"main"})
    replacement = tmp_path / "replacement.sqlite"

    async def replace_file() -> None:
        replacement.write_bytes(database.read_bytes())
        os.replace(replacement, database)

    try:
        await assert_change_detection_contract(
            lambda: backend.probe(
                agent_id=_AGENT, source_id=adapter.registration.id, cursor=None
            ),
            [
                ProbeMutation(
                    "rows written",
                    _run(database, "INSERT INTO orders (total) VALUES (1.5)"),
                ),
                ProbeMutation(
                    "table added",
                    _run(database, "CREATE TABLE users (id INTEGER, name TEXT)"),
                    main,
                ),
                ProbeMutation(
                    "column added",
                    _run(database, "ALTER TABLE users ADD COLUMN email TEXT"),
                    main,
                ),
                ProbeMutation(
                    "column renamed",
                    _run(database, "ALTER TABLE users RENAME COLUMN email TO mail"),
                    main,
                ),
                ProbeMutation(
                    "index added",
                    _run(database, "CREATE INDEX users_name ON users (name)"),
                    main,
                ),
                ProbeMutation(
                    "index dropped", _run(database, "DROP INDEX users_name"), main
                ),
                ProbeMutation(
                    "view created",
                    _run(database, "CREATE VIEW big AS SELECT * FROM orders"),
                    main,
                ),
                ProbeMutation(
                    "table with a foreign key added",
                    _run(
                        database,
                        "CREATE TABLE lines (id INTEGER, order_id INTEGER "
                        "REFERENCES orders (id))",
                    ),
                    main,
                ),
                ProbeMutation(
                    "table dropped", _run(database, "DROP TABLE lines"), main
                ),
                ProbeMutation("file replaced", replace_file, main, exact=False),
            ],
            revisions=lambda: scoped_revisions(
                adapter, _request(adapter), lambda entry: "main"
            ),
        )
    finally:
        await adapter.close()


async def test_sqlite_probe_failures_carry_a_stable_code(tmp_path: Path) -> None:
    from daita.adapters.protocols import ResourceAdapterError

    database = tmp_path / "gone.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE t (id INTEGER)")
    adapter, backend = await _open(database)
    try:
        database.unlink()
        with pytest.raises(ResourceAdapterError) as failure:
            await backend.probe(
                agent_id=_AGENT, source_id=adapter.registration.id, cursor=None
            )
        assert failure.value.code == "source_path_invalid"
        assert str(tmp_path) not in str(failure.value)
    finally:
        await adapter.close()


async def test_sqlite_fill_describes_what_lacks_detail_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    database = tmp_path / "many.sqlite"
    with sqlite3.connect(database) as connection:
        for index in range(6):
            connection.execute(f"CREATE TABLE t{index} (id INTEGER)")
    adapter, _ = await _open(database)
    try:
        first = await reconcile_pass(adapter, _request(adapter, max_resources=2))
        previous = first.detail.entries
        assert len(previous) == 2
        second = await reconcile_pass(
            adapter, _request(adapter, max_resources=2, previous=previous, fill=True)
        )
        names = {entry.resource.name for entry in second.detail.entries}
        assert len(names) == 4 and {e.resource.name for e in previous} <= names
        assert second.detail.truncated is True
        done = await reconcile_pass(
            adapter,
            _request(
                adapter,
                max_resources=2,
                previous=second.detail.entries,
                fill=True,
            ),
        )
        assert len(done.detail.entries) == 6 and done.detail.truncated is False
        with pytest.raises(ValueError):
            _request(adapter, fill=True)
    finally:
        await adapter.close()
