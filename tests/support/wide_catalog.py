"""Seed a deterministic 46-table, 257-column SQLite source for catalog tests.

Names do not encode size. Row counts come from this manifest, never from SQL
under test. Table 0 is empty and table 1 needs identifier quoting. Tables 2
through 5 form a three-hop foreign-key chain for relationship resolution.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

TABLE_COUNT = 46
COLUMN_COUNT = 257
CHAIN = ("table_02", "table_03", "table_04", "table_05")


def _table_name(index: int) -> str:
    if index == 0:
        return "empty_ledger"
    if index == 1:
        return 'odd "table'
    return f"table_{index:02}"


class WideTable(TypedDict):
    table: str
    columns: int
    rows: int


MANIFEST: tuple[WideTable, ...] = tuple(
    {
        "table": _table_name(index),
        "columns": 6 if index < 27 else 5,
        "rows": 0 if index == 0 else (index * 17 + 11) % 73,
    }
    for index in range(TABLE_COUNT)
)
TOTAL_ROWS = sum(item["rows"] for item in MANIFEST)
assert sum(item["columns"] for item in MANIFEST) == COLUMN_COUNT


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def seed_wide_sqlite(path: Path) -> Path:
    """Create the fixture database at ``path`` and verify every seeded count."""

    with sqlite3.connect(path) as connection:
        for index, table in enumerate(MANIFEST):
            name = table["table"]
            filler = [
                f"analytical_dimension_{column}"
                for column in range(table["columns"] - 3)
            ]
            parent = (
                CHAIN[CHAIN.index(name) + 1]
                if name in CHAIN and name != CHAIN[-1]
                else None
            )
            link = (
                f"link_id INTEGER REFERENCES {quote(parent)}(record_id)"
                if parent is not None
                else "link_id INTEGER"
            )
            definitions = [
                "record_id INTEGER PRIMARY KEY",
                "control_value INTEGER NOT NULL",
                link,
                *(f"{quote(column)} TEXT" for column in filler),
            ]
            connection.execute(f"CREATE TABLE {quote(name)} ({', '.join(definitions)})")
            connection.executemany(
                f"INSERT INTO {quote(name)} (record_id, control_value) VALUES (?, ?)",
                ((row, (row * 7 + index) % 5) for row in range(table["rows"])),
            )
        for table in MANIFEST:
            actual = connection.execute(
                f"SELECT count(*) FROM {quote(table['table'])}"
            ).fetchone()[0]
            assert actual == table["rows"]
    return path


__all__ = [
    "CHAIN",
    "COLUMN_COUNT",
    "MANIFEST",
    "TABLE_COUNT",
    "TOTAL_ROWS",
    "WideTable",
    "quote",
    "seed_wide_sqlite",
]
