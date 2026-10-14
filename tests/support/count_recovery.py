"""Independent disposable count fixture: 46 base tables and 257 columns.

Names do not encode size. Counts and field-control expectations come from this
manifest, never from evaluated SQL or a model's answer.
"""

from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path

from daita._json import canonical_json

SOURCE_NAMES = ("Juniper", "Meridian")
MANIFEST = {
    name: tuple(
        {
            "table": 'odd "table' if i == 1 else f"records_{i:02}",
            "rows": 0 if i == 0 else (i * 17 + offset) % 73,
            "columns": 6 if ordinal < 27 else 5,
        }
        for i, ordinal in ((i, i + start) for i in range(size))
    )
    for name, size, start, offset in (("Juniper", 8, 0, 11), ("Meridian", 38, 8, 3))
}
FIXTURE_DIGEST = "sha256:" + sha256(canonical_json(MANIFEST).encode()).hexdigest()
TOTALS = {
    name: sum(int(t["rows"]) for t in tables) for name, tables in MANIFEST.items()
}
WINNER = max(TOTALS, key=lambda name: TOTALS[name])


def quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def seed_sqlite(directory: Path) -> dict[str, Path]:
    paths = {}
    for name, tables in MANIFEST.items():
        path = directory / f"{name}.sqlite"
        with sqlite3.connect(path) as connection:
            for table in tables:
                identifier = quote(str(table["table"]))
                columns = ["record_id", "control_value"] + [
                    f"analytical_dimension_{i}"
                    for i in range(int(table["columns"]) - 2)
                ]
                connection.execute(
                    f"CREATE TABLE {identifier} ("
                    + ",".join(quote(c) + " INTEGER" for c in columns)
                    + ")"
                )
                connection.executemany(
                    f"INSERT INTO {identifier} VALUES ("
                    + ",".join("?" for _ in columns)
                    + ")",
                    (
                        (i, i % 3, *([None] * (len(columns) - 2)))
                        for i in range(int(table["rows"]))
                    ),
                )
                actual = connection.execute(
                    f"SELECT count(*) FROM {identifier}"
                ).fetchone()[0]
                assert actual == table["rows"]
            connection.execute(
                "CREATE VIEW distractor_view AS SELECT * FROM records_02"
            )
        paths[name] = path
    return paths
