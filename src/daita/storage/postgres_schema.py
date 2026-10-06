"""Render PostgreSQL tables from the authoritative agent-home schema.

There is no second table definition or revision sequence. Backend differences
(namespace keys, identity allocation, ordering and JSON predicates) live here.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from typing import cast

from .schema_contract import require_schema, schema_from_sql
from .sqlite_schema import CURRENT_DATABASE_SQL, CURRENT_SCHEMA

POSTGRES_TABLES = tuple(
    name for name in CURRENT_SCHEMA.tables if name != "agent_home_migrations"
)


def _identifier(name: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError(f"unsupported state schema identifier: {name}")
    return '"' + name + '"'


def _columns(names) -> str:
    return ", ".join(_identifier(name) for name in ("namespace_id", *names))


def render_postgres_schema(database_sql: str) -> str:
    """Project a declared home schema; reject features the adapter cannot preserve.

    SQLite parses its own canonical DDL. We use relational metadata rather than
    rewrite SQL with string replacement or maintain a copy of every table.
    No application database, credential or filesystem is accessed.
    """
    schema = schema_from_sql(database_sql)
    tables = tuple(name for name in schema.tables if name != "agent_home_migrations")
    if set(tables) & {"namespaces", "namespace_roles", "schema_version"}:
        raise ValueError("home schema collides with PostgreSQL administrative tables")
    statements: list[str] = []
    foreign_keys: list[str] = []
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(database_sql)
        require_schema(connection, schema)
        for table in tables:
            columns = schema.tables[table]
            primary_key = tuple(
                str(row[0])
                for row in sorted(columns, key=lambda row: cast(int, row[4]))
                if row[4]
            )
            if not primary_key:
                raise ValueError(f"state table needs an explicit primary key: {table}")
            definitions = [
                "namespace_id UUID NOT NULL DEFAULT current_setting('daita.store_id')::uuid REFERENCES namespaces(namespace_id)"
            ]
            for column, kind, required, default, primary in columns:
                name = str(column)
                if name in {"namespace_id", "insertion_id"}:
                    raise ValueError(
                        "home column collides with PostgreSQL backend metadata"
                    )
                if kind not in {"INTEGER", "TEXT"}:
                    raise ValueError(f"unsupported portable state column type: {kind}")
                value_type = "BIGINT" if kind == "INTEGER" else 'TEXT COLLATE "C"'
                definition = f"{_identifier(name)} {value_type}"
                if kind == "INTEGER" and primary and len(primary_key) == 1:
                    definition += " GENERATED ALWAYS AS IDENTITY"
                if required or primary:
                    definition += " NOT NULL"
                if default is not None:
                    if (
                        not isinstance(default, str)
                        or re.fullmatch(r"-?\d+|'(?:[^']|'')*'|NULL", default) is None
                    ):
                        raise ValueError(
                            f"unsupported portable state column default: {name}"
                        )
                    definition += " DEFAULT " + default
                definitions.append(definition)
            definitions.append(f"PRIMARY KEY ({_columns(primary_key)})")
            for unique_columns in sorted(schema.unique_constraints.get(table, ())):
                definitions.append(f"UNIQUE ({_columns(unique_columns)})")
            for check in schema.required_sql_fragments.get(table, ()):
                # This is the sole current nonstandard function in home checks.
                check = re.sub(
                    r"json_valid\(([a-z][a-z0-9_]*)\)",
                    r"(\1::jsonb IS NOT NULL)",
                    check,
                )
                definitions.append(check)
            if table == "runs":
                definitions.append("insertion_id BIGINT GENERATED ALWAYS AS IDENTITY")
            statements.append(
                f"CREATE TABLE {_identifier(table)} (\n    "
                + ",\n    ".join(definitions)
                + "\n);"
            )
            grouped: dict[int, list[tuple]] = defaultdict(list)
            for row in connection.execute(
                f"PRAGMA foreign_key_list({_identifier(table)})"
            ):
                grouped[row[0]].append(row)
            for rows in grouped.values():
                rows.sort(key=lambda row: row[1])
                _, _, target, _, _, on_update, on_delete, match = rows[0]
                if (
                    target not in tables
                    or match != "NONE"
                    or on_update not in {"NO ACTION", "RESTRICT", "CASCADE"}
                    or on_delete not in {"NO ACTION", "RESTRICT", "CASCADE"}
                ):
                    raise ValueError(f"unsupported portable state foreign key: {table}")
                source_columns = tuple(row[3] for row in rows)
                target_columns = tuple(row[4] for row in rows)
                foreign_keys.append(
                    f"ALTER TABLE {_identifier(table)} ADD FOREIGN KEY ({_columns(source_columns)}) REFERENCES {_identifier(target)} ({_columns(target_columns)}) ON UPDATE {on_update} ON DELETE {on_delete};"
                )
        for index, (table, is_unique, index_columns) in sorted(
            schema.named_indexes.items()
        ):
            if table in tables:
                statements.append(
                    f"CREATE {'UNIQUE ' if is_unique else ''}INDEX {_identifier(index)} ON {_identifier(table)} ({_columns(index_columns)});"
                )
        if "runs" in tables:
            statements.append(
                "CREATE INDEX runs_insertion_order ON runs(namespace_id, agent_id, insertion_id DESC);"
            )
    finally:
        connection.close()
    return "\n\n".join((*statements, *foreign_keys)) + "\n"


POSTGRES_SCHEMA_SQL = render_postgres_schema(CURRENT_DATABASE_SQL)
