"""Immutable physical SQLite schema contracts and exact validators."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

TableSchema = Mapping[str, tuple[tuple[object, ...], ...]]
ForeignKeySchema = Mapping[str, tuple[tuple[object, ...], ...]]
UniqueConstraintSchema = Mapping[str, frozenset[tuple[str, ...]]]
NamedIndexSchema = Mapping[str, tuple[str, bool, tuple[str, ...]]]
RequiredSQLSchema = Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SQLiteSchema:
    """Immutable complete contract for one agent-home SQLite shape."""

    tables: TableSchema
    foreign_keys: ForeignKeySchema
    unique_constraints: UniqueConstraintSchema
    named_indexes: NamedIndexSchema
    required_sql_fragments: RequiredSQLSchema

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tables",
            MappingProxyType(
                {table: tuple(columns) for table, columns in self.tables.items()}
            ),
        )
        object.__setattr__(
            self,
            "foreign_keys",
            MappingProxyType(
                {
                    table: tuple(foreign_keys)
                    for table, foreign_keys in self.foreign_keys.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "unique_constraints",
            MappingProxyType(
                {
                    table: frozenset(constraints)
                    for table, constraints in self.unique_constraints.items()
                }
            ),
        )
        object.__setattr__(
            self, "named_indexes", MappingProxyType(dict(self.named_indexes))
        )
        object.__setattr__(
            self,
            "required_sql_fragments",
            MappingProxyType(
                {
                    table: tuple(fragments)
                    for table, fragments in self.required_sql_fragments.items()
                }
            ),
        )


def table_names(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )


def schema_matches(connection: sqlite3.Connection, schema: SQLiteSchema) -> bool:
    try:
        require_schema(connection, schema)
    except (sqlite3.Error, ValueError):
        return False
    return True


def require_schema(connection: sqlite3.Connection, schema: SQLiteSchema) -> None:
    tables = schema.tables
    if table_names(connection) != set(tables):
        raise ValueError("state tables do not match the declared revision")
    for table, expected in tables.items():
        actual = tuple(
            (row[1], str(row[2]).upper(), row[3], row[4], row[5], row[6])
            for row in connection.execute(f'PRAGMA table_xinfo("{table}")')
        )
        if actual != tuple((*column, 0) for column in expected):
            raise ValueError(f"state table does not match its revision: {table}")

    for table in tables:
        actual_foreign_keys = tuple(
            (row[2], row[3], row[4], row[5], row[6], row[7])
            for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
        )
        if actual_foreign_keys != schema.foreign_keys.get(table, ()):
            raise ValueError(f"state foreign keys are invalid: {table}")

    for table in tables:
        index_rows = tuple(connection.execute(f'PRAGMA index_list("{table}")'))
        if any(row[4] != 0 for row in index_rows):
            raise ValueError(f"state partial indexes are invalid: {table}")
        actual_unique_constraints = frozenset(
            tuple(
                column[2]
                for column in connection.execute(f'PRAGMA index_info("{index[1]}")')
            )
            for index in index_rows
            if index[3] == "u"
        )
        if actual_unique_constraints != schema.unique_constraints.get(
            table, frozenset()
        ):
            raise ValueError(f"state unique constraints are invalid: {table}")

    named_indexes = {
        str(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if named_indexes != {
        name: definition[0] for name, definition in schema.named_indexes.items()
    }:
        raise ValueError("state named indexes do not match the declared revision")
    for name, (
        table,
        expected_unique,
        expected_columns,
    ) in schema.named_indexes.items():
        indexes = {
            str(row[1]): bool(row[2])
            for row in connection.execute(f'PRAGMA index_list("{table}")')
            if not str(row[1]).startswith("sqlite_autoindex")
        }
        if indexes != {
            index_name: definition[1]
            for index_name, definition in schema.named_indexes.items()
            if definition[0] == table
        }:
            raise ValueError(f"state index is invalid: {name}")
        columns = tuple(
            str(row[2]) for row in connection.execute(f'PRAGMA index_info("{name}")')
        )
        if bool(indexes[name]) != expected_unique or columns != expected_columns:
            raise ValueError(f"state index columns are invalid: {name}")

    for table in tables:
        for index in connection.execute(f'PRAGMA index_list("{table}")'):
            key_columns = tuple(
                row
                for row in connection.execute(f'PRAGMA index_xinfo("{index[1]}")')
                if row[5] == 1
            )
            if any(row[3] != 0 or row[4] != "BINARY" for row in key_columns):
                raise ValueError(f"state index ordering is invalid: {index[1]}")

    for table in tables:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if row is None or not isinstance(row[0], str):
            raise ValueError(f"state table SQL is unavailable: {table}")
        normalized_sql = " ".join(row[0].split())
        fragments = schema.required_sql_fragments.get(table, ())
        if any(fragment not in normalized_sql for fragment in fragments):
            raise ValueError(f"state table checks are invalid: {table}")
        if len(re.findall(r"\bCHECK\s*\(", normalized_sql, re.IGNORECASE)) != len(
            fragments
        ):
            raise ValueError(f"state table checks are invalid: {table}")
        if re.search(
            r"\b(COLLATE|DEFERRABLE|GENERATED|STRICT|WITHOUT\s+ROWID)\b",
            normalized_sql,
            re.IGNORECASE,
        ):
            raise ValueError(f"state table options are invalid: {table}")

    extra_objects = tuple(
        connection.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE type IN ('trigger', 'view') AND name NOT LIKE 'sqlite_%'"
        )
    )
    if extra_objects:
        raise ValueError("state database has unexpected triggers or views")


def require_healthy(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
        raise ValueError("state database integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("state database foreign-key check failed")


def _check_fragments(sql: str) -> tuple[str, ...]:
    normalized = " ".join(sql.split())
    fragments: list[str] = []
    for match in re.finditer(r"\bCHECK\s*\(", normalized, re.IGNORECASE):
        depth = 0
        end = match.end() - 1
        for index in range(end, len(normalized)):
            character = normalized[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    fragments.append(normalized[match.start() : index + 1])
                    break
        else:
            raise ValueError("schema contains an unclosed CHECK clause")
    return tuple(fragments)


def schema_from_sql(sql: str) -> SQLiteSchema:
    """Freeze one complete validator contract from immutable DDL."""

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(sql)
        tables = {
            table: tuple(
                (row[1], str(row[2]).upper(), row[3], row[4], row[5])
                for row in connection.execute(f'PRAGMA table_xinfo("{table}")')
            )
            for table in sorted(table_names(connection))
        }
        foreign_keys = {
            table: tuple(
                (row[2], row[3], row[4], row[5], row[6], row[7])
                for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
            )
            for table in tables
            if connection.execute(f'PRAGMA foreign_key_list("{table}")').fetchone()
            is not None
        }
        unique_constraints: dict[str, frozenset[tuple[str, ...]]] = {}
        named_indexes: dict[str, tuple[str, bool, tuple[str, ...]]] = {}
        required_sql_fragments: dict[str, tuple[str, ...]] = {}
        for table in tables:
            index_rows = tuple(connection.execute(f'PRAGMA index_list("{table}")'))
            constraints = frozenset(
                tuple(
                    str(column[2])
                    for column in connection.execute(f'PRAGMA index_info("{index[1]}")')
                )
                for index in index_rows
                if index[3] == "u"
            )
            if constraints:
                unique_constraints[table] = constraints
            for index in index_rows:
                if str(index[1]).startswith("sqlite_autoindex"):
                    continue
                named_indexes[str(index[1])] = (
                    table,
                    bool(index[2]),
                    tuple(
                        str(column[2])
                        for column in connection.execute(
                            f'PRAGMA index_info("{index[1]}")'
                        )
                    ),
                )
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if row is None or not isinstance(row[0], str):
                raise ValueError(f"table SQL is unavailable: {table}")
            checks = _check_fragments(row[0])
            if checks:
                required_sql_fragments[table] = checks
        return SQLiteSchema(
            tables=tables,
            foreign_keys=foreign_keys,
            unique_constraints=unique_constraints,
            named_indexes=named_indexes,
            required_sql_fragments=required_sql_fragments,
        )
    finally:
        connection.close()


__all__ = [
    "SQLiteSchema",
    "require_healthy",
    "require_schema",
    "schema_from_sql",
    "schema_matches",
    "table_names",
]
