"""Define migration records and checksums for the staged-copy SQLite engine."""

from __future__ import annotations

import inspect
import json
import sqlite3
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from ..sqlite_schema import TableSchema


def _callable_source(
    function: Callable[[sqlite3.Connection], None] | None,
) -> str | None:
    if function is None:
        return None
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as error:
        raise ValueError("migration callables must have inspectable source") from error
    return textwrap.dedent(source).replace("\r\n", "\n").rstrip() + "\n"


def _schema_material(schema: TableSchema) -> dict[str, list[list[object]]]:
    return {
        table: [list(column) for column in columns] for table, columns in schema.items()
    }


def migration_checksum(
    *,
    ordinal: int,
    migration_id: str,
    definition: str,
    source_schema: TableSchema,
    target_schema: TableSchema,
    apply: Callable[[sqlite3.Connection], None],
    validate_target: Callable[[sqlite3.Connection], None] | None,
) -> str:
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("migration definition must be non-empty text")
    material = json.dumps(
        {
            "apply": _callable_source(apply),
            "definition": definition,
            "migration_id": migration_id,
            "ordinal": ordinal,
            "source_schema": _schema_material(source_schema),
            "target_schema": _schema_material(target_schema),
            "validate_target": _callable_source(validate_target),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SQLiteMigration:
    ordinal: int
    migration_id: str
    definition: str
    source_schema: TableSchema
    target_schema: TableSchema
    apply: Callable[[sqlite3.Connection], None]
    validate_target: Callable[[sqlite3.Connection], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ValueError("migration ordinal must be positive")
        if not self.migration_id or self.migration_id != self.migration_id.strip():
            raise ValueError("migration ID must be non-empty text")
        self.checksum

    @property
    def checksum(self) -> str:
        return migration_checksum(
            ordinal=self.ordinal,
            migration_id=self.migration_id,
            definition=self.definition,
            source_schema=self.source_schema,
            target_schema=self.target_schema,
            apply=self.apply,
            validate_target=self.validate_target,
        )
