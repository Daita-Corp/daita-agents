"""Define immutable whole-agent-home migration records."""

from __future__ import annotations

import inspect
import json
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from ..schema_contract import SQLiteSchema

HomeMigrationApply = Callable[[Path, str | None], None]


def _module_source(function: HomeMigrationApply) -> str:
    module = sys.modules.get(function.__module__)
    if module is None:
        raise ValueError("migration implementation module is unavailable")
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as error:
        raise ValueError("migration implementation source is unavailable") from error
    return textwrap.dedent(source).replace("\r\n", "\n").rstrip() + "\n"


def _schema_material(schema: SQLiteSchema) -> dict[str, object]:
    return {
        "foreign_keys": {
            table: [list(item) for item in foreign_keys]
            for table, foreign_keys in schema.foreign_keys.items()
        },
        "named_indexes": {
            name: [table, unique, list(columns)]
            for name, (table, unique, columns) in schema.named_indexes.items()
        },
        "required_sql_fragments": {
            table: list(fragments)
            for table, fragments in schema.required_sql_fragments.items()
        },
        "tables": {
            table: [list(column) for column in columns]
            for table, columns in schema.tables.items()
        },
        "unique_constraints": {
            table: [list(columns) for columns in sorted(constraints)]
            for table, constraints in schema.unique_constraints.items()
        },
    }


@dataclass(frozen=True, slots=True)
class HomeMigration:
    """One append-only transition in the sole agent-home revision sequence."""

    revision: int
    migration_id: str
    definition: str
    affected_paths: tuple[str, ...]
    target_schema: SQLiteSchema
    apply: HomeMigrationApply
    implementation_material: tuple[str, ...] = ()
    _checksum: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
        ):
            raise ValueError("home migration revision must be a positive integer")
        if not self.migration_id or self.migration_id != self.migration_id.strip():
            raise ValueError("home migration ID must be non-empty text")
        if not self.definition.strip():
            raise ValueError("home migration definition must be non-empty text")
        if not self.affected_paths or self.affected_paths[0] != "state.db":
            raise ValueError("home migrations must stage state.db first")
        if len(set(self.affected_paths)) != len(self.affected_paths):
            raise ValueError("home migration affected paths must be unique")
        for relative in self.affected_paths:
            candidate = Path(relative)
            if (
                candidate.is_absolute()
                or not candidate.parts
                or ".." in candidate.parts
                or candidate.as_posix() != relative
            ):
                raise ValueError("home migration path is invalid")
        if any(
            not isinstance(item, str) or not item
            for item in self.implementation_material
        ):
            raise ValueError("home migration implementation material is invalid")
        object.__setattr__(self, "_checksum", self._calculate_checksum())

    @property
    def checksum(self) -> str:
        return self._checksum

    def _calculate_checksum(self) -> str:
        material = json.dumps(
            {
                "affected_paths": self.affected_paths,
                "definition": self.definition,
                "implementation": _module_source(self.apply),
                "implementation_material": self.implementation_material,
                "migration_id": self.migration_id,
                "revision": self.revision,
                "target_schema": _schema_material(self.target_schema),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(material.encode("utf-8")).hexdigest()


__all__ = ["HomeMigration", "HomeMigrationApply"]
