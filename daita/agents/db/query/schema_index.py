"""Query-focused schema index for resolving tables and columns."""

from __future__ import annotations

from typing import Any, Optional

from ..utils import unique_preserving_order
from ..schema.metadata import (
    column_name,
    normalize_identifier,
    split_identifier,
    table_name,
)
from .ir import FieldRef


class QuerySchemaIndex:
    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.tables = {
            table_name(table): table
            for table in schema.get("tables") or []
            if isinstance(table, dict) and table_name(table)
        }
        self.table_lookup: dict[str, str] = {}
        for name in self.tables:
            self.table_lookup[name.lower()] = name
            self.table_lookup[name.split(".")[-1].lower()] = name

    def resolve_tables(self, names: list[str]) -> list[str]:
        return unique_preserving_order(
            [table for table in (self.resolve_table(name) for name in names) if table]
        )

    def resolve_table(self, name: Any) -> Optional[str]:
        key = str(name or "").strip().lower()
        if not key:
            return None
        return self.table_lookup.get(key)

    def has_column(self, table_name: str, column_ref: str) -> bool:
        return any(
            column_name(column).lower() == column_ref.lower()
            for column in self._columns(table_name)
            if isinstance(column, dict)
        )

    def resolve_field(
        self, name: str, *, preferred_tables: list[str]
    ) -> Optional[FieldRef]:
        raw = str(name or "").strip()
        if not raw:
            return None
        if "." in raw:
            table_ref, column_ref = raw.rsplit(".", 1)
            table = self.resolve_table(table_ref)
            if table and self.has_column(table, column_ref):
                return FieldRef(table, column_ref)

        tokens = set(split_identifier(raw))
        matches: list[tuple[int, str, str]] = []
        table_order = preferred_tables + [
            table for table in self.tables if table not in preferred_tables
        ]
        for table in table_order:
            for column in self._columns(table):
                col_name = column_name(column)
                column_tokens = set(split_identifier(col_name))
                score = 0
                if normalize_identifier(col_name) == normalize_identifier(raw):
                    score += 8
                score += len(tokens & column_tokens) * 3
                if score:
                    matches.append((score, table, col_name))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], table_order.index(item[1]), item[2]))
        _, table, column = matches[0]
        return FieldRef(table, column)

    def primary_key_or_identity(self, table_name: str) -> Optional[str]:
        columns = self._columns(table_name)
        for column in columns:
            if bool(column.get("is_primary_key")):
                return column_name(column)
        names = [column_name(column) for column in columns]
        short = table_name.split(".")[-1].rstrip("s")
        preferred = [
            f"{short}_id",
            "id",
            "uuid",
            "key",
        ]
        for name in preferred:
            for col_name in names:
                if col_name.lower() == name.lower():
                    return col_name
        for col_name in names:
            lowered = col_name.lower()
            if lowered.endswith("_id") or lowered == "id":
                return col_name
        return None

    def timestamp_column(self, table_name: str) -> Optional[str]:
        names = [column_name(column) for column in self._columns(table_name)]
        preferred = [
            "created_at",
            "created",
            "timestamp",
            "event_time",
            "event_timestamp",
            "date",
            "updated_at",
        ]
        for name in preferred:
            for col_name in names:
                if col_name.lower() == name:
                    return col_name
        for col_name in names:
            lowered = col_name.lower()
            if any(token in lowered for token in ("time", "date", "created_at")):
                return col_name
        return None

    def _columns(self, table_name: str) -> list[dict[str, Any]]:
        table = self.tables.get(table_name) or {}
        return [
            column for column in table.get("columns") or [] if isinstance(column, dict)
        ]
