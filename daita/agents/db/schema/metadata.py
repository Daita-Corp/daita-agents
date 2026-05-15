"""Shared metadata helpers for normalized ``from_db`` schemas."""

from __future__ import annotations

import re
from typing import Any


def column_name(column: Any) -> str:
    if isinstance(column, dict):
        return str(column.get("name") or column.get("column_name") or "").strip()
    if isinstance(column, str):
        return column.split(":", 1)[0].strip()
    return ""


def table_name(table: Any) -> str:
    if not isinstance(table, dict):
        return ""
    return str(table.get("name") or table.get("table_name") or "").strip()


def short_table_name(name: str) -> str:
    return str(name).split(".")[-1].lower()


def schema_table_columns(schema: dict[str, Any]) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for table in schema.get("tables") or []:
        name = table_name(table)
        if not name:
            continue
        columns = {
            col_name.lower()
            for col_name in (
                column_name(column) for column in table.get("columns") or []
            )
            if col_name
        }
        tables[name.lower()] = columns
        if "." in name:
            tables[short_table_name(name)] = columns
    return tables


def matching_tables(
    schema_or_tables: dict[str, Any] | list[dict[str, Any]], table_ref: str
) -> list[dict[str, Any]]:
    wanted = str(table_ref or "").strip().lower()
    if not wanted:
        return []
    tables = (
        schema_or_tables.get("tables", []) or []
        if isinstance(schema_or_tables, dict)
        else schema_or_tables
    )
    exact = [table for table in tables if table_name(table).lower() == wanted]
    if exact:
        return exact
    return [table for table in tables if short_table_name(table_name(table)) == wanted]


def split_identifier(value: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z0-9_]+", str(value).lower())
    tokens: list[str] = []
    for item in raw:
        tokens.extend(part for part in re.split(r"[_\W]+", item) if part)
    return [token for token in tokens if token]


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
