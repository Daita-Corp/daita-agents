"""SQL planner metadata helpers for normalized catalog snapshots."""

from __future__ import annotations

import re
from typing import Any, Literal

IdentityMode = Literal[
    "declared_only",
    "declared_or_conventional",
    "count_stable_row",
]

FIELD_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "of",
    "record",
    "records",
    "the",
    "to",
    "total",
}


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


def identity_column(
    table: Any,
    *,
    mode: IdentityMode = "declared_or_conventional",
) -> str | None:
    """Return the best row identity column for a normalized table."""
    if mode not in {
        "declared_only",
        "declared_or_conventional",
        "count_stable_row",
    }:
        raise ValueError("unknown identity resolution mode")
    if not isinstance(table, dict):
        return None

    columns = [
        column for column in table.get("columns") or [] if isinstance(column, dict)
    ]
    for column in columns:
        if bool(column.get("is_primary_key")):
            return column_name(column)
    if mode == "declared_only":
        return None

    names = [name for name in (column_name(column) for column in columns) if name]
    short = short_table_name(table_name(table)).rstrip("s")
    preferred = [f"{short}_id", "id", "uuid", "key"]
    for wanted in preferred:
        for name in names:
            if name.lower() == wanted.lower():
                return name
    for name in names:
        lowered = name.lower()
        if lowered.endswith("_id") or lowered == "id":
            return name
    return None


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


def normalize_field_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def field_phrase_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in normalize_field_phrase(value).split():
        if not token or token in FIELD_TOKEN_STOPWORDS:
            continue
        tokens.add(token[:-1] if token.endswith("s") and len(token) > 3 else token)
    return tokens


def required_field_matches_output(required_field: str, output_name: str) -> bool:
    normalized_required = normalize_identifier(required_field)
    normalized_output = normalize_identifier(output_name)
    if not normalized_required or not normalized_output:
        return False
    if normalized_required == normalized_output:
        return True

    required_tokens = field_phrase_tokens(required_field)
    output_tokens = field_phrase_tokens(output_name)
    if not required_tokens or not output_tokens:
        return False
    shared = required_tokens & output_tokens
    if required_tokens <= output_tokens or output_tokens <= required_tokens:
        return True
    return len(shared) >= min(2, len(required_tokens), len(output_tokens))


def field_ref_matches_required(field: Any, required_field: str) -> bool:
    table = str(getattr(field, "table", "") or "")
    column = str(getattr(field, "column", "") or "")
    return required_field_matches_output(
        required_field, column
    ) or required_field_matches_output(required_field, f"{table}.{column}")


def metric_matches_required(metric: Any, required_field: str) -> bool:
    if required_field_matches_output(required_field, getattr(metric, "name", "")):
        return True
    column = str(getattr(metric, "column", "") or "")
    return bool(column and required_field_matches_output(required_field, column))
