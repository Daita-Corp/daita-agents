"""Shared database type metadata helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping

_NATIVE_TYPE_ALIASES = {
    "timestamp": "datetime",
    "numeric": "decimal",
    "int": "integer",
    "double": "float",
    "bool": "boolean",
    "text": "string",
}


def native_type_from_db_type(db_type: Any) -> str | None:
    """Return the framework-native semantic type for a database type string."""

    lowered = str(db_type or "").strip().lower()
    if not lowered:
        return None
    if "timestamp" in lowered or "datetime" in lowered:
        return "datetime"
    if re.search(r"\bdate\b", lowered):
        return "date"
    if re.search(r"\btime\b", lowered):
        return "time"
    if "uuid" in lowered:
        return "uuid"
    if "json" in lowered:
        return "json"
    if any(token in lowered for token in ("numeric", "decimal", "money")):
        return "decimal"
    if any(token in lowered for token in ("bigint", "smallint", "integer", "serial")):
        return "integer"
    if re.fullmatch(r"int(\d+)?", lowered):
        return "integer"
    if any(token in lowered for token in ("double", "float", "real")):
        return "float"
    if any(token in lowered for token in ("bool", "bit(1)")):
        return "boolean"
    if any(token in lowered for token in ("char", "text", "clob", "string")):
        return "string"
    if lowered in {"integer", "real", "text", "blob"}:
        return {"integer": "integer", "real": "float", "text": "string"}.get(lowered)
    return None


def native_type_from_param_spec(spec: Mapping[str, Any]) -> str | None:
    """Return the native type declared by a typed parameter spec."""

    native_type = str(spec.get("native_type") or "").strip().lower()
    if native_type:
        normalized = native_type.replace("-", "_")
        if normalized in {
            "datetime",
            "timestamp",
            "date",
            "time",
            "uuid",
            "decimal",
            "numeric",
            "json",
            "integer",
            "int",
            "float",
            "double",
            "boolean",
            "bool",
            "string",
            "text",
        }:
            return _NATIVE_TYPE_ALIASES.get(normalized, normalized)
    return native_type_from_db_type(spec.get("db_type"))


def nullable_value(value: Any) -> bool | None:
    """Normalize common schema nullable flags."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"yes", "true", "1", "nullable", "null"}:
        return True
    if lowered in {"no", "false", "0", "not null", "not_nullable"}:
        return False
    return None


def column_type_metadata(
    table: Mapping[str, Any],
    column: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the typed monitor parameter column contract from schema metadata."""

    db_type = (
        column.get("data_type")
        or column.get("type")
        or column.get("column_type")
        or column.get("db_type")
    )
    native_type = column.get("native_type") or native_type_from_db_type(db_type)
    dialect = schema.get("sql_dialect") or schema.get("database_type")
    if not (db_type and native_type and dialect):
        return None
    payload: dict[str, Any] = {
        "table": str(table.get("name") or table.get("table_name") or ""),
        "column": _column_name(column),
        "db_type": str(db_type),
        "native_type": str(native_type),
        "dialect": str(dialect),
    }
    nullable = nullable_value(column.get("is_nullable"))
    if nullable is not None:
        payload["nullable"] = nullable
    return payload


def is_timezone_type(spec: Mapping[str, Any]) -> bool:
    db_type = str(spec.get("db_type") or "").lower()
    return "with time zone" in db_type or "timestamptz" in db_type


def _column_name(column: Any) -> str:
    if isinstance(column, Mapping):
        return str(column.get("name") or column.get("column_name") or "").strip()
    if isinstance(column, str):
        return column.split(":", 1)[0].strip()
    return ""
