"""Strict primitives for explicit SQLite record-family codecs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TypeVar

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
_EnumT = TypeVar("_EnumT", bound=Enum)


def dump_payload(value: JsonValue) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def load_payload(value: str) -> JsonValue:
    parsed = json.loads(value)
    if not _is_json_value(parsed):
        raise ValueError("stored payload is not bounded JSON")
    return parsed


def plain_encode(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): plain_encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_encode(item) for item in value]
    raise TypeError(f"unsupported stored JSON value: {type(value).__name__}")


def plain_decode(value: JsonValue) -> object:
    if isinstance(value, list):
        return tuple(plain_decode(item) for item in value)
    if isinstance(value, dict):
        return {key: plain_decode(item) for key, item in value.items()}
    return value


def record(name: str, values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"__record__": name, "fields": dict(values)}


def record_fields(
    value: JsonValue,
    name: str,
    required: tuple[str, ...],
    *,
    optional: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != {"__record__", "fields"}:
        raise ValueError(f"stored {name} record envelope is invalid")
    if value["__record__"] != name:
        raise ValueError(f"stored record is not {name}")
    fields = value["fields"]
    if not isinstance(fields, dict):
        raise ValueError(f"stored {name} fields are invalid")
    defaults = dict(optional or {})
    allowed = set(required) | set(defaults)
    unknown = set(fields) - allowed
    missing = set(required) - set(fields)
    if unknown:
        raise ValueError(f"stored {name} has unknown fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"stored {name} is missing fields: {sorted(missing)!r}")
    return {key: fields.get(key, default) for key, default in defaults.items()} | {
        key: fields[key] for key in required
    }


def enum_encode(value: Enum, name: str) -> dict[str, JsonValue]:
    if type(value).__name__ != name:
        raise TypeError(f"stored enum must be {name}")
    raw = value.value
    if not isinstance(raw, str):
        raise TypeError(f"stored {name} value must be text")
    return {"__enum__": name, "value": raw}


def enum_decode(value: JsonValue, enum: type[_EnumT], name: str) -> _EnumT:
    if (
        not isinstance(value, dict)
        or set(value) != {"__enum__", "value"}
        or value["__enum__"] != name
        or not isinstance(value["value"], str)
    ):
        raise ValueError(f"stored {name} enum is invalid")
    return enum(value["value"])


def datetime_encode(value: datetime) -> dict[str, JsonValue]:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("stored datetime must be timezone-aware")
    timestamp = value.isoformat()
    offset = value.utcoffset()
    if offset is not None and offset.total_seconds() == 0:
        timestamp = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"__datetime__": timestamp}


def datetime_decode(value: JsonValue) -> datetime:
    if (
        not isinstance(value, dict)
        or set(value) != {"__datetime__"}
        or not isinstance(value["__datetime__"], str)
    ):
        raise ValueError("stored datetime is invalid")
    decoded = datetime.fromisoformat(value["__datetime__"])
    if decoded.tzinfo is None or decoded.utcoffset() is None:
        raise ValueError("stored datetime must be timezone-aware")
    return decoded


def optional_datetime_encode(value: datetime | None) -> JsonValue:
    return None if value is None else datetime_encode(value)


def optional_datetime_decode(value: JsonValue) -> datetime | None:
    return None if value is None else datetime_decode(value)


def decimal_encode(value: Decimal) -> dict[str, JsonValue]:
    if not isinstance(value, Decimal):
        raise TypeError("stored decimal must be Decimal")
    if not value.is_finite():
        raise ValueError("stored decimal must be finite")
    return {"__decimal__": str(value)}


def decimal_decode(value: JsonValue) -> Decimal:
    if (
        not isinstance(value, dict)
        or set(value) != {"__decimal__"}
        or not isinstance(value["__decimal__"], str)
    ):
        raise ValueError("stored decimal is invalid")
    try:
        decoded = Decimal(value["__decimal__"])
    except InvalidOperation:
        raise ValueError("stored decimal is invalid") from None
    if not decoded.is_finite():
        raise ValueError("stored decimal must be finite")
    return decoded


def optional_decimal_encode(value: Decimal | None) -> JsonValue:
    return None if value is None else decimal_encode(value)


def optional_decimal_decode(value: JsonValue) -> Decimal | None:
    return None if value is None else decimal_decode(value)


def sequence(value: JsonValue, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"stored {label} must be a list")
    return value


def mapping(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"stored {label} must be an object")
    return value


def text(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"stored {label} must be text")
    return value


def optional_text(value: JsonValue, label: str) -> str | None:
    return None if value is None else text(value, label)


def integer(value: JsonValue, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"stored {label} must be an integer")
    return value


def optional_integer(value: JsonValue, label: str) -> int | None:
    return None if value is None else integer(value, label)


def boolean(value: JsonValue, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"stored {label} must be a boolean")
    return value


def number(value: JsonValue, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"stored {label} must be a number")
    decoded = float(value)
    if not math.isfinite(decoded):
        raise ValueError(f"stored {label} must be finite")
    return decoded


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False
