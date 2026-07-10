"""Generic helpers for the DB agent loop."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _safe_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _first_string_list_from_mappings(
    mappings: Iterable[Mapping[str, Any]],
    *keys: str,
) -> list[str]:
    for mapping in mappings:
        for key in keys:
            values = _string_list(mapping.get(key))
            if values:
                return values
    return []


def _safe_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _first_present(
    *sources: Mapping[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source.get(key) is not None:
                return source[key]
    return default


def _split_column_ref(value: Any) -> tuple[str, str]:
    parts = [
        part.strip().strip('"`[]')
        for part in str(value or "").split(".")
        if part.strip().strip('"`[]')
    ]
    if len(parts) >= 2:
        return ".".join(parts[:-1]), parts[-1]
    return "", ""


def _dedupe_dicts(
    values: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, Any]] = []
    for value in values:
        key = tuple(str(value.get(item) or "") for item in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _dedupe_json_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _float_option(value: Mapping[str, Any], key: str, default: float) -> float:
    raw = value.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _ordered_unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _json_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    copied = dict(value or {})
    try:
        json.dumps(copied, sort_keys=True, default=str)
    except TypeError as exc:
        raise TypeError("DB agent loop mappings must be JSON serializable") from exc
    return copied
