"""SQL connector parameter coercion helpers."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
import json
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID

from daita.core.db_type_metadata import is_timezone_type, native_type_from_param_spec

JsonBinding = Literal["native", "text"]


class SQLParameterCoercionError(ValueError):
    """Raised when a typed SQL parameter cannot be coerced safely."""

    def __init__(
        self,
        *,
        index: int,
        spec: Mapping[str, Any],
        value: Any,
        target_type: str,
        error: Exception,
    ) -> None:
        ref = spec.get("ref")
        db_type = spec.get("db_type") or spec.get("native_type") or target_type
        super().__init__(
            "Failed to coerce SQL parameter "
            f"${index}"
            f"{f' ({ref})' if ref else ''} "
            f"to {db_type!r}: raw type {type(value).__name__}; {error}"
        )
        self.index = index
        self.ref = str(ref) if ref else None
        self.db_type = str(db_type)
        self.raw_type = type(value).__name__


def coerce_sql_params(
    params: Sequence[Any],
    specs: Sequence[Mapping[str, Any]] | None,
    *,
    dialect: str,
    json_binding: JsonBinding = "text",
) -> list[Any]:
    if not specs:
        return list(params)
    coerced: list[Any] = []
    for index, value in enumerate(params, start=1):
        spec = specs[index - 1] if index - 1 < len(specs) else {}
        if not _has_type_metadata(spec):
            coerced.append(value)
            continue
        try:
            coerced.append(
                _coerce_value(
                    value,
                    spec,
                    dialect=dialect,
                    json_binding=json_binding,
                )
            )
        except SQLParameterCoercionError:
            raise
        except Exception as exc:
            target_type = native_type_from_param_spec(spec) or "unknown"
            raise SQLParameterCoercionError(
                index=index,
                spec=spec,
                value=value,
                target_type=target_type,
                error=exc,
            ) from exc
    return coerced


def param_specs_from_payload(args: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in args.get("param_specs") or () if isinstance(item, Mapping)]


def _coerce_value(
    value: Any,
    spec: Mapping[str, Any],
    *,
    dialect: str,
    json_binding: JsonBinding,
) -> Any:
    if value is None:
        return None
    target = native_type_from_param_spec(spec)
    if target is None:
        return value
    if target == "datetime":
        return _coerce_datetime(value, timezone_aware=is_timezone_type(spec))
    if target == "date":
        return _coerce_date(value)
    if target == "time":
        return _coerce_time(value)
    if target == "uuid":
        return value if isinstance(value, UUID) else UUID(str(value))
    if target == "decimal":
        return value if isinstance(value, Decimal) else Decimal(str(value))
    if target == "json":
        return _coerce_json(value, binding=json_binding)
    if target == "integer":
        return _coerce_int(value)
    if target == "float":
        return value if isinstance(value, float) else float(value)
    if target == "boolean":
        return _coerce_bool(value)
    if target == "string":
        return value if isinstance(value, str) else str(value)
    return value


def _coerce_datetime(value: Any, *, timezone_aware: bool) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.min)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        result = datetime.fromisoformat(text)
    else:
        raise TypeError("expected datetime/date or ISO datetime string")
    if timezone_aware and result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result


def _coerce_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip().split("T", 1)[0])
    raise TypeError("expected date/datetime or ISO date string")


def _coerce_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, str):
        return time.fromisoformat(value.strip())
    raise TypeError("expected time/datetime or ISO time string")


def _coerce_json(value: Any, *, binding: JsonBinding) -> Any:
    if binding == "native":
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, separators=(",", ":"))


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "f", "0", "no", "n", "off"}:
            return False
    raise TypeError("expected boolean or boolean-like string")


def _has_type_metadata(spec: Mapping[str, Any]) -> bool:
    return any(spec.get(key) for key in ("db_type", "native_type", "dialect"))
