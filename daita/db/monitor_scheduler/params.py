"""Typed parameter contract helpers for DB monitor observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..monitors import DbMonitorState

_MISSING = object()


@dataclass(frozen=True)
class MonitorObservationParamSpec:
    """Normalized typed parameter spec for a persisted monitor observation plan."""

    ref: str | None = None
    source: str | None = None
    path: tuple[str, ...] = ()
    table: str | None = None
    column: str | None = None
    db_type: str | None = None
    native_type: str | None = None
    dialect: str | None = None
    nullable: bool | None = None
    value: Any = _MISSING

    @property
    def has_value(self) -> bool:
        return self.value is not _MISSING

    @property
    def has_type_metadata(self) -> bool:
        return any(
            item is not None
            for item in (
                self.db_type,
                self.native_type,
                self.dialect,
                self.table,
                self.column,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.ref:
            payload["ref"] = self.ref
        if self.source:
            payload["source"] = self.source
        if self.path:
            payload["path"] = list(self.path)
        if self.table:
            payload["table"] = self.table
        if self.column:
            payload["column"] = self.column
        if self.db_type:
            payload["db_type"] = self.db_type
        if self.native_type:
            payload["native_type"] = self.native_type
        if self.dialect:
            payload["dialect"] = self.dialect
        if self.nullable is not None:
            payload["nullable"] = self.nullable
        if self.has_value:
            payload["value"] = self.value
        return payload


def normalize_observation_param_spec(value: Any) -> MonitorObservationParamSpec:
    """Normalize a structured observation parameter."""

    if isinstance(value, Mapping):
        payload = dict(value)
        ref = _string_or_none(payload.get("ref"))
        path = _path_tuple(payload.get("path"))
        if not path and ref:
            path = _path_from_ref(ref)
        source = _string_or_none(payload.get("source")) or _source_from_ref(ref)
        nullable = _nullable_value(payload.get("nullable"))
        literal = payload["value"] if "value" in payload else _MISSING
        return MonitorObservationParamSpec(
            ref=ref,
            source=source,
            path=path,
            table=_string_or_none(payload.get("table")),
            column=_string_or_none(payload.get("column")),
            db_type=_string_or_none(
                payload.get("db_type")
                or payload.get("database_type")
                or payload.get("column_type")
            ),
            native_type=_string_or_none(payload.get("native_type")),
            dialect=_string_or_none(
                payload.get("dialect") or payload.get("sql_dialect")
            ),
            nullable=nullable,
            value=literal,
        )
    raise TypeError(
        "monitor observation parameters must be structured specs; use "
        "{'ref': 'monitor.state.cursor.<key>', ...} or {'value': ...}"
    )


def resolve_observation_params(
    plan: Mapping[str, Any],
    state: DbMonitorState,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Resolve monitor observation params and preserve their normalized specs."""

    params: list[Any] = []
    specs: list[dict[str, Any]] = []
    for value in list(plan.get("parameters") or plan.get("params") or ()):
        spec = normalize_observation_param_spec(value)
        params.append(resolve_observation_param(spec, state))
        specs.append(spec.to_dict())
    return params, specs


def resolve_observation_param(
    spec: MonitorObservationParamSpec,
    state: DbMonitorState,
) -> Any:
    if spec.has_value:
        return spec.value
    if spec.ref:
        return resolve_monitor_state_ref(spec.ref, state)
    return None


def resolve_monitor_state_ref(ref: str, state: DbMonitorState) -> Any:
    if ref.startswith("monitor.state.cursor."):
        current: Any = state.cursor
        parts = ref.removeprefix("monitor.state.cursor.").split(".")
    elif ref.startswith("cursor."):
        current = state.cursor
        parts = ref.removeprefix("cursor.").split(".")
    else:
        raise ValueError(f"unsupported monitor state parameter ref: {ref}")
    for part in parts:
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _source_from_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    if ref.startswith("monitor.state.") or ref.startswith("cursor."):
        return "monitor_state"
    return None


def _path_from_ref(ref: str) -> tuple[str, ...]:
    if ref.startswith("monitor.state.cursor."):
        return ("cursor", *ref.removeprefix("monitor.state.cursor.").split("."))
    if ref.startswith("cursor."):
        return ("cursor", *ref.removeprefix("cursor.").split("."))
    return ()


def _path_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part for part in value.split(".") if part)
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in value if str(part))
    return ()


def _nullable_value(value: Any) -> bool | None:
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


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
