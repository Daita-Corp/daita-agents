"""State, cursor, and scheduling gate helpers for DB monitor ticks."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping

from .types import DbMonitorObservationResult
from ..monitor_commands.types import DbMonitorValidation
from ..monitors import DbMonitor, DbMonitorState


def _cursor_updates_from_plan(
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return updates
    cursor_update = plan.get("cursor_update")
    if not isinstance(cursor_update, Mapping):
        return updates
    for cursor_key, rule in cursor_update.items():
        if not isinstance(cursor_key, str) or not isinstance(rule, str):
            continue
        match = re.fullmatch(r"max\(rows\.([A-Za-z_][A-Za-z0-9_]*)\)", rule)
        if not match:
            continue
        field = match.group(1)
        values = [
            row.get(field)
            for row in rows
            if isinstance(row, Mapping) and row.get(field) is not None
        ]
        if values:
            updates[cursor_key] = max(values)
    return updates


def _is_due(
    monitor: DbMonitor,
    state: DbMonitorState | None,
    now: datetime,
) -> tuple[bool, str]:
    if monitor.status == "paused":
        return False, "paused"
    if monitor.status == "disabled":
        return False, "disabled"
    if monitor.status == "error":
        return False, "error"
    state = state or DbMonitorState(monitor_id=monitor.id)
    blocked = _time_gate_reason(state, now)
    if blocked is not None:
        return False, blocked
    interval = _schedule_interval_seconds(monitor.schedule)
    if interval is None:
        return True, "due"
    if state.last_tick_at is None:
        return True, "due"
    elapsed = (now - _parse_iso(state.last_tick_at)).total_seconds()
    return (elapsed >= interval, "due" if elapsed >= interval else "not_due")


def _blocked_reason(
    monitor: DbMonitor,
    state: DbMonitorState,
    now: datetime,
) -> str | None:
    if monitor.status == "paused":
        return "paused"
    if monitor.status in {"disabled", "error"}:
        return monitor.status
    time_gate = _time_gate_reason(state, now)
    if time_gate is not None:
        return time_gate
    validation = _validation_for_monitor(monitor)
    if validation is None:
        return None
    if (
        not validation.accepted
        or validation.errors
        or validation.missing_capabilities
        or validation.unsupported_actions
    ):
        return "validation_blocked"
    return None


def _time_gate_reason(state: DbMonitorState, now: datetime) -> str | None:
    if state.paused_until is not None and _parse_iso(state.paused_until) > now:
        return "paused"
    if state.cooldown_until is not None and _parse_iso(state.cooldown_until) > now:
        return "cooldown"
    backoff_until = (state.error or {}).get("backoff_until")
    if backoff_until and _parse_iso(str(backoff_until)) > now:
        return "backoff"
    return None


def _state_for_scheduler_decision(
    state: DbMonitorState,
    reason: str,
    now: datetime,
) -> DbMonitorState | None:
    if reason == "lease_lost":
        return None
    if reason == "paused" and state.paused_until is not None:
        return state
    if reason in {"cooldown", "backoff", "not_due"}:
        return state
    if reason in {"disabled", "error"}:
        return DbMonitorState.from_dict(
            {**state.to_dict(), "error": {"reason": reason, "checked_at": _iso(now)}}
        )
    return state


def _state_for_blocked_decision(
    state: DbMonitorState,
    reason: str,
    now: datetime,
    operation_id: str,
) -> DbMonitorState:
    if reason == "validation_blocked":
        return DbMonitorState.from_dict(
            {
                **state.to_dict(),
                "last_tick_at": _iso(now),
                "last_operation_id": operation_id,
                "last_tick_operation_id": operation_id,
                "consecutive_failures": state.consecutive_failures + 1,
                "error": {"reason": reason, "checked_at": _iso(now)},
            }
        )
    return state


def _error_backoff_seconds(monitor: DbMonitor, failures: int) -> float:
    raw = monitor.budgets.get("error_backoff_seconds") or monitor.budgets.get(
        "backoff_seconds"
    )
    try:
        base = float(raw if raw is not None else 60.0)
    except (TypeError, ValueError):
        base = 60.0
    return max(1.0, base * max(1, failures))


def _schedule_interval_seconds(schedule: Mapping[str, Any] | None) -> float | None:
    if not schedule:
        return None
    for key in ("interval_seconds", "every_seconds"):
        if key in schedule:
            try:
                return max(0.0, float(schedule[key]))
            except (TypeError, ValueError):
                return None
    expression = str(schedule.get("expression") or "")
    every = re.search(r"every\s+(\d+)\s+(second|minute|hour)s?", expression, re.I)
    if every:
        amount = int(every.group(1))
        unit = every.group(2).lower()
        multiplier = {"second": 1, "minute": 60, "hour": 3600}[unit]
        return float(amount * multiplier)
    cron_every = re.match(r"^\*/(\d+)\s+\*\s+\*\s+\*\s+\*$", expression)
    if cron_every:
        return float(int(cron_every.group(1)) * 60)
    return None


def _validation_for_monitor(monitor: DbMonitor) -> DbMonitorValidation | None:
    data = monitor.metadata.get("validation")
    if not isinstance(data, Mapping):
        return None
    return DbMonitorValidation.from_dict(dict(data))


def _validation_payload(monitor: DbMonitor) -> dict[str, Any] | None:
    validation = _validation_for_monitor(monitor)
    return None if validation is None else validation.to_dict()


def _state_value_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _coerce_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        return _parse_iso(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
