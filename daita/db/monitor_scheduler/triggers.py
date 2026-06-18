"""Trigger evaluation helpers for durable DB monitor ticks."""

from __future__ import annotations

from typing import Any, Mapping

from daita.runtime.monitors import (
    MonitorSpec,
    monitor_trigger_matches,
    summarize_monitor_value,
)

from ..monitors import DbMonitor, DbMonitorState
from .state import _validation_payload


def _compile_monitor_spec(
    monitor: DbMonitor,
    state: DbMonitorState,
    *,
    tick_operation_id: str,
    run_id: str,
) -> MonitorSpec:
    return MonitorSpec(
        id=monitor.id,
        name=monitor.name,
        schedule=monitor.schedule,
        stream=monitor.stream,
        trigger={},
        action_input={"action_plan": monitor.action_plan},
        cooldown_seconds=_cooldown_seconds(monitor) or None,
        cursor=state.cursor,
        metadata={
            "parent_operation_id": tick_operation_id,
            "tick_operation_id": tick_operation_id,
            "monitor_run_id": run_id,
            "db_monitor": {
                "id": monitor.id,
                "tick_operation_id": tick_operation_id,
                "run_id": run_id,
                "source_scope": list(monitor.source_scope),
                "observation_plan": monitor.observation_plan,
                "policy": monitor.policy,
                "budgets": monitor.budgets,
                "owner": monitor.owner,
                "validation": _validation_payload(monitor),
            },
        },
    )


def _runtime_spec_for_decision(spec: MonitorSpec, trigger_ready: bool) -> MonitorSpec:
    return MonitorSpec(
        id=spec.id,
        name=spec.name,
        schedule=spec.schedule,
        stream=spec.stream,
        trigger={} if trigger_ready else {"truthy": True},
        action_input=spec.action_input,
        cooldown_seconds=spec.cooldown_seconds,
        cursor=spec.cursor,
        metadata=spec.metadata,
    )


def _trigger_decision(
    value: Any,
    trigger: Mapping[str, Any],
) -> tuple[bool, Any, str]:
    if not trigger:
        return True, summarize_monitor_value(value), "trigger_matched"
    if trigger.get("type") in {"schedule", "scheduled"}:
        return True, summarize_monitor_value(value), "schedule"
    if "force_trigger" in trigger:
        return bool(trigger["force_trigger"]), summarize_monitor_value(value), "forced"
    if value is None:
        return False, None, "observation_unavailable"
    trigger_value = (
        {"rows": value}
        if trigger.get("path") == "rows" and isinstance(value, list)
        else value
    )
    try:
        matched = monitor_trigger_matches(trigger_value, trigger)
    except TypeError:
        return False, summarize_monitor_value(value), "trigger_type_mismatch"
    return (
        matched,
        summarize_monitor_value(value),
        "trigger_matched" if matched else "no_match",
    )


def _required_consecutive_matches(monitor: DbMonitor) -> int:
    raw = (
        monitor.trigger.get("consecutive_matches")
        or monitor.budgets.get("consecutive_matches")
        or monitor.policy.get("consecutive_matches")
        or 1
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _cooldown_seconds(monitor: DbMonitor) -> float:
    raw = (
        monitor.budgets.get("cooldown_seconds")
        or monitor.budgets.get("cooldown")
        or monitor.policy.get("cooldown_seconds")
        or 0
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0
