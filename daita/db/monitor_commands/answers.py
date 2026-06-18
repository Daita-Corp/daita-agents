"""User-facing answer formatting for DB monitor commands."""

from __future__ import annotations

from typing import Any

from ..monitors import DbMonitor
from .types import DbMonitorCommand, DbMonitorResolution, DbMonitorValidation


def _create_monitor_answer(monitor: DbMonitor) -> str:
    schedule = ""
    if monitor.schedule:
        schedule = f" on {monitor.schedule.get('expression', 'its schedule')}"
    return f"Created monitor {monitor.name} ({monitor.id}){schedule}."


def _monitor_validation_answer(validation: DbMonitorValidation) -> str:
    missing = ", ".join(validation.missing_capabilities)
    if missing:
        return f"Monitor was not created because required capabilities are missing: {missing}."
    return "Monitor was not created because its definition did not pass validation."


def _list_monitors_answer(monitors: tuple[DbMonitor, ...]) -> str:
    if not monitors:
        return "No monitors are currently defined."
    lines = ["Monitors:"]
    for monitor in monitors:
        lines.append(f"- {monitor.id}: {monitor.name} [{monitor.status}]")
    return "\n".join(lines)


def _inspect_monitor_answer(
    inspection: Any,
    *,
    command: DbMonitorCommand,
) -> str:
    monitor = inspection.monitor
    if command.kind == "explain_run":
        if not inspection.runs:
            return f"Monitor {monitor.name} ({monitor.id}) has no recorded runs yet."
        last_run = inspection.runs[-1]
        return (
            f"Monitor {monitor.name} ({monitor.id}) last run "
            f"{last_run.status}; operation {last_run.operation_id}."
        )
    schedule = ""
    if monitor.schedule:
        schedule = f", schedule {monitor.schedule.get('expression')}"
    return f"Monitor {monitor.name} ({monitor.id}) is {monitor.status}{schedule}."


def _approval_action_answer(
    action: str,
    monitor_id: str,
    approval_id: str,
    *,
    operation_status: str,
) -> str:
    verb = {
        "approve": "Approved",
        "reject": "Rejected",
        "cancel": "Cancelled",
    }.get(action, "Updated")
    return (
        f"{verb} monitor approval {approval_id} for {monitor_id}; "
        f"operation is {operation_status}."
    )


def _resolution_failure_answer(
    reason: str,
    resolution: DbMonitorResolution,
) -> str:
    if reason == "monitor_reference_required":
        return "Please specify which monitor to manage."
    if reason == "monitor_reference_ambiguous":
        matches = ", ".join(monitor.id for monitor in resolution.matches)
        return f"Monitor reference is ambiguous; matching monitors: {matches}."
    return f"Monitor {resolution.monitor_ref!r} was not found."
