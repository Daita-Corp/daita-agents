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


def _monitor_detail_answer(
    resolution: DbMonitorResolution,
    *,
    command: DbMonitorCommand,
) -> str:
    detail = str(command.patch.get("detail") or "plan")
    context = _resolution_monitor_context(resolution)
    plan = context["observation_plan"]
    sql = str(plan.get("sql") or "").strip()
    schedule = _schedule_text(context.get("schedule"))
    cursor = _cursor_text(plan)
    suffix = _schedule_cursor_sentence(schedule, cursor)

    if detail == "sql":
        if sql:
            answer = f"The monitor SQL is:\n{sql}"
        else:
            answer = "That monitor does not have SQL in its observation plan."
        return f"{answer}\n\n{suffix}" if suffix else answer
    if detail == "schedule":
        if suffix:
            return suffix
        return "That monitor does not have a polling schedule or cursor recorded."

    target = plan.get("target_name") or context.get("name") or context.get("monitor_id")
    target_text = f" watches {target}" if target else ""
    if sql:
        answer = f"The monitor{target_text} and runs:\n{sql}"
    else:
        answer = f"The monitor{target_text} uses its recorded observation plan."
    return f"{answer}\n\n{suffix}" if suffix else answer


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


def _resolution_monitor_context(resolution: DbMonitorResolution) -> dict[str, Any]:
    if resolution.monitor is not None:
        return {
            "monitor_id": resolution.monitor.id,
            "name": resolution.monitor.name,
            "observation_plan": dict(resolution.monitor.observation_plan),
            "schedule": resolution.monitor.schedule,
        }
    if resolution.definition_evidence is not None:
        monitor = resolution.definition_evidence.payload.get("monitor")
        if isinstance(monitor, dict):
            return {
                "monitor_id": monitor.get("id"),
                "name": monitor.get("name"),
                "observation_plan": dict(monitor.get("observation_plan") or {}),
                "schedule": monitor.get("schedule"),
            }
    if resolution.proposal_evidence is not None:
        payload = resolution.proposal_evidence.payload
        return {
            "monitor_id": payload.get("monitor_id"),
            "name": payload.get("name"),
            "observation_plan": dict(payload.get("observation_plan") or {}),
            "schedule": payload.get("schedule"),
        }
    return {"observation_plan": {}, "schedule": None}


def _schedule_text(schedule: Any) -> str:
    if not isinstance(schedule, dict):
        return ""
    interval_seconds = schedule.get("interval_seconds") or schedule.get("every_seconds")
    if isinstance(interval_seconds, (int, float)) and interval_seconds > 0:
        if interval_seconds % 3600 == 0:
            hours = int(interval_seconds // 3600)
            return "hourly" if hours == 1 else f"every {hours} hours"
        if interval_seconds % 60 == 0:
            minutes = int(interval_seconds // 60)
            return "every minute" if minutes == 1 else f"every {minutes} minutes"
        seconds = int(interval_seconds)
        return "every second" if seconds == 1 else f"every {seconds} seconds"
    expression = str(schedule.get("expression") or "").strip()
    if not expression:
        return ""
    parts = expression.split()
    if (
        len(parts) >= 5
        and parts[0].startswith("*/")
        and parts[1:5]
        == [
            "*",
            "*",
            "*",
            "*",
        ]
    ):
        minutes = parts[0][2:]
        if minutes == "1":
            return "every minute"
        return f"every {minutes} minutes"
    if len(parts) >= 5 and parts[:2] == ["0", "*"]:
        return "hourly"
    return f"on {expression}"


def _cursor_text(plan: dict[str, Any]) -> str:
    cursor = plan.get("cursor")
    if isinstance(cursor, dict):
        field = cursor.get("field")
        if field:
            return str(field)
    if isinstance(cursor, str) and cursor.strip():
        return cursor.strip()
    return ""


def _schedule_cursor_sentence(schedule: str, cursor: str) -> str:
    if schedule and cursor:
        return f"It polls {schedule} and uses {cursor} as the cursor."
    if schedule:
        return f"It polls {schedule}."
    if cursor:
        return f"It uses {cursor} as the cursor."
    return ""
