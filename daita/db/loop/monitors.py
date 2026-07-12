"""Monitor action helpers for the DB agent loop."""

from __future__ import annotations

from typing import Any

from ..planner_protocol import DbPlannerAction, DbPlannerActionKind

_MONITOR_ACTION_CAPABILITIES: dict[DbPlannerActionKind, str] = {
    DbPlannerActionKind.PLAN_MONITOR_CREATE: "db.monitor.plan_create",
    DbPlannerActionKind.COMMIT_MONITOR_CREATE: "db.monitor.commit_create",
    DbPlannerActionKind.PLAN_MONITOR_LIFECYCLE: "db.monitor.plan_lifecycle",
    DbPlannerActionKind.COMMIT_MONITOR_LIFECYCLE: "db.monitor.commit_lifecycle",
    DbPlannerActionKind.READ_MONITOR_STATE: "db.monitor.read",
    DbPlannerActionKind.RESOLVE_MONITOR_APPROVAL: "db.monitor.resolve_approval",
}


def _monitor_action_output_evidence(
    action: DbPlannerAction,
) -> tuple[str, str | None]:
    if action.kind is DbPlannerActionKind.PLAN_MONITOR_CREATE:
        return "monitor.proposal", None
    if action.kind is DbPlannerActionKind.COMMIT_MONITOR_CREATE:
        return "monitor.definition", None
    if action.kind is DbPlannerActionKind.PLAN_MONITOR_LIFECYCLE:
        return "monitor.proposal", None
    if action.kind is DbPlannerActionKind.COMMIT_MONITOR_LIFECYCLE:
        lifecycle_action = _monitor_lifecycle_action_label(
            action.input.get("action") or action.input.get("operation_type")
        )
        if lifecycle_action is None:
            return "", "missing_monitor_lifecycle_action"
        return _monitor_lifecycle_evidence_kind(lifecycle_action), None
    if action.kind is DbPlannerActionKind.READ_MONITOR_STATE:
        read_kind = str(action.input.get("read_kind") or "list").lower()
        evidence_kind = {
            "list": "monitor.listing",
            "inspect": "monitor.inspection",
            "explain_run": "monitor.run_summary",
            "approvals": "monitor.approval_state",
        }.get(read_kind)
        if evidence_kind is None:
            return "", f"unsupported_monitor_read_kind:{read_kind}"
        return evidence_kind, None
    if action.kind is DbPlannerActionKind.RESOLVE_MONITOR_APPROVAL:
        return "monitor.approval_resolution", None
    return "", "unsupported_monitor_action_kind"


def _monitor_lifecycle_action_label(value: Any) -> str | None:
    normalized = str(value or "").removeprefix("monitor.").replace("_", ".").lower()
    if normalized in {"update", "pause", "resume", "delete", "disable"}:
        return normalized
    if normalized == "disabled":
        return "disable"
    return None


def _monitor_lifecycle_evidence_kind(action: str) -> str:
    if action == "delete":
        return "monitor.deleted"
    if action == "disable":
        return "monitor.disabled"
    if action == "pause":
        return "monitor.paused"
    if action == "resume":
        return "monitor.resumed"
    return "monitor.state_update"
