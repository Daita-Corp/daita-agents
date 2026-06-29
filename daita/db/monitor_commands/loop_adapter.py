"""Typed loop adapter for compatibility monitor commands."""

from __future__ import annotations

from typing import Any, Mapping

from ..planner_protocol import DbPlannerAction, DbPlannerDecision


class DbMonitorCommandLoopPlanner:
    """Emit typed monitor planner actions from an already-routed command."""

    def __init__(self, command: Mapping[str, Any]) -> None:
        self.command = dict(command)

    def decide(
        self, context: Mapping[str, Any], observations: Any
    ) -> DbPlannerDecision:
        evidence = _evidence(context, observations)
        kind = str(self.command.get("kind") or "")
        if kind == "create":
            return self._create_decision(context, evidence)
        if kind in {"update", "pause", "resume", "delete"}:
            return self._lifecycle_decision(context, kind, evidence)
        if kind in {"list", "inspect", "explain_run"}:
            if not _has_evidence(evidence, "monitor.snapshot"):
                return DbPlannerDecision(
                    actions=(
                        DbPlannerAction(
                            kind="inspect_monitor",
                            payload={
                                "command": self.command,
                                "monitor_id": self.command.get("monitor_id"),
                                "status": dict(self.command.get("patch") or {}).get(
                                    "status"
                                ),
                                "detail": dict(self.command.get("patch") or {}).get(
                                    "detail"
                                ),
                            },
                            action_id="monitor.inspect",
                        ),
                    )
                )
            return _finish("Monitor inspection completed.")
        if kind == "execute":
            if not _has_evidence(evidence, "monitor.execution"):
                return DbPlannerDecision(
                    actions=(
                        DbPlannerAction(
                            kind="execute_monitor",
                            payload={"command": self.command},
                            action_id="monitor.execute",
                        ),
                    )
                )
            return _finish("Monitor execution completed.")
        return _finish("Monitor command completed.")

    def _create_decision(
        self,
        context: Mapping[str, Any],
        evidence: tuple[dict[str, Any], ...],
    ) -> DbPlannerDecision:
        proposal = _latest_evidence(evidence, "monitor.proposal")
        if proposal is None:
            request = dict(context.get("request") or {})
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="create_monitor",
                        payload={
                            "phase": "plan",
                            "command": self.command,
                            "prompt": request.get("prompt")
                            or self.command.get("prompt"),
                            "source_scope": list(request.get("source_scope") or ()),
                            "owner": _owner_from_request(request),
                        },
                        action_id="monitor.create.plan",
                    ),
                )
            )
        if not bool(proposal.get("accepted", True)):
            return _finish("Monitor proposal did not pass validation.")
        if _latest_evidence(evidence, "monitor.definition") is not None:
            return _finish("Monitor create completed.")
        diagnostics = dict(self.command.get("diagnostics") or {})
        if diagnostics.get("approval_required") and not _has_task(
            context,
            "db.monitor.commit_create",
        ):
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind="clarify",
                        payload={"message": "Monitor create requires approval."},
                    ),
                )
            )
        if _has_task(context, "db.monitor.commit_create"):
            return _finish("Monitor create is waiting for commit.")
        payload = dict(proposal.get("payload") or {})
        return DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind="create_monitor",
                    payload={
                        "phase": "commit",
                        "proposal_evidence_id": proposal.get("id"),
                        "proposal_fingerprint": payload.get("proposal_fingerprint"),
                    },
                    action_id="monitor.create.commit",
                ),
            )
        )

    def _lifecycle_decision(
        self,
        context: Mapping[str, Any],
        kind: str,
        evidence: tuple[dict[str, Any], ...],
    ) -> DbPlannerDecision:
        proposal = _latest_evidence(evidence, "monitor.proposal")
        action_kind = {
            "update": "update_monitor",
            "pause": "pause_monitor",
            "resume": "resume_monitor",
            "delete": "delete_monitor",
        }[kind]
        if proposal is None:
            patch = dict(self.command.get("patch") or {})
            return DbPlannerDecision(
                actions=(
                    DbPlannerAction(
                        kind=action_kind,
                        payload={
                            "phase": "plan",
                            "action": kind,
                            "monitor_id": self.command.get("monitor_id"),
                            "patch": patch,
                            "paused_until": patch.get("paused_until"),
                        },
                        action_id=f"monitor.{kind}.plan",
                    ),
                )
            )
        if not bool(proposal.get("accepted", True)):
            return _finish("Monitor lifecycle proposal did not pass validation.")
        payload = dict(proposal.get("payload") or {})
        if _has_lifecycle_commit(evidence, str(payload.get("action") or kind)):
            return _finish(f"Monitor {kind} completed.")
        if _has_task(context, "db.monitor.commit_lifecycle"):
            return _finish(f"Monitor {kind} is waiting for commit.")
        return DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind=action_kind,
                    payload={
                        "phase": "commit",
                        "proposal_evidence_id": proposal.get("id"),
                        "proposal_fingerprint": payload.get("proposal_fingerprint"),
                    },
                    action_id=f"monitor.{kind}.commit",
                ),
            )
        )


def _finish(message: str) -> DbPlannerDecision:
    return DbPlannerDecision(
        actions=(DbPlannerAction(kind="finish", payload={"message": message}),)
    )


def _owner_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    owner: dict[str, Any] = {}
    if request.get("user_id") is not None:
        owner["user_id"] = request["user_id"]
    if request.get("session_id") is not None:
        owner["session_id"] = request["session_id"]
    return owner


def _evidence(
    context: Mapping[str, Any],
    observations: Any,
) -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    for item in context.get("evidence_observations") or ():
        if isinstance(item, dict):
            items.append(dict(item))
    for observation in observations or ():
        payload = getattr(observation, "payload", {}) or {}
        for item in payload.get("evidence") or ():
            if isinstance(item, dict):
                items.append(dict(item))
    return tuple(items)


def _latest_evidence(
    evidence: tuple[dict[str, Any], ...],
    kind: str,
) -> dict[str, Any] | None:
    for item in reversed(evidence):
        if item.get("kind") == kind:
            return item
    return None


def _has_evidence(evidence: tuple[dict[str, Any], ...], kind: str) -> bool:
    return any(
        item.get("kind") == kind and bool(item.get("accepted", True))
        for item in evidence
    )


def _has_lifecycle_commit(
    evidence: tuple[dict[str, Any], ...],
    action: str,
) -> bool:
    expected = {
        "update": "monitor.state_update",
        "pause": "monitor.paused",
        "resume": "monitor.resumed",
        "delete": "monitor.deleted",
        "disable": "monitor.disabled",
    }.get(action, "monitor.state_update")
    return _has_evidence(evidence, expected)


def _has_task(context: Mapping[str, Any], capability_id: str) -> bool:
    return any(
        isinstance(item, dict) and item.get("capability_id") == capability_id
        for item in context.get("task_observations") or ()
    )
