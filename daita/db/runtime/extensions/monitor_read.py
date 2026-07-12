"""Runtime-owned DB monitor read and approval executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from daita.runtime import Evidence, Operation, Task

from ...fingerprints import persisted_fingerprint
from ...monitor_commands.resolver import DbMonitorResolver
from ...monitor_commands.types import DbMonitorCommand, DbMonitorResolution

if TYPE_CHECKING:
    from .plugin import DbRuntimePlanningPlugin


@dataclass(frozen=True)
class DbMonitorReadExecutor:
    """Executor that reads monitor state and persists monitor evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.read"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.read"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        read_kind = str(task.input.get("read_kind") or "list").lower()
        if read_kind == "list":
            status = task.input.get("status")
            monitors = await runtime.list_monitors(
                status=str(status) if status is not None else None
            )
            payload = {
                "read_kind": "list",
                "status": status,
                "monitors": [monitor.to_dict() for monitor in monitors],
            }
            return [_monitor_evidence("monitor.listing", operation, task, payload)]

        if read_kind in {"inspect", "explain_run"}:
            resolution = await _resolve_monitor(runtime, task.input)
            evidence_kind = (
                "monitor.run_summary"
                if read_kind == "explain_run"
                else "monitor.inspection"
            )
            if not resolution.accepted or resolution.monitor is None:
                payload = {
                    "read_kind": read_kind,
                    "resolution": resolution.to_dict(),
                    "errors": list(resolution.errors),
                }
                return [
                    _monitor_evidence(
                        evidence_kind,
                        operation,
                        task,
                        payload,
                        accepted=False,
                    )
                ]
            inspection = await runtime.inspect_monitor(resolution.monitor.id)
            if inspection is None:
                payload = {
                    "read_kind": read_kind,
                    "resolution": resolution.to_dict(),
                    "errors": ["monitor_not_found"],
                }
                return [
                    _monitor_evidence(
                        evidence_kind,
                        operation,
                        task,
                        payload,
                        accepted=False,
                    )
                ]
            payload = {
                "read_kind": read_kind,
                "resolution": resolution.to_dict(),
                "inspection": inspection.to_dict(),
            }
            return [_monitor_evidence(evidence_kind, operation, task, payload)]

        if read_kind == "approvals":
            monitor_id = _optional_string(task.input.get("monitor_id"))
            approvals = await runtime.list_monitor_approvals(
                monitor_id=monitor_id,
                monitor_run_id=_optional_string(task.input.get("monitor_run_id")),
                pending_only=bool(task.input.get("pending_only", True)),
            )
            payload = {
                "read_kind": "approvals",
                "monitor_id": monitor_id,
                "approvals": [dict(item) for item in approvals],
                "pending_only": bool(task.input.get("pending_only", True)),
            }
            return [
                _monitor_evidence("monitor.approval_state", operation, task, payload)
            ]

        raise ValueError(f"unsupported monitor read kind: {read_kind!r}")


@dataclass(frozen=True)
class DbMonitorResolveApprovalExecutor:
    """Executor that mutates monitor approval state through approval channels."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.resolve_approval"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.resolve_approval"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        action = str(task.input.get("approval_action") or "approve").lower()
        if action not in {"approve", "reject", "cancel"}:
            raise ValueError(f"unsupported monitor approval action: {action!r}")
        approval_id = _optional_string(task.input.get("approval_id"))
        monitor_id = _optional_string(task.input.get("monitor_id"))
        approvals: tuple[dict[str, Any], ...] = tuple(
            dict(item)
            for item in await runtime.list_monitor_approvals(
                monitor_id=monitor_id,
                pending_only=True,
            )
        )
        if approval_id is not None:
            approvals = tuple(
                approval
                for approval in approvals
                if approval.get("approval_id") == approval_id
            )
        payload: dict[str, Any] = {
            "approval_action": action,
            "approval_id": approval_id,
            "monitor_id": monitor_id,
            "matched_approvals": [dict(item) for item in approvals],
        }
        matched_approval = next(iter(approvals), None)
        if matched_approval is None:
            payload["status"] = "not_found"
            return [
                _monitor_evidence(
                    "monitor.approval_resolution",
                    operation,
                    task,
                    payload,
                )
            ]
        if len(approvals) > 1:
            payload["status"] = "ambiguous"
            return [
                _monitor_evidence(
                    "monitor.approval_resolution",
                    operation,
                    task,
                    payload,
                )
            ]

        resolved_id = str(matched_approval["approval_id"])
        if action == "reject":
            approval = await runtime.reject_monitor_approval(resolved_id)
        elif action == "cancel":
            approval = await runtime.cancel_monitor_approval(resolved_id)
        else:
            approval = await runtime.approve_monitor_approval(resolved_id)
        payload.update(
            {
                "status": "resolved",
                "approval_id": approval.approval_id,
                "approval_status": approval.status_value,
                "operation_id": approval.operation_id,
            }
        )
        return [
            _monitor_evidence(
                "monitor.approval_resolution",
                operation,
                task,
                payload,
            )
        ]


async def _resolve_monitor(
    runtime: Any, data: Mapping[str, Any]
) -> DbMonitorResolution:
    monitor_ref = _optional_string(data.get("monitor_id") or data.get("monitor_ref"))
    monitors = await runtime.list_monitors()
    return DbMonitorResolver().resolve(
        DbMonitorCommand(kind="inspect", monitor_id=monitor_ref),
        monitors,
    )


def _monitor_evidence(
    kind: str,
    operation: Operation,
    task: Task,
    payload: dict[str, Any],
    *,
    accepted: bool = True,
) -> Evidence:
    return Evidence(
        kind=kind,
        owner="db_runtime",
        operation_id=operation.id,
        task_id=task.id,
        accepted=accepted,
        payload=payload,
        metadata={"payload_fingerprint": persisted_fingerprint(payload)},
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
