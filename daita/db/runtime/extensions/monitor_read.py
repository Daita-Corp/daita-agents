"""Runtime-owned DB monitor read and approval executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from daita.runtime import Evidence, Operation, Task

from ...fingerprints import persisted_fingerprint
from ...monitor_commands.resolver import DbMonitorResolver
from ...monitor_commands.types import DbMonitorCommand, DbMonitorResolution
from .monitor_evidence import evidence_matches_dependency

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
            pending_only = bool(task.input.get("pending_only", True))
            approvals = await runtime.list_monitor_approvals(
                monitor_id=monitor_id,
                monitor_run_id=_optional_string(task.input.get("monitor_run_id")),
                pending_only=pending_only,
            )
            payload = _bounded_approval_state_payload(
                {
                    "read_kind": "approvals",
                    "monitor_id": monitor_id,
                    "approvals": [dict(item) for item in approvals],
                    "pending_only": pending_only,
                }
            )
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
        approval_id = _exact_approval_id(task.input.get("approval_id"))
        monitor_id = _optional_string(task.input.get("monitor_id"))
        approvals: tuple[dict[str, Any], ...]
        candidate_count: int
        candidates_truncated = False
        if approval_id is not None:
            approvals = tuple(
                dict(item)
                for item in await runtime.list_monitor_approvals(
                    monitor_id=monitor_id,
                    pending_only=True,
                )
                if item.get("approval_id") == approval_id
            )
            candidate_count = len(approvals)
        else:
            approval_state = await _approval_state_dependency(
                runtime,
                operation,
                task,
                monitor_id=monitor_id,
            )
            if approval_state is None:
                payload = _bounded_approval_resolution_payload(
                    {
                        "approval_action": action,
                        "approval_id": None,
                        "monitor_id": monitor_id,
                        "matched_approvals": [],
                        "matched_approval_count": 0,
                        "matched_approvals_truncated": False,
                        "status": "inbox_required",
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
            raw_approvals = approval_state.payload.get("approvals")
            approvals = tuple(
                dict(item)
                for item in (
                    raw_approvals if isinstance(raw_approvals, (list, tuple)) else ()
                )
                if isinstance(item, Mapping)
            )
            reported_count = approval_state.payload.get("approval_count")
            candidate_count = len(approvals)
            if (
                isinstance(reported_count, int)
                and not isinstance(reported_count, bool)
                and reported_count >= candidate_count
            ):
                candidate_count = reported_count
            candidates_truncated = approval_state.payload.get(
                "approvals_truncated"
            ) is True or candidate_count > len(approvals)
        payload: dict[str, Any] = {
            "approval_action": action,
            "approval_id": approval_id,
            "monitor_id": monitor_id,
            "matched_approvals": [dict(item) for item in approvals],
            "matched_approval_count": candidate_count,
            "matched_approvals_truncated": candidates_truncated,
        }
        if candidate_count == 0:
            payload["status"] = "not_found"
            return [
                _monitor_evidence(
                    "monitor.approval_resolution",
                    operation,
                    task,
                    _bounded_approval_resolution_payload(payload),
                )
            ]
        if candidate_count > 1:
            payload["status"] = "ambiguous"
            return [
                _monitor_evidence(
                    "monitor.approval_resolution",
                    operation,
                    task,
                    _bounded_approval_resolution_payload(payload),
                )
            ]
        matched_approval = next(iter(approvals), None)
        if matched_approval is None:
            payload["status"] = "inbox_incomplete"
            return [
                _monitor_evidence(
                    "monitor.approval_resolution",
                    operation,
                    task,
                    _bounded_approval_resolution_payload(payload),
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
                _bounded_approval_resolution_payload(payload),
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


def _exact_approval_id(value: Any) -> str | None:
    if not isinstance(value, str) or value == "":
        return None
    return value


async def _approval_state_dependency(
    runtime: Any,
    operation: Operation,
    task: Task,
    *,
    monitor_id: str | None,
) -> Evidence | None:
    evidence = await runtime.store.list_evidence(operation.id)
    for dependency in task.dependencies:
        if dependency.kind_value != "evidence":
            continue
        if dependency.evidence_kind != "monitor.approval_state":
            continue
        for item in reversed(evidence):
            if not evidence_matches_dependency(item, dependency):
                continue
            payload = item.payload if isinstance(item.payload, Mapping) else {}
            if payload.get("read_kind") != "approvals":
                continue
            if payload.get("pending_only") is not True:
                continue
            if (
                monitor_id is not None
                and _optional_string(payload.get("monitor_id")) != monitor_id
            ):
                continue
            return item
    return None


def _bounded_approval_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...loop.summaries import _monitor_approval_state_summary

    return _monitor_approval_state_summary(payload)


def _bounded_approval_resolution_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    from ...loop.summaries import _bounded_monitor_approval_resolution_payload

    return _bounded_monitor_approval_resolution_payload(payload)
