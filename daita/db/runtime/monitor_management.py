"""Monitor runtime behavior for ``DbRuntime``."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Evidence,
    Operation,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStore,
    SQLiteRuntimeStore,
)

from ..models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from ..monitors import (
    DbMonitor,
    DbMonitorInspection,
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
    DbMonitorStore,
    InMemoryDbMonitorStore,
    SQLiteDbMonitorStore,
    monitor_with_updates,
)
from ..monitor_scheduler import DbMonitorScheduler




class DbRuntimeMonitorManagementMixin:
    async def _latest_monitor_action_result(
        self,
        operation_id: str,
        *,
        action_plan_fingerprint: str,
    ) -> Evidence | None:
        return await self._latest_evidence(
            operation_id,
            "monitor.action_result",
            payload={"action_plan_fingerprint": action_plan_fingerprint},
        )

    async def create_monitor(self, monitor: DbMonitor) -> DbMonitor:
        """Persist a DB monitor definition and audit the create operation."""
        operation, evidence, events = self._monitor_management_artifacts(
            "create",
            monitor=monitor,
            evidence_kind="monitor.definition",
            payload={"monitor": monitor.to_dict()},
        )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="create",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_after=monitor,
                state_after=DbMonitorState(
                    monitor_id=monitor.id,
                    last_operation_id=operation.id,
                    last_management_operation_id=operation.id,
                ),
            )
        )
        return monitor

    async def list_monitors(
        self, *, status: str | None = None
    ) -> tuple[DbMonitor, ...]:
        """List durable DB monitor definitions."""
        return await self.monitor_store.list_monitors(status=status)

    async def inspect_monitor(self, monitor_id: str) -> DbMonitorInspection | None:
        """Return a monitor definition with durable state and run summaries."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            return None
        state = await self.monitor_store.load_monitor_state(monitor_id)
        runs = await self.monitor_store.list_monitor_runs(monitor_id)
        return DbMonitorInspection(monitor=monitor, state=state, runs=runs)

    async def update_monitor(
        self,
        monitor_id: str,
        patch: dict[str, Any],
    ) -> DbMonitor:
        """Patch a durable monitor definition and audit the update."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        updated = monitor_with_updates(monitor, patch)
        operation, evidence, events = self._monitor_management_artifacts(
            "update",
            monitor=updated,
            evidence_kind="monitor.state_update",
            payload={
                "monitor_id": monitor_id,
                "before": monitor.to_dict(),
                "after": updated.to_dict(),
                "patch": dict(patch),
            },
        )
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="update",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
                monitor_after=updated,
                state_after=DbMonitorState.from_dict(
                    {
                        **state.to_dict(),
                        "last_operation_id": operation.id,
                        "last_management_operation_id": operation.id,
                    }
                ),
            )
        )
        return updated

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        paused_until: str | None = None,
    ) -> DbMonitor:
        """Mark a monitor paused and persist its pause state."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        updated = monitor_with_updates(monitor, {"status": "paused"})
        operation, evidence, events = self._monitor_management_artifacts(
            "pause",
            monitor=updated,
            evidence_kind="monitor.state_update",
            payload={
                "monitor_id": monitor_id,
                "before": monitor.to_dict(),
                "after": updated.to_dict(),
                "paused_until": paused_until,
            },
        )
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="pause",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
                monitor_after=updated,
                state_after=DbMonitorState.from_dict(
                    {
                        **state.to_dict(),
                        "last_operation_id": operation.id,
                        "last_management_operation_id": operation.id,
                        "paused_until": paused_until,
                    }
                ),
            )
        )
        return updated

    async def resume_monitor(self, monitor_id: str) -> DbMonitor:
        """Resume a paused monitor."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        updated = monitor_with_updates(monitor, {"status": "active"})
        operation, evidence, events = self._monitor_management_artifacts(
            "resume",
            monitor=updated,
            evidence_kind="monitor.state_update",
            payload={
                "monitor_id": monitor_id,
                "before": monitor.to_dict(),
                "after": updated.to_dict(),
            },
        )
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="resume",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
                monitor_after=updated,
                state_after=DbMonitorState.from_dict(
                    {
                        **state.to_dict(),
                        "last_operation_id": operation.id,
                        "last_management_operation_id": operation.id,
                        "paused_until": None,
                    }
                ),
            )
        )
        return updated

    async def delete_monitor(self, monitor_id: str) -> DbMonitor:
        """Delete a monitor control-plane record and audit the removal."""
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            raise ValueError(f"monitor {monitor_id!r} does not exist")
        operation, evidence, events = self._monitor_management_artifacts(
            "delete",
            monitor=monitor,
            evidence_kind="monitor.definition",
            payload={"deleted_monitor": monitor.to_dict()},
        )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="delete",
                operation=operation,
                evidence=evidence,
                events=events,
                monitor_before=monitor,
            )
        )
        return monitor

    async def list_monitor_approvals(
        self,
        *,
        monitor_id: str | None = None,
        monitor_run_id: str | None = None,
        pending_only: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        """List monitor-related approvals without mutating approval state."""

        approvals: list[dict[str, Any]] = []
        for approval in await self.store.list_approval_requests():
            if pending_only and approval.status is not ApprovalStatus.PENDING:
                continue
            context = await self._monitor_approval_context(approval)
            if not context:
                continue
            if monitor_id is not None and context.get("monitor_id") != monitor_id:
                continue
            if (
                monitor_run_id is not None
                and context.get("monitor_run_id") != monitor_run_id
            ):
                continue
            approvals.append(
                {
                    "approval_id": approval.approval_id,
                    "operation_id": approval.operation_id,
                    "task_id": getattr(approval, "task_id", None),
                    "status": approval.status.value,
                    "reason": approval.reason,
                    "risk": approval.risk.value,
                    "requested_by_policy_id": approval.requested_by_policy_id,
                    "owner": approval.owner,
                    "context": context,
                }
            )
        return tuple(approvals)

    async def approve_monitor_approval(self, approval_id: str) -> ApprovalRequest:
        """Approve a monitor approval through the configured approval channel."""

        return await self.approval_channel.approve(approval_id)

    async def reject_monitor_approval(self, approval_id: str) -> ApprovalRequest:
        """Reject a monitor approval through the configured approval channel."""

        return await self.approval_channel.reject(approval_id)

    async def cancel_monitor_approval(self, approval_id: str) -> ApprovalRequest:
        """Cancel a monitor approval through the configured approval channel."""

        return await self.approval_channel.cancel(approval_id)

    async def tick_monitors(
        self, *, now: datetime | str | None = None
    ) -> tuple[DbMonitorRun, ...]:
        """Run one durable DB monitor scheduler pass."""
        if not self._is_setup:
            await self.setup()
        scheduler = DbMonitorScheduler(runtime=self)
        results = await scheduler.run_once(now=now)
        return tuple(result.run for result in results)

    async def record_monitor_command_result(
        self,
        *,
        request: DbRequest,
        kind: str,
        command: dict[str, Any],
        status: OperationStatus,
        answer: str,
        operation_id: str | None = None,
        evidence: tuple[Evidence, ...] = (),
        evidence_kind: str | None = None,
        payload: dict[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
        diagnostics: dict[str, Any] | None = None,
        persist_operation: bool = True,
    ) -> DbOperationResult:
        """Record a prompt-level monitor command result through runtime audit.

        This is a control-plane audit path only. It persists operation/evidence/
        event records for monitor management commands without planning SQL,
        creating tasks, invoking executors, or evaluating governance.
        """
        command_payload = dict(command)
        operation_id = operation_id or f"monitor-command-{kind}-{uuid4()}"
        persisted_evidence = tuple(evidence)
        if persist_operation:
            operation = Operation(
                id=operation_id,
                operation_type=f"monitor.{kind}",
                status=status,
                request={
                    "kind": f"monitor.{kind}",
                    "prompt": request.prompt,
                    "command": command_payload,
                },
                required_evidence=(
                    frozenset({evidence_kind})
                    if evidence_kind is not None
                    else frozenset()
                ),
                metadata={
                    "runtime_id": self.runtime_id,
                    "runtime_kind": self.runtime_kind,
                    "control_plane": "db.monitor",
                    "monitor_id": command_payload.get("monitor_id"),
                    "command_kind": kind,
                },
            )
            if evidence_kind is not None:
                persisted_evidence = (
                    Evidence(
                        id=f"monitor-command-evidence-{uuid4()}",
                        kind=evidence_kind,
                        owner="db.monitor",
                        operation_id=operation_id,
                        payload=dict(payload or {}),
                        accepted=status is OperationStatus.SUCCEEDED,
                    ),
                )
            created_event = RuntimeEvent(
                id=f"monitor-command-event-{uuid4()}",
                type=RuntimeEventType.OPERATION_CREATED,
                operation_id=operation_id,
                runtime_id=self.runtime_id,
                runtime_kind=self.runtime_kind,
                message=f"Monitor command operation {operation_id} created.",
                payload={"operation_type": operation.operation_type},
            )
            completed_event = RuntimeEvent(
                id=f"monitor-command-event-{uuid4()}",
                type=RuntimeEventType.OPERATION_UPDATED,
                operation_id=operation_id,
                runtime_id=self.runtime_id,
                runtime_kind=self.runtime_kind,
                evidence_id=(persisted_evidence[0].id if persisted_evidence else None),
                message=f"Monitor command {kind} finished with {status.value}.",
                payload={
                    "status": status.value,
                    "command_kind": kind,
                    "monitor_id": command_payload.get("monitor_id"),
                    "warnings": list(warnings),
                },
            )
            await self.store.save_operation(operation)
            for item in persisted_evidence:
                await self.store.save_evidence(item)
            await self.store.append_event(created_event)
            await self.store.append_event(completed_event)

        result = DbOperationResult(
            operation_id=operation_id,
            request=request,
            intent=DbIntent(
                kind=DbIntentKind.ADMIN,
                confidence=float(command_payload.get("confidence") or 0.0),
                access=AccessMode.NONE,
                diagnostics={
                    "command_kind": kind,
                    **dict(command_payload.get("diagnostics") or {}),
                },
            ),
            contract=DbOperationContract(
                operation_type=f"monitor.{kind}",
                required_evidence=tuple(item.kind for item in persisted_evidence),
                access=AccessMode.NONE,
                metadata={
                    "control_plane": "db.monitor",
                    "command": command_payload,
                },
            ),
            status=status,
            answer=answer,
            evidence=persisted_evidence,
            warnings=warnings,
            diagnostics={
                "command": command_payload,
                **dict(diagnostics or {}),
            },
        )
        return await self._record_operation_result(result)

    def _monitor_management_artifacts(
        self,
        action: str,
        *,
        monitor: DbMonitor,
        evidence_kind: str,
        payload: dict[str, Any],
    ) -> tuple[Operation, tuple[Evidence, ...], tuple[RuntimeEvent, ...]]:
        operation_id = f"monitor-{action}-{uuid4()}"
        operation = Operation(
            id=operation_id,
            operation_type=f"monitor.{action}",
            status=OperationStatus.SUCCEEDED,
            request={
                "kind": f"monitor.{action}",
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
            },
            required_evidence=frozenset({evidence_kind}),
            metadata={
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "monitor_status": monitor.status,
                "control_plane": "db.monitor",
            },
        )
        evidence = Evidence(
            id=f"monitor-evidence-{uuid4()}",
            kind=evidence_kind,
            owner="db.monitor",
            operation_id=operation_id,
            payload=payload,
        )
        created_event = RuntimeEvent(
            id=f"monitor-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_CREATED,
            operation_id=operation_id,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            message=f"Monitor operation {operation_id} created.",
            payload={"operation_type": operation.operation_type},
        )
        completed_event = RuntimeEvent(
            id=f"monitor-event-{uuid4()}",
            type=RuntimeEventType.OPERATION_UPDATED,
            operation_id=operation_id,
            runtime_id=self.runtime_id,
            runtime_kind=self.runtime_kind,
            evidence_id=evidence.id,
            message=f"Monitor {monitor.id} {action} committed.",
            payload={
                "status": OperationStatus.SUCCEEDED.value,
                "monitor_id": monitor.id,
                "monitor_name": monitor.name,
                "action": action,
            },
        )
        return operation, (evidence,), (created_event, completed_event)



def _default_monitor_store(store: RuntimeStore) -> DbMonitorStore:
    if isinstance(store, SQLiteRuntimeStore):
        return SQLiteDbMonitorStore(store.path)
    return InMemoryDbMonitorStore(store)
