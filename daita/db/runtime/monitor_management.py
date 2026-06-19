"""Monitor runtime behavior for ``DbRuntime``."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
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
    TaskDependency,
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
)
from ..monitor_scheduler.scheduler import DbMonitorScheduler
from .types import DbRuntimeGovernanceBlocked


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
        _raise_if_non_executable_active_monitor(monitor)
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
        """Patch a durable monitor definition through runtime lifecycle tasks."""
        result = await self.execute_monitor_lifecycle_operation(
            {
                "operation_type": "monitor.update",
                "monitor_id": monitor_id,
                "patch": dict(patch),
            }
        )
        return _monitor_from_lifecycle_result(result)

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        paused_until: str | None = None,
    ) -> DbMonitor:
        """Pause a monitor through runtime lifecycle tasks."""
        result = await self.execute_monitor_lifecycle_operation(
            {
                "operation_type": "monitor.pause",
                "monitor_id": monitor_id,
                "paused_until": paused_until,
            }
        )
        return _monitor_from_lifecycle_result(result)

    async def resume_monitor(self, monitor_id: str) -> DbMonitor:
        """Resume a monitor through runtime lifecycle tasks."""
        result = await self.execute_monitor_lifecycle_operation(
            {
                "operation_type": "monitor.resume",
                "monitor_id": monitor_id,
            }
        )
        return _monitor_from_lifecycle_result(result)

    async def delete_monitor(self, monitor_id: str) -> DbMonitor:
        """Delete a monitor through runtime lifecycle tasks."""
        result = await self.execute_monitor_lifecycle_operation(
            {
                "operation_type": "monitor.delete",
                "monitor_id": monitor_id,
            }
        )
        return _monitor_from_lifecycle_result(result)

    async def execute_monitor_lifecycle_operation(
        self,
        request: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        operation_id: str | None = None,
        source: str = "runtime",
    ) -> DbOperationResult:
        """Run update/delete/pause/resume through proposal and commit tasks."""

        if not self._is_setup:
            await self.setup()
        operation_type = str(request.get("operation_type") or "monitor.update")
        action = _monitor_lifecycle_action(operation_type)
        monitor_id = str(request.get("monitor_id") or "")
        if not monitor_id:
            raise ValueError("monitor lifecycle operation requires monitor_id")
        patch = dict(request.get("patch") or {})
        paused_until = request.get("paused_until")
        required_evidence = frozenset(
            {
                "monitor.proposal",
                _monitor_lifecycle_evidence_kind(action),
            }
        )
        operation = await self.kernel.create_operation(
            operation_id=operation_id,
            operation_type=operation_type,
            request={
                "kind": operation_type,
                "monitor_id": monitor_id,
                "patch": patch,
                "paused_until": paused_until,
                "source": source,
                **dict(context or {}),
            },
            required_evidence=required_evidence,
            metadata={
                "runtime_id": self.runtime_id,
                "runtime_kind": self.runtime_kind,
                "control_plane": "db.monitor",
                "monitor_id": monitor_id,
                "command_kind": action,
                "access": AccessMode.WRITE.value,
                "resume_context": {
                    "request": {
                        "operation_type": operation_type,
                        "monitor_id": monitor_id,
                        "patch": patch,
                        "paused_until": paused_until,
                    },
                    "contract": {
                        "operation_type": operation_type,
                        "required_evidence": sorted(required_evidence),
                        "access": AccessMode.WRITE.value,
                    },
                },
            },
            evaluate_governance=False,
        )
        plan_task = await self.kernel.plan_task(
            operation_id=operation.id,
            capability_id="db.monitor.plan_lifecycle",
            owner="db_runtime",
            input={
                "action": action,
                "monitor_id": monitor_id,
                "patch": patch,
                "paused_until": paused_until,
            },
            metadata={
                "reason": f"monitor_{action}_planning",
                "sequence": 1,
                "idempotency_key": _stable_monitor_lifecycle_hash(
                    {
                        "operation_type": operation_type,
                        "monitor_id": monitor_id,
                        "patch": patch,
                        "paused_until": paused_until,
                    }
                ),
            },
        )
        plan_evidence = await self.execute_task(plan_task, operation)
        proposal_evidence = next(
            (item for item in plan_evidence if item.kind == "monitor.proposal"),
            None,
        )
        if proposal_evidence is None:
            raise RuntimeError(
                "monitor lifecycle planning did not produce proposal evidence"
            )
        proposal = dict(proposal_evidence.payload)
        validation = _validation_from_lifecycle_proposal(proposal)
        if not validation.accepted:
            await self.kernel.block_operation(
                operation.id,
                message="Monitor lifecycle proposal is incomplete or unsupported.",
                payload={"validation": validation.to_dict()},
            )
            return await self._record_monitor_lifecycle_result(
                request=request,
                operation=operation,
                action=action,
                status=OperationStatus.BLOCKED,
                answer=_monitor_lifecycle_validation_answer(action, validation),
                evidence=tuple(plan_evidence),
                warnings=("db_monitor_validation_failed", *validation.warnings),
                diagnostics={
                    "proposal": proposal,
                    "validation": validation.to_dict(),
                },
            )

        commit_task = await self.kernel.plan_task(
            operation_id=operation.id,
            capability_id="db.monitor.commit_lifecycle",
            owner="db_runtime",
            input={
                "proposal_evidence_id": proposal_evidence.id,
                "proposal_fingerprint": proposal["proposal_fingerprint"],
            },
            metadata={
                "reason": f"monitor_{action}_commit",
                "sequence": 2,
                "idempotency_key": proposal["proposal_fingerprint"],
            },
            dependencies=(
                TaskDependency(
                    kind="evidence",
                    evidence_kind="monitor.proposal",
                    evidence_id=proposal_evidence.id,
                    evidence_owner="db_runtime",
                    producer_task_id=plan_task.id,
                    producer_capability_id=plan_task.capability_id,
                    producer_executor_id=plan_task.executor_id,
                    evidence_payload={
                        "proposal_fingerprint": proposal["proposal_fingerprint"],
                    },
                    evidence_accepted=True,
                    operation_id=operation.id,
                ),
            ),
        )
        try:
            commit_evidence = await self.execute_task(commit_task, operation)
        except DbRuntimeGovernanceBlocked as exc:
            snapshot = await self.inspect_operation(operation.id)
            evidence = tuple(plan_evidence)
            if snapshot is not None:
                evidence = tuple(snapshot.evidence)
            return await self._record_monitor_lifecycle_result(
                request=request,
                operation=exc.operation,
                action=action,
                status=OperationStatus.BLOCKED,
                answer="This operation requires approval before changing the monitor.",
                evidence=evidence,
                warnings=("db_runtime_approval_required",),
                diagnostics={
                    "proposal": proposal,
                    "validation": validation.to_dict(),
                    "governance": exc.governance.to_dict(),
                },
            )

        lifecycle_evidence = next(
            (
                item
                for item in commit_evidence
                if item.kind == _monitor_lifecycle_evidence_kind(action)
            ),
            None,
        )
        if lifecycle_evidence is None:
            raise RuntimeError(
                "monitor lifecycle commit did not produce commit evidence"
            )
        monitor_payload = (
            lifecycle_evidence.payload.get("monitor")
            or lifecycle_evidence.payload.get("after")
            or lifecycle_evidence.payload.get("before")
        )
        monitor = (
            DbMonitor.from_dict(monitor_payload)
            if isinstance(monitor_payload, dict)
            else None
        )
        await self.kernel.complete_operation(
            operation.id,
            status=OperationStatus.SUCCEEDED,
            message=f"Monitor {monitor_id} {action} committed.",
            payload={"monitor_id": monitor_id, "action": action},
        )
        return await self._record_monitor_lifecycle_result(
            request=request,
            operation=operation,
            action=action,
            status=OperationStatus.SUCCEEDED,
            answer=_monitor_lifecycle_answer(action, monitor_id),
            evidence=(*plan_evidence, *commit_evidence),
            diagnostics={
                "proposal": proposal,
                "validation": validation.to_dict(),
                "monitor": None if monitor is None else monitor.to_dict(),
            },
        )

    async def _record_monitor_lifecycle_result(
        self,
        *,
        request: dict[str, Any],
        operation: Operation,
        action: str,
        status: OperationStatus,
        answer: str,
        evidence: tuple[Evidence, ...],
        warnings: tuple[str, ...] = (),
        diagnostics: dict[str, Any] | None = None,
    ) -> DbOperationResult:
        result = DbOperationResult(
            operation_id=operation.id,
            request=DbRequest(
                prompt=str(
                    request.get("prompt") or request.get("operation_type") or ""
                ),
                metadata={
                    key: value for key, value in request.items() if key != "prompt"
                },
            ),
            intent=DbIntent(
                kind=DbIntentKind.ADMIN,
                confidence=1.0,
                access=AccessMode.WRITE,
                diagnostics={"command_kind": action},
            ),
            contract=DbOperationContract(
                operation_type=operation.operation_type,
                required_evidence=tuple(item.kind for item in evidence),
                access=AccessMode.WRITE,
                metadata={
                    "control_plane": "db.monitor",
                    "monitor_id": operation.metadata.get("monitor_id"),
                    "action": action,
                },
            ),
            status=status,
            answer=answer,
            evidence=evidence,
            warnings=warnings,
            diagnostics=diagnostics or {},
        )
        return await self._record_operation_result(result, operation=operation)

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
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "source_scope": list(request.source_scope),
                    "metadata": request.metadata,
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
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "source_scope": list(request.source_scope),
                    "request_metadata": request.metadata,
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


def _raise_if_non_executable_active_monitor(monitor: DbMonitor) -> None:
    if monitor.status != "active":
        return
    reason = _non_executable_observation_plan_reason(monitor.observation_plan)
    if reason is not None:
        raise ValueError(
            "active monitors require an executable observation_plan; " f"{reason}"
        )


def _monitor_from_lifecycle_result(result: DbOperationResult) -> DbMonitor:
    if result.status is OperationStatus.BLOCKED:
        if "db_monitor_validation_failed" in result.warnings:
            validation = result.diagnostics.get("validation")
            if isinstance(validation, dict) and validation.get("errors"):
                message = ", ".join(str(item) for item in validation["errors"])
                message = message.replace("monitor.lifecycle:", "")
                raise ValueError(message)
            raise ValueError(result.answer or "Monitor lifecycle proposal is invalid.")
        raise PermissionError(
            result.answer or "Monitor lifecycle operation is blocked."
        )
    monitor = result.diagnostics.get("monitor")
    if not isinstance(monitor, dict):
        raise RuntimeError("monitor lifecycle operation did not return a monitor")
    return DbMonitor.from_dict(monitor)


def _monitor_lifecycle_action(operation_type: str) -> str:
    normalized = operation_type.removeprefix("monitor.").lower()
    if normalized in {"update", "pause", "resume", "delete", "disable"}:
        return normalized
    raise ValueError(f"unsupported monitor lifecycle operation: {operation_type!r}")


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


def _validation_from_lifecycle_proposal(proposal: dict[str, Any]) -> Any:
    validation = proposal.get("validation")
    if isinstance(validation, dict):
        from ..monitor_commands.types import DbMonitorValidation

        return DbMonitorValidation.from_dict(validation)
    from ..monitor_commands.types import DbMonitorValidation

    return DbMonitorValidation(accepted=bool(proposal.get("accepted", True)))


def _monitor_lifecycle_validation_answer(action: str, validation: Any) -> str:
    errors = ", ".join(str(item) for item in validation.errors)
    return f"Monitor {action} proposal is incomplete or unsupported" + (
        f": {errors}" if errors else "."
    )


def _monitor_lifecycle_answer(action: str, monitor_id: str) -> str:
    verb = {
        "update": "Updated",
        "pause": "Paused",
        "resume": "Resumed",
        "delete": "Deleted",
        "disable": "Disabled",
    }.get(action, "Updated")
    return f"{verb} monitor {monitor_id}."


def _stable_monitor_lifecycle_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_executable_observation_plan_reason(plan: dict[str, Any]) -> str | None:
    kind = (plan or {}).get("kind")
    if kind not in {"planned_read", "metric_sql", "freshness_sql", "plugin_source"}:
        return "missing executable kind"
    if kind in {"planned_read", "metric_sql", "freshness_sql"}:
        if not isinstance(plan.get("sql"), str) or not plan["sql"].strip():
            return "missing observation SQL"
        if not plan.get("value_path"):
            return "missing value_path"
    if kind == "planned_read":
        if not plan.get("cursor") or not plan.get("cursor_update"):
            return "missing cursor strategy"
    if kind == "plugin_source":
        if not plan.get("capability_id") and not plan.get("source_kind"):
            return "missing plugin source capability"
    return None


def _default_monitor_store(store: RuntimeStore) -> DbMonitorStore:
    if isinstance(store, SQLiteRuntimeStore):
        return SQLiteDbMonitorStore(store.path)
    return InMemoryDbMonitorStore(store)
