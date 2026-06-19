"""Command service orchestration for DB monitor management."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Evidence,
    OperationStatus,
    RiskLevel,
    RuntimeEvent,
    RuntimeEventType,
    TaskDependency,
)

from ..models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from ..monitors import DbMonitor
from .answers import (
    _approval_action_answer,
    _create_monitor_answer,
    _inspect_monitor_answer,
    _list_monitors_answer,
    _monitor_detail_answer,
    _monitor_validation_answer,
    _resolution_failure_answer,
)
from .planner import _stable_monitor_payload_hash
from .prompt_parsing import (
    _create_name_phrase,
    _monitor_id_from_phrase,
    _title_from_phrase,
)
from .resolver import DbMonitorResolver
from .router import DbCommandRouter
from .types import DbMonitorCommand, DbMonitorResolution, DbMonitorValidation


class DbMonitorCommandService:
    """Execute prompt-managed monitor commands through the DB runtime control plane."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.router = DbCommandRouter()
        self.resolver = DbMonitorResolver()

    async def run(self, request: DbRequest) -> DbOperationResult | None:
        has_monitor_context = _request_has_monitor_context(request)
        if not has_monitor_context and request.session_id:
            has_monitor_context = await self._has_session_monitor_context(
                request.session_id
            )
        command = self.router.route(
            request.prompt,
            has_monitor_context=has_monitor_context,
        )
        if command is None:
            return None
        try:
            return await self._run_command(command, request)
        except ValueError as exc:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command),
                status=OperationStatus.FAILED,
                answer=str(exc),
                evidence_kind="monitor.command.error",
                payload={
                    "command": _command_payload(command),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                warnings=("db_monitor_command_failed",),
            )

    async def _run_command(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        if command.kind == "create":
            return await self._create(command, request)
        if command.kind == "list":
            return await self._list(command, request)
        if command.kind in {"inspect", "explain_run"}:
            return await self._inspect(command, request)
        if command.kind == "update":
            return await self._update(command, request)
        if command.kind == "pause":
            return await self._pause(command, request)
        if command.kind == "resume":
            return await self._resume(command, request)
        if command.kind == "delete":
            return await self._delete(command, request)
        if command.kind == "approve_action":
            return await self._approve_action(command, request)
        raise AssertionError(f"unsupported monitor command kind: {command.kind}")

    async def _create(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        if not self.runtime.is_setup:
            await self.runtime.setup()
        monitor_id = command.monitor_id or _monitor_id_from_phrase(
            _title_from_phrase(_create_name_phrase(request.prompt))
        )
        monitor_name = _title_from_phrase(_create_name_phrase(request.prompt))

        operation = await self.runtime.kernel.create_operation(
            operation_type="monitor.create",
            request={
                "kind": "monitor.create",
                "prompt": request.prompt,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "source_scope": list(request.source_scope),
                "metadata": request.metadata,
                "command": _command_payload(command),
            },
            required_evidence=frozenset({"monitor.proposal", "monitor.definition"}),
            metadata={
                "control_plane": "db.monitor",
                "command_kind": "create",
                "monitor_id": monitor_id,
                "monitor_name": monitor_name,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "source_scope": list(request.source_scope),
                "request_metadata": request.metadata,
                "resume_context": {
                    "request": {
                        "prompt": request.prompt,
                        "user_id": request.user_id,
                        "session_id": request.session_id,
                        "source_scope": list(request.source_scope),
                        "metadata": request.metadata,
                    },
                    "intent": {
                        "kind": DbIntentKind.ADMIN.value,
                        "confidence": command.confidence,
                        "access": AccessMode.WRITE.value,
                    },
                    "contract": {
                        "operation_type": "monitor.create",
                        "required_evidence": [
                            "monitor.proposal",
                            "monitor.definition",
                        ],
                        "access": AccessMode.WRITE.value,
                    },
                },
            },
            evaluate_governance=False,
        )
        plan_task = await self.runtime.kernel.plan_task(
            operation_id=operation.id,
            capability_id="db.monitor.plan_create",
            owner="db_runtime",
            input={
                "command": _command_payload(command),
                "prompt": request.prompt,
                "source_scope": list(request.source_scope),
                "owner": _owner_from_request(request),
            },
            metadata={
                "reason": "monitor_create_planning",
                "sequence": 1,
                "idempotency_key": _stable_monitor_payload_hash(
                    {
                        "prompt": request.prompt,
                        "command": _command_payload(command),
                        "source_scope": list(request.source_scope),
                    }
                ),
            },
        )
        plan_evidence = await self.runtime.execute_task(plan_task, operation)
        proposal_evidence = next(
            (item for item in plan_evidence if item.kind == "monitor.proposal"),
            None,
        )
        if proposal_evidence is None:
            raise RuntimeError("monitor planning did not produce proposal evidence")
        proposal = dict(proposal_evidence.payload)
        validation = DbMonitorValidation.from_dict(
            dict(proposal.get("validation") or {})
        )
        operation = replace(
            operation,
            metadata={
                **operation.metadata,
                "monitor_id": proposal["monitor_id"],
                "monitor_name": proposal["name"],
                "proposal_fingerprint": proposal["proposal_fingerprint"],
            },
        )
        await self.runtime.store.save_operation(operation)
        if not validation.accepted:
            await self.runtime.kernel.block_operation(
                operation.id,
                message="Monitor proposal is incomplete or unsupported.",
                payload={"validation": validation.to_dict()},
            )
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command),
                status=OperationStatus.BLOCKED,
                answer=_monitor_validation_answer(validation),
                operation_id=operation.id,
                evidence=tuple(plan_evidence),
                payload={
                    "command": _command_payload(command),
                    "proposal": proposal,
                    "validation": validation.to_dict(),
                },
                warnings=("db_monitor_validation_failed", *validation.warnings),
                diagnostics={"validation": validation.to_dict()},
                persist_operation=False,
            )
        commit_task = await self.runtime.kernel.plan_task(
            operation_id=operation.id,
            capability_id="db.monitor.commit_create",
            owner="db_runtime",
            input={
                "proposal_evidence_id": proposal_evidence.id,
                "proposal_fingerprint": proposal["proposal_fingerprint"],
            },
            metadata={
                "reason": "monitor_create_commit",
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
        if command.diagnostics.get("approval_required") is True:
            return await self._create_pending_approval(
                command,
                request,
                proposal,
                proposal_evidence,
                commit_task.id,
                validation,
            )
        commit_evidence = await self.runtime.execute_task(commit_task, operation)
        definition = next(
            (item for item in commit_evidence if item.kind == "monitor.definition"),
            None,
        )
        if definition is None:
            raise RuntimeError("monitor commit did not produce definition evidence")
        committed = DbMonitor.from_dict(definition.payload["monitor"])
        inspection = await self.runtime.inspect_monitor(committed.id)
        await self.runtime.kernel.complete_operation(
            operation.id,
            status=OperationStatus.SUCCEEDED,
            message=f"Monitor {committed.id} created from proposal.",
            payload={"monitor_id": committed.id},
        )
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.SUCCEEDED,
            answer=_create_monitor_answer(committed),
            operation_id=operation.id,
            evidence=(*plan_evidence, *commit_evidence),
            warnings=validation.warnings,
            diagnostics={
                "monitor": committed.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
                "validation": validation.to_dict(),
                "proposal_evidence_id": proposal_evidence.id,
            },
            persist_operation=False,
        )

    async def _create_pending_approval(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        proposal: dict[str, Any],
        proposal_evidence: Evidence,
        commit_task_id: str,
        validation: "DbMonitorValidation",
    ) -> DbOperationResult:
        operation = await self.runtime.store.load_operation(
            proposal_evidence.operation_id
        )
        if operation is None:
            raise RuntimeError(
                f"monitor create operation {proposal_evidence.operation_id} missing"
            )
        command_payload = _command_payload(command)
        approval = ApprovalRequest(
            approval_id=f"{operation.id}:runtime.approval_required:monitor.create",
            operation_id=operation.id,
            reason="Creating a monitor changes runtime monitor configuration.",
            proposed_action={
                "operation_type": "monitor.create",
                "request": operation.request,
                "monitor": {
                    "id": proposal["monitor_id"],
                    "name": proposal["name"],
                    "trigger": proposal["trigger"],
                    "action_plan": proposal["action_plan"],
                },
                "approval": "monitor.create",
                "proposal_evidence_id": proposal_evidence.id,
                "proposal_fingerprint": proposal["proposal_fingerprint"],
            },
            risk=RiskLevel.MEDIUM,
            requested_by_policy_id="runtime.approval_required",
            owner="db.monitor",
            metadata={
                "command": command_payload,
                "commit_task_id": commit_task_id,
                "proposal_evidence_id": proposal_evidence.id,
                "proposal_fingerprint": proposal["proposal_fingerprint"],
                "validation": validation.to_dict(),
            },
        )
        await self.runtime.store.save_approval_request(approval)
        commit_task = await self.runtime.store.load_task(commit_task_id)
        if commit_task is not None:
            from dataclasses import replace

            await self.runtime.store.save_task(
                replace(
                    commit_task,
                    dependencies=(
                        *commit_task.dependencies,
                        TaskDependency(
                            kind="approval",
                            approval_id=approval.approval_id,
                            approval_status=ApprovalStatus.APPROVED,
                            operation_id=operation.id,
                        ),
                    ),
                )
            )
        await self.runtime.store.append_event(
            RuntimeEvent(
                id=f"monitor-command-event-{operation.id}:approval-requested",
                type=RuntimeEventType.APPROVAL_REQUESTED,
                operation_id=operation.id,
                runtime_id=self.runtime.runtime_id,
                runtime_kind=self.runtime.runtime_kind,
                approval_id=approval.approval_id,
                policy_id=approval.requested_by_policy_id,
                message=f"Approval {approval.approval_id} requested.",
                payload={"approval": approval.to_dict()},
            )
        )
        await self.runtime.kernel.block_operation(
            operation.id,
            message="Monitor create operation is waiting for approval.",
            payload={
                "approval_id": approval.approval_id,
                "proposal_evidence_id": proposal_evidence.id,
            },
        )
        result = DbOperationResult(
            operation_id=operation.id,
            request=request,
            intent=DbIntent(
                kind=DbIntentKind.ADMIN,
                confidence=command.confidence,
                access=AccessMode.WRITE,
                diagnostics={
                    "command_kind": "create",
                    **dict(command.diagnostics),
                },
            ),
            contract=DbOperationContract(
                operation_type="monitor.create",
                required_evidence=("monitor.definition",),
                access=AccessMode.WRITE,
                policy_ids=("runtime.approval_required",),
                metadata={
                    "control_plane": "db.monitor",
                    "approval_required": True,
                    "command": command_payload,
                    "proposal": proposal,
                    "proposal_evidence_id": proposal_evidence.id,
                    "validation": validation.to_dict(),
                },
            ),
            status=OperationStatus.BLOCKED,
            answer="This operation requires approval before creating the monitor.",
            evidence=(proposal_evidence,),
            warnings=("db_monitor_approval_required", *validation.warnings),
            diagnostics={
                "proposal": proposal,
                "validation": validation.to_dict(),
                "approval_id": approval.approval_id,
                "proposal_evidence_id": proposal_evidence.id,
            },
        )
        for item in result.evidence:
            await self.runtime.store.save_evidence(item)
        return await self.runtime._record_operation_result(result, operation=operation)

    async def _list(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        monitors = await self.runtime.list_monitors(status=command.patch.get("status"))
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.SUCCEEDED,
            answer=_list_monitors_answer(monitors),
            evidence_kind="monitor.listing",
            payload={
                "command": _command_payload(command),
                "status": command.patch.get("status"),
                "monitors": [monitor.to_dict() for monitor in monitors],
            },
            diagnostics={"monitors": [monitor.to_dict() for monitor in monitors]},
        )

    async def _inspect(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        if command.patch.get("detail"):
            monitor_id = (
                resolution.monitor.id if resolution.monitor is not None else None
            )
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command, monitor_id=monitor_id),
                status=OperationStatus.SUCCEEDED,
                answer=_monitor_detail_answer(resolution, command=command),
                evidence_kind="monitor.inspection",
                payload={
                    "command": _command_payload(command, monitor_id=monitor_id),
                    "resolution": resolution.to_dict(),
                    "detail": command.patch.get("detail"),
                },
                warnings=resolution.warnings,
                diagnostics={
                    "resolution": resolution.to_dict(),
                    "detail": command.patch.get("detail"),
                },
            )
        assert resolution.monitor is not None
        inspection = await self.runtime.inspect_monitor(resolution.monitor.id)
        if inspection is None:
            return await self._resolution_failure(
                command,
                request,
                DbMonitorResolution(
                    None,
                    resolution.monitor_ref,
                    errors=("monitor_not_found",),
                ),
            )
        evidence_kind = (
            "monitor.run_summary"
            if command.kind == "explain_run"
            else "monitor.inspection"
        )
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.SUCCEEDED,
            answer=_inspect_monitor_answer(inspection, command=command),
            evidence_kind=evidence_kind,
            payload={
                "command": _command_payload(command),
                "resolution": resolution.to_dict(),
                "inspection": inspection.to_dict(),
            },
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "inspection": inspection.to_dict(),
            },
        )

    async def _update(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        updated = await self.runtime.update_monitor(
            resolution.monitor.id, command.patch
        )
        inspection = await self.runtime.inspect_monitor(updated.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=updated.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Updated monitor {updated.name} ({updated.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": updated.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
            },
            persist_operation=False,
        )

    async def _pause(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        paused = await self.runtime.pause_monitor(
            resolution.monitor.id,
            paused_until=command.patch.get("paused_until"),
        )
        inspection = await self.runtime.inspect_monitor(paused.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=paused.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Paused monitor {paused.name} ({paused.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": paused.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
            },
            persist_operation=False,
        )

    async def _resume(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        resumed = await self.runtime.resume_monitor(resolution.monitor.id)
        inspection = await self.runtime.inspect_monitor(resumed.id)
        operation_id = _inspection_operation_id(inspection)
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=resumed.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Resumed monitor {resumed.name} ({resumed.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": resumed.to_dict(),
                "inspection": None if inspection is None else inspection.to_dict(),
            },
            persist_operation=False,
        )

    async def _delete(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        deleted = await self.runtime.delete_monitor(resolution.monitor.id)
        operation_id = await _latest_monitor_operation_id(
            self.runtime,
            "delete",
            deleted.id,
        )
        evidence = await _operation_evidence(self.runtime, operation_id)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=deleted.id),
            status=OperationStatus.SUCCEEDED,
            answer=f"Deleted monitor {deleted.name} ({deleted.id}).",
            operation_id=operation_id,
            evidence=tuple(evidence),
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "monitor": deleted.to_dict(),
            },
            persist_operation=False,
        )

    async def _approve_action(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbOperationResult:
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            return await self._resolution_failure(command, request, resolution)
        assert resolution.monitor is not None
        action = str(command.patch.get("approval_action") or "approve")
        approvals = await self.runtime.list_monitor_approvals(
            monitor_id=resolution.monitor.id,
            pending_only=True,
        )
        if not approvals:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command, monitor_id=resolution.monitor.id),
                status=OperationStatus.FAILED,
                answer=f"Monitor {resolution.monitor.id} has no pending approvals.",
                evidence_kind="monitor.command.approval",
                payload={
                    "command": _command_payload(
                        command, monitor_id=resolution.monitor.id
                    ),
                    "resolution": resolution.to_dict(),
                    "approval_action": action,
                    "approvals": [],
                    "status": "not_found",
                },
                warnings=("db_monitor_approval_not_found",),
                diagnostics={"resolution": resolution.to_dict()},
            )
        if len(approvals) > 1:
            return await self.runtime.record_monitor_command_result(
                request=request,
                kind=command.kind,
                command=_command_payload(command, monitor_id=resolution.monitor.id),
                status=OperationStatus.FAILED,
                answer=(
                    f"Monitor {resolution.monitor.id} has multiple pending approvals; "
                    "use the approval id to choose one."
                ),
                evidence_kind="monitor.command.approval",
                payload={
                    "command": _command_payload(
                        command, monitor_id=resolution.monitor.id
                    ),
                    "resolution": resolution.to_dict(),
                    "approval_action": action,
                    "approvals": [dict(item) for item in approvals],
                    "status": "ambiguous",
                },
                warnings=("db_monitor_approval_ambiguous",),
                diagnostics={
                    "resolution": resolution.to_dict(),
                    "approval_ids": [item["approval_id"] for item in approvals],
                },
            )

        approval_context = dict(approvals[0])
        approval_id = str(approval_context["approval_id"])
        if action == "reject":
            approval = await self.runtime.reject_monitor_approval(approval_id)
        elif action == "cancel":
            approval = await self.runtime.cancel_monitor_approval(approval_id)
        else:
            approval = await self.runtime.approve_monitor_approval(approval_id)

        operation_id = str(approval_context.get("operation_id") or "")
        resumed = None
        if operation_id:
            resumed = await self.runtime.resume_operation(operation_id)
        status = (
            resumed.operation.status
            if resumed is not None
            else OperationStatus.SUCCEEDED
        )
        answer = _approval_action_answer(
            action,
            resolution.monitor.id,
            approval_id,
            operation_status=status.value,
        )
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command, monitor_id=resolution.monitor.id),
            status=status,
            answer=answer,
            evidence_kind="monitor.command.approval",
            payload={
                "command": _command_payload(command, monitor_id=resolution.monitor.id),
                "resolution": resolution.to_dict(),
                "approval_action": action,
                "approval_id": approval_id,
                "approval_status": approval.status.value,
                "operation_id": operation_id or None,
                "operation_status": status.value,
                "approval_context": approval_context.get("context"),
            },
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "approval_id": approval_id,
                "approval_status": approval.status.value,
                "operation_id": operation_id or None,
                "operation_status": status.value,
            },
        )

    async def _resolve(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbMonitorResolution:
        monitors = await self.runtime.list_monitors()
        resolution = self.resolver.resolve(command, monitors)
        if command.monitor_id:
            return resolution
        if resolution.accepted or not _allows_contextual_resolution(command):
            return resolution
        contextual = await self._resolve_from_request_context(
            command,
            request,
            monitors,
        )
        if contextual is not None:
            return contextual
        return resolution

    async def _resolve_from_request_context(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        monitors: tuple[DbMonitor, ...],
    ) -> DbMonitorResolution | None:
        last_monitor_id = request.metadata.get("last_monitor_id")
        if last_monitor_id:
            resolution = self.resolver.resolve(
                replace(command, monitor_id=str(last_monitor_id)),
                monitors,
            )
            if resolution.accepted:
                return replace(
                    resolution,
                    resolution_source="request.metadata.last_monitor_id",
                )

        operation_id = request.metadata.get("last_runtime_operation_id")
        if operation_id:
            resolution = await self._resolve_from_operation(
                str(operation_id),
                monitors,
                source="request.metadata.last_runtime_operation_id",
            )
            if resolution is not None:
                return resolution

        approval_id = request.metadata.get("last_approval_id")
        if approval_id:
            approval_operation_id = await self._monitor_operation_from_approval(
                str(approval_id)
            )
            if approval_operation_id:
                resolution = await self._resolve_from_operation(
                    approval_operation_id,
                    monitors,
                    source="request.metadata.last_approval_id",
                )
                if resolution is not None:
                    return resolution

        if request.session_id:
            resolution = await self._resolve_from_latest_session_operation(
                request.session_id,
                monitors,
            )
            if resolution is not None:
                return resolution
        return None

    async def _resolve_from_latest_session_operation(
        self,
        session_id: str,
        monitors: tuple[DbMonitor, ...],
    ) -> DbMonitorResolution | None:
        for operation in await self._session_monitor_operations(session_id):
            resolution = await self._resolve_from_operation(
                operation.id,
                monitors,
                source="request.session_id",
            )
            if resolution is not None:
                return resolution
        return None

    async def _has_session_monitor_context(self, session_id: str) -> bool:
        operations = await self._session_monitor_operations(session_id)
        return bool(operations)

    async def _session_monitor_operations(
        self,
        session_id: str,
    ) -> tuple[Any, ...]:
        operations = await self.runtime.store.list_operations()
        matches = []
        for operation in reversed(operations):
            if not operation.operation_type.startswith("monitor."):
                continue
            if operation.metadata.get("control_plane") != "db.monitor":
                continue
            operation_session_id = operation.metadata.get(
                "session_id",
                operation.request.get("session_id"),
            )
            if operation_session_id == session_id:
                matches.append(operation)
        return tuple(matches)

    async def _resolve_from_operation(
        self,
        operation_id: str,
        monitors: tuple[DbMonitor, ...],
        *,
        source: str,
    ) -> DbMonitorResolution | None:
        definition = await self._latest_monitor_definition(operation_id)
        if definition is not None:
            monitor = _monitor_from_definition_evidence(definition)
            matches: tuple[DbMonitor, ...] = ()
            if monitor is not None:
                committed = self.resolver.resolve(
                    DbMonitorCommand(kind="inspect", monitor_id=monitor.id),
                    monitors,
                )
                monitor = committed.monitor or monitor
                matches = (monitor,)
            return DbMonitorResolution(
                monitor=monitor,
                monitor_ref=None if monitor is None else monitor.id,
                matches=matches,
                definition_evidence=definition,
                operation_id=operation_id,
                resolution_source=f"{source}.monitor.definition",
            )

        proposal = await self._latest_monitor_proposal(operation_id)
        if proposal is not None:
            monitor_id = proposal.payload.get("monitor_id")
            monitor = None
            matches: tuple[DbMonitor, ...] = ()
            if monitor_id:
                committed = self.resolver.resolve(
                    DbMonitorCommand(kind="inspect", monitor_id=str(monitor_id)),
                    monitors,
                )
                if committed.monitor is not None:
                    monitor = committed.monitor
                    matches = committed.matches
            return DbMonitorResolution(
                monitor=monitor,
                monitor_ref=None if monitor_id is None else str(monitor_id),
                matches=matches,
                proposal_evidence=proposal,
                operation_id=operation_id,
                resolution_source=f"{source}.monitor.proposal",
            )

        operation = await self.runtime.store.load_operation(operation_id)
        monitor_id = None if operation is None else operation.metadata.get("monitor_id")
        if monitor_id:
            resolution = self.resolver.resolve(
                DbMonitorCommand(kind="inspect", monitor_id=str(monitor_id)),
                monitors,
            )
            if resolution.accepted:
                return replace(
                    resolution,
                    operation_id=operation_id,
                    resolution_source=f"{source}.operation.monitor_id",
                )
        return None

    async def _latest_monitor_definition(self, operation_id: str) -> Evidence | None:
        evidence = await self.runtime.store.list_evidence(operation_id)
        for item in reversed(evidence):
            if item.kind == "monitor.definition" and item.accepted:
                return item
        return None

    async def _latest_monitor_proposal(self, operation_id: str) -> Evidence | None:
        evidence = await self.runtime.store.list_evidence(operation_id)
        for item in reversed(evidence):
            if (
                item.kind == "monitor.proposal"
                and item.accepted
                and isinstance(item.payload.get("observation_plan"), dict)
            ):
                return item
        return None

    async def _monitor_operation_from_approval(
        self,
        approval_id: str,
    ) -> str | None:
        for approval in await self.runtime.store.list_approval_requests():
            if approval.approval_id == approval_id:
                return approval.operation_id
        return None

    async def _resolution_failure(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        resolution: DbMonitorResolution,
    ) -> DbOperationResult:
        reason = resolution.errors[0] if resolution.errors else "monitor_not_found"
        answer = _resolution_failure_answer(reason, resolution)
        return await self.runtime.record_monitor_command_result(
            request=request,
            kind=command.kind,
            command=_command_payload(command),
            status=OperationStatus.FAILED,
            answer=answer,
            evidence_kind="monitor.command.resolution",
            payload={
                "command": _command_payload(command),
                "resolution": resolution.to_dict(),
            },
            warnings=(f"db_{reason}",),
            diagnostics={"resolution": resolution.to_dict()},
        )


def _owner_from_request(request: DbRequest) -> dict[str, Any]:
    owner: dict[str, Any] = {}
    if request.user_id is not None:
        owner["user_id"] = request.user_id
    if request.session_id is not None:
        owner["session_id"] = request.session_id
    return owner


def _request_has_monitor_context(request: DbRequest) -> bool:
    return any(
        request.metadata.get(key)
        for key in (
            "last_monitor_id",
            "last_runtime_operation_id",
            "last_approval_id",
        )
    )


def _allows_contextual_resolution(command: DbMonitorCommand) -> bool:
    return command.kind == "inspect" and bool(command.patch.get("detail"))


def _monitor_from_definition_evidence(evidence: Evidence) -> DbMonitor | None:
    monitor = evidence.payload.get("monitor")
    if not isinstance(monitor, dict):
        return None
    return DbMonitor.from_dict(monitor)


def _command_payload(
    command: DbMonitorCommand,
    *,
    monitor_id: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": command.kind,
        "monitor_id": monitor_id or command.monitor_id,
        "patch": command.patch,
        "prompt": command.prompt,
        "confidence": command.confidence,
        "diagnostics": command.diagnostics,
    }


def _inspection_operation_id(inspection: Any) -> str | None:
    if inspection is None or inspection.state is None:
        return None
    return inspection.state.last_operation_id


async def _operation_evidence(
    runtime: Any, operation_id: str | None
) -> tuple[Any, ...]:
    if not operation_id:
        return ()
    return tuple(await runtime.store.list_evidence(operation_id))


async def _latest_monitor_operation_id(
    runtime: Any,
    action: str,
    monitor_id: str,
) -> str | None:
    operations = await runtime.store.list_operations()
    for operation in reversed(operations):
        if (
            operation.operation_type == f"monitor.{action}"
            and operation.metadata.get("monitor_id") == monitor_id
        ):
            return operation.id
    return None
