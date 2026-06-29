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
    RuntimeEventType,
    TaskDependency,
    TaskStatus,
)

from ..models import (
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from ..monitors import DbMonitor
from ..runtime.tasks import DbTaskSpec
from ..session_context import db_session_context_from_request
from .answers import (
    _approval_action_answer,
    _create_monitor_answer,
    _inspect_monitor_answer,
    _list_monitors_answer,
    _monitor_detail_answer,
    _monitor_validation_answer,
    _resolution_failure_answer,
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
            if command.kind == "approve_action":
                return await self._approve_action(command, request)
            command = await self._resolved_loop_command(command, request)
            loop_result = await self.runtime.run(_loop_request(command, request))
            pending_approval = await self._maybe_create_loop_approval(
                command,
                request,
                loop_result,
            )
            if pending_approval is not None:
                return pending_approval
            projected = _project_monitor_loop_result(command, loop_result)
            if (
                projected.status is OperationStatus.BLOCKED
                and "db_monitor_validation_failed" in projected.warnings
            ):
                await self.runtime.kernel.block_operation(
                    projected.operation_id,
                    message="Monitor proposal is incomplete or unsupported.",
                    payload={"status": "blocked"},
                )
            return projected
        except ValueError as exc:
            warning = (
                "db_monitor_reference_required"
                if str(exc) == "Please specify which monitor to manage."
                else "db_monitor_command_failed"
            )
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
                warnings=(warning,),
            )

    async def _resolved_loop_command(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
    ) -> DbMonitorCommand:
        if command.kind in {"create", "list"}:
            return command
        resolution = await self._resolve(command, request)
        if not resolution.accepted:
            reason = resolution.errors[0] if resolution.errors else "monitor_not_found"
            raise ValueError(_resolution_failure_answer(reason, resolution))
        diagnostics = {**command.diagnostics, "resolution": resolution.to_dict()}
        if resolution.proposal_evidence is not None:
            diagnostics["proposal_evidence"] = resolution.proposal_evidence.to_dict()
        if resolution.definition_evidence is not None:
            diagnostics["definition_evidence"] = (
                resolution.definition_evidence.to_dict()
            )
        return replace(
            command,
            monitor_id=resolution.monitor_ref or command.monitor_id,
            diagnostics=diagnostics,
        )

    async def _maybe_create_loop_approval(
        self,
        command: DbMonitorCommand,
        request: DbRequest,
        result: DbOperationResult,
    ) -> DbOperationResult | None:
        if command.kind != "create" or not command.diagnostics.get("approval_required"):
            return None
        evidence = await self.runtime.store.list_evidence(result.operation_id)
        proposal_evidence = next(
            (item for item in evidence if item.kind == "monitor.proposal"),
            None,
        )
        if proposal_evidence is None or not proposal_evidence.accepted:
            return None
        proposal = dict(proposal_evidence.payload)
        tasks = await self.runtime.store.list_tasks(result.operation_id)
        commit_task = next(
            (
                task
                for task in reversed(tasks)
                if task.capability_id == "db.monitor.commit_create"
            ),
            None,
        )
        if commit_task is None:
            operation = await self.runtime.store.load_operation(result.operation_id)
            if operation is None:
                return None
            commit_task = self.runtime.materialize_task_specs(
                operation,
                (
                    DbTaskSpec(
                        capability_id="db.monitor.commit_create",
                        owner="db_runtime",
                        input={
                            "proposal_evidence_id": proposal_evidence.id,
                            "proposal_fingerprint": proposal["proposal_fingerprint"],
                        },
                        reason="monitor_create_commit",
                        sequence=2,
                        metadata={
                            "idempotency_key": proposal["proposal_fingerprint"],
                        },
                    ),
                ),
            )[0]
            commit_task = await self.runtime._plan_kernel_task(commit_task)
        validation = DbMonitorValidation.from_dict(
            dict(proposal.get("validation") or {})
        )
        return await self._create_pending_approval(
            command,
            request,
            proposal,
            proposal_evidence,
            commit_task.id,
            validation,
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
        operation = replace(
            operation,
            metadata={
                **operation.metadata,
                "control_plane": "db.monitor",
                "command_kind": command.kind,
                "monitor_id": proposal["monitor_id"],
                "monitor_name": proposal["name"],
                "proposal_fingerprint": proposal["proposal_fingerprint"],
                "proposal_evidence_id": proposal_evidence.id,
            },
        )
        await self.runtime.store.save_operation(operation)
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
            retained_dependencies = tuple(
                dependency
                for dependency in commit_task.dependencies
                if not (
                    dependency.kind.value == "approval"
                    and dependency.approval_id is None
                    and dependency.approval_policy_id == "runtime.approval_required"
                )
            )
            await self.runtime.store.save_task(
                replace(
                    commit_task,
                    status=TaskStatus.PENDING,
                    dependencies=(
                        *retained_dependencies,
                        TaskDependency(
                            kind="approval",
                            approval_id=approval.approval_id,
                            approval_status=ApprovalStatus.APPROVED,
                            operation_id=operation.id,
                        ),
                    ),
                )
            )
        await self.runtime.kernel.append_event(
            RuntimeEventType.APPROVAL_REQUESTED,
            operation_id=operation.id,
            approval_id=approval.approval_id,
            policy_id=approval.requested_by_policy_id,
            message=f"Approval {approval.approval_id} requested.",
            payload={"approval": approval.to_dict()},
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
        status = OperationStatus.SUCCEEDED
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
                "operation_status": "approval_state_changed",
                "approval_context": approval_context.get("context"),
            },
            warnings=resolution.warnings,
            diagnostics={
                "resolution": resolution.to_dict(),
                "approval_id": approval_id,
                "approval_status": approval.status.value,
                "operation_id": operation_id or None,
                "operation_status": "approval_state_changed",
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
        session_context = db_session_context_from_request(request)
        if session_context is not None:
            for monitor_id in session_context.referents.monitors:
                resolution = self.resolver.resolve(
                    replace(command, monitor_id=str(monitor_id)),
                    monitors,
                )
                if resolution.accepted:
                    return replace(
                        resolution,
                        resolution_source="db_session_context.referents.monitors",
                    )
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


def _loop_request(command: DbMonitorCommand, request: DbRequest) -> DbRequest:
    metadata = {
        **request.metadata,
        "db_monitor_command": _command_payload(command),
    }
    return replace(
        request,
        mode=_operation_type_for_command(command),
        metadata=metadata,
    )


def _operation_type_for_command(command: DbMonitorCommand) -> str:
    if command.kind == "create":
        return "monitor.create"
    if command.kind == "list":
        return "monitor.list"
    if command.kind == "inspect":
        return "monitor.inspect"
    if command.kind == "explain_run":
        return "monitor.explain_run"
    return f"monitor.{command.kind}"


def _project_monitor_loop_result(
    command: DbMonitorCommand,
    result: DbOperationResult,
) -> DbOperationResult:
    evidence = tuple(result.evidence)
    diagnostics = dict(result.diagnostics)
    if command.patch.get("detail") and (
        command.diagnostics.get("proposal_evidence")
        or command.diagnostics.get("definition_evidence")
    ):
        proposal_evidence = (
            Evidence.from_dict(command.diagnostics["proposal_evidence"])
            if command.diagnostics.get("proposal_evidence")
            else None
        )
        definition_evidence = (
            Evidence.from_dict(command.diagnostics["definition_evidence"])
            if command.diagnostics.get("definition_evidence")
            else None
        )
        resolution = DbMonitorResolution(
            monitor=None,
            monitor_ref=command.monitor_id,
            proposal_evidence=proposal_evidence,
            definition_evidence=definition_evidence,
            operation_id=dict(command.diagnostics.get("resolution") or {}).get(
                "operation_id"
            ),
            resolution_source=dict(command.diagnostics.get("resolution") or {}).get(
                "resolution_source"
            ),
        )
        return replace(
            result,
            answer=_monitor_detail_answer(resolution, command=command),
            diagnostics={**diagnostics, "resolution": resolution.to_dict()},
        )
    proposal = _latest_evidence_by_kind(evidence, "monitor.proposal")
    if proposal is not None:
        validation = DbMonitorValidation.from_dict(
            dict(proposal.payload.get("validation") or {})
        )
        diagnostics.setdefault("proposal", proposal.payload)
        diagnostics.setdefault("validation", validation.to_dict())
        if not validation.accepted:
            return replace(
                result,
                status=OperationStatus.BLOCKED,
                answer=_monitor_validation_answer(validation),
                warnings=("db_monitor_validation_failed", *validation.warnings),
                diagnostics=diagnostics,
            )
    if command.kind == "create":
        definition = _latest_evidence_by_kind(evidence, "monitor.definition")
        if definition is not None and isinstance(
            definition.payload.get("monitor"), dict
        ):
            monitor = DbMonitor.from_dict(dict(definition.payload["monitor"]))
            diagnostics.setdefault("monitor", monitor.to_dict())
            return replace(
                result,
                answer=_create_monitor_answer(monitor),
                diagnostics=diagnostics,
            )
    if command.kind in {"update", "pause", "resume", "delete"}:
        committed = _latest_lifecycle_evidence(evidence)
        if committed is not None:
            monitor_payload = (
                committed.payload.get("monitor")
                or committed.payload.get("after")
                or committed.payload.get("before")
            )
            monitor_name = command.monitor_id or str(
                committed.payload.get("monitor_id")
            )
            monitor_id = monitor_name
            if isinstance(monitor_payload, dict):
                monitor = DbMonitor.from_dict(monitor_payload)
                monitor_name = monitor.name
                monitor_id = monitor.id
                diagnostics.setdefault("monitor", monitor.to_dict())
            verb = {
                "update": "Updated",
                "pause": "Paused",
                "resume": "Resumed",
                "delete": "Deleted",
            }[command.kind]
            return replace(
                result,
                answer=f"{verb} monitor {monitor_name} ({monitor_id}).",
                diagnostics=diagnostics,
            )
    snapshot = _latest_evidence_by_kind(evidence, "monitor.snapshot")
    if snapshot is not None:
        payload = dict(snapshot.payload)
        diagnostics.setdefault("snapshot", payload)
        if command.kind == "list":
            monitors = tuple(
                DbMonitor.from_dict(dict(item))
                for item in payload.get("monitors") or ()
                if isinstance(item, dict)
            )
            return replace(
                result,
                answer=_list_monitors_answer(monitors),
                diagnostics={
                    **diagnostics,
                    "monitors": [m.to_dict() for m in monitors],
                },
            )
        inspections = payload.get("inspections") or ()
        if inspections and isinstance(inspections[0], dict):
            from ..monitors import DbMonitorInspection, DbMonitorRun, DbMonitorState

            inspection_payload = dict(inspections[0])
            inspection = DbMonitorInspection(
                monitor=DbMonitor.from_dict(dict(inspection_payload["monitor"])),
                state=(
                    None
                    if inspection_payload.get("state") is None
                    else DbMonitorState.from_dict(dict(inspection_payload["state"]))
                ),
                runs=tuple(
                    DbMonitorRun.from_dict(dict(item))
                    for item in inspection_payload.get("runs") or ()
                    if isinstance(item, dict)
                ),
            )
            if command.patch.get("detail"):
                resolution = DbMonitorResolution(
                    monitor=inspection.monitor,
                    monitor_ref=inspection.monitor.id,
                    matches=(inspection.monitor,),
                    operation_id=dict(command.diagnostics.get("resolution") or {}).get(
                        "operation_id"
                    ),
                    resolution_source=dict(
                        command.diagnostics.get("resolution") or {}
                    ).get("resolution_source"),
                )
                return replace(
                    result,
                    answer=_monitor_detail_answer(resolution, command=command),
                    diagnostics={
                        **diagnostics,
                        "inspection": inspection.to_dict(),
                        "resolution": resolution.to_dict(),
                    },
                )
            return replace(
                result,
                answer=_inspect_monitor_answer(inspection, command=command),
                diagnostics={**diagnostics, "inspection": inspection.to_dict()},
            )
    return result


def _latest_evidence_by_kind(
    evidence: tuple[Evidence, ...],
    kind: str,
) -> Evidence | None:
    for item in reversed(evidence):
        if item.kind == kind:
            return item
    return None


def _latest_lifecycle_evidence(evidence: tuple[Evidence, ...]) -> Evidence | None:
    for kind in (
        "monitor.state_update",
        "monitor.paused",
        "monitor.resumed",
        "monitor.deleted",
        "monitor.disabled",
    ):
        item = _latest_evidence_by_kind(evidence, kind)
        if item is not None:
            return item
    return None


def _request_has_monitor_context(request: DbRequest) -> bool:
    if any(
        request.metadata.get(key)
        for key in (
            "last_monitor_id",
            "last_runtime_operation_id",
            "last_approval_id",
        )
    ):
        return True
    session_context = db_session_context_from_request(request)
    return bool(
        session_context
        and (
            session_context.referents.monitors
            or session_context.durable_ids.get("last_monitor_id")
            or session_context.durable_ids.get("monitor_id")
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
