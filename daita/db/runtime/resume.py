"""Operation inspection and resume helpers for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, TYPE_CHECKING

from daita.runtime import (
    AccessMode,
    Evidence,
    Operation,
    OperationSnapshot,
    OperationStatus,
    RuntimeEventType,
    RuntimeKernel,
    RuntimeStore,
    Task,
    TaskStatus,
)

from ..evidence import evidence_in_task_plan_order
from ..loop import DbAgentLoop, DbLoopResult
from ..models import (
    DbIntent,
    DbIntentKind,
    DbLimits,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from ..planner_protocol import DbAgentPlanner
from .tasks.runtime import DbTaskRuntime
from .types import DbRuntimeGovernanceBlocked, DbRuntimeTaskNotRunnable


class DbRuntimeResumeMixin:
    if TYPE_CHECKING:
        store: RuntimeStore
        kernel: RuntimeKernel
        tasks: DbTaskRuntime
        runtime_id: str
        _is_setup: bool

        async def setup(self, *, agent_id: str | None = None) -> None: ...

        def _analysis_progress_payload(
            self,
            snapshot: OperationSnapshot | None,
            *,
            plan_evidence: Evidence | None = None,
        ) -> dict[str, Any]: ...

        async def _finalize_resumed_monitor_action(
            self,
            snapshot: OperationSnapshot,
        ) -> None: ...

        async def _finalize_resumed_monitor_delivery(
            self,
            snapshot: OperationSnapshot,
        ) -> None: ...

        def _has_pending_approvals(self, snapshot: OperationSnapshot) -> bool: ...

        async def execute_task(
            self,
            task: Task,
            operation: Operation,
            context: dict[str, Any] | None = None,
        ) -> tuple[Evidence, ...]: ...

        async def _run_multi_step_analysis(
            self,
            request: DbRequest,
            intent: DbIntent,
            contract: DbOperationContract,
            operation: Operation,
            *,
            base_diagnostics: dict[str, Any],
            reuse_existing_plan: bool = False,
        ) -> DbOperationResult: ...

        async def _try_finalize_run_operation_from_snapshot(
            self,
            snapshot: OperationSnapshot,
            *,
            request: DbRequest,
            fallback_intent: DbIntent,
            fallback_contract: DbOperationContract,
            base_diagnostics: dict[str, Any] | None = None,
        ) -> OperationSnapshot | None: ...

        def _select_db_agent_planner(self) -> DbAgentPlanner | None: ...

        async def _finalize_run_operation(
            self,
            *,
            operation_id: str,
            request: DbRequest,
            fallback_intent: DbIntent,
            fallback_contract: DbOperationContract,
            loop_result: DbLoopResult | None = None,
            base_diagnostics: dict[str, Any] | None = None,
        ) -> DbOperationResult: ...

    async def inspect_operation(self, operation_id: str) -> OperationSnapshot | None:
        """Inspect persisted state for one operation."""
        inspect = getattr(self.store, "inspect_operation", None)
        if inspect is not None:
            snapshot = await inspect(operation_id)
            if snapshot is None:
                return None
            return replace(
                snapshot,
                evidence=evidence_in_task_plan_order(
                    snapshot.evidence,
                    snapshot.tasks,
                ),
            )
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            return None
        tasks = tuple(await self.store.list_tasks(operation_id))
        completed_task_ids = tuple(
            task.id
            for task in tasks
            if task.status
            in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.SKIPPED,
            }
        )
        resumable_task_ids = tuple(
            task.id
            for task in tasks
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
        )
        return OperationSnapshot(
            operation=operation,
            tasks=tasks,
            evidence=evidence_in_task_plan_order(
                await self.store.list_evidence(operation_id),
                tasks,
            ),
            events=tuple(await self.store.list_events(operation_id)),
            policy_decisions=tuple(
                await self.store.list_policy_decisions(operation_id)
            ),
            governance_audit_records=tuple(
                await self.store.list_governance_audit_records(operation_id)
            ),
            approval_requests=tuple(
                await self.store.list_approval_requests(operation_id)
            ),
            resumable_task_ids=resumable_task_ids,
            completed_task_ids=completed_task_ids,
        )

    async def inspect_analysis_operation(self, operation_id: str) -> dict[str, Any]:
        """Inspect persisted multi-step analysis progress for one operation."""
        snapshot = await self.inspect_operation(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        return self._analysis_progress_payload(snapshot)

    async def resume_operation(self, operation_id: str) -> OperationSnapshot:
        """Resume persisted operation state without re-running completed tasks."""
        if not self._is_setup:
            await self.setup()
        snapshot = await self.inspect_operation(operation_id)
        if snapshot is None:
            raise KeyError(operation_id)
        await self.kernel.append_event(
            RuntimeEventType.OPERATION_RESUMED,
            operation_id=operation_id,
            message=f"Operation {operation_id} resume requested.",
            payload={
                "completed_task_ids": list(snapshot.completed_task_ids),
                "resumable_task_ids": list(snapshot.resumable_task_ids),
            },
        )
        for task in snapshot.tasks:
            if task.id in snapshot.completed_task_ids:
                await self.kernel.append_event(
                    RuntimeEventType.TASK_SKIPPED,
                    operation_id=operation_id,
                    task=task,
                    capability=self.tasks.capability_for_task(task),
                    message=f"Task {task.id} already completed; not re-running.",
                )

        terminal_operation = await self.kernel.apply_terminal_approval_state(
            operation_id
        )
        if terminal_operation is not None:
            terminal_snapshot = await self.inspect_operation(operation_id)
            if terminal_snapshot is None:
                raise KeyError(operation_id)
            if _monitor_action_context(terminal_snapshot.operation):
                await self._finalize_resumed_monitor_action(terminal_snapshot)
                terminal_snapshot = await self.inspect_operation(operation_id)
                if terminal_snapshot is None:
                    raise KeyError(operation_id)
            if _monitor_delivery_context(terminal_snapshot.operation):
                await self._finalize_resumed_monitor_delivery(terminal_snapshot)
                terminal_snapshot = await self.inspect_operation(operation_id)
                if terminal_snapshot is None:
                    raise KeyError(operation_id)
            return terminal_snapshot

        refreshed = await self.inspect_operation(operation_id)
        if refreshed is None:
            raise KeyError(operation_id)
        snapshot = refreshed

        if self._has_pending_approvals(snapshot):
            await self.kernel.block_operation(operation_id)
            resumed = await self.inspect_operation(operation_id)
            if resumed is None:
                raise KeyError(operation_id)
            return resumed

        recovered_tasks = await self.kernel.recover_expired_task_claims(operation_id)
        if recovered_tasks:
            recovered = await self.inspect_operation(operation_id)
            if recovered is None:
                raise KeyError(operation_id)
            snapshot = recovered

        resumable_tasks = tuple(
            task
            for task in _tasks_in_resume_order(snapshot.tasks)
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
            and not task.metadata.get("manual_recovery_required")
        )
        operation = snapshot.operation
        if resumable_tasks:
            operation = await self.kernel.update_operation(
                operation_id,
                OperationStatus.RUNNING,
                message=f"Operation {operation_id} resumed execution.",
            )
        for task in resumable_tasks:
            try:
                await self.execute_task(task, operation)
            except DbRuntimeGovernanceBlocked:
                resumed = await self.inspect_operation(operation_id)
                if resumed is None:
                    raise KeyError(operation_id)
                return resumed
            except DbRuntimeTaskNotRunnable:
                resumed = await self.inspect_operation(operation_id)
                if resumed is None:
                    raise KeyError(operation_id)
                return resumed
            except Exception as exc:
                if _monitor_action_context(operation) and str(exc).startswith(
                    "monitor_write_"
                ):
                    await self.kernel.block_operation(
                        operation.id,
                        message=f"Monitor write execution blocked: {exc}.",
                    )
                    resumed = await self.inspect_operation(operation_id)
                    if resumed is None:
                        raise KeyError(operation_id)
                    await self._finalize_resumed_monitor_action(resumed)
                    finalized = await self.inspect_operation(operation_id)
                    if finalized is None:
                        raise KeyError(operation_id)
                    return finalized
                await self.kernel.fail_operation_if_active(operation.id, exc)
                raise

        completed = await self.inspect_operation(operation_id)
        if completed is None:
            raise KeyError(operation_id)
        if _has_running_tasks(completed.tasks):
            return completed
        if completed.resumable_task_ids:
            return completed
        if _monitor_action_context(completed.operation):
            await self._finalize_resumed_monitor_action(completed)
            action_finalized = await self.inspect_operation(operation_id)
            if action_finalized is None:
                raise KeyError(operation_id)
            if _monitor_delivery_context(action_finalized.operation):
                await self._finalize_resumed_monitor_delivery(action_finalized)
                delivery_finalized = await self.inspect_operation(operation_id)
                if delivery_finalized is None:
                    raise KeyError(operation_id)
                return delivery_finalized
            return action_finalized
        if _monitor_delivery_context(completed.operation):
            await self._finalize_resumed_monitor_delivery(completed)
            finalized = await self.inspect_operation(operation_id)
            if finalized is None:
                raise KeyError(operation_id)
            return finalized
        if completed.operation.status is OperationStatus.SUCCEEDED:
            return completed

        if _snapshot_has_incomplete_analysis(completed):
            request = _db_request_from_context(completed.operation)
            intent = _db_intent_from_context(completed.operation)
            contract = _db_contract_from_context(completed.operation)
            operation = await self.kernel.update_operation(
                operation_id,
                OperationStatus.RUNNING,
                message=f"Operation {operation_id} resumed analysis materialization.",
            )
            await self._run_multi_step_analysis(
                request,
                intent,
                contract,
                operation,
                base_diagnostics={
                    "runtime_id": self.runtime_id,
                    "resume": {
                        "operation_id": operation_id,
                        "completed_task_ids": list(completed.completed_task_ids),
                    },
                },
                reuse_existing_plan=True,
            )
            resumed_analysis = await self.inspect_operation(operation_id)
            if resumed_analysis is None:
                raise KeyError(operation_id)
            if _monitor_action_context(resumed_analysis.operation):
                await self._finalize_resumed_monitor_action(resumed_analysis)
                finalized = await self.inspect_operation(operation_id)
                if finalized is None:
                    raise KeyError(operation_id)
                if _monitor_delivery_context(finalized.operation):
                    await self._finalize_resumed_monitor_delivery(finalized)
                    delivery_finalized = await self.inspect_operation(operation_id)
                    if delivery_finalized is None:
                        raise KeyError(operation_id)
                    return delivery_finalized
                return finalized
            return resumed_analysis

        if _monitor_create_context(completed.operation):
            definition = _latest_monitor_definition_from_snapshot(completed)
            if definition is None:
                await self.kernel.complete_operation(
                    operation_id,
                    status=OperationStatus.FAILED,
                    message=(
                        f"Operation {operation_id} failed because monitor creation "
                        "did not produce a committed monitor definition."
                    ),
                    payload={"reason": "missing_monitor_definition"},
                )
                failed = await self.inspect_operation(operation_id)
                if failed is None:
                    raise KeyError(operation_id)
                return failed
            payload = {}
            monitor = definition.payload.get("monitor")
            if isinstance(monitor, dict):
                payload["monitor_id"] = monitor.get("id")
            await self.kernel.complete_operation(
                operation_id,
                message=f"Operation {operation_id} succeeded after monitor commit.",
                payload=payload,
            )
            resumed = await self.inspect_operation(operation_id)
            if resumed is None:
                raise KeyError(operation_id)
            return resumed

        if _monitor_lifecycle_context(completed.operation):
            await self.kernel.complete_operation(
                operation_id,
                message=f"Operation {operation_id} succeeded after monitor lifecycle commit.",
                payload={
                    "monitor_id": completed.operation.metadata.get("monitor_id"),
                    "action": completed.operation.metadata.get("command_kind"),
                },
            )
            resumed = await self.inspect_operation(operation_id)
            if resumed is None:
                raise KeyError(operation_id)
            return resumed

        if _operation_has_run_context(completed.operation):
            return await self._resume_run_operation_through_agent_loop(completed)
        elif (
            completed.tasks
            and completed.operation.status is not OperationStatus.SUCCEEDED
        ):
            await self.kernel.complete_operation(
                operation_id,
                message=f"Operation {operation_id} succeeded after resume.",
            )
        resumed = await self.inspect_operation(operation_id)
        if resumed is None:
            raise KeyError(operation_id)
        return resumed

    async def _resume_run_operation_through_agent_loop(
        self,
        snapshot: OperationSnapshot,
    ) -> OperationSnapshot:
        operation_id = snapshot.operation.id
        base_diagnostics = {
            "runtime_id": self.runtime_id,
            "resume": {
                "operation_id": operation_id,
                "completed_task_ids": list(snapshot.completed_task_ids),
            },
        }
        request = _db_request_from_context(snapshot.operation)
        intent = _db_intent_from_context(snapshot.operation)
        contract = _db_contract_from_context(snapshot.operation)
        finalized = await self._try_finalize_run_operation_from_snapshot(
            snapshot,
            request=request,
            fallback_intent=intent,
            fallback_contract=contract,
            base_diagnostics=base_diagnostics,
        )
        if finalized is not None:
            return finalized

        refreshed = await self.inspect_operation(operation_id)
        if refreshed is None:
            raise KeyError(operation_id)
        if self._has_pending_approvals(refreshed):
            await self.kernel.block_operation(operation_id)
            blocked = await self.inspect_operation(operation_id)
            if blocked is None:
                raise KeyError(operation_id)
            return blocked
        if refreshed.resumable_task_ids:
            return refreshed
        snapshot = refreshed

        planner = self._select_db_agent_planner()
        if planner is None:
            await self.kernel.block_operation(
                operation_id,
                message=(
                    f"Operation {operation_id} requires semantic DB planning "
                    "before resume can continue."
                ),
                payload={
                    "warnings": ["db_runtime_llm_configuration_required"],
                    "configuration_required": True,
                },
            )
            blocked = await self.inspect_operation(operation_id)
            if blocked is None:
                raise KeyError(operation_id)
            return blocked

        operation = await self.kernel.update_operation(
            operation_id,
            OperationStatus.RUNNING,
            message=f"Operation {operation_id} resumed DB agent loop.",
        )
        safety_frame = operation.metadata.get("safety_frame")
        loop_result = await DbAgentLoop(self, planner).run(
            operation,
            safety_frame=safety_frame if isinstance(safety_frame, dict) else None,
        )
        if loop_result.status == "finished":
            await self._finalize_run_operation(
                operation_id=operation_id,
                request=request,
                fallback_intent=intent,
                fallback_contract=contract,
                loop_result=loop_result,
                base_diagnostics=base_diagnostics,
            )
        else:
            payload = {
                "loop_status": loop_result.status,
                "warnings": list(loop_result.warnings),
            }
            if loop_result.status in {
                "blocked",
                "configuration_required",
                "clarification_required",
            }:
                await self.kernel.block_operation(
                    operation_id,
                    message=(
                        f"Operation {operation_id} blocked after DB agent loop resume."
                    ),
                    payload=payload,
                )
            elif loop_result.status == "budget_exhausted":
                await self.kernel.block_operation(
                    operation_id,
                    message=(
                        f"Operation {operation_id} exhausted planner turns after resume."
                    ),
                    payload=payload,
                )
            else:
                await self.kernel.complete_operation(
                    operation_id,
                    status=OperationStatus.FAILED,
                    message=(
                        f"Operation {operation_id} failed after DB agent loop resume."
                    ),
                    payload=payload,
                )
        resumed = await self.inspect_operation(operation_id)
        if resumed is None:
            raise KeyError(operation_id)
        return resumed


def _tasks_in_resume_order(tasks: tuple[Task, ...]) -> tuple[Task, ...]:
    return tuple(
        sorted(
            tasks,
            key=lambda task: (
                int(task.metadata.get("sequence") or 0),
                task.id,
            ),
        )
    )


def _operation_has_run_context(operation: Operation) -> bool:
    return operation.operation_type == "db.run" and isinstance(
        operation.metadata.get("resume_context"), dict
    )


def _monitor_create_context(operation: Operation) -> bool:
    return (
        operation.operation_type == "monitor.create"
        and operation.metadata.get("control_plane") == "db.monitor"
    )


def _monitor_lifecycle_context(operation: Operation) -> bool:
    return (
        operation.operation_type
        in {
            "monitor.update",
            "monitor.pause",
            "monitor.resume",
            "monitor.delete",
            "monitor.disable",
        }
        and operation.metadata.get("control_plane") == "db.monitor"
    )


def _latest_monitor_definition_from_snapshot(
    snapshot: OperationSnapshot,
) -> Evidence | None:
    definitions = [
        evidence
        for evidence in snapshot.evidence
        if evidence.kind == "monitor.definition" and evidence.accepted
    ]
    return definitions[-1] if definitions else None


def _snapshot_has_incomplete_analysis(snapshot: OperationSnapshot) -> bool:
    has_plan = any(
        evidence.kind == "analysis.plan" and evidence.accepted
        for evidence in snapshot.evidence
    )
    has_validation = any(
        evidence.kind == "analysis.plan.validation"
        and evidence.accepted
        and evidence.payload.get("valid") is True
        for evidence in snapshot.evidence
    )
    has_synthesis = any(
        evidence.kind == "analysis.synthesis"
        and evidence.accepted
        and not _analysis_synthesis_is_partial(evidence)
        for evidence in snapshot.evidence
    )
    return has_plan and has_validation and not has_synthesis


def _analysis_synthesis_is_partial(evidence: Evidence | None) -> bool:
    if evidence is None:
        return False
    return bool(evidence.payload.get("partial") or evidence.metadata.get("partial"))


def _latest_final_analysis_synthesis_from_snapshot(
    snapshot: OperationSnapshot,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in snapshot.evidence
        if evidence.kind == "analysis.synthesis"
        and evidence.accepted
        and not _analysis_synthesis_is_partial(evidence)
    ]
    return matches[-1] if matches else None


def _has_running_tasks(tasks: tuple[Task, ...]) -> bool:
    return any(task.status is TaskStatus.RUNNING for task in tasks)


def _monitor_action_context(operation: Operation) -> dict[str, Any]:
    context = operation.metadata.get("monitor_action_context")
    return context if isinstance(context, dict) else {}


def _monitor_delivery_context(operation: Operation) -> dict[str, Any]:
    context = operation.metadata.get("monitor_delivery_context")
    return context if isinstance(context, dict) else {}


def _resume_context(operation: Operation) -> dict[str, Any]:
    context = operation.metadata.get("resume_context")
    return context if isinstance(context, dict) else {}


def _db_request_from_context(operation: Operation) -> DbRequest:
    context = _resume_context(operation).get("request") or operation.request
    return DbRequest(
        prompt=str(context.get("prompt") or ""),
        user_id=context.get("user_id"),
        session_id=context.get("session_id"),
        source_scope=tuple(context.get("source_scope") or ()),
        mode=context.get("mode"),
        requested_capabilities=tuple(context.get("requested_capabilities") or ()),
        constraints=dict(context.get("constraints") or {}),
        metadata=dict(context.get("metadata") or {}),
        session_context=(
            dict(context.get("session_context"))
            if isinstance(context.get("session_context"), dict)
            else None
        ),
    )


def _db_intent_from_context(operation: Operation) -> DbIntent:
    context = _resume_context(operation).get("intent") or {}
    return DbIntent(
        kind=DbIntentKind(str(context.get("kind") or operation.operation_type)),
        confidence=float(context.get("confidence", 1.0)),
        access=AccessMode(str(context.get("access") or AccessMode.NONE.value)),
        evidence_mode=str(context.get("evidence_mode") or "none"),
        requested_outputs=tuple(context.get("requested_outputs") or ()),
        constraints=dict(context.get("constraints") or {}),
        diagnostics=dict(context.get("diagnostics") or {}),
    )


def _db_contract_from_context(operation: Operation) -> DbOperationContract:
    context = _resume_context(operation).get("contract") or {}
    limits = dict(context.get("limits") or {})
    return DbOperationContract(
        operation_type=str(context.get("operation_type") or operation.operation_type),
        required_capabilities=tuple(context.get("required_capabilities") or ()),
        required_evidence=tuple(
            context.get("required_evidence") or sorted(operation.required_evidence)
        ),
        access=AccessMode(str(context.get("access") or AccessMode.NONE.value)),
        limits=DbLimits(**limits) if limits else DbLimits(),
        policy_ids=tuple(context.get("policy_ids") or ()),
        metadata=dict(context.get("metadata") or {}),
    )


def _db_request_context(request: DbRequest) -> dict[str, Any]:
    return {
        "prompt": request.prompt,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "source_scope": list(request.source_scope),
        "mode": request.mode,
        "requested_capabilities": list(request.requested_capabilities),
        "constraints": request.constraints,
        "metadata": request.metadata,
        "session_context": request.session_context,
    }


def _db_intent_context(intent: DbIntent) -> dict[str, Any]:
    return {
        "kind": intent.kind.value,
        "confidence": intent.confidence,
        "access": intent.access.value,
        "evidence_mode": intent.evidence_mode,
        "requested_outputs": list(intent.requested_outputs),
        "constraints": intent.constraints,
        "diagnostics": intent.diagnostics,
    }


def _db_contract_context(contract: DbOperationContract) -> dict[str, Any]:
    return {
        "operation_type": contract.operation_type,
        "required_capabilities": list(contract.required_capabilities),
        "required_evidence": list(contract.required_evidence),
        "access": contract.access.value,
        "limits": contract.limits.to_dict(),
        "policy_ids": list(contract.policy_ids),
        "metadata": contract.metadata,
    }


def _answer_from_synthesis_evidence(evidence: Evidence) -> str:
    answer = evidence.payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("accepted answer.synthesis evidence is missing answer")
    return answer
