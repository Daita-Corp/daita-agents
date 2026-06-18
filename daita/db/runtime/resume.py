"""Operation inspection and resume helpers for ``DbRuntime``."""

from __future__ import annotations

from typing import Any

from daita.runtime import (
    AccessMode,
    Evidence,
    Operation,
    OperationSnapshot,
    OperationStatus,
    RuntimeEventType,
    Task,
    TaskStatus,
)

from ..models import (
    DbIntent,
    DbIntentKind,
    DbLimits,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from .types import DbRuntimeGovernanceBlocked, DbRuntimeTaskNotRunnable


class DbRuntimeResumeMixin:
    async def inspect_operation(self, operation_id: str) -> OperationSnapshot | None:
        """Inspect persisted state for one operation."""
        inspect = getattr(self.store, "inspect_operation", None)
        if inspect is not None:
            return await inspect(operation_id)
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
            evidence=tuple(await self.store.list_evidence(operation_id)),
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
                    capability=self._capability_for_task(task),
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
            await self._complete_resumed_run_operation(completed)
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

    async def _complete_resumed_run_operation(
        self,
        snapshot: OperationSnapshot,
    ) -> None:
        request = _db_request_from_context(snapshot.operation)
        intent = _db_intent_from_context(snapshot.operation)
        contract = _db_contract_from_context(snapshot.operation)
        evidence = tuple(await self.store.list_evidence(snapshot.operation.id))
        tasks = tuple(await self.store.list_tasks(snapshot.operation.id))
        verification = self.verifier.verify(contract, intent, evidence, tasks)
        if not verification.passed:
            await self._record_operation_result(
                DbOperationResult(
                    operation_id=snapshot.operation.id,
                    request=request,
                    intent=intent,
                    contract=contract,
                    status=OperationStatus.FAILED,
                    answer="DB operation could not be verified against required evidence.",
                    evidence=evidence,
                    warnings=verification.warnings,
                    diagnostics={"verification": verification.to_dict()},
                ),
                operation=snapshot.operation,
            )
            return

        verification_evidence = await self._persist_verification_result_evidence(
            snapshot.operation,
            verification,
            evidence,
        )
        synthesis_evidence, synthesis_task = await self._execute_answer_synthesis(
            operation=snapshot.operation,
            intent=intent,
            outcome_evidence=(*evidence, verification_evidence),
        )
        final_evidence = (*evidence, verification_evidence, synthesis_evidence)
        final_tasks = (*tasks, synthesis_task) if synthesis_task not in tasks else tasks
        await self._record_operation_result(
            DbOperationResult(
                operation_id=snapshot.operation.id,
                request=request,
                intent=intent,
                contract=contract,
                status=OperationStatus.SUCCEEDED,
                answer=_answer_from_synthesis_evidence(synthesis_evidence),
                evidence=final_evidence,
                diagnostics={
                    "verification": verification.to_dict(),
                    "synthesis": synthesis_evidence.payload,
                    "execution": {
                        "task_count": len(final_tasks),
                        "tasks": [task.to_dict() for task in final_tasks],
                    },
                },
            ),
            operation=snapshot.operation,
        )


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
    return isinstance(operation.metadata.get("resume_context"), dict)


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
