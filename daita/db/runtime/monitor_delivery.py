"""Monitor runtime behavior for ``DbRuntime``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from daita.runtime import (
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    Evidence,
    Operation,
    OperationSnapshot,
    OperationStatus,
    RuntimeEvent,
    RuntimeEventType,
    Task,
    TaskDependency,
    TaskStatus,
)

from ..analysis import evidence_ref
from ..monitor_plugin_planning import (
    MonitorPluginPlanner,
    MonitorPluginPlanningBlocked,
    monitor_delivery_source_refs,
    monitor_report_fingerprint,
)
from ..monitors import (
    DbMonitorMutation,
    DbMonitorRun,
    DbMonitorState,
)
from .analysis import _payload_fingerprint, _stable_hash
from .resume import _monitor_delivery_context
from .types import (
    _TERMINAL_TASK_STATUSES,
    DbRuntimeGovernanceBlocked,
)


from .monitor_helpers import _terminal_monitor_approval_reason


class DbRuntimeMonitorDeliveryMixin:
    async def execute_monitor_delivery(
        self,
        operation_id: str,
        *,
        monitor_id: str,
        monitor_name: str,
        monitor_run_id: str,
        tick_operation_id: str,
        report_evidence_id: str | None = None,
        governed: bool = False,
    ) -> dict[str, Any]:
        """Deliver Phase 5 monitor report evidence through a registered capability."""

        if not self._is_setup:
            await self.setup()
        operation = await self.store.load_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        report = await self._monitor_report_for_delivery(
            operation.id,
            report_evidence_id=report_evidence_id,
        )
        if report is None:
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=None,
                capability=None,
                action_plan_fingerprint="",
                report_fingerprint="",
                source_evidence_refs=(),
                task_ids=(),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key="",
                block_reason="missing_monitor_report",
            )

        intent = dict(report.payload.get("delivery_intent") or {})
        action_fingerprint = str(report.payload.get("action_plan_fingerprint") or "")
        report_fingerprint = monitor_report_fingerprint(report)
        source_refs = monitor_delivery_source_refs(
            report,
            tuple(await self.store.list_evidence(operation.id)),
        )
        if not intent:
            return {
                "status": "skipped",
                "block_reason": "missing_delivery_intent",
                "report_evidence_id": report.id,
                "source_evidence_refs": [dict(item) for item in source_refs],
            }

        operation = await self._prepare_monitor_delivery_operation(
            operation,
            monitor_id=monitor_id,
            monitor_name=monitor_name,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            report=report,
            source_evidence_refs=source_refs,
        )
        existing = await self._latest_evidence(
            operation.id,
            "monitor.delivery_result",
            payload={"report_fingerprint": report_fingerprint},
        )
        if existing is not None and not await self._monitor_delivery_can_resume(
            operation.id, existing
        ):
            return dict(existing.payload)

        try:
            plan = MonitorPluginPlanner(
                tuple(self.registry.capabilities)
            ).plan_delivery(
                intent,
                report=report,
                source_evidence_refs=source_refs,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
            )
        except MonitorPluginPlanningBlocked as exc:
            blocked_plan_evidence = await self._persist_monitor_delivery_plan(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_intent=intent,
                report=report,
                source_evidence_refs=source_refs,
                accepted=False,
                block_reason=exc.reason,
                details=exc.details,
            )
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(
                    intent.get("delivery_kind") or intent.get("mode") or ""
                ),
                capability=None,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=(),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key="",
                block_reason=exc.reason,
                plan_evidence=blocked_plan_evidence,
            )

        capability = plan.capability
        idempotency_key = plan.idempotency_key
        existing = await self._latest_evidence(
            operation.id,
            "monitor.delivery_result",
            payload={"idempotency_key": idempotency_key},
        )
        if existing is not None and not await self._monitor_delivery_can_resume(
            operation.id, existing
        ):
            return dict(existing.payload)

        plan_evidence = await self._persist_monitor_delivery_plan(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            delivery_intent=intent,
            report=report,
            source_evidence_refs=source_refs,
            capability=capability,
            idempotency_key=idempotency_key,
            accepted=True,
        )
        governance_decision = await self.evaluate_monitor_effect_governance(
            operation,
            capability=capability,
            intent=plan.intent_payload,
            phase="delivery",
            mutate_approvals=governed,
        )
        if not governance_decision.allowed:
            task_ids: tuple[str, ...] = ()
            if (
                governed
                and governance_decision.result.pending_approval
                and governance_decision.result.approval_requests
            ):
                task = self._monitor_plugin_task_for_capability(
                    operation,
                    capability,
                    input_payload=plan.input_payload,
                    input_hash=plan.input_hash,
                    idempotency_key=idempotency_key,
                    reason=plan.reason,
                    sequence=plan.sequence,
                    metadata=plan.metadata,
                    approval_requests=governance_decision.result.approval_requests,
                )
                await self._plan_kernel_task(task)
                task_ids = (task.id,)
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(plan.intent_payload.get("delivery_kind") or ""),
                capability=capability,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=task_ids,
                plugin_result_evidence=(),
                status=governance_decision.status,
                idempotency_key=idempotency_key,
                block_reason=governance_decision.reason,
                plan_evidence=plan_evidence,
            )

        task = self._monitor_plugin_task_for_capability(
            operation,
            capability,
            input_payload=plan.input_payload,
            input_hash=plan.input_hash,
            idempotency_key=idempotency_key,
            reason=plan.reason,
            sequence=plan.sequence,
            metadata=plan.metadata,
        )
        try:
            plugin_evidence = await self._execute_or_reuse_monitor_plugin_task(
                task,
                operation,
                context={
                    "monitor_id": monitor_id,
                    "monitor_run_id": monitor_run_id,
                    "tick_operation_id": tick_operation_id,
                    "db_monitor_phase": 6,
                    "monitor_action_role": "delivery",
                },
            )
        except DbRuntimeGovernanceBlocked:
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(
                    intent.get("delivery_kind") or intent.get("mode") or ""
                ),
                capability=capability,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=(task.id,),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key=idempotency_key,
                block_reason="governance_blocked",
                plan_evidence=plan_evidence,
            )
        except Exception as exc:
            return await self._persist_monitor_delivery_result(
                operation,
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
                delivery_kind=str(
                    intent.get("delivery_kind") or intent.get("mode") or ""
                ),
                capability=capability,
                action_plan_fingerprint=action_fingerprint,
                report_fingerprint=report_fingerprint,
                source_evidence_refs=source_refs,
                task_ids=(task.id,),
                plugin_result_evidence=(),
                status="failed",
                idempotency_key=idempotency_key,
                block_reason=str(exc),
                plan_evidence=plan_evidence,
            )
        return await self._persist_monitor_delivery_result(
            operation,
            monitor_id=monitor_id,
            monitor_run_id=monitor_run_id,
            tick_operation_id=tick_operation_id,
            delivery_kind=str(intent.get("delivery_kind") or intent.get("mode") or ""),
            capability=capability,
            action_plan_fingerprint=action_fingerprint,
            report_fingerprint=report_fingerprint,
            source_evidence_refs=source_refs,
            task_ids=(task.id,),
            plugin_result_evidence=plugin_evidence,
            status="succeeded",
            idempotency_key=idempotency_key,
            plan_evidence=plan_evidence,
            supersede_approval_block=True,
        )

    async def _monitor_report_for_delivery(
        self,
        operation_id: str,
        *,
        report_evidence_id: str | None,
    ) -> Evidence | None:
        reports = [
            item
            for item in await self.store.list_evidence(operation_id)
            if item.kind == "monitor.report" and item.accepted
        ]
        if report_evidence_id is not None:
            return next(
                (item for item in reports if item.id == report_evidence_id), None
            )
        return reports[-1] if reports else None

    async def _prepare_monitor_delivery_operation(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_name: str,
        monitor_run_id: str,
        tick_operation_id: str,
        report: Evidence,
        source_evidence_refs: tuple[dict[str, Any], ...],
    ) -> Operation:
        metadata = {
            **operation.metadata,
            "monitor_delivery_context": {
                "monitor_id": monitor_id,
                "monitor_name": monitor_name,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "report_evidence_id": report.id,
                "report_fingerprint": (
                    report.metadata.get("payload_fingerprint")
                    or _payload_fingerprint(report.payload)
                ),
                "action_plan_fingerprint": report.payload.get(
                    "action_plan_fingerprint"
                ),
                "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            },
        }
        updated = replace(
            operation,
            status=OperationStatus.RUNNING,
            required_evidence=frozenset(
                {
                    *operation.required_evidence,
                    "monitor.delivery_plan",
                    "monitor.delivery_result",
                }
            ),
            metadata=metadata,
        )
        await self.store.save_operation(updated)
        return updated

    async def _persist_monitor_delivery_plan(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        delivery_intent: dict[str, Any],
        report: Evidence,
        source_evidence_refs: tuple[dict[str, Any], ...],
        capability: Capability | None = None,
        idempotency_key: str | None = None,
        accepted: bool,
        block_reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Evidence:
        report_fingerprint = str(
            report.metadata.get("payload_fingerprint") or ""
        ) or _payload_fingerprint(report.payload)
        existing = await self._latest_evidence(
            operation.id,
            "monitor.delivery_plan",
            payload={"report_fingerprint": report_fingerprint},
            accepted=accepted,
        )
        if existing is not None:
            return existing
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "delivery_kind": delivery_intent.get("delivery_kind")
            or delivery_intent.get("mode"),
            "capability_id": capability.id if capability is not None else None,
            "capability_owner": capability.owner if capability is not None else None,
            "report_evidence_id": report.id,
            "report_fingerprint": report_fingerprint,
            "action_plan_fingerprint": report.payload.get("action_plan_fingerprint"),
            "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            "delivery_intent": dict(delivery_intent),
            "idempotency_key": idempotency_key,
            "status": "planned" if accepted else "blocked",
            "block_reason": block_reason,
            "details": dict(details or {}),
        }
        evidence = Evidence(
            id=f"monitor-delivery-plan-{uuid4()}",
            kind="monitor.delivery_plan",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=accepted,
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_delivery_kind": payload["delivery_kind"],
                "monitor_report_fingerprint": report_fingerprint,
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        return evidence

    async def _persist_monitor_delivery_result(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        delivery_kind: str | None,
        capability: Capability | None,
        action_plan_fingerprint: str,
        report_fingerprint: str,
        source_evidence_refs: tuple[dict[str, Any], ...],
        task_ids: tuple[str, ...],
        plugin_result_evidence: tuple[Evidence, ...],
        status: str,
        idempotency_key: str,
        block_reason: str | None = None,
        plan_evidence: Evidence | None = None,
        supersede_approval_block: bool = False,
    ) -> dict[str, Any]:
        if idempotency_key:
            existing = await self._latest_evidence(
                operation.id,
                "monitor.delivery_result",
                payload={"idempotency_key": idempotency_key},
            )
            if existing is not None and not (
                supersede_approval_block
                and existing.payload.get("block_reason")
                == "governance_approval_required"
            ):
                return dict(existing.payload)
        evidence_id = f"monitor-delivery-result-{uuid4()}"
        plugin_refs = [evidence_ref(item) for item in plugin_result_evidence if item.id]
        delivery_intent = (
            dict(plan_evidence.payload.get("delivery_intent") or {})
            if plan_evidence is not None
            else {}
        )
        delivery_target = delivery_intent.get("target")
        delivery_target = delivery_target if isinstance(delivery_target, dict) else {}
        delivery_channel = (
            delivery_target.get("channel")
            or delivery_target.get("type")
            or delivery_target.get("recipient")
            or delivery_target.get("address")
            or delivery_target.get("url")
            or delivery_target.get("endpoint")
            or delivery_target.get("path")
        )
        payload = {
            "monitor_id": monitor_id,
            "monitor_run_id": monitor_run_id,
            "tick_operation_id": tick_operation_id,
            "delivery_operation_id": operation.id,
            "delivery_kind": delivery_kind,
            "delivery_target": dict(delivery_target),
            "delivery_channel": delivery_channel,
            "capability_id": capability.id if capability is not None else None,
            "capability_owner": capability.owner if capability is not None else None,
            "action_plan_fingerprint": action_plan_fingerprint,
            "report_fingerprint": report_fingerprint,
            "source_evidence_refs": [dict(item) for item in source_evidence_refs],
            "task_ids": list(task_ids),
            "plugin_result_evidence_refs": plugin_refs,
            "status": status,
            "idempotency_key": idempotency_key,
            "block_reason": block_reason,
            "plan_evidence_id": plan_evidence.id if plan_evidence is not None else None,
            "delivery_result_evidence_id": evidence_id,
            "report_delivery_status": (
                "delivered"
                if status == "succeeded"
                else "blocked" if status == "blocked" else "failed"
            ),
        }
        evidence = Evidence(
            id=evidence_id,
            kind="monitor.delivery_result",
            owner="db_runtime",
            operation_id=operation.id,
            accepted=status == "succeeded",
            payload=payload,
            metadata={
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "tick_operation_id": tick_operation_id,
                "monitor_delivery_kind": delivery_kind,
                "monitor_delivery_channel": delivery_channel,
                "monitor_report_fingerprint": report_fingerprint,
                "payload_fingerprint": _payload_fingerprint(payload),
            },
        )
        await self.store.save_evidence(evidence)
        if status == "succeeded":
            await self.kernel.complete_operation(
                operation.id,
                status=OperationStatus.SUCCEEDED,
                message=f"Monitor delivery {operation.id} succeeded.",
            )
        elif status == "blocked":
            await self.kernel.block_operation(
                operation.id,
                message=f"Monitor delivery blocked: {block_reason}.",
            )
        else:
            await self.kernel.fail_operation_if_active(
                operation.id, RuntimeError(block_reason or status)
            )
        return payload

    def _monitor_plugin_task_for_capability(
        self,
        operation: Operation,
        capability: Capability,
        *,
        input_payload: dict[str, Any],
        input_hash: str,
        idempotency_key: str,
        reason: str,
        sequence: int,
        metadata: dict[str, Any],
        approval_requests: tuple[ApprovalRequest, ...] = (),
    ) -> Task:
        task_key = _stable_hash(
            {
                "operation_id": operation.id,
                "capability_id": capability.id,
                "capability_owner": capability.owner,
                "idempotency_key": idempotency_key,
            }
        )
        task = Task(
            id=f"monitor-plugin-task-{task_key[:32]}",
            operation_id=operation.id,
            capability_id=capability.id,
            executor_id=capability.executor,
            input={**input_payload, "input_hash": input_hash},
            required_evidence=capability.output_evidence,
            metadata={
                **dict(metadata),
                "owner": capability.owner,
                "reason": reason,
                "sequence": sequence,
                "input_hash": input_hash,
                "idempotency_key": idempotency_key,
                "idempotent": capability.idempotent,
                "replay_safe": capability.replay_safe,
                "side_effecting": capability.side_effecting,
            },
        )
        approval_dependencies = tuple(
            TaskDependency(
                kind="approval",
                approval_status=ApprovalStatus.APPROVED,
                approval_id=request.approval_id,
                approval_policy_id=request.requested_by_policy_id,
                approval_name=str(request.proposed_action.get("approval") or ""),
                operation_id=operation.id,
            )
            for request in approval_requests
        )
        if approval_dependencies:
            task = replace(
                task,
                dependencies=(*task.dependencies, *approval_dependencies),
            )
        return task

    async def _execute_or_reuse_monitor_plugin_task(
        self,
        task: Task,
        operation: Operation,
        *,
        context: dict[str, Any],
    ) -> tuple[Evidence, ...]:
        stored = await self.store.load_task(task.id)
        if stored is None:
            stored = await self._plan_kernel_task(task)
        elif stored.status in _TERMINAL_TASK_STATUSES:
            return tuple(
                item
                for item in await self.store.list_evidence(operation.id)
                if item.task_id == stored.id
            )
        elif stored.input != task.input and stored.status is TaskStatus.PENDING:
            stored = replace(
                stored,
                input=task.input,
                metadata={**stored.metadata, **task.metadata},
            )
            await self.store.save_task(stored)
        return await self.execute_task(stored, operation, context=context)

    async def _monitor_delivery_can_resume(
        self,
        operation_id: str,
        result: Evidence,
    ) -> bool:
        if result.payload.get("block_reason") != "governance_approval_required":
            return False
        approvals = await self.store.list_approval_requests(operation_id)
        if not approvals:
            return False
        if any(
            approval.status
            in {
                ApprovalStatus.REJECTED,
                ApprovalStatus.CANCELLED,
                ApprovalStatus.EXPIRED,
            }
            for approval in approvals
        ):
            return False
        return all(approval.status is ApprovalStatus.APPROVED for approval in approvals)

    async def _monitor_approval_context(
        self,
        approval: ApprovalRequest,
    ) -> dict[str, Any]:
        operation = await self.store.load_operation(approval.operation_id)
        metadata = operation.metadata if operation is not None else {}
        context: dict[str, Any] = {}
        action_context = metadata.get("monitor_action_context")
        if isinstance(action_context, dict):
            context.update(
                {
                    "kind": f"monitor.{action_context.get('action_kind')}",
                    "monitor_id": action_context.get("monitor_id"),
                    "monitor_run_id": action_context.get("monitor_run_id"),
                    "tick_operation_id": action_context.get("tick_operation_id"),
                    "operation_id": approval.operation_id,
                    "action_plan_fingerprint": action_context.get(
                        "action_plan_fingerprint"
                    ),
                    "source_evidence_refs": list(
                        action_context.get("cited_tick_evidence_refs") or ()
                    ),
                }
            )
        delivery_context = metadata.get("monitor_delivery_context")
        if isinstance(delivery_context, dict):
            context.update(
                {
                    "kind": "monitor.delivery",
                    "monitor_id": delivery_context.get("monitor_id"),
                    "monitor_run_id": delivery_context.get("monitor_run_id"),
                    "tick_operation_id": delivery_context.get("tick_operation_id"),
                    "operation_id": approval.operation_id,
                    "report_fingerprint": delivery_context.get("report_fingerprint"),
                    "action_plan_fingerprint": delivery_context.get(
                        "action_plan_fingerprint"
                    ),
                    "source_evidence_refs": list(
                        delivery_context.get("source_evidence_refs") or ()
                    ),
                }
            )
        governance_facts = (
            approval.proposed_action.get("request", {}).get("governance_facts", {})
            if isinstance(approval.proposed_action.get("request"), dict)
            else {}
        )
        monitor_effect = (
            governance_facts.get("monitor_effect")
            if isinstance(governance_facts, dict)
            else {}
        )
        monitor_effect = monitor_effect if isinstance(monitor_effect, dict) else {}
        intent = monitor_effect.get("intent")
        intent = intent if isinstance(intent, dict) else {}
        context.update(
            {
                key: value
                for key, value in {
                    "capability_id": (
                        approval.proposed_action.get("request", {})
                        .get("capability", {})
                        .get("id")
                        if isinstance(approval.proposed_action.get("request"), dict)
                        and isinstance(
                            approval.proposed_action["request"].get("capability"),
                            dict,
                        )
                        else None
                    ),
                    "capability_owner": (
                        approval.proposed_action.get("request", {})
                        .get("capability", {})
                        .get("owner")
                        if isinstance(approval.proposed_action.get("request"), dict)
                        and isinstance(
                            approval.proposed_action["request"].get("capability"),
                            dict,
                        )
                        else None
                    ),
                    "target": intent.get("target"),
                    "risk": approval.risk.value,
                    "reason": approval.reason,
                }.items()
                if value is not None
            }
        )
        return context if context.get("monitor_id") else {}

    async def _finalize_resumed_monitor_delivery(
        self,
        snapshot: OperationSnapshot,
    ) -> None:
        context = _monitor_delivery_context(snapshot.operation)
        if not context:
            return
        report_id = context.get("report_evidence_id")
        result = await self._latest_evidence(
            snapshot.operation.id,
            "monitor.delivery_result",
            payload={
                "report_fingerprint": str(context.get("report_fingerprint") or "")
            },
        )
        terminal_reason = _terminal_monitor_approval_reason(snapshot.approval_requests)
        if result is not None and terminal_reason:
            payload = await self._persist_monitor_delivery_result(
                snapshot.operation,
                monitor_id=str(context.get("monitor_id") or ""),
                monitor_run_id=str(context.get("monitor_run_id") or ""),
                tick_operation_id=str(context.get("tick_operation_id") or ""),
                delivery_kind=result.payload.get("delivery_kind"),
                capability=(
                    self.registry.get_capability(
                        str(result.payload.get("capability_id")),
                        owner=str(result.payload.get("capability_owner")),
                    )
                    if result.payload.get("capability_id")
                    and result.payload.get("capability_owner")
                    else None
                ),
                action_plan_fingerprint=str(
                    result.payload.get("action_plan_fingerprint") or ""
                ),
                report_fingerprint=str(result.payload.get("report_fingerprint") or ""),
                source_evidence_refs=tuple(
                    dict(item)
                    for item in result.payload.get("source_evidence_refs") or ()
                    if isinstance(item, dict)
                ),
                task_ids=tuple(
                    str(item) for item in result.payload.get("task_ids") or ()
                ),
                plugin_result_evidence=(),
                status="blocked",
                idempotency_key=str(result.payload.get("idempotency_key") or ""),
                block_reason=terminal_reason,
                supersede_approval_block=True,
            )
            await self._refresh_monitor_delivery_run_summary(
                snapshot.operation,
                result_payload=payload,
            )
            return
        if result is not None and not await self._monitor_delivery_can_resume(
            snapshot.operation.id, result
        ):
            await self._refresh_monitor_delivery_run_summary(
                snapshot.operation,
                result_payload=dict(result.payload),
            )
            return
        payload = await self.execute_monitor_delivery(
            snapshot.operation.id,
            monitor_id=str(context.get("monitor_id") or ""),
            monitor_name=str(context.get("monitor_name") or ""),
            monitor_run_id=str(context.get("monitor_run_id") or ""),
            tick_operation_id=str(context.get("tick_operation_id") or ""),
            report_evidence_id=str(report_id) if report_id else None,
            governed=True,
        )
        await self._refresh_monitor_delivery_run_summary(
            snapshot.operation,
            result_payload=payload,
        )

    async def _refresh_monitor_delivery_run_summary(
        self,
        operation: Operation,
        *,
        result_payload: dict[str, Any],
    ) -> None:
        monitor_id = str(result_payload.get("monitor_id") or "")
        monitor_run_id = str(result_payload.get("monitor_run_id") or "")
        if not monitor_id or not monitor_run_id:
            return
        monitor = await self.monitor_store.load_monitor(monitor_id)
        if monitor is None:
            return
        runs = await self.monitor_store.list_monitor_runs(monitor_id)
        run = next((item for item in runs if item.id == monitor_run_id), None)
        if run is None:
            return
        summary = {
            **run.summary,
            "delivery_status": result_payload.get("status"),
            "delivery_kind": result_payload.get("delivery_kind"),
            "delivery_target": dict(result_payload.get("delivery_target") or {}),
            "delivery_channel": result_payload.get("delivery_channel"),
            "delivery_operation_id": result_payload.get("delivery_operation_id"),
            "delivery_result_evidence_id": result_payload.get(
                "delivery_result_evidence_id"
            ),
            "delivery_plugin_result_evidence_refs": list(
                result_payload.get("plugin_result_evidence_refs") or ()
            ),
            "delivery_task_ids": list(result_payload.get("task_ids") or ()),
            "delivery_block_reason": result_payload.get("block_reason"),
            "delivery_idempotency_key": result_payload.get("idempotency_key"),
        }
        if summary == run.summary:
            return
        updated_run = DbMonitorRun.from_dict({**run.to_dict(), "summary": summary})
        state = await self.monitor_store.load_monitor_state(monitor_id)
        state_after = state
        if state is not None:
            state_after = DbMonitorState.from_dict(
                {
                    **state.to_dict(),
                    "cursor": {
                        **state.cursor,
                        "last_delivery_status": result_payload.get("status"),
                        "last_delivery_result_evidence_id": result_payload.get(
                            "delivery_result_evidence_id"
                        ),
                    },
                }
            )
        await self.monitor_store.commit_monitor_mutation(
            DbMonitorMutation(
                action="run",
                operation=operation,
                events=(
                    RuntimeEvent(
                        id=f"monitor-delivery-resume-event-{uuid4()}",
                        type=RuntimeEventType.OPERATION_UPDATED,
                        operation_id=operation.id,
                        runtime_id=self.runtime_id,
                        runtime_kind=self.runtime_kind,
                        evidence_id=result_payload.get("delivery_result_evidence_id"),
                        message=(
                            f"Monitor {monitor_id} delivery resume summary refreshed."
                        ),
                        payload={
                            "monitor_id": monitor_id,
                            "monitor_run_id": monitor_run_id,
                            "status": result_payload.get("status"),
                        },
                    ),
                ),
                monitor_before=monitor,
                state_before=state,
                state_after=state_after,
                run_after=updated_run,
            )
        )
