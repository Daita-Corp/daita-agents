"""Monitor runtime behavior for ``DbRuntime``."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from daita.runtime import Operation

from ..analysis import evidence_ref
from ..monitor_plugin_planning import (
    MonitorPluginPlanner,
    MonitorPluginPlanningBlocked,
    monitor_source_observed_value,
)
from .types import DbRuntimeGovernanceBlocked

if TYPE_CHECKING:
    from daita.plugins import ExtensionRegistry
    from daita.runtime import ApprovalRequest, Capability, Evidence, Task


class DbRuntimeMonitorObservationMixin:
    if TYPE_CHECKING:
        registry: ExtensionRegistry
        _is_setup: bool

        async def setup(self, *, agent_id: str | None = None) -> None: ...

        async def _plan_monitor_plugin_task_for_capability(
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
        ) -> Task: ...

        async def _execute_or_reuse_monitor_plugin_task(
            self,
            task: Task,
            operation: Operation,
            *,
            context: dict[str, Any],
        ) -> tuple[Evidence, ...]: ...

    async def execute_monitor_source_observation(
        self,
        operation: Operation,
        *,
        monitor_id: str,
        monitor_run_id: str,
        tick_operation_id: str,
        source_step: dict[str, Any],
        cursor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one deterministic read-only plugin source step for a tick."""

        if not self._is_setup:
            await self.setup()
        step = dict(source_step or {})
        try:
            plan = MonitorPluginPlanner(tuple(self.registry.capabilities)).plan_source(
                step,
                cursor=cursor or {},
                monitor_id=monitor_id,
                monitor_run_id=monitor_run_id,
                tick_operation_id=tick_operation_id,
            )
        except MonitorPluginPlanningBlocked as exc:
            return {
                "status": "blocked",
                "block_reason": exc.reason,
                "details": exc.details,
                "task_ids": [],
                "plugin_evidence_refs": [],
            }

        task = await self._plan_monitor_plugin_task_for_capability(
            operation,
            plan.capability,
            input_payload=plan.input_payload,
            input_hash=plan.input_hash,
            idempotency_key=plan.idempotency_key,
            reason=plan.reason,
            sequence=plan.sequence,
            metadata=plan.metadata,
        )
        try:
            evidence = await self._execute_or_reuse_monitor_plugin_task(
                task,
                operation,
                context={
                    "monitor_id": monitor_id,
                    "monitor_run_id": monitor_run_id,
                    "tick_operation_id": tick_operation_id,
                    "db_monitor_phase": 6,
                    "monitor_observation_role": "plugin_source",
                },
            )
        except DbRuntimeGovernanceBlocked:
            return {
                "status": "blocked",
                "block_reason": "governance_blocked",
                "capability_id": plan.capability.id,
                "capability_owner": plan.capability.owner,
                "task_ids": [task.id],
                "plugin_evidence_refs": [],
            }
        except Exception as exc:
            return {
                "status": "failed",
                "block_reason": "plugin_source_failed",
                "details": {"type": type(exc).__name__, "message": str(exc)},
                "capability_id": plan.capability.id,
                "capability_owner": plan.capability.owner,
                "task_ids": [task.id],
                "plugin_evidence_refs": [],
            }
        if not evidence:
            return {
                "status": "blocked",
                "block_reason": "plugin_source_evidence_missing",
                "capability_id": plan.capability.id,
                "capability_owner": plan.capability.owner,
                "task_ids": [task.id],
                "plugin_evidence_refs": [],
            }
        return {
            "status": "succeeded",
            "source_kind": plan.intent_payload.get("source_kind"),
            "capability_id": plan.capability.id,
            "capability_owner": plan.capability.owner,
            "task_ids": [task.id],
            "plugin_evidence_refs": [evidence_ref(item) for item in evidence],
            "value": monitor_source_observed_value(step, evidence),
        }
