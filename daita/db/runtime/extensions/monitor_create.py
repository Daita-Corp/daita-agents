"""Runtime-owned DB monitor create extension executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from daita.runtime import Evidence, Operation, Task

from ...fingerprints import persisted_fingerprint
from ...monitor_commands.planner import (
    DbMonitorPlanner,
    _monitor_from_proposal,
    monitor_create_intent_from_dict,
)
from ...monitor_commands.types import DbMonitorCommand, DbMonitorValidation
from ...monitors import DbMonitorMutation, DbMonitorState
from ..tasks.models import DbTaskSpec
from .monitor_evidence import load_monitor_proposal_evidence


@dataclass(frozen=True)
class DbMonitorPlanCreateExecutor:
    """Executor that persists accepted or blocked monitor proposal evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.plan_create"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.plan_create"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        task_input = dict(task.input)
        source_scope = tuple(
            str(item)
            for item in (
                task.input.get("source_scope")
                or operation.request.get("source_scope")
                or ()
            )
        )
        planner = DbMonitorPlanner(
            registry=runtime.registry,
            limits=runtime.config.limits.to_dict(),
            delivery_default=_hosted_delivery_default(runtime),
        )
        proposal_input = _structured_proposal_input(task_input)
        intent = None
        if proposal_input is None and isinstance(task_input.get("intent"), dict):
            intent = monitor_create_intent_from_dict(dict(task_input["intent"]))
        target = (
            str(proposal_input.get("target_name") or proposal_input.get("target") or "")
            if proposal_input is not None
            else (intent.target.name if intent is not None else "")
        )
        schema_evidence = await _inspect_monitor_target_schema(
            runtime,
            operation,
            target=target,
        )
        if proposal_input is not None:
            proposal, validation = planner.create_structured_proposal(
                proposal_input,
                source_scope=source_scope,
                owner=dict(task.input.get("owner") or {}),
                schema_evidence_id=(
                    schema_evidence.id if schema_evidence is not None else None
                ),
            )
        elif intent is not None:
            command = DbMonitorCommand(
                kind="create",
                monitor_id=_optional_string(task_input.get("monitor_id")),
                prompt=str(
                    task_input.get("prompt") or operation.request.get("prompt") or ""
                ),
                confidence=float(task_input.get("confidence") or 1.0),
                diagnostics={"intent": intent.to_dict()},
            )
            proposal, validation = planner.create_proposal(
                command,
                source_scope=source_scope,
                owner=dict(task.input.get("owner") or {}),
                schema=(
                    schema_evidence.payload if schema_evidence is not None else None
                ),
                schema_evidence_id=(
                    schema_evidence.id if schema_evidence is not None else None
                ),
                intent=intent,
            )
        else:
            proposal, validation = _blocked_structured_input_required(task_input)
        fingerprint = str(
            proposal.get("proposal_fingerprint") or persisted_fingerprint(proposal)
        )
        proposal.setdefault("proposal_fingerprint", fingerprint)
        proposal.setdefault("kind", "monitor.proposal")
        proposal["validation"] = validation.to_dict()
        return [
            Evidence(
                kind="monitor.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=validation.accepted,
                payload=proposal,
                metadata={
                    "payload_fingerprint": fingerprint,
                    "monitor_id": proposal.get("monitor_id"),
                    "validation_accepted": validation.accepted,
                },
            )
        ]


def _structured_proposal_input(data: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("proposal", "monitor"):
        value = data.get(key)
        if isinstance(value, dict):
            return dict(value)
    if isinstance(data.get("observation_plan"), dict):
        return {
            key: value
            for key, value in data.items()
            if key
            in {
                "monitor_id",
                "id",
                "name",
                "display_name",
                "description",
                "status",
                "target_type",
                "target_name",
                "target",
                "source_scope",
                "schedule",
                "stream",
                "trigger",
                "observation_plan",
                "action_plan",
                "initial_state",
                "policy",
                "budgets",
                "owner",
                "governance",
                "metadata",
            }
        }
    return None


def _blocked_structured_input_required(
    data: dict[str, Any],
) -> tuple[dict[str, Any], DbMonitorValidation]:
    validation = DbMonitorValidation(
        accepted=False,
        errors=("monitor.proposal_incomplete:structured_input",),
        diagnostics={"input_keys": sorted(str(key) for key in data)},
    )
    proposal: dict[str, Any] = {
        "kind": "monitor.proposal",
        "monitor_id": _optional_string(data.get("monitor_id")) or "db_monitor",
        "name": _optional_string(data.get("name")) or "DB Monitor",
        "description": "",
        "status": "blocked",
        "target_type": "table",
        "target_name": _optional_string(data.get("target_name")) or "",
        "source_scope": list(data.get("source_scope") or ()),
        "schedule": None,
        "stream": None,
        "trigger": {},
        "observation_plan": {},
        "action_plan": {"kind": "none", "steps": []},
        "initial_state": {},
        "policy": {},
        "budgets": {},
        "owner": dict(data.get("owner") or {}),
        "runtime_limits": {},
        "governance": {"approval_required": False, "risk": "low"},
        "metadata": {
            "created_from_structured_planner_action": True,
            "validation": validation.to_dict(),
        },
        "validation": validation.to_dict(),
    }
    proposal["proposal_fingerprint"] = persisted_fingerprint(proposal)
    proposal["metadata"]["proposal_fingerprint"] = proposal["proposal_fingerprint"]
    return proposal, validation


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _hosted_delivery_default(runtime: Any) -> str | None:
    host_runtime = runtime.config.metadata.get("host_runtime")
    if not isinstance(host_runtime, dict):
        return None
    delivery_defaults = host_runtime.get("delivery_defaults")
    if not isinstance(delivery_defaults, list) or not delivery_defaults:
        return None
    default = delivery_defaults[0]
    return default if isinstance(default, str) and default else None


@dataclass(frozen=True)
class DbMonitorCommitCreateExecutor:
    """Executor that idempotently commits a monitor proposal."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.commit_create"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.commit_create"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        proposal_evidence = await load_monitor_proposal_evidence(
            runtime,
            operation,
            task,
            task.input.get("proposal_evidence_id"),
        )
        if proposal_evidence is None:
            raise RuntimeError("monitor proposal evidence is required")
        if not proposal_evidence.accepted:
            raise RuntimeError("monitor proposal evidence was not accepted")
        proposal = dict(proposal_evidence.payload)
        expected_fingerprint = task.input.get("proposal_fingerprint")
        actual_fingerprint = proposal.get(
            "proposal_fingerprint"
        ) or persisted_fingerprint(proposal)
        if expected_fingerprint and expected_fingerprint != actual_fingerprint:
            raise RuntimeError("monitor proposal fingerprint mismatch")

        validation = DbMonitorValidation.from_dict(
            dict(
                proposal.get("validation")
                or proposal.get("metadata", {}).get("validation")
                or {}
            )
        )
        monitor = _monitor_from_proposal(proposal, validation=validation)
        existing = await runtime.inspect_monitor(monitor.id)
        committed_existing = existing is not None
        if existing is None:
            initial_state = dict(proposal.get("initial_state") or {})
            await runtime.commit_monitor_mutation(
                DbMonitorMutation(
                    action="create",
                    operation=operation,
                    monitor_after=monitor,
                    state_after=DbMonitorState(
                        monitor_id=monitor.id,
                        cursor=dict(initial_state.get("cursor") or {}),
                        last_operation_id=operation.id,
                        last_management_operation_id=operation.id,
                    ),
                )
            )
        committed_state = await runtime.monitor_store.load_monitor_state(monitor.id)
        if committed_state is None:
            committed_state = DbMonitorState(monitor_id=monitor.id)
        return [
            Evidence(
                kind="monitor.definition",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload={
                    "monitor": monitor.to_dict(),
                    "monitor_state": committed_state.to_dict(),
                    "proposal_evidence_id": proposal_evidence.id,
                    "proposal_fingerprint": actual_fingerprint,
                    "idempotent_existing": committed_existing,
                },
                metadata={
                    "payload_fingerprint": actual_fingerprint,
                    "proposal_evidence_id": proposal_evidence.id,
                    "monitor_id": monitor.id,
                    "idempotent_existing": committed_existing,
                },
            )
        ]


async def _inspect_monitor_target_schema(
    runtime: Any,
    operation: Operation,
    *,
    target: str,
) -> Evidence | None:
    cached = runtime.cached_schema_evidence(operation_id=operation.id)
    if cached is not None:
        return await _persist_monitor_schema_evidence(runtime, operation, cached)
    persisted = runtime.persisted_schema_evidence(operation_id=operation.id)
    if persisted is not None:
        return await _persist_monitor_schema_evidence(runtime, operation, persisted)
    capability = _first_capability(runtime, "db.schema.inspect")
    if capability is None:
        return None
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id=capability.id,
                owner=capability.owner,
                input={"tables": [target] if target else []},
                reason="monitor_create_schema_context",
                sequence=0,
                metadata={"target_table": target},
            ),
        ),
    )
    evidence = await runtime.execute_task(plan.tasks[0], operation)
    schema_evidence = next(
        (item for item in evidence if item.kind == "schema.asset_profile"),
        None,
    )
    if schema_evidence is not None:
        runtime.remember_schema_evidence(schema_evidence)
    return schema_evidence


async def _persist_monitor_schema_evidence(
    runtime: Any,
    operation: Operation,
    evidence: Evidence,
) -> Evidence:
    persisted = Evidence(
        id=evidence.id or f"monitor-schema-{uuid4()}",
        kind=evidence.kind,
        owner=evidence.owner,
        operation_id=operation.id,
        accepted=evidence.accepted,
        payload=dict(evidence.payload),
        metadata={
            **dict(evidence.metadata),
            "monitor_planning_schema_context": True,
        },
    )
    await runtime.store.save_evidence(persisted)
    return persisted


def _first_capability(runtime: Any, capability_id: str) -> Any | None:
    matches = [
        capability
        for capability in runtime.registry.capabilities
        if capability.id == capability_id
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda item: item.owner)[0]
