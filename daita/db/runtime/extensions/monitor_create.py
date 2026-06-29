"""Runtime-owned DB monitor create extension executors."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from daita.runtime import Evidence, Operation, Task

from ...analysis import stable_fingerprint
from ...models import DbRequest
from ...monitor_commands.extractor import DeterministicMonitorIntentExtractor
from ...monitor_commands.planner import DbMonitorPlanner, _monitor_from_proposal
from ...monitor_commands.types import DbMonitorCommand, DbMonitorValidation
from ...monitors import DbMonitorMutation, DbMonitorState

if TYPE_CHECKING:
    from .plugin import DbRuntimePlanningPlugin


def _bound_runtime(plugin: DbRuntimePlanningPlugin) -> Any:
    runtime = plugin.runtime
    if runtime is None:
        raise RuntimeError("DB runtime planning plugin is not bound to a runtime")
    return runtime


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
        runtime = _bound_runtime(self.plugin)
        command = DbMonitorCommand(
            **dict(task.input.get("command") or operation.request.get("command") or {})
        )
        source_scope = tuple(str(item) for item in task.input.get("source_scope") or ())
        preliminary_intent = DeterministicMonitorIntentExtractor().extract(
            command,
            DbRequest(prompt=command.prompt, source_scope=source_scope),
            host_defaults={"delivery_default": _hosted_delivery_default(runtime)},
        )
        target = preliminary_intent.target.name or ""
        schema_evidence = await _inspect_monitor_target_schema(
            runtime,
            operation,
            target=target,
        )
        proposal, validation = DbMonitorPlanner(
            registry=runtime.registry,
            limits=runtime.config.limits.to_dict(),
            delivery_default=_hosted_delivery_default(runtime),
        ).create_proposal(
            command,
            source_scope=source_scope,
            owner=dict(task.input.get("owner") or {}),
            schema=(schema_evidence.payload if schema_evidence is not None else None),
            schema_evidence_id=(
                schema_evidence.id if schema_evidence is not None else None
            ),
        )
        fingerprint = str(
            proposal.get("proposal_fingerprint") or stable_fingerprint(proposal)
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
        runtime = _bound_runtime(self.plugin)
        proposal_evidence = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("proposal_evidence_id"),
        )
        if proposal_evidence is None:
            raise RuntimeError("monitor proposal evidence is required")
        if not proposal_evidence.accepted:
            raise RuntimeError("monitor proposal evidence was not accepted")
        proposal = dict(proposal_evidence.payload)
        expected_fingerprint = task.input.get("proposal_fingerprint")
        actual_fingerprint = proposal.get("proposal_fingerprint") or stable_fingerprint(
            proposal
        )
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
            await runtime.monitor_store.commit_monitor_mutation(
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
        await runtime.store.save_operation(
            replace(
                operation,
                metadata={
                    **operation.metadata,
                    "monitor_id": monitor.id,
                    "monitor_name": monitor.name,
                    "proposal_fingerprint": actual_fingerprint,
                    "proposal_evidence_id": proposal_evidence.id,
                },
            )
        )
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


async def _load_evidence(
    runtime: Any,
    operation_id: str,
    evidence_id: Any,
) -> Evidence | None:
    if not evidence_id:
        return None
    for evidence in await runtime.store.list_evidence(operation_id):
        if evidence.id == evidence_id:
            return evidence
    return None


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
    schema_task = await runtime.kernel.plan_task(
        operation_id=operation.id,
        capability_id=capability.id,
        owner=capability.owner,
        input={"tables": [target] if target else []},
        metadata={
            "reason": "monitor_create_schema_context",
            "sequence": 0,
            "target_table": target,
        },
    )
    evidence = await runtime.execute_task(schema_task, operation)
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
