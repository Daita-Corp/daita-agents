"""Runtime-owned DB monitor create extension executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from daita.runtime import Evidence, Operation, Task

from ...fingerprints import persisted_fingerprint
from ...planning_context import catalog_schema_from_evidence
from ...monitor_commands.planner import (
    DbMonitorPlanner,
    _monitor_from_proposal,
    monitor_create_intent_from_dict,
)
from ...monitor_commands.types import DbMonitorCommand, DbMonitorValidation
from ...monitors import DbMonitorMutation, DbMonitorState
from .monitor_evidence import load_monitor_proposal_evidence

if TYPE_CHECKING:
    from .plugin import DbRuntimePlanningPlugin


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
        raw_target = (
            proposal_input.get("target_name") or proposal_input.get("target")
            if proposal_input is not None
            else (intent.target.name if intent is not None else None)
        )
        target = str(raw_target or "")
        schema_evidence, catalog_schema, catalog_errors = (
            await _catalog_asset_dependency_evidence(
                runtime,
                operation,
                task,
                target=target,
            )
        )
        if catalog_errors:
            proposal, validation = _blocked_catalog_dependency(
                task_input,
                target=target,
                errors=catalog_errors,
            )
        elif proposal_input is not None:
            proposal, validation = planner.create_structured_proposal(
                proposal_input,
                source_scope=source_scope,
                owner=dict(task.input.get("owner") or {}),
                schema_evidence_id=(
                    schema_evidence.id if schema_evidence is not None else None
                ),
            )
        elif intent is not None:
            grounding_evidence_by_id = await _grounding_evidence_by_id(
                runtime,
                operation,
                intent.to_dict().get("observation"),
                schema_evidence=schema_evidence,
            )
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
                schema=catalog_schema,
                schema_evidence_id=(
                    schema_evidence.id if schema_evidence is not None else None
                ),
                intent=intent,
                grounding_evidence_by_id=grounding_evidence_by_id,
            )
        else:
            proposal, validation = _blocked_structured_input_required(task_input)
        if schema_evidence is not None:
            metadata = proposal.setdefault("metadata", {})
            metadata["schema_evidence_id"] = schema_evidence.id
            metadata["catalog_selection_evidence_ids"] = list(
                task_input.get("catalog_selection_evidence_ids") or ()
            )
            proposal.pop("proposal_fingerprint", None)
            metadata.pop("proposal_fingerprint", None)
        fingerprint = str(
            proposal.get("proposal_fingerprint") or persisted_fingerprint(proposal)
        )
        proposal.setdefault("proposal_fingerprint", fingerprint)
        proposal.setdefault("metadata", {})["proposal_fingerprint"] = fingerprint
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


async def _catalog_asset_dependency_evidence(
    runtime: Any,
    operation: Operation,
    task: Task,
    *,
    target: str,
) -> tuple[Evidence | None, dict[str, Any] | None, tuple[str, ...]]:
    dependencies = [
        item
        for item in task.dependencies
        if item.kind_value == "evidence"
        and item.evidence_kind == "schema.asset_profile"
    ]
    if len(dependencies) != 1:
        return None, None, ("monitor.catalog_dependency_required",)
    dependency = dependencies[0]
    if (
        dependency.evidence_owner != "catalog"
        or dependency.producer_task_id is None
        or dependency.producer_capability_id != "catalog.asset.inspect"
        or dependency.producer_executor_id != "catalog.inspect_asset"
        or dependency.input_hash is None
        or dependency.evidence_accepted is not True
        or dependency.operation_id != operation.id
    ):
        return None, None, ("monitor.catalog_dependency_identity_mismatch",)
    producer = await runtime.store.load_task(dependency.producer_task_id)
    if producer is None:
        return None, None, ("monitor.catalog_dependency_task_missing",)
    if (
        producer.operation_id != operation.id
        or producer.capability_id != dependency.producer_capability_id
        or producer.executor_id != dependency.producer_executor_id
        or producer.metadata.get("owner") != dependency.evidence_owner
        or producer.metadata.get("input_hash") != dependency.input_hash
    ):
        return None, None, ("monitor.catalog_dependency_identity_mismatch",)
    selected = await runtime.tasks.accepted_evidence_for_dependency(
        operation.id,
        dependency,
    )
    if selected is None:
        return None, None, ("monitor.catalog_dependency_evidence_missing",)
    if selected.payload.get("success") is False:
        return None, None, ("monitor.catalog_dependency_evidence_rejected",)

    expected_asset_ref = str(task.input.get("catalog_asset_ref") or "")
    expected_asset_name = str(task.input.get("catalog_asset_name") or "")
    expected_store = str(task.input.get("catalog_store_id") or "")
    producer_asset_ref = str(producer.input.get("asset_ref") or "")
    producer_store = str(producer.input.get("store_id") or "")
    payload = selected.payload
    asset = payload.get("asset")
    asset = asset if isinstance(asset, Mapping) else {}
    evidence_asset_name = str(asset.get("name") or payload.get("name") or "")
    evidence_asset_ref = str(
        asset.get("asset_ref") or payload.get("asset_ref") or evidence_asset_name
    )
    evidence_store = str(payload.get("store_id") or asset.get("store_id") or "")
    if (
        not target
        or not expected_asset_ref
        or not expected_asset_name
        or not expected_store
        or target != expected_asset_name
        or producer_asset_ref != expected_asset_ref
        or producer_store != expected_store
        or evidence_asset_ref != expected_asset_ref
        or evidence_asset_name != expected_asset_name
        or evidence_store != expected_store
    ):
        return None, None, ("monitor.catalog_dependency_selection_mismatch",)
    source_scopes: list[tuple[str, ...]] = []
    raw_intent = task.input.get("intent")
    intent_target = (
        raw_intent.get("target") if isinstance(raw_intent, Mapping) else None
    )
    if isinstance(intent_target, Mapping) and intent_target.get("source_scope"):
        source_scopes.append(
            tuple(str(item) for item in intent_target.get("source_scope") or ())
        )
    structured_proposal = _structured_proposal_input(dict(task.input))
    if structured_proposal is not None and structured_proposal.get("source_scope"):
        source_scopes.append(
            tuple(str(item) for item in structured_proposal.get("source_scope") or ())
        )
    if task.input.get("source_scope"):
        source_scopes.append(
            tuple(str(item) for item in task.input.get("source_scope") or ())
        )
    if not source_scopes or any(set(scope) != {target} for scope in source_scopes):
        return None, None, ("monitor.catalog_dependency_source_scope_mismatch",)
    schema = catalog_schema_from_evidence((selected,), ())
    normalized_assets = {
        (
            str(item.get("name") or ""),
            str(
                (item.get("metadata") or {}).get("catalog_asset_ref")
                if isinstance(item.get("metadata"), Mapping)
                else ""
            ),
        )
        for item in schema.get("tables") or ()
        if isinstance(item, Mapping)
    }
    if (expected_asset_name, expected_asset_ref) not in normalized_assets:
        return None, None, ("monitor.catalog_dependency_target_mismatch",)
    return selected, schema, ()


async def _grounding_evidence_by_id(
    runtime: Any,
    operation: Operation,
    observation: Any,
    *,
    schema_evidence: Evidence | None,
) -> dict[str, dict[str, Any]]:
    raw_filters = observation.get("filters") if isinstance(observation, Mapping) else ()
    evidence_ids = {
        str(evidence_id)
        for item in (raw_filters or ())
        if isinstance(item, Mapping)
        for evidence_id in item.get("evidence_ids") or ()
        if str(evidence_id)
    }
    if schema_evidence is not None and schema_evidence.id:
        evidence_ids.add(schema_evidence.id)
    return {
        item.id: {
            "id": item.id,
            "kind": item.kind,
            "owner": item.owner,
            "accepted": item.accepted,
            "payload": dict(item.payload),
        }
        for item in await runtime.store.list_evidence(operation.id)
        if item.id in evidence_ids
    }


def _blocked_catalog_dependency(
    data: dict[str, Any],
    *,
    target: str,
    errors: tuple[str, ...],
) -> tuple[dict[str, Any], DbMonitorValidation]:
    validation = DbMonitorValidation(
        accepted=False,
        errors=errors,
        diagnostics={"catalog_dependency": "rejected"},
    )
    proposal: dict[str, Any] = {
        "kind": "monitor.proposal",
        "monitor_id": _optional_string(data.get("monitor_id")) or "db_monitor",
        "name": _optional_string(data.get("name")) or "DB Monitor",
        "description": "",
        "status": "blocked",
        "target_type": "table",
        "target_name": target,
        "source_scope": list(_monitor_create_source_scope_from_input(data)),
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


def _monitor_create_source_scope_from_input(data: Mapping[str, Any]) -> tuple[str, ...]:
    intent = data.get("intent")
    target = intent.get("target") if isinstance(intent, Mapping) else None
    if isinstance(target, Mapping) and target.get("source_scope"):
        return tuple(str(item) for item in target.get("source_scope") or ())
    return tuple(str(item) for item in data.get("source_scope") or ())
