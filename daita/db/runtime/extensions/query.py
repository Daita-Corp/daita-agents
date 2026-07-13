"""Runtime-owned DB query planning extension executors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, TYPE_CHECKING

from daita.runtime import Evidence, Operation, Task

from ...evidence import load_evidence, load_evidence_refs_or_latest
from ...plan_validation import DbQueryPlanValidator
from ...planning_context import DbPlanningContextBuilder, _evidence_ref
from ...query_plan import DbQueryPlan
from ...session_context import session_scope_binding_evidence_for

if TYPE_CHECKING:
    from .plugin import DbRuntimePlanningPlugin


@dataclass(frozen=True)
class DbPlanningContextExecutor:
    """Executor that persists `planning.context` evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.planning.context.build"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.planning.context.build"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        builder = DbPlanningContextBuilder(runtime.config)
        base_request = runtime._db_request_from_operation(operation)
        prompt = task.input.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            base_request = replace(base_request, prompt=prompt)
        schema_evidence = await load_evidence(
            runtime,
            operation.id,
            task.input.get("schema_evidence_id"),
        )
        if schema_evidence is None:
            schema_evidence = await _latest_accepted_schema_evidence(
                runtime,
                operation.id,
            )
        catalog_evidence = await _load_catalog_evidence_refs_or_latest(
            runtime,
            operation.id,
            task.input.get("catalog_evidence_ids", ()),
            kinds=(
                "catalog.source_registered",
                "catalog.profile",
                "schema.asset_profile",
                "schema.search_result",
                "schema.column_value_profile",
                "schema.column_value_search_result",
                "schema.column_value_hint",
                "catalog.value_grounding.plan",
            ),
        )
        relationship_evidence = await load_evidence_refs_or_latest(
            runtime,
            operation.id,
            task.input.get("relationship_evidence_ids", ()),
            kinds=("schema.relationship_path",),
        )
        memory_recall_evidence_ids = task.input.get("memory_recall_evidence_ids", ())
        memory_recall_evidence = await load_evidence_refs_or_latest(
            runtime,
            operation.id,
            memory_recall_evidence_ids,
            kinds=("memory.semantic.recall",),
        )
        if not memory_recall_evidence_ids:
            dependency_recall = await _load_dependency_evidence(
                runtime,
                operation.id,
                task,
                "memory.semantic.recall",
            )
            if dependency_recall is not None:
                memory_recall_evidence = (dependency_recall,)
        memory_recall_diagnostics = task.input.get("memory_recall_diagnostics")
        planning_context = builder.build(
            request=base_request,
            intent=runtime._db_intent_from_operation(operation),
            operation=operation,
            schema_evidence=schema_evidence,
            catalog_evidence=catalog_evidence,
            relationship_evidence=relationship_evidence,
            memory_recall_evidence=memory_recall_evidence,
            memory_recall_diagnostics=(
                memory_recall_diagnostics
                if isinstance(memory_recall_diagnostics, dict)
                else None
            ),
            capability_summaries=_planner_capability_summaries(runtime),
            source=_runtime_source_plugin(runtime),
        )
        validation_repair = task.input.get("validation_grounding_repair")
        if isinstance(validation_repair, Mapping):
            planning_context = replace(
                planning_context,
                diagnostics={
                    **planning_context.diagnostics,
                    "validation_grounding_repair_attempted": True,
                    "validation_grounding_repair": dict(validation_repair),
                },
            )
        memory_recall_binding = task.input.get("memory_recall_binding")
        if isinstance(memory_recall_binding, Mapping):
            planning_context = replace(
                planning_context,
                diagnostics={
                    **planning_context.diagnostics,
                    "memory_recall_binding": dict(memory_recall_binding),
                },
            )
        memory_selection = builder.memory_selection_evidence_for(
            planning_context,
            task_id=task.id,
        )
        memory_selection_ref = (
            _evidence_ref(memory_selection) if memory_selection is not None else None
        )
        memory_contracts = builder.memory_contracts_evidence_for(
            planning_context,
            selection_evidence_ref=memory_selection_ref,
            task_id=task.id,
        )
        planning_context = builder.with_memory_artifact_refs(
            planning_context,
            selection_evidence=memory_selection,
            contracts_evidence=memory_contracts,
        )
        evidence: list[Evidence] = []
        if memory_selection is not None:
            evidence.append(memory_selection)
        if memory_contracts is not None:
            evidence.append(memory_contracts)
        evidence.append(builder.evidence_for(planning_context))
        return evidence


@dataclass(frozen=True)
class DbQueryPlanValidationExecutor:
    """Executor that persists `query.plan.validation` evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.query.plan.validate"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.query.plan.validate"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        plan_evidence = await load_evidence(
            runtime,
            operation.id,
            task.input.get("plan_evidence_id"),
        )
        context_evidence = await load_evidence(
            runtime,
            operation.id,
            task.input.get("planning_context_evidence_id"),
        )
        if context_evidence is None:
            context_evidence = await _load_dependency_evidence(
                runtime,
                operation.id,
                task,
                "planning.context",
            )
        if plan_evidence is None or context_evidence is None:
            raise RuntimeError("plan and planning context evidence are required")
        plan_payload = (
            plan_evidence.payload.get("structured_plan") or plan_evidence.payload
        )
        plan = DbQueryPlan.from_mapping(plan_payload)
        binding = session_scope_binding_evidence_for(
            operation,
            plan,
            context_evidence.payload,
            plan_payload=plan_payload if isinstance(plan_payload, Mapping) else None,
            task_id=task.id,
        )
        validation_context = dict(context_evidence.payload)
        if binding is not None:
            validation_context["session_scope_binding"] = dict(binding.payload)
        validation = DbQueryPlanValidator().validate(plan, validation_context)
        payload = {
            **validation.to_dict(),
            "plan_evidence_id": plan_evidence.id,
            "planning_context_evidence_id": context_evidence.id,
            "schema_fingerprint": context_evidence.payload.get("schema_fingerprint"),
            "planning_context_fingerprint": (
                context_evidence.metadata.get("payload_fingerprint")
            ),
            "session_context_fingerprint": plan_evidence.payload.get(
                "session_context_fingerprint"
            ),
            "contract_fingerprint": plan_evidence.payload.get("contract_fingerprint"),
        }
        if binding is not None:
            payload["session_scope_binding_evidence_id"] = binding.id
            payload["session_scope_binding_fingerprint"] = binding.metadata.get(
                "payload_fingerprint"
            )
        evidence = []
        if binding is not None:
            evidence.append(binding)
        evidence.append(
            Evidence(
                kind="query.plan.validation",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=validation.valid,
                payload=payload,
                metadata={
                    "plan_evidence_id": plan_evidence.id,
                    "planning_context_evidence_id": context_evidence.id,
                    "payload_fingerprint": validation.plan_fingerprint,
                    "sql_fingerprint": validation.sql_fingerprint,
                },
            )
        )
        return evidence


async def _load_dependency_evidence(
    runtime: Any,
    operation_id: str,
    task: Task,
    kind: str,
) -> Evidence | None:
    for dependency in reversed(task.dependencies):
        if dependency.kind_value != "evidence":
            continue
        if dependency.evidence_kind != kind:
            continue
        matches = [
            evidence
            for evidence in await runtime.store.list_evidence(operation_id)
            if evidence.kind == kind
            and evidence.accepted is dependency.evidence_accepted
            and (
                dependency.evidence_id is None or evidence.id == dependency.evidence_id
            )
            and (
                dependency.evidence_owner is None
                or evidence.owner == dependency.evidence_owner
            )
            and (
                dependency.producer_task_id is None
                or evidence.task_id == dependency.producer_task_id
            )
        ]
        if matches:
            return matches[-1]
    return None


async def _latest_accepted_evidence(
    runtime: Any,
    operation_id: str,
    kind: str,
) -> Evidence | None:
    matches = [
        evidence
        for evidence in await runtime.store.list_evidence(operation_id)
        if evidence.kind == kind and evidence.accepted
    ]
    return matches[-1] if matches else None


async def _latest_accepted_schema_evidence(
    runtime: Any,
    operation_id: str,
) -> Evidence | None:
    evidence = [
        item
        for item in await runtime.store.list_evidence(operation_id)
        if item.kind == "schema.asset_profile" and item.accepted
    ]
    catalog = [item for item in evidence if item.owner == "catalog"]
    if catalog:
        return catalog[-1]
    return evidence[-1] if evidence else None


async def _load_catalog_evidence_refs_or_latest(
    runtime: Any,
    operation_id: str,
    evidence_ids: Any,
    *,
    kinds: tuple[str, ...],
) -> tuple[Evidence, ...]:
    evidence = await load_evidence_refs_or_latest(
        runtime,
        operation_id,
        evidence_ids,
        kinds=kinds,
    )
    return tuple(
        item
        for item in evidence
        if item.owner == "catalog" or item.kind.startswith("catalog.")
    )


def _planner_capability_summaries(runtime: Any) -> tuple[dict[str, Any], ...]:
    interesting_owners = {"catalog", "memory", "lineage", "data_quality", "metrics"}
    interesting_prefixes = ("catalog.", "memory.", "lineage.", "quality.", "metric.")
    summaries = []
    for capability in runtime.registry.capabilities:
        if capability.owner not in interesting_owners and not capability.id.startswith(
            interesting_prefixes
        ):
            continue
        summaries.append(
            {
                "id": capability.id,
                "owner": capability.owner,
                "description": capability.description,
                "access": capability.access.value,
                "risk": capability.risk.value,
                "output_evidence": sorted(capability.output_evidence),
                "runtime_only": capability.runtime_only,
                "side_effecting": capability.side_effecting,
            }
        )
    return tuple(sorted(summaries, key=lambda item: (item["owner"], item["id"])))


def _runtime_source_plugin(runtime: Any) -> Any:
    for plugin in getattr(getattr(runtime, "config", None), "plugins", ()) or ():
        if getattr(plugin, "sql_dialect", None) and hasattr(plugin, "query"):
            return plugin
    return getattr(runtime, "source", None)
