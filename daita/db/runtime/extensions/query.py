"""Runtime-owned DB query planning extension executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from daita.runtime import Evidence, Operation, Task

from ...plan_validation import DbQueryPlanValidator
from ...planning_context import DbPlanningContextBuilder
from ...query_plan import DbQueryPlan


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
            from dataclasses import replace

            base_request = replace(base_request, prompt=prompt)
        schema_evidence = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("schema_evidence_id"),
        )
        catalog_evidence = tuple(
            item
            for item in [
                await _load_evidence(runtime, operation.id, evidence_id)
                for evidence_id in task.input.get("catalog_evidence_ids", ())
            ]
            if item is not None
        )
        relationship_evidence = tuple(
            item
            for item in [
                await _load_evidence(runtime, operation.id, evidence_id)
                for evidence_id in task.input.get("relationship_evidence_ids", ())
            ]
            if item is not None
        )
        memory_recall_evidence = tuple(
            item
            for item in [
                await _load_evidence(runtime, operation.id, evidence_id)
                for evidence_id in task.input.get("memory_recall_evidence_ids", ())
            ]
            if item is not None
        )
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
        return [builder.evidence_for(planning_context)]


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
        plan_evidence = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("plan_evidence_id"),
        )
        context_evidence = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("planning_context_evidence_id"),
        )
        if plan_evidence is None or context_evidence is None:
            raise RuntimeError("plan and planning context evidence are required")
        plan = DbQueryPlan.from_mapping(
            plan_evidence.payload.get("structured_plan") or plan_evidence.payload
        )
        validation = DbQueryPlanValidator().validate(plan, context_evidence.payload)
        payload = {
            **validation.to_dict(),
            "plan_evidence_id": plan_evidence.id,
            "planning_context_evidence_id": context_evidence.id,
            "schema_fingerprint": context_evidence.payload.get("schema_fingerprint"),
        }
        return [
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
