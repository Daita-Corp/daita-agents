"""Runtime-owned DB planning extension declarations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from daita.plugins import PluginContext, PluginKind, PluginManifest
from daita.plugins.base import RuntimeExtensionPlugin
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
    Task,
    Operation,
)

from .analysis import (
    DbAnalysisBudgets,
    DbAnalysisPlan,
    analysis_metadata,
    evidence_ref,
    parse_analysis_plan_json,
    stable_fingerprint,
    validate_analysis_plan_payload,
)
from .llm_planner import DbLLMPlannerExecutor, DbLLMRepairExecutor
from .plan_validation import DbQueryPlanValidator
from .planning_context import DbPlanningContextBuilder
from .query_plan import DbQueryPlan
from .synthesis import DbAnswerSynthesisExecutor


class DbRuntimePlanningPlugin(RuntimeExtensionPlugin):
    """Built-in runtime extension for DB planning evidence and tasks."""

    manifest = PluginManifest(
        id="db_runtime",
        display_name="DB Runtime Planning",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
        provides=frozenset({"planning"}),
    )

    def __init__(self, *, llm_capable: bool = False) -> None:
        self.llm_capable = llm_capable
        self.runtime = None

    async def setup(self, context: PluginContext) -> None:
        self.runtime = context.services.require("db_runtime")

    def declare_capabilities(self) -> tuple[Capability, ...]:
        common_schema = {"type": "object"}
        capabilities = [
            Capability(
                id="db.planning.context.build",
                owner="db_runtime",
                description="Build evidence-backed context for DB planning.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query", "query.plan"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"planning.context"}),
                executor="db_runtime.planning.context.build",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.query.plan.validate",
                owner="db_runtime",
                description="Validate a structured DB query plan proposal.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query", "query.plan"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"query.plan.validation"}),
                executor="db_runtime.query.plan.validate",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.answer.synthesize",
                owner="db_runtime",
                description="Synthesize a final DB answer from accepted evidence.",
                domains=frozenset({"db"}),
                operation_types=frozenset(
                    {"data.query", "schema.query", "data.query.catalog_assisted"}
                ),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"answer.synthesis"}),
                executor="db.answer.synthesize.runtime",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.analysis.plan",
                owner="db_runtime",
                description="Plan a serial multi-step DB analysis DAG.",
                domains=frozenset({"db"}),
                operation_types=frozenset(
                    {"data.query", "data.query.catalog_assisted"}
                ),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"analysis.plan"}),
                executor="db.analysis.plan.llm",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.analysis.plan.validate",
                owner="db_runtime",
                description="Validate a multi-step DB analysis DAG.",
                domains=frozenset({"db"}),
                operation_types=frozenset(
                    {"data.query", "data.query.catalog_assisted"}
                ),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"analysis.plan.validation"}),
                executor="db.analysis.plan.validate.runtime",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.analysis.checkpoint",
                owner="db_runtime",
                description="Checkpoint accepted multi-step DB analysis evidence.",
                domains=frozenset({"db"}),
                operation_types=frozenset(
                    {"data.query", "data.query.catalog_assisted"}
                ),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"analysis.checkpoint"}),
                executor="db.analysis.checkpoint.runtime",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.analysis.summarize",
                owner="db_runtime",
                description="Synthesize an analysis answer from accepted evidence.",
                domains=frozenset({"db"}),
                operation_types=frozenset(
                    {"data.query", "data.query.catalog_assisted"}
                ),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"analysis.synthesis"}),
                executor="db.analysis.summarize.runtime",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.analysis.replan",
                owner="db_runtime",
                description="Persist an auditable multi-step analysis plan revision.",
                domains=frozenset({"db"}),
                operation_types=frozenset(
                    {"data.query", "data.query.catalog_assisted"}
                ),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"analysis.plan.revision"}),
                executor="db.analysis.replan.runtime",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
        ]
        if self.llm_capable:
            capabilities.extend(
                [
                    Capability(
                        id="db.query.plan",
                        owner="db_runtime",
                        description="Generate a structured DB query plan proposal.",
                        domains=frozenset({"db"}),
                        operation_types=frozenset({"data.query", "query.plan"}),
                        access=AccessMode.METADATA_READ,
                        risk=RiskLevel.LOW,
                        input_schema=common_schema,
                        output_evidence=frozenset({"query.plan.proposal"}),
                        executor="db_runtime.query.plan.llm",
                        runtime_only=True,
                        side_effecting=False,
                    ),
                    Capability(
                        id="db.query.repair",
                        owner="db_runtime",
                        description="Repair a failed DB query plan proposal.",
                        domains=frozenset({"db"}),
                        operation_types=frozenset({"data.query", "query.plan"}),
                        access=AccessMode.METADATA_READ,
                        risk=RiskLevel.LOW,
                        input_schema=common_schema,
                        output_evidence=frozenset(
                            {"query.plan.repair", "query.plan.proposal"}
                        ),
                        executor="db_runtime.query.repair.llm",
                        runtime_only=True,
                        side_effecting=False,
                    ),
                ]
            )
        return tuple(capabilities)

    def get_executors(self) -> tuple[Any, ...]:
        executors: list[Any] = [
            DbPlanningContextExecutor(self),
            DbQueryPlanValidationExecutor(self),
            DbAnswerSynthesisExecutor(runtime=self),
            DbAnalysisPlanExecutor(self),
            DbAnalysisPlanValidationExecutor(self),
            DbAnalysisCheckpointExecutor(self),
            DbAnalysisSummarizeExecutor(self),
            DbAnalysisReplanExecutor(self),
        ]
        if self.llm_capable:
            executors.append(DbLLMPlannerExecutor(runtime=self))
            executors.append(DbLLMRepairExecutor(runtime=self))
        return tuple(executors)

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        object_schema = {"type": "object"}
        return (
            EvidenceSchema(
                kind="planning.context",
                owner="db_runtime",
                json_schema=object_schema,
                description="Evidence-backed context used by DB planners.",
            ),
            EvidenceSchema(
                kind="query.plan.proposal",
                owner="db_runtime",
                json_schema=object_schema,
                description="Structured query plan proposal.",
            ),
            EvidenceSchema(
                kind="query.plan.validation",
                owner="db_runtime",
                json_schema=object_schema,
                description="Deterministic query plan validation result.",
            ),
            EvidenceSchema(
                kind="query.plan.repair",
                owner="db_runtime",
                json_schema=object_schema,
                description="LLM query plan repair diagnostics.",
            ),
            EvidenceSchema(
                kind="verification.result",
                owner="db_runtime",
                json_schema=object_schema,
                description="Compact runtime verification result.",
            ),
            EvidenceSchema(
                kind="answer.synthesis",
                owner="db_runtime",
                json_schema=object_schema,
                description="Final answer synthesized from accepted evidence.",
            ),
            EvidenceSchema(
                kind="analysis.plan",
                owner="db_runtime",
                json_schema=object_schema,
                description="Declarative multi-step DB analysis DAG.",
            ),
            EvidenceSchema(
                kind="analysis.plan.validation",
                owner="db_runtime",
                json_schema=object_schema,
                description="Deterministic multi-step analysis DAG validation.",
            ),
            EvidenceSchema(
                kind="analysis.checkpoint",
                owner="db_runtime",
                json_schema=object_schema,
                description="Checkpoint over accepted analysis step evidence.",
            ),
            EvidenceSchema(
                kind="analysis.synthesis",
                owner="db_runtime",
                json_schema=object_schema,
                description="Final or partial synthesis over accepted analysis evidence.",
            ),
            EvidenceSchema(
                kind="analysis.plan.revision",
                owner="db_runtime",
                json_schema=object_schema,
                description="Auditable multi-step analysis replan or repair revision.",
            ),
        )


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
        planning_context = builder.build(
            request=base_request,
            intent=runtime._db_intent_from_operation(operation),
            operation=operation,
            schema_evidence=schema_evidence,
            catalog_evidence=catalog_evidence,
            relationship_evidence=relationship_evidence,
            capability_summaries=_planner_capability_summaries(runtime),
            source=runtime.source,
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


async def _analysis_context_ref_errors(
    runtime: Any,
    operation_id: str,
    plan_payload: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    evidence_by_id = {
        item.id: item
        for item in await runtime.store.list_evidence(operation_id)
        if item.id
    }
    for step in plan_payload.get("steps") or ():
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("id") or "")
        for ref in step.get("context_evidence_refs") or ():
            if not isinstance(ref, Mapping):
                continue
            evidence_id = ref.get("id")
            if not evidence_id:
                errors.append(f"context_evidence_id_required:{step_id}")
                continue
            evidence = evidence_by_id.get(str(evidence_id))
            if evidence is None:
                errors.append(f"context_evidence_missing:{step_id}:{evidence_id}")
                continue
            if not evidence.accepted or evidence.operation_id != operation_id:
                errors.append(
                    "context_evidence_not_accepted_same_operation:"
                    f"{step_id}:{evidence_id}"
                )
                continue
            fingerprint = ref.get("payload_fingerprint")
            actual = evidence.metadata.get("payload_fingerprint") or stable_fingerprint(
                evidence.payload
            )
            if fingerprint is not None and str(fingerprint) != actual:
                errors.append(
                    f"context_evidence_fingerprint_mismatch:{step_id}:{evidence_id}"
                )
    return errors


@dataclass(frozen=True)
class DbAnalysisPlanExecutor:
    """Executor that proposes a multi-step analysis DAG."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db.analysis.plan.llm"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.analysis.plan"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        analysis_id = str(task.input.get("analysis_id") or f"analysis-{operation.id}")
        planning_context = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("planning_context_evidence_id"),
        )
        if planning_context is None:
            raise RuntimeError("planning.context evidence is required")
        metadata = analysis_metadata(
            analysis_id=analysis_id,
            step_id="analysis_plan",
            phase="plan",
        )
        if not runtime.db_llm_service.available:
            payload = {
                "analysis_id": analysis_id,
                "goal": operation.request.get("prompt"),
                "steps": [],
                "budgets": _analysis_budget_overrides(runtime),
                "diagnostics": {
                    "mode": "unavailable",
                    "failure": "db_llm_service_unavailable",
                    "planning_context_evidence_id": planning_context.id,
                },
                "clarification_question": (
                    "This request needs multi-step analysis, but no DB LLM "
                    "planner is configured."
                ),
            }
            evidence_id = _predicted_evidence_id(
                operation, task, "analysis.plan", payload
            )
            return [
                Evidence(
                    id=evidence_id,
                    kind="analysis.plan",
                    owner=self.owner,
                    operation_id=operation.id,
                    task_id=task.id,
                    accepted=False,
                    payload=payload,
                    metadata={
                        **metadata,
                        "analysis_plan_evidence_id": evidence_id,
                    },
                )
            ]
        response = await runtime.db_llm_service.generate_json(
            _analysis_plan_messages(
                planning_context.payload,
                analysis_id=analysis_id,
                budgets=_analysis_budget_overrides(runtime),
            )
        )
        try:
            plan = parse_analysis_plan_json(response.content)
            payload = {
                **plan.to_dict(),
                "analysis_id": plan.analysis_id or analysis_id,
                "diagnostics": {
                    **plan.diagnostics,
                    "mode": "llm",
                    "planning_context_evidence_id": planning_context.id,
                    "model": response.diagnostics.get("model"),
                    "provider": response.diagnostics.get("provider"),
                    "llm": response.diagnostics,
                },
                "raw_model_response": response.content,
            }
            accepted = True
        except Exception as exc:
            payload = {
                "analysis_id": analysis_id,
                "goal": operation.request.get("prompt"),
                "steps": [],
                "budgets": _analysis_budget_overrides(runtime),
                "diagnostics": {
                    "mode": "llm",
                    "failure": "analysis_plan_json_invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "planning_context_evidence_id": planning_context.id,
                    "llm": response.diagnostics,
                },
                "raw_model_response": response.content,
                "clarification_question": (
                    "I could not parse a safe multi-step analysis plan. Please "
                    "clarify the analysis steps or narrow the question."
                ),
            }
            accepted = False
        evidence_id = _predicted_evidence_id(operation, task, "analysis.plan", payload)
        return [
            Evidence(
                id=evidence_id,
                kind="analysis.plan",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=accepted,
                payload=payload,
                metadata={
                    **analysis_metadata(
                        analysis_id=str(payload.get("analysis_id") or analysis_id),
                        step_id="analysis_plan",
                        plan_evidence_id=evidence_id,
                        phase="plan",
                    ),
                    "payload_fingerprint": stable_fingerprint(payload),
                },
            )
        ]


@dataclass(frozen=True)
class DbAnalysisPlanValidationExecutor:
    """Executor that validates an analysis DAG before step materialization."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db.analysis.plan.validate.runtime"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.analysis.plan.validate"})

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
            task.input.get("analysis_plan_evidence_id"),
        )
        if plan_evidence is None:
            raise RuntimeError("analysis.plan evidence is required")
        validation = validate_analysis_plan_payload(
            plan_evidence.payload,
            plan_evidence=plan_evidence,
            registered_capabilities={
                (capability.owner, capability.id): capability
                for capability in runtime.registry.capabilities
            },
        )
        payload = validation.to_dict()
        context_errors = await _analysis_context_ref_errors(
            runtime,
            operation.id,
            plan_evidence.payload,
        )
        if context_errors:
            payload = {
                **payload,
                "valid": False,
                "errors": [
                    *list(payload.get("errors") or ()),
                    *context_errors,
                ],
            }
        if not plan_evidence.accepted:
            errors = list(payload.get("errors") or ())
            errors.append(
                str(
                    (plan_evidence.payload.get("diagnostics") or {}).get("failure")
                    or plan_evidence.payload.get("failure")
                    or "analysis_plan_not_accepted"
                )
            )
            payload = {
                **payload,
                "valid": False,
                "errors": list(dict.fromkeys(errors)),
            }
        metadata = analysis_metadata(
            analysis_id=str(
                validation.analysis_id
                or plan_evidence.metadata.get("analysis_id")
                or ""
            ),
            step_id="analysis_plan_validation",
            plan_evidence_id=plan_evidence.id,
            phase="validation",
        )
        return [
            Evidence(
                kind="analysis.plan.validation",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=bool(payload.get("valid")),
                payload=payload,
                metadata={
                    **metadata,
                    "payload_fingerprint": stable_fingerprint(payload),
                    "analysis_plan_fingerprint": validation.plan_fingerprint,
                },
            )
        ]


@dataclass(frozen=True)
class DbAnalysisCheckpointExecutor:
    """Executor that emits checkpoint evidence from accepted dependencies."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db.analysis.checkpoint.runtime"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.analysis.checkpoint"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        cited = await _accepted_dependency_evidence(runtime, task, operation)
        rows = sum(_row_count(item) for item in cited if item.kind == "query.result")
        analysis_id = str(
            task.metadata.get("analysis_id") or task.input.get("analysis_id") or ""
        )
        payload = {
            "analysis_id": analysis_id,
            "completed_step_evidence_refs": [evidence_ref(item) for item in cited],
            "warnings": list(task.input.get("warnings") or ()),
            "budget_usage": {
                "cited_evidence_count": len(cited),
                "query_result_rows": rows,
                **dict(task.input.get("budget_usage") or {}),
            },
            "remaining_step_ids": list(task.input.get("remaining_step_ids") or ()),
            "diagnostics": dict(task.input.get("diagnostics") or {}),
        }
        return [
            Evidence(
                kind="analysis.checkpoint",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload,
                metadata={
                    **analysis_metadata(
                        analysis_id=analysis_id,
                        step_id=str(
                            task.metadata.get("analysis_step_id")
                            or task.input.get("analysis_step_id")
                            or "checkpoint"
                        ),
                        step_kind="checkpoint",
                        plan_evidence_id=task.metadata.get("analysis_plan_evidence_id"),
                    ),
                    "payload_fingerprint": stable_fingerprint(payload),
                },
            )
        ]


@dataclass(frozen=True)
class DbAnalysisSummarizeExecutor:
    """Executor that synthesizes final analysis evidence from accepted refs."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db.analysis.summarize.runtime"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.analysis.summarize"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        cited = await _accepted_dependency_evidence(runtime, task, operation)
        analysis_id = str(
            task.metadata.get("analysis_id") or task.input.get("analysis_id") or ""
        )
        payload = None
        fallback_reason = None
        if runtime.db_llm_service.available:
            try:
                generator = getattr(
                    runtime.db_llm_service,
                    "generate_synthesis_json",
                    runtime.db_llm_service.generate_json,
                )
                response = await generator(
                    _analysis_synthesis_messages(operation, cited)
                )
                payload = _validate_analysis_synthesis_response(
                    response.content,
                    cited,
                    diagnostics=response.diagnostics,
                )
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}:{exc}"
        else:
            fallback_reason = "db_llm_service_unavailable"
        if payload is None:
            query_count = sum(1 for item in cited if item.kind == "query.result")
            payload = {
                "answer": (
                    f"Completed analysis from {query_count} query result"
                    f"{'' if query_count == 1 else 's'}."
                ),
                "reasoning_summary": "Deterministic analysis summary from accepted evidence.",
                "cited_evidence_refs": [evidence_ref(item) for item in cited],
                "assumptions": [],
                "limitations": [],
                "warnings": [],
                "sufficiency": "answered" if cited else "insufficient_evidence",
                "confidence": 0.8 if cited else 0.0,
                "diagnostics": {
                    "mode": "deterministic_fallback",
                    "fallback_reason": fallback_reason,
                },
            }
        payload = {
            **payload,
            "analysis_id": analysis_id,
            "partial": bool(task.input.get("partial")),
            "pause_reason": task.input.get("pause_reason"),
            "remaining_step_ids": list(task.input.get("remaining_step_ids") or ()),
            "diagnostics": {
                **dict(payload.get("diagnostics") or {}),
                "dependency_evidence_refs": [evidence_ref(item) for item in cited],
            },
        }
        return [
            Evidence(
                kind="analysis.synthesis",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload,
                metadata={
                    **analysis_metadata(
                        analysis_id=analysis_id,
                        step_id=str(
                            task.metadata.get("analysis_step_id")
                            or task.input.get("analysis_step_id")
                            or "synthesis"
                        ),
                        step_kind="synthesis",
                        plan_evidence_id=task.metadata.get("analysis_plan_evidence_id"),
                    ),
                    "cited_evidence_refs": [item.id for item in cited if item.id],
                    "partial": bool(task.input.get("partial")),
                    "payload_fingerprint": stable_fingerprint(payload),
                },
            )
        ]


@dataclass(frozen=True)
class DbAnalysisReplanExecutor:
    """Executor that persists typed analysis revision evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db.analysis.replan.runtime"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.analysis.replan"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        parent = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("parent_plan_evidence_id"),
        )
        trigger_ids = [
            str(item) for item in task.input.get("trigger_evidence_ids") or ()
        ]
        triggers = tuple(
            item
            for item in [
                await _load_evidence(runtime, operation.id, evidence_id)
                for evidence_id in trigger_ids
            ]
            if item is not None
        )
        errors: list[str] = []
        if parent is None:
            errors.append("parent_plan_evidence_missing")
            analysis_id = str(
                task.input.get("analysis_id") or f"analysis-{operation.id}"
            )
            parent_ref = None
            parent_steps: tuple[Any, ...] = ()
        else:
            analysis_id = str(
                parent.payload.get("analysis_id")
                or parent.metadata.get("analysis_id")
                or task.input.get("analysis_id")
                or f"analysis-{operation.id}"
            )
            parent_ref = evidence_ref(parent)
            if not parent.accepted or parent.operation_id != operation.id:
                errors.append("parent_plan_evidence_not_accepted_same_operation")
            try:
                parent_steps = DbAnalysisPlan.from_mapping(parent.payload).steps
            except Exception as exc:
                errors.append(f"parent_plan_invalid:{type(exc).__name__}")
                parent_steps = ()
        if len(triggers) != len(trigger_ids):
            errors.append("trigger_evidence_missing")
        if any(
            not item.accepted or item.operation_id != operation.id for item in triggers
        ):
            errors.append("trigger_evidence_not_accepted_same_operation")
        existing_revisions = [
            item
            for item in await runtime.store.list_evidence(operation.id)
            if item.kind == "analysis.plan.revision"
            and item.accepted
            and item.payload.get("parent_plan_evidence_id")
            == (parent.id if parent is not None else None)
        ]
        failed_step_ids = [
            str(item) for item in task.input.get("failed_step_ids") or ()
        ]
        replacement_steps = [
            dict(item)
            for item in task.input.get("replacement_steps") or ()
            if isinstance(item, Mapping)
        ]
        unchanged_step_ids = [
            step.id for step in parent_steps if step.id not in set(failed_step_ids)
        ]
        prior_sql_fingerprints = sorted(
            {
                str(item.metadata.get("sql_fingerprint"))
                for item in await runtime.store.list_evidence(operation.id)
                if item.metadata.get("sql_fingerprint")
            }
        )
        payload = {
            "analysis_id": analysis_id,
            "parent_plan_evidence_id": parent.id if parent is not None else None,
            "parent_plan_evidence_ref": parent_ref,
            "revision_number": len(existing_revisions) + 1,
            "trigger_evidence_refs": [evidence_ref(item) for item in triggers],
            "failed_or_invalidated_step_ids": failed_step_ids,
            "replacement_steps": replacement_steps,
            "unchanged_step_ids": unchanged_step_ids,
            "budget_usage": dict(task.input.get("budget_usage") or {}),
            "retry_rationale": str(task.input.get("retry_rationale") or ""),
            "prior_sql_fingerprints": prior_sql_fingerprints,
            "diagnostics": {
                "mode": "runtime",
                "errors": errors,
            },
        }
        return [
            Evidence(
                kind="analysis.plan.revision",
                owner=self.owner,
                operation_id=operation.id,
                task_id=task.id,
                accepted=not errors,
                payload=payload,
                metadata={
                    **analysis_metadata(
                        analysis_id=analysis_id,
                        step_id="analysis_replan",
                        phase="replan",
                        plan_evidence_id=parent.id if parent is not None else None,
                    ),
                    "payload_fingerprint": stable_fingerprint(payload),
                    "analysis_revision_number": payload["revision_number"],
                },
            )
        ]


async def _accepted_dependency_evidence(
    runtime: Any,
    task: Task,
    operation: Operation,
) -> tuple[Evidence, ...]:
    evidence = []
    for dependency in task.dependencies:
        if dependency.kind.value != "evidence":
            continue
        item = await runtime._accepted_evidence_for_dependency(operation.id, dependency)
        if item is not None and item.accepted and item.operation_id == operation.id:
            evidence.append(item)
    return tuple(evidence)


def _analysis_plan_messages(
    context_payload: dict[str, Any],
    *,
    analysis_id: str,
    budgets: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Plan a serial database analysis DAG inside a governed runtime. "
                "Return strict JSON only. Step kinds are only query, checkpoint, "
                "and synthesis. Never execute SQL, call tools, or request connector "
                "access."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{context_payload.get('rendered_context', '')}\n\n"
                "Source evidence refs: "
                f"{json.dumps(context_payload.get('source_evidence_refs') or [], sort_keys=True, default=str)}\n"
                "Capability summaries: "
                f"{json.dumps(context_payload.get('capability_summaries') or [], sort_keys=True, default=str)}\n"
                "Return JSON with analysis_id, goal, steps, budgets, diagnostics. "
                f"Use analysis_id {analysis_id!r}. Budgets: "
                f"{budgets}. Each step needs id, kind, purpose, depends_on, "
                "input_refs, expected_evidence, and budgets."
            ),
        },
    ]


def _analysis_synthesis_messages(
    operation: Operation,
    evidence: tuple[Evidence, ...],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Synthesize a database analysis answer from accepted evidence only. "
                "Return strict JSON. Never execute SQL, call tools, or ask for DB work."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt": operation.request.get("prompt"),
                    "evidence": [
                        {
                            **evidence_ref(item),
                            "payload": _compact_payload(item.payload),
                        }
                        for item in evidence
                    ],
                    "schema": {
                        "answer": "string",
                        "reasoning_summary": "string",
                        "cited_evidence_refs": [
                            {
                                "id": "accepted evidence id",
                                "kind": "evidence kind",
                                "purpose": "why cited",
                            }
                        ],
                        "assumptions": ["string"],
                        "limitations": ["string"],
                        "warnings": ["string"],
                        "sufficiency": "answered|partial|needs_clarification|insufficient_evidence",
                        "confidence": "number 0..1",
                    },
                },
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _validate_analysis_synthesis_response(
    content: str,
    evidence: tuple[Evidence, ...],
    *,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    import json
    import re

    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    parsed = json.loads(match.group(1).strip() if match else stripped)
    if not isinstance(parsed, dict):
        raise ValueError("analysis_synthesis_json_not_object")
    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        raise ValueError("analysis_synthesis_answer_empty")
    accepted_ids = {item.id for item in evidence if item.id}
    citations = parsed.get("cited_evidence_refs")
    if not isinstance(citations, list) or not citations:
        raise ValueError("analysis_synthesis_citations_missing")
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("id") not in accepted_ids:
            raise ValueError("analysis_synthesis_unknown_citation")
    confidence = parsed.get("confidence", 0.8)
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("analysis_synthesis_confidence_invalid")
    if _requests_db_work(parsed):
        raise ValueError("analysis_synthesis_requests_db_work")
    return {
        "answer": answer,
        "reasoning_summary": str(parsed.get("reasoning_summary") or ""),
        "cited_evidence_refs": [dict(item) for item in citations],
        "assumptions": [str(item) for item in parsed.get("assumptions") or ()],
        "limitations": [str(item) for item in parsed.get("limitations") or ()],
        "warnings": [str(item) for item in parsed.get("warnings") or ()],
        "sufficiency": str(parsed.get("sufficiency") or "answered"),
        "confidence": float(confidence),
        "diagnostics": {"mode": "llm", **diagnostics},
    }


def _analysis_budget_overrides(runtime: Any) -> dict[str, Any]:
    options = getattr(getattr(runtime, "config", None), "metadata", {}).get(
        "from_db_options"
    )
    if not isinstance(options, dict):
        options = {}
    defaults = DbAnalysisBudgets().to_dict()
    return {
        key: int(options.get(f"analysis_{key}", value))
        for key, value in defaults.items()
    }


def _row_count(evidence: Evidence) -> int:
    rows = evidence.payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return int(evidence.payload.get("total_rows") or 0)


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {key: payload.get(key) for key in ("rows", "total_rows", "sql", "valid")}
    if isinstance(compact.get("rows"), list):
        compact["rows"] = compact["rows"][:10]
    return {key: value for key, value in compact.items() if value is not None}


def _requests_db_work(parsed: dict[str, Any]) -> bool:
    rendered = json.dumps(parsed, sort_keys=True, default=str).lower()
    blocked = (
        "execute sql",
        "run sql",
        "query the database",
        "call connector",
        "tool call",
    )
    return any(phrase in rendered for phrase in blocked)


def _predicted_evidence_id(
    operation: Operation,
    task: Task,
    kind: str,
    payload: dict[str, Any],
) -> str:
    return f"evidence-{stable_fingerprint({'operation_id': operation.id, 'task_id': task.id, 'kind': kind, 'payload': payload})}"
