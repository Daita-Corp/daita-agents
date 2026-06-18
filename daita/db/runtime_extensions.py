"""Runtime-owned DB planning extension declarations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping
from uuid import uuid4

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
from .monitor_commands import (
    DbMonitorCommand,
    DbMonitorPlanner,
    DbMonitorValidation,
    _monitor_from_proposal,
    _target_resource_from_prompt,
)
from .monitors import (
    DbMonitor,
    DbMonitorMutation,
    DbMonitorState,
    monitor_with_updates,
)
from .plan_validation import DbQueryPlanValidator
from .planning_context import DbPlanningContextBuilder
from .query_plan import DbQueryPlan
from .query_planning import DbQueryPlanner
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
                id="db.query.prepare_read",
                owner="db_runtime",
                description="Prepare and compactly validate a deterministic DB read plan.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query", "query.plan"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset(
                    {"query.plan.proposal", "query.plan.validation"}
                ),
                executor="db_runtime.query.prepare_read",
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
            Capability(
                id="db.monitor.plan_create",
                owner="db_runtime",
                description="Plan an executable DB monitor proposal from a monitor create prompt.",
                domains=frozenset({"db", "monitor"}),
                operation_types=frozenset({"monitor.create"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"monitor.proposal"}),
                executor="db_runtime.monitor.plan_create",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.monitor.commit_create",
                owner="db_runtime",
                description="Commit an approved executable DB monitor proposal.",
                domains=frozenset({"db", "monitor"}),
                operation_types=frozenset({"monitor.create"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.MEDIUM,
                input_schema=common_schema,
                output_evidence=frozenset({"monitor.definition"}),
                executor="db_runtime.monitor.commit_create",
                runtime_only=True,
                side_effecting=True,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.monitor.plan_lifecycle",
                owner="db_runtime",
                description="Plan an auditable DB monitor lifecycle proposal.",
                domains=frozenset({"db", "monitor"}),
                operation_types=frozenset(
                    {
                        "monitor.update",
                        "monitor.pause",
                        "monitor.resume",
                        "monitor.delete",
                        "monitor.disable",
                    }
                ),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"monitor.proposal"}),
                executor="db_runtime.monitor.plan_lifecycle",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.monitor.commit_lifecycle",
                owner="db_runtime",
                description="Commit an approved DB monitor lifecycle proposal.",
                domains=frozenset({"db", "monitor"}),
                operation_types=frozenset(
                    {
                        "monitor.update",
                        "monitor.pause",
                        "monitor.resume",
                        "monitor.delete",
                        "monitor.disable",
                    }
                ),
                access=AccessMode.WRITE,
                risk=RiskLevel.MEDIUM,
                input_schema=common_schema,
                output_evidence=frozenset(
                    {
                        "monitor.state_update",
                        "monitor.definition",
                        "monitor.deleted",
                        "monitor.disabled",
                    }
                ),
                executor="db_runtime.monitor.commit_lifecycle",
                runtime_only=True,
                side_effecting=True,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="monitor.delivery.local",
                owner="db_runtime",
                description="Record a local monitor notification delivery for CLI/runtime use.",
                domains=frozenset({"monitor", "local"}),
                operation_types=frozenset({"monitor.delivery"}),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema={
                    "type": "object",
                    "required": ["delivery_kind", "target", "payload_source"],
                    "properties": {
                        "delivery_kind": {"type": "string"},
                        "target": {"type": "object"},
                        "format": {"type": "string"},
                        "subject": {"type": "string"},
                        "idempotency_key": {"type": "string"},
                        "payload_source": {"type": "object"},
                    },
                },
                output_evidence=frozenset({"local.notification.delivery"}),
                executor="db_runtime.monitor.delivery.local",
                runtime_only=True,
                side_effecting=True,
                replay_safe=True,
                idempotent=True,
                metadata={
                    "monitor_roles": ["delivery"],
                    "delivery_kind": "local",
                    "accepted_payload_kinds": [
                        "monitor.report",
                        "monitor.action_result",
                        "analysis.synthesis",
                    ],
                    "accepted_formats": ["markdown", "plain", "text"],
                    "accepted_target_types": [
                        "runtime_console",
                        "terminal",
                        "stdout",
                        "callback",
                    ],
                    "supports_idempotency_key": True,
                },
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
            DbQueryPrepareReadExecutor(self),
            DbQueryPlanValidationExecutor(self),
            DbAnswerSynthesisExecutor(runtime=self),
            DbAnalysisPlanExecutor(self),
            DbAnalysisPlanValidationExecutor(self),
            DbAnalysisCheckpointExecutor(self),
            DbAnalysisSummarizeExecutor(self),
            DbAnalysisReplanExecutor(self),
            DbMonitorPlanCreateExecutor(self),
            DbMonitorCommitCreateExecutor(self),
            DbMonitorPlanLifecycleExecutor(self),
            DbMonitorCommitLifecycleExecutor(self),
            DbMonitorLocalDeliveryExecutor(self),
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
                kind="query.zero_row_diagnosis",
                owner="db_runtime",
                json_schema=object_schema,
                description="Bounded diagnosis for a read query that returned no rows.",
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
            EvidenceSchema(
                kind="monitor.proposal",
                owner="db_runtime",
                json_schema=object_schema,
                description="Executable monitor proposal produced by DB monitor planning.",
            ),
            EvidenceSchema(
                kind="monitor.definition",
                owner="db_runtime",
                json_schema=object_schema,
                description="Accepted DB monitor definition committed from a proposal.",
            ),
            EvidenceSchema(
                kind="monitor.action_plan",
                owner="db_runtime",
                json_schema=object_schema,
                description="Normalized monitor action plan selected for a triggered run.",
            ),
            EvidenceSchema(
                kind="monitor.report",
                owner="db_runtime",
                json_schema=object_schema,
                description="Durable monitor notification or report payload descriptor.",
            ),
            EvidenceSchema(
                kind="monitor.action_result",
                owner="db_runtime",
                json_schema=object_schema,
                description="Monitor action execution result linked to a trigger run.",
            ),
            EvidenceSchema(
                kind="monitor.delivery_plan",
                owner="db_runtime",
                json_schema=object_schema,
                description="Resolved notification delivery capability plan.",
            ),
            EvidenceSchema(
                kind="monitor.delivery_result",
                owner="db_runtime",
                json_schema=object_schema,
                description="Durable notification delivery status for a monitor run.",
            ),
            EvidenceSchema(
                kind="local.notification.delivery",
                owner="db_runtime",
                json_schema=object_schema,
                description="Local runtime notification delivery result.",
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
class DbQueryPrepareReadExecutor:
    """Executor that prepares deterministic read evidence without full context."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.query.prepare_read"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.query.prepare_read"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
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
        planning_context_evidence = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("planning_context_evidence_id"),
        )
        schema = dict(schema_evidence.payload) if schema_evidence is not None else {}
        plan = DbQueryPlanner().plan_read_query(
            base_request,
            runtime._db_intent_from_operation(operation),
            operation,
            schema,
            planning_context=(
                dict(planning_context_evidence.payload)
                if planning_context_evidence is not None
                else None
            ),
        )
        plan_payload = dict(plan.evidence.payload)
        plan_evidence_id = _predicted_evidence_id(
            operation,
            task,
            "query.plan.proposal",
            plan_payload,
        )
        plan_evidence = Evidence(
            id=plan_evidence_id,
            kind="query.plan.proposal",
            owner="db_runtime",
            operation_id=operation.id,
            task_id=task.id,
            payload=plan_payload,
            metadata={
                **plan.evidence.metadata,
                "prepare_read": True,
                "payload_fingerprint": stable_fingerprint(plan_payload),
            },
        )
        compact_context = (
            planning_context_evidence.payload
            if planning_context_evidence is not None
            else _compact_prepare_context(
                runtime=runtime,
                operation=operation,
                schema_evidence=schema_evidence,
                schema=schema,
            )
        )
        structured = DbQueryPlan.from_mapping(
            plan_evidence.payload.get("structured_plan") or plan_evidence.payload
        )
        validation = DbQueryPlanValidator().validate(structured, compact_context)
        validation_evidence = Evidence(
            kind="query.plan.validation",
            owner="db_runtime",
            operation_id=operation.id,
            task_id=task.id,
            accepted=validation.valid,
            payload={
                **validation.to_dict(),
                "plan_evidence_id": plan_evidence_id,
                "planning_context_evidence_id": (
                    planning_context_evidence.id
                    if planning_context_evidence is not None
                    else None
                ),
                "schema_fingerprint": compact_context.get("schema_fingerprint"),
                "prepare_read": True,
            },
            metadata={
                "prepare_read": True,
                "payload_fingerprint": validation.plan_fingerprint,
                "sql_fingerprint": validation.sql_fingerprint,
            },
        )
        return [plan_evidence, validation_evidence]


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
        command = DbMonitorCommand(
            **dict(task.input.get("command") or operation.request.get("command") or {})
        )
        target = _target_resource_from_prompt(command.prompt)
        schema_evidence = await _inspect_monitor_target_schema(
            runtime,
            operation,
            target=target,
        )
        proposal, validation = DbMonitorPlanner(
            registry=runtime.registry,
            limits=runtime.config.limits.to_dict(),
        ).create_proposal(
            command,
            source_scope=tuple(
                str(item) for item in task.input.get("source_scope") or ()
            ),
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
        return [
            Evidence(
                kind="monitor.definition",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload={
                    "monitor": monitor.to_dict(),
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


@dataclass(frozen=True)
class DbMonitorPlanLifecycleExecutor:
    """Executor that persists accepted or blocked monitor lifecycle proposals."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.plan_lifecycle"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.plan_lifecycle"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        action = _monitor_lifecycle_action(
            str(task.input.get("action") or operation.operation_type)
        )
        monitor_id = str(
            task.input.get("monitor_id") or operation.metadata.get("monitor_id") or ""
        )
        if not monitor_id:
            raise RuntimeError("monitor lifecycle proposal requires monitor_id")

        monitor = await runtime.monitor_store.load_monitor(monitor_id)
        errors: list[str] = []
        before = monitor.to_dict() if monitor is not None else None
        after = None
        patch = dict(task.input.get("patch") or {})
        paused_until = task.input.get("paused_until")
        if monitor is None:
            errors.append("monitor.lifecycle:monitor_not_found")
        else:
            try:
                updated = _monitor_after_lifecycle_action(
                    monitor,
                    action=action,
                    patch=patch,
                    paused_until=paused_until,
                )
                after = None if updated is None else updated.to_dict()
                reason = (
                    None
                    if updated is None
                    else _non_executable_active_monitor_reason(
                        updated.observation_plan,
                    )
                )
                if updated is not None and updated.status == "active" and reason:
                    errors.append(f"monitor.lifecycle:{reason}")
            except ValueError as exc:
                errors.append(f"monitor.lifecycle:{str(exc)}")

        validation = DbMonitorValidation(
            accepted=not errors,
            errors=tuple(errors),
            diagnostics={
                "action": action,
                "operation_type": operation.operation_type,
            },
        )
        proposal = {
            "kind": "monitor.proposal",
            "operation_type": operation.operation_type,
            "action": action,
            "monitor_id": monitor_id,
            "before": before,
            "after": after,
            "patch": patch,
            "paused_until": paused_until,
            "validation": validation.to_dict(),
        }
        fingerprint = stable_fingerprint(proposal)
        proposal["proposal_fingerprint"] = fingerprint
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
                    "monitor_id": monitor_id,
                    "action": action,
                    "validation_accepted": validation.accepted,
                },
            )
        ]


@dataclass(frozen=True)
class DbMonitorCommitLifecycleExecutor:
    """Executor that idempotently commits a monitor lifecycle proposal."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.commit_lifecycle"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"db.monitor.commit_lifecycle"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        proposal_evidence = await _load_evidence(
            runtime,
            operation.id,
            task.input.get("proposal_evidence_id"),
        )
        if proposal_evidence is None:
            raise RuntimeError("monitor lifecycle proposal evidence is required")
        if not proposal_evidence.accepted:
            raise RuntimeError("monitor lifecycle proposal evidence was not accepted")
        proposal = dict(proposal_evidence.payload)
        expected_fingerprint = task.input.get("proposal_fingerprint")
        actual_fingerprint = proposal.get("proposal_fingerprint") or stable_fingerprint(
            proposal
        )
        if expected_fingerprint and expected_fingerprint != actual_fingerprint:
            raise RuntimeError("monitor lifecycle proposal fingerprint mismatch")

        action = _monitor_lifecycle_action(str(proposal.get("action") or "update"))
        monitor_id = str(proposal.get("monitor_id") or "")
        before_payload = proposal.get("before")
        after_payload = proposal.get("after")
        before = (
            DbMonitor.from_dict(before_payload)
            if isinstance(before_payload, Mapping)
            else None
        )
        after = (
            DbMonitor.from_dict(after_payload)
            if isinstance(after_payload, Mapping)
            else None
        )
        state = await runtime.monitor_store.load_monitor_state(monitor_id)
        state = state or DbMonitorState(monitor_id=monitor_id)
        existing = await runtime.monitor_store.load_monitor(monitor_id)
        idempotent_existing = _monitor_lifecycle_already_committed(
            existing,
            after=after,
            action=action,
        )
        if not idempotent_existing:
            await runtime.monitor_store.commit_monitor_mutation(
                DbMonitorMutation(
                    action=_monitor_lifecycle_mutation_action(action),
                    operation=operation,
                    monitor_before=before,
                    monitor_after=after,
                    state_after=(
                        None
                        if action == "delete"
                        else DbMonitorState.from_dict(
                            {
                                **state.to_dict(),
                                "last_operation_id": operation.id,
                                "last_management_operation_id": operation.id,
                                "paused_until": (
                                    proposal.get("paused_until")
                                    if action == "pause"
                                    else None
                                    if action == "resume"
                                    else state.paused_until
                                ),
                            }
                        )
                    ),
                )
            )

        evidence_kind = _monitor_lifecycle_commit_evidence_kind(action)
        payload = {
            "monitor_id": monitor_id,
            "action": action,
            "before": before_payload,
            "after": after_payload,
            "patch": dict(proposal.get("patch") or {}),
            "proposal_evidence_id": proposal_evidence.id,
            "proposal_fingerprint": actual_fingerprint,
            "idempotent_existing": idempotent_existing,
        }
        if action == "delete":
            payload["monitor"] = before_payload
        elif after_payload is not None:
            payload["monitor"] = after_payload
        return [
            Evidence(
                kind=evidence_kind,
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload,
                metadata={
                    "payload_fingerprint": actual_fingerprint,
                    "proposal_evidence_id": proposal_evidence.id,
                    "monitor_id": monitor_id,
                    "action": action,
                    "idempotent_existing": idempotent_existing,
                },
            )
        ]


@dataclass(frozen=True)
class DbMonitorLocalDeliveryExecutor:
    """Executor that records local monitor notification delivery evidence."""

    plugin: DbRuntimePlanningPlugin
    id: str = "db_runtime.monitor.delivery.local"
    owner: str = "db_runtime"
    capability_ids: frozenset[str] = frozenset({"monitor.delivery.local"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        runtime = self.plugin.runtime
        payload_source = dict(task.input.get("payload_source") or {})
        report = await _load_evidence(
            runtime,
            operation.id,
            payload_source.get("report_evidence_id"),
        )
        if report is None:
            raise RuntimeError("monitor report evidence is required for local delivery")
        delivery_kind = str(task.input.get("delivery_kind") or "")
        if delivery_kind != "local":
            raise RuntimeError("local delivery executor only supports local delivery")
        target = dict(task.input.get("target") or {})
        target_type = str(target.get("type") or target.get("channel") or "")
        if target_type not in {"runtime_console", "terminal", "stdout", "callback"}:
            raise RuntimeError("unsupported_local_delivery_target")
        payload = {
            "monitor_id": task.metadata.get("monitor_id"),
            "monitor_run_id": task.metadata.get("monitor_run_id"),
            "tick_operation_id": task.metadata.get("tick_operation_id"),
            "delivery_operation_id": operation.id,
            "delivery_kind": delivery_kind,
            "target": target,
            "target_channel": target.get("channel") or target_type,
            "format": task.input.get("format"),
            "subject": task.input.get("subject"),
            "status": "delivered",
            "idempotency_key": (
                task.input.get("idempotency_key")
                or task.metadata.get("idempotency_key")
            ),
            "report_evidence_id": report.id,
            "report_fingerprint": payload_source.get("report_fingerprint"),
            "action_plan_fingerprint": payload_source.get("action_plan_fingerprint"),
            "source_evidence_refs": list(
                payload_source.get("source_evidence_refs") or ()
            ),
        }
        return [
            Evidence(
                kind="local.notification.delivery",
                owner="db_runtime",
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload=payload,
                metadata={
                    "monitor_id": payload["monitor_id"],
                    "monitor_run_id": payload["monitor_run_id"],
                    "tick_operation_id": payload["tick_operation_id"],
                    "monitor_delivery_kind": delivery_kind,
                    "monitor_report_fingerprint": payload["report_fingerprint"],
                    "monitor_action_fingerprint": payload["action_plan_fingerprint"],
                    "idempotency_key": payload["idempotency_key"],
                    "payload_fingerprint": stable_fingerprint(payload),
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


def _monitor_lifecycle_action(value: str) -> str:
    normalized = value.removeprefix("monitor.").replace("_", ".").lower()
    if normalized in {"update", "pause", "resume", "delete", "disable"}:
        return normalized
    if normalized == "disabled":
        return "disable"
    raise ValueError(f"unsupported monitor lifecycle action: {value!r}")


def _monitor_after_lifecycle_action(
    monitor: DbMonitor,
    *,
    action: str,
    patch: dict[str, Any],
    paused_until: Any = None,
) -> DbMonitor | None:
    if action == "delete":
        return None
    if action == "update":
        return monitor_with_updates(monitor, patch)
    if action == "pause":
        return monitor_with_updates(monitor, {"status": "paused", **patch})
    if action == "resume":
        return monitor_with_updates(monitor, {"status": "active", **patch})
    if action == "disable":
        return monitor_with_updates(monitor, {"status": "disabled", **patch})
    raise ValueError(f"unsupported monitor lifecycle action: {action!r}")


def _monitor_lifecycle_commit_evidence_kind(action: str) -> str:
    if action == "delete":
        return "monitor.deleted"
    if action == "disable":
        return "monitor.disabled"
    if action == "pause":
        return "monitor.paused"
    if action == "resume":
        return "monitor.resumed"
    return "monitor.state_update"


def _monitor_lifecycle_mutation_action(action: str) -> str:
    if action == "disable":
        return "update"
    return action


def _monitor_lifecycle_already_committed(
    existing: DbMonitor | None,
    *,
    after: DbMonitor | None,
    action: str,
) -> bool:
    if action == "delete":
        return existing is None
    return existing == after


def _non_executable_active_monitor_reason(plan: dict[str, Any]) -> str | None:
    kind = (plan or {}).get("kind")
    if kind not in {"planned_read", "metric_sql", "freshness_sql", "plugin_source"}:
        return "missing_executable_kind"
    if kind in {"planned_read", "metric_sql", "freshness_sql"}:
        if not isinstance(plan.get("sql"), str) or not plan["sql"].strip():
            return "missing_observation_sql"
        if not plan.get("value_path"):
            return "missing_value_path"
    if kind == "planned_read":
        if not plan.get("cursor") or not plan.get("cursor_update"):
            return "missing_cursor_strategy"
    if kind == "plugin_source":
        if not plan.get("capability_id") and not plan.get("source_kind"):
            return "missing_plugin_source_capability"
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


def _compact_prepare_context(
    *,
    runtime: Any,
    operation: Operation,
    schema_evidence: Evidence | None,
    schema: dict[str, Any],
) -> dict[str, Any]:
    dialect = (
        str(schema.get("database_type") or getattr(runtime.source, "sql_dialect", ""))
        or None
    )
    return {
        "operation_id": operation.id,
        "prompt": operation.request.get("prompt"),
        "intent_kind": runtime._db_intent_from_operation(operation).kind.value,
        "dialect": dialect,
        "schema": {
            "database_type": schema.get("database_type"),
            "database_name": schema.get("database_name"),
            "table_count": schema.get("table_count")
            or len(schema.get("tables", []) or []),
            "tables": [
                {
                    "name": table.get("name"),
                    "columns": [
                        {
                            "name": column.get("name"),
                            "data_type": column.get("data_type"),
                            "is_primary_key": column.get("is_primary_key"),
                        }
                        for column in table.get("columns", []) or []
                        if column.get("name")
                    ],
                }
                for table in schema.get("tables", []) or []
                if table.get("name")
            ],
            "foreign_keys": list(schema.get("foreign_keys", []) or []),
        },
        "schema_evidence_refs": [schema_evidence.id] if schema_evidence else [],
        "catalog_evidence_refs": [],
        "relationship_evidence_refs": [],
        "column_value_evidence_refs": [],
        "column_value_hints": [],
        "included_sections": ["schema"],
        "schema_fingerprint": stable_fingerprint(schema),
        "diagnostics": {"mode": "prepare_read_compact"},
    }


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
