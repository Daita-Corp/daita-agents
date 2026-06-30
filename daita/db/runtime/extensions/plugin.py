"""Runtime-owned DB planning extension plugin."""

from __future__ import annotations

from typing import Any

from daita.plugins import PluginContext, PluginKind, PluginManifest
from daita.plugins.base import RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, EvidenceSchema, RiskLevel, Worker

from ...llm_planner import DbLLMPlannerExecutor, DbLLMRepairExecutor
from ...synthesis import DbAnswerSynthesisExecutor
from .analysis import (
    DbAnalysisCheckpointExecutor,
    DbAnalysisPlanExecutor,
    DbAnalysisPlanValidationExecutor,
    DbAnalysisReplanExecutor,
    DbAnalysisSummarizeExecutor,
)
from .monitor_create import (
    DbMonitorCommitCreateExecutor,
    DbMonitorPlanCreateExecutor,
)
from .memory_update import (
    DbMemoryCommitUpdateExecutor,
    DbMemoryPlanUpdateExecutor,
)
from .memory_learning import (
    DbMemoryLearningEnqueueExecutor,
    DbMemoryLearningRunExecutor,
)
from .monitor_lifecycle import (
    DbMonitorCommitLifecycleExecutor,
    DbMonitorLocalDeliveryExecutor,
    DbMonitorPlanLifecycleExecutor,
)
from .query import (
    DbPlanningContextExecutor,
    DbQueryPlanValidationExecutor,
)


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
                operation_types=frozenset(
                    {
                        "data.query",
                        "query.plan",
                        "schema.query",
                        "schema.relationship_query",
                    }
                ),
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
                    {
                        "data.query",
                        "schema.query",
                        "schema.relationship_query",
                        "data.query.catalog_assisted",
                    }
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
            Capability(
                id="db.memory.plan_update",
                owner="db_runtime",
                description="Plan and validate an explicit DB memory update proposal.",
                domains=frozenset({"db", "memory"}),
                operation_types=frozenset({"memory.update"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"db.memory.proposal"}),
                executor="db_runtime.memory.plan_update",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.memory.commit_update",
                owner="db_runtime",
                description="Commit an accepted explicit DB memory proposal.",
                domains=frozenset({"db", "memory"}),
                operation_types=frozenset({"memory.update"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.MEDIUM,
                input_schema=common_schema,
                output_evidence=frozenset(
                    {"db.memory.definition", "memory.semantic.write"}
                ),
                executor="db_runtime.memory.commit_update",
                runtime_only=True,
                side_effecting=True,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.memory.learning.enqueue",
                owner="db_runtime",
                description="Plan a worker-owned DB memory learning task.",
                domains=frozenset({"db", "memory"}),
                operation_types=frozenset({"db.memory.learning"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema=common_schema,
                output_evidence=frozenset({"db.memory.learning.enqueue"}),
                executor="db_runtime.memory.learning.enqueue",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.memory.learning.run",
                owner="db_runtime",
                description="Extract and promote deterministic DB memory candidates.",
                domains=frozenset({"db", "memory"}),
                operation_types=frozenset({"db.memory.learning"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.MEDIUM,
                input_schema=common_schema,
                output_evidence=frozenset(
                    {
                        "db.memory.candidate",
                        "db.memory.promotion",
                        "db.memory.rejection",
                        "memory.semantic.write",
                    }
                ),
                executor="db_runtime.memory.learning.run",
                runtime_only=True,
                side_effecting=True,
                replay_safe=True,
                idempotent=True,
                retry_safe=True,
                metadata={
                    "queue": "memory_learning",
                    "worker_id": "db.memory.learner",
                    "worker_owner": "db_runtime",
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
            DbQueryPlanValidationExecutor(self),
            DbAnswerSynthesisExecutor(runtime=self),
            DbAnalysisPlanExecutor(self),
            DbAnalysisPlanValidationExecutor(self),
            DbAnalysisCheckpointExecutor(self),
            DbAnalysisSummarizeExecutor(self),
            DbAnalysisReplanExecutor(self),
            DbMonitorPlanCreateExecutor(self),
            DbMonitorCommitCreateExecutor(self),
            DbMemoryPlanUpdateExecutor(self),
            DbMemoryCommitUpdateExecutor(self),
            DbMemoryLearningEnqueueExecutor(self),
            DbMemoryLearningRunExecutor(self),
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
                kind="planner.decision",
                owner="db_runtime",
                json_schema=object_schema,
                description="Persisted DB agent planner decision.",
            ),
            EvidenceSchema(
                kind="planner.observation",
                owner="db_runtime",
                json_schema=object_schema,
                description="Compact runtime observation returned to the DB planner.",
            ),
            EvidenceSchema(
                kind="planner.compilation",
                owner="db_runtime",
                json_schema=object_schema,
                description="Runtime compilation of planner actions to governed tasks.",
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
                kind="db.memory.proposal",
                owner="db_runtime",
                json_schema=object_schema,
                description="Validated explicit DB memory proposal.",
            ),
            EvidenceSchema(
                kind="db.memory.definition",
                owner="db_runtime",
                json_schema=object_schema,
                description="Explicit DB memory definition committed from a proposal.",
            ),
            EvidenceSchema(
                kind="db.memory.learning.enqueue",
                owner="db_runtime",
                json_schema=object_schema,
                description="Worker-owned DB memory learning handoff.",
            ),
            EvidenceSchema(
                kind="db.memory.candidate",
                owner="db_runtime",
                json_schema=object_schema,
                description="Deterministic DB memory candidate from accepted evidence.",
            ),
            EvidenceSchema(
                kind="db.memory.promotion",
                owner="db_runtime",
                json_schema=object_schema,
                description="DB memory candidate promoted through memory semantic write.",
            ),
            EvidenceSchema(
                kind="db.memory.rejection",
                owner="db_runtime",
                json_schema=object_schema,
                description="DB memory candidate rejected by automatic gates.",
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

    def get_workers(self) -> tuple[Worker, ...]:
        return (
            Worker(
                id="db.memory.learner",
                owner="db_runtime",
                role="db_memory_learning",
                capability_ids=frozenset({"db.memory.learning.run"}),
                input_schema={"type": "object"},
                output_evidence=frozenset(
                    {
                        "db.memory.candidate",
                        "db.memory.promotion",
                        "db.memory.rejection",
                    }
                ),
                max_concurrency=1,
                metadata={"queue": "memory_learning"},
            ),
        )
