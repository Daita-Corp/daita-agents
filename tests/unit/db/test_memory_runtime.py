from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from unittest.mock import AsyncMock, MagicMock

from daita.db import (
    DbIntent,
    DbIntentKind,
    DbRequest,
    DbRuntime,
    DbRuntimeConfig,
)
from daita.db.analysis import structural_schema_fingerprint
from daita.db.loop import DbAgentLoop
from daita.db.llm_planner import _planner_messages
from daita.db.memory import (
    DBMemoryRecord,
    calibrate_db_memory,
    db_memory_planning_recall_decision,
    db_memory_planning_recall_query,
    db_memory_refs_from_recall_evidence,
    db_memory_selection_artifact_payload,
    has_db_memory_marker,
    normalize_db_memory_record,
    recall_db_memory_records,
    write_db_memory_record,
    write_db_memory_records,
)
from daita.db.memory_contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    confidence_value,
    db_memory_contract_refs,
    db_memory_contracts_artifact_payload,
    extract_db_memory_semantic_contract,
    meaningful_tokens,
    project_db_memory_semantic_contracts,
    safe_omission_summaries,
    schema_refs_known_schema,
)
from daita.db.planning_context import DbPlanningContextBuilder
from daita.db.planner_protocol import (
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.catalog import CatalogPlugin
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.plugins.memory.db_semantic_store import DBSemanticMemoryStore
from daita.plugins.memory.local_backend import LocalMemoryBackend
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import (
    AccessMode,
    Capability,
    ContextAudience,
    Evidence,
    EvidenceSchema,
    Operation,
    OperationStatus,
    RiskLevel,
    Task,
    TaskDependency,
)

import pytest


def _memory(tmp_path: Path) -> MemoryPlugin:
    plugin = MemoryPlugin(embedder=MockEmbeddingProvider(dim=8))
    plugin.backend = LocalMemoryBackend(
        workspace="runtime_memory",
        agent_id="memory-test",
        scope="project",
        base_dir=tmp_path,
        embedder=MockEmbeddingProvider(dim=8),
    )
    return plugin


def _runtime_with_memory_source(*plugins) -> DbRuntime:
    return DbRuntime(
        config=DbRuntimeConfig(
            plugins=tuple(plugins),
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.45,
                        "retrieval_mode": "structured",
                        "workspace_scope": "source",
                        "source_identity": "test:memory-runtime-source",
                    }
                }
            },
        )
    )


class _ScriptedPlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        if not self.decisions:
            raise AssertionError("planner was called after scripted decisions ended")
        return self.decisions.pop(0)


def _board_revenue_schema():
    return {
        "database_type": "sqlite",
        "tables": [
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "total", "data_type": "REAL"},
                    {"name": "status", "data_type": "TEXT"},
                ],
            },
            {
                "name": "refunds",
                "columns": [
                    {"name": "id", "data_type": "INTEGER"},
                    {"name": "order_id", "data_type": "INTEGER"},
                    {"name": "amount", "data_type": "REAL"},
                ],
            },
        ],
        "foreign_keys": [
            {
                "source_table": "refunds",
                "source_column": "order_id",
                "target_table": "orders",
                "target_column": "id",
            }
        ],
    }


def _mock_db_memory_backend(results: list[dict]) -> MagicMock:
    backend = MagicMock()
    backend.embedding_available = False
    backend.structured_index = True
    backend.recall_db_records = AsyncMock(return_value=results)
    backend.recall = AsyncMock(return_value=[])
    backend.list_by_category = AsyncMock(return_value=[])
    return backend


def _db_memory_recall_result(
    *,
    key: str,
    text: str,
    source_identity: str,
    table: str | None = None,
    score: float = 0.91,
) -> dict:
    metadata = {
        "source_identity": source_identity,
        "workspace_scope": "source",
        "active": True,
        "confidence": 0.9,
    }
    if table:
        metadata["table"] = table
    return {
        "chunk_id": f"mem-{key}",
        "content": text,
        "metadata": {
            "db_memory": {
                "kind": "schema_interpretation",
                "key": key,
                "text": text,
                "metadata": metadata,
                "importance": 0.8,
                "category": "db_semantics",
            }
        },
        "score": score,
    }


def _board_revenue_contract_metadata(source_identity: str) -> dict:
    return {
        "source_identity": source_identity,
        "workspace_scope": "source",
        "active": True,
        "confidence": 0.95,
        "semantic_contract_status": "validated",
        "subject": {
            "type": "metric",
            "key": "metric:board_revenue",
            "aliases": ["board revenue"],
        },
        "requirements": {
            "refs": [
                {"kind": "column", "ref": "orders.total", "role": "measure"},
                {"kind": "column", "ref": "refunds.amount", "role": "adjustment"},
                {"kind": "column", "ref": "orders.status", "role": "filter"},
            ],
            "relationships": [
                {"from": "refunds.order_id", "to": "orders.id", "role": "join"}
            ],
            "filters": [
                {
                    "ref": "orders.status",
                    "operator": "semantic_equals",
                    "value": "complete",
                    "value_source": "literal_or_catalog_value",
                }
            ],
            "aggregations": [
                {"function": "sum", "ref": "orders.total", "role": "base_measure"},
                {
                    "function": "sum",
                    "ref": "refunds.amount",
                    "role": "subtractive_adjustment",
                },
            ],
            "result_shape": {"grain": "single_aggregate"},
        },
        "schema_refs": [
            "orders.total",
            "refunds.amount",
            "orders.status",
            "refunds.order_id",
            "orders.id",
        ],
    }


class SchemaInspectExecutor:
    id = "schema_probe.schema.inspect"
    capability_ids = frozenset({"db.schema.inspect"})

    async def execute(self, task: Task, operation: Operation, context):
        return [
            Evidence(
                kind="schema.asset_profile",
                owner="schema_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "database_type": "probe",
                    "tables": [
                        {
                            "name": "orders",
                            "columns": [
                                {"name": "total_cents", "type": "real"},
                            ],
                        }
                    ],
                },
            )
        ]


class SchemaInspectPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="schema_probe",
        display_name="Schema Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.executor = SchemaInspectExecutor()

    def declare_capabilities(self):
        return [
            Capability(
                id="db.schema.inspect",
                owner="schema_probe",
                description="Inspect schema for calibration.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"source.profile"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.asset_profile"}),
                executor=self.executor.id,
                runtime_only=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [self.executor]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="schema.asset_profile",
                owner="schema_probe",
                json_schema={"type": "object"},
            )
        ]


async def test_memory_registers_domain_service_capabilities_and_context_provider(
    tmp_path,
):
    memory = _memory(tmp_path)
    runtime = DbRuntime(plugins=(memory,))

    inspection = await runtime.inspect()

    assert inspection.plugin_ids == ("memory",)
    assert "memory:memory.semantic.recall" in inspection.capability_ids
    assert "memory:memory.semantic.write" in inspection.capability_ids
    assert "memory:memory.fact.query" in inspection.capability_ids
    assert "memory:memory.context.render" in inspection.capability_ids
    assert "memory.semantic.write" in inspection.executor_ids
    assert "memory:memory.context" in inspection.evidence_schema_kinds
    assert "memory:memory.context" in inspection.context_provider_ids
    recall = runtime.registry.get_capability(
        "memory.semantic.recall",
        owner="memory",
    )
    assert {
        "schema.query",
        "schema.relationship_query",
    } <= recall.operation_types
    planning_context = runtime.registry.get_capability(
        "db.planning.context.build",
        owner="db_runtime",
    )
    assert {
        "schema.query",
        "schema.relationship_query",
    } <= planning_context.operation_types
    answer_synthesis = runtime.registry.get_capability(
        "db.answer.synthesize",
        owner="db_runtime",
    )
    assert {
        "schema.query",
        "schema.relationship_query",
    } <= answer_synthesis.operation_types
    schema_contract = runtime.build_contract(
        DbRequest("What is the operations table?", mode="schema.query")
    )
    assert "memory.semantic.recall" not in schema_contract.required_capabilities
    assert "memory.semantic.recall" not in schema_contract.required_evidence


async def test_memory_teardown_flushes_runtime_state():
    backend = MagicMock()
    backend.flush = AsyncMock()
    backend.prune = AsyncMock()
    memory = MemoryPlugin(auto_curate="manual")
    memory.backend = backend

    await memory.teardown()

    backend.flush.assert_awaited_once()
    backend.prune.assert_awaited_once()


def test_db_runtime_can_require_memory_write_when_service_registered(tmp_path):
    runtime = _runtime_with_memory_source(_memory(tmp_path))

    contract = runtime.build_contract(
        DbRequest("Remember that revenue excludes tax", mode="memory.update")
    )

    assert contract.required_capabilities == (
        "db.memory.plan_update",
        "db.memory.commit_update",
    )
    assert set(contract.required_evidence) == {
        "db.memory.proposal",
        "db.memory.definition",
        "memory.semantic.write",
    }
    assert contract.metadata["missing_capabilities"] == []
    selected = contract.metadata["selected_capabilities"]
    assert selected[0]["owner"] == "db_runtime"
    assert selected[0]["executor"] == "db_runtime.memory.plan_update"
    assert selected[1]["owner"] == "db_runtime"
    assert selected[1]["executor"] == "db_runtime.memory.commit_update"


async def test_memory_executors_return_typed_evidence(tmp_path):
    memory = _memory(tmp_path)
    runtime = DbRuntime(plugins=(memory,))

    written = await runtime.execute_capability(
        "memory.semantic.write",
        owner="memory",
        operation_type="memory.update",
        input={
            "content": "Orders revenue excludes tax",
            "importance": 0.9,
            "category": "db_semantic",
        },
    )
    recalled = await runtime.execute_capability(
        "memory.semantic.recall",
        owner="memory",
        operation_type="memory.recall",
        input={"query": "Orders revenue excludes tax", "score_threshold": 0.0},
    )
    context = await runtime.execute_capability(
        "memory.context.render",
        owner="memory",
        operation_type="context.render",
        input={"prompt": "Orders revenue excludes tax", "token_budget": 1000},
    )

    assert written[0].kind == "memory.semantic.write"
    assert written[0].payload["result"]["status"] == "success"
    assert recalled[0].kind == "memory.semantic.recall"
    assert "Orders revenue excludes tax" in str(recalled[0].payload["results"])
    assert context[0].kind == "memory.context"
    assert context[0].payload["rendered"] is True
    assert "Relevant Memory" in context[0].payload["content"]


async def test_planning_time_db_memory_recall_adds_bounded_context(tmp_path):
    db_path = tmp_path / "planning_memory.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            total REAL NOT NULL
        );
        INSERT INTO orders (total) VALUES (10.0), (20.0);
        """)
    source_identity = "sqlite:from_db:memory-test-source"
    memory = MemoryPlugin()
    backend = MagicMock()
    backend.recall_db_records = AsyncMock(
        return_value=[
            {
                "chunk_id": "mem-1",
                "content": (
                    "DB memory record:\n"
                    '{"kind":"metric_definition","key":"metric:revenue",'
                    '"text":"Revenue excludes refunded orders.",'
                    '"metadata":{"source_identity":"sqlite:from_db:memory-test-source",'
                    '"workspace_scope":"source","active":true,"confidence":0.9},'
                    '"importance":0.8,"category":"db_semantics"}'
                ),
                "metadata": {
                    "db_memory": {
                        "kind": "metric_definition",
                        "key": "metric:revenue",
                        "text": "Revenue excludes refunded orders.",
                        "metadata": {
                            "source_identity": source_identity,
                            "workspace_scope": "source",
                            "active": True,
                            "confidence": 0.9,
                        },
                        "importance": 0.8,
                        "category": "db_semantics",
                    }
                },
                "score": 0.91,
            }
        ]
    )
    backend.recall = AsyncMock(return_value=[])
    backend.list_by_category = AsyncMock(return_value=[])
    memory.backend = backend
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:memory-test-source",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "learning": "safe",
                        "limit": 3,
                        "char_budget": 120,
                        "score_threshold": 0.0,
                        "workspace_scope": "source",
                        "source_identity": source_identity,
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        request = DbRequest("How should revenue be calculated?")
        intent = runtime.classify_request(request)
        contract = runtime.build_contract(request, intent)
        operation = await runtime.kernel.create_operation(
            operation_type=contract.operation_type,
            request={
                "prompt": request.prompt,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "source_scope": list(request.source_scope),
                "mode": request.mode,
                "requested_capabilities": list(request.requested_capabilities),
                "constraints": request.constraints,
                "metadata": request.metadata,
            },
            required_evidence=frozenset(contract.required_evidence),
            metadata={
                "intent_kind": intent.kind.value,
                "access": contract.access.value,
                "resume_context": {
                    "request": {
                        "prompt": request.prompt,
                        "user_id": request.user_id,
                        "session_id": request.session_id,
                        "source_scope": list(request.source_scope),
                        "mode": request.mode,
                        "requested_capabilities": list(request.requested_capabilities),
                        "constraints": request.constraints,
                        "metadata": request.metadata,
                    },
                    "intent": {
                        "kind": intent.kind.value,
                        "confidence": intent.confidence,
                        "access": intent.access.value,
                        "evidence_mode": intent.evidence_mode,
                        "requested_outputs": list(intent.requested_outputs),
                        "constraints": intent.constraints,
                        "diagnostics": intent.diagnostics,
                    },
                    "contract": {
                        "operation_type": contract.operation_type,
                        "required_capabilities": list(contract.required_capabilities),
                        "required_evidence": list(contract.required_evidence),
                        "access": contract.access.value,
                        "limits": contract.limits.to_dict(),
                        "policy_ids": list(contract.policy_ids),
                        "metadata": contract.metadata,
                    },
                },
            },
        )
        schema_evidence = (
            await runtime.execute_capability(
                "db.schema.inspect",
                owner="sqlite",
                operation_type="source.profile",
                input={},
            )
        )[0]
        schema_evidence = replace(
            schema_evidence,
            id=schema_evidence.id or "schema-memory-runtime",
            operation_id=operation.id,
            task_id=None,
        )
        await runtime.store.save_evidence(schema_evidence)
        tasks = []
        recall_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="memory.semantic.recall",
                    owner="memory",
                    input={
                        "query": request.prompt,
                        "category": "db_semantics",
                        "limit": 9,
                        "score_threshold": 0.0,
                        "retrieval_mode": "structured",
                        "source_identity": source_identity,
                    },
                    reason="planning_memory_recall",
                    metadata={"memory_recall": "planning"},
                    deterministic_key="memory-runtime-planning-recall",
                ),
            ),
            contract=contract,
        )
        tasks.extend(recall_plan.tasks)
        memory_evidence = await runtime.execute_task(recall_plan.tasks[0], operation)
        recall = next(
            item for item in memory_evidence if item.kind == "memory.semantic.recall"
        )
        context_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.planning.context.build",
                    owner="db_runtime",
                    input={
                        "prompt": request.prompt,
                        "schema_evidence_id": schema_evidence.id,
                        "catalog_evidence_ids": [],
                        "relationship_evidence_ids": [],
                        "memory_recall_evidence_ids": [recall.id],
                        "memory_recall_diagnostics": {
                            "registered": True,
                            "queried": True,
                            "decision": {"recall": True},
                            "evidence_count": len(memory_evidence),
                        },
                    },
                    reason="planning_context",
                    sequence=2,
                    dependencies=(
                        TaskDependency(
                            kind="evidence",
                            evidence_kind=schema_evidence.kind,
                            evidence_id=schema_evidence.id,
                            evidence_owner=schema_evidence.owner,
                            evidence_accepted=True,
                            operation_id=operation.id,
                        ),
                        TaskDependency(
                            kind="evidence",
                            evidence_kind=recall.kind,
                            evidence_id=recall.id,
                            evidence_owner=recall.owner,
                            evidence_accepted=True,
                            operation_id=operation.id,
                        ),
                    ),
                    deterministic_key="memory-runtime-planning-context",
                ),
            ),
            contract=contract,
        )
        tasks.extend(context_plan.tasks)
        context_evidence = await runtime.execute_task(context_plan.tasks[0], operation)
        selection = next(
            item for item in context_evidence if item.kind == "db.memory.selection"
        )
        contracts = next(
            item for item in context_evidence if item.kind == "db.memory.contracts"
        )
        context = next(
            item for item in context_evidence if item.kind == "planning.context"
        )
    finally:
        await runtime.teardown()

    assert [task.capability_id for task in tasks][-2:] == [
        "memory.semantic.recall",
        "db.planning.context.build",
    ]
    assert recall.owner == "memory"
    assert selection.owner == "db_runtime"
    assert contracts.owner == "db_runtime"
    assert backend.recall_db_records.await_args.kwargs["category"] == "db_semantics"
    assert backend.recall.await_count == 0
    assert selection.payload["source_identity"] == source_identity
    assert selection.payload["recall_evidence_refs"] == [recall.id]
    assert selection.payload["raw_candidate_count"] == 1
    assert selection.payload["included_count"] == 1
    assert selection.payload["included_refs"][0]["key"] == "metric:revenue"
    assert selection.payload["omitted_counts_by_reason"] == {}
    assert selection.payload["budget_usage"]["char_budget"] == 120
    assert contracts.payload["selection_evidence_ref"]["id"] == selection.id
    assert contracts.payload["recall_evidence_refs"] == [recall.id]
    assert contracts.payload["enforceable_contracts"] == []
    assert contracts.payload["contract_omission_reasons"] == {}
    assert context.payload["db_memory_refs"] == [
        {
            "chunk_id": "mem-1",
            "kind": "metric_definition",
            "key": "metric:revenue",
            "text": "Revenue excludes refunded orders.",
            "confidence": 0.9,
            "importance": 0.8,
            "source_identity": source_identity,
            "evidence_refs": [],
            "schema_fingerprint": None,
        }
    ]
    assert context.payload["db_memory_selection_evidence_ref"]["id"] == selection.id
    assert context.payload["db_memory_contracts_evidence_ref"]["id"] == contracts.id
    assert "Database memory:" in context.payload["rendered_context"]
    assert "DB memory record:" not in context.payload["rendered_context"]


def test_llm_planner_includes_db_memory_advisory_contract():
    system_message = _planner_messages({"rendered_context": "Database memory:\n- x"})[
        0
    ]["content"]
    assert "DB memory is semantic business context" in system_message
    assert "must satisfy the memory contract" in system_message
    assert "Schema, catalog, policy, SQL validation" in system_message


async def test_agent_loop_build_planning_context_adds_memory_recall_prerequisite(
    tmp_path,
):
    memory = _memory(tmp_path)
    runtime = DbRuntime(
        plugins=(memory,),
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "learning": "safe",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "retrieval_mode": "structured",
                        "source_identity": "sqlite:from_db:loop-memory",
                    }
                }
            }
        ),
    )
    await runtime.setup()
    try:
        state = DbLoopState(
            operation_id="op-memory-context",
            normalized_user_request={
                "prompt": "Calculate recognized revenue from orders.total."
            },
            safety_frame={"max_access": "read"},
            available_action_kinds=tuple(DbPlannerActionKind),
            memory_context={
                "enabled": True,
                "source_identity": "sqlite:from_db:loop-memory",
                "retrieval_mode": "structured",
                "limit": 3,
                "score_threshold": 0.0,
                "recall_decision": {
                    "recall": True,
                    "reason": "semantic_prompt",
                    "query": "recognized revenue orders.total complete",
                },
            },
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            actions=(
                DbPlannerAction(
                    action_id="context",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={},
                ),
            ),
        )

        compilation = DbAgentLoop(runtime, object()).compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "memory.semantic.recall",
        "db.planning.context.build",
    ]
    recall, context = compilation.task_specs
    assert recall.input == {
        "query": "recognized revenue orders.total complete",
        "category": "db_semantics",
        "limit": 9,
        "score_threshold": 0.0,
        "retrieval_mode": "structured",
        "source_identity": "sqlite:from_db:loop-memory",
    }
    assert context.dependencies[0].evidence_kind == "memory.semantic.recall"
    assert context.dependencies[0].producer_task_id == recall.task_id
    assert context.input["memory_recall_diagnostics"]["queried"] is True


async def test_required_memory_recall_runs_before_planner_can_clarify(tmp_path):
    source_identity = "sqlite:from_db:phase-zero-recall"
    memory = _memory(tmp_path)
    runtime = DbRuntime(
        plugins=(memory,),
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "learning": "safe",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "retrieval_mode": "structured",
                        "source_identity": source_identity,
                    }
                }
            }
        ),
    )
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="db.run",
        request={
            "prompt": "Calculate recognized revenue from orders.total.",
            "source_scope": ["sqlite"],
        },
        required_evidence=frozenset({"query.result"}),
        metadata={"intent_kind": "metric.query"},
        evaluate_governance=False,
    )
    planner = _ScriptedPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CLARIFY,
            intent={"operation_type": "metric.query"},
            clarification_question="What does recognized revenue mean?",
        )
    )
    try:
        result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert result.status != "clarification_required"
    assert planner.states == []
    capabilities = [task.capability_id for task in tasks]
    assert "memory.semantic.recall" in capabilities
    assert "db.planning.context.build" in capabilities
    assert capabilities.index("memory.semantic.recall") < capabilities.index(
        "db.planning.context.build"
    )


async def test_required_memory_recall_is_in_first_planner_state_and_reused(tmp_path):
    source_identity = "test:memory-runtime-source"
    memory = MemoryPlugin()
    backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="metric:recognized_revenue",
                text="Recognized revenue uses orders.total_cents.",
                source_identity=source_identity,
                table="orders",
            )
        ]
    )
    memory.backend = backend
    runtime = _runtime_with_memory_source(SchemaInspectPlugin(), memory)
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="data.query",
        request={
            "prompt": "What does recognized revenue mean for orders?",
            "source_scope": ["schema_probe"],
        },
        required_evidence=frozenset({"query.result"}),
        metadata={"intent_kind": "metric.query"},
        evaluate_governance=False,
    )
    planner = _ScriptedPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CLARIFY,
            intent={"operation_type": "metric.query"},
            clarification_question="What does recognized revenue mean?",
        )
    )
    resumed_planner = _ScriptedPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CLARIFY,
            intent={"operation_type": "metric.query"},
            clarification_question="What does recognized revenue mean?",
        )
    )
    try:
        result = await DbAgentLoop(runtime, planner).run(operation, max_turns=2)
        resumed = await DbAgentLoop(runtime, resumed_planner).run(
            operation,
            max_turns=1,
        )
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert result.status == "clarification_required"
    assert resumed.status == "clarification_required"
    assert len(planner.states) == 1
    planning_context = next(
        summary
        for summary in planner.states[0].accepted_evidence_summaries
        if summary["kind"] == "planning.context"
    )
    assert planning_context["db_memory_refs"][0]["key"] == ("metric:recognized_revenue")
    assert len(resumed_planner.states) == 1
    assert backend.recall_db_records.await_count == 1
    assert sum(task.capability_id == "memory.semantic.recall" for task in tasks) == 1
    assert sum(task.capability_id == "db.planning.context.build" for task in tasks) == 1


async def test_mismatched_recall_evidence_does_not_satisfy_required_continuation(
    tmp_path,
):
    memory = MemoryPlugin()
    backend = _mock_db_memory_backend([])
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="db.run",
        request={
            "prompt": "Calculate recognized revenue.",
            "source_scope": ["memory"],
        },
        required_evidence=frozenset({"query.result"}),
        metadata={"intent_kind": "metric.query"},
        evaluate_governance=False,
    )
    await runtime.store.save_evidence(
        Evidence(
            id="mismatched-recall",
            kind="memory.semantic.recall",
            owner="memory",
            operation_id=operation.id,
            task_id="mismatched-recall-task",
            payload={
                "query": "a different prompt",
                "results": [],
                "recall_binding": {"recall_fingerprint": "wrong"},
            },
            metadata={"task_input_hash": "wrong"},
        )
    )
    planner = _ScriptedPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CLARIFY,
            intent={"operation_type": "metric.query"},
            clarification_question="What does recognized revenue mean?",
        )
    )
    try:
        result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)
        evidence = await runtime.store.list_evidence(operation.id)
    finally:
        await runtime.teardown()

    recalls = [item for item in evidence if item.kind == "memory.semantic.recall"]
    assert result.status != "clarification_required"
    assert planner.states == []
    assert len(recalls) == 2
    assert recalls[-1].payload["recall_binding"]["recall_fingerprint"] != "wrong"
    assert backend.recall_db_records.await_count == 1


async def test_prior_turn_memory_proposal_dependency_recovers_to_latest_proposal(
    tmp_path,
):
    runtime = _runtime_with_memory_source(_memory(tmp_path))
    await runtime.setup()
    try:
        state = DbLoopState(
            operation_id="op-memory-commit",
            normalized_user_request={
                "prompt": "Remember the board revenue metric definition",
                "mode": "memory.update",
            },
            explicit_mode="memory.update",
            safety_frame={"max_access": "write"},
            available_action_kinds=tuple(DbPlannerActionKind),
            accepted_evidence_summaries=(
                {
                    "id": "proposal-accepted",
                    "kind": "db.memory.proposal",
                    "owner": "db_runtime",
                    "accepted": True,
                    "task_id": "plan-task",
                    "payload_fingerprint": "payload-fingerprint",
                    "proposal_fingerprint": "proposal-fingerprint",
                },
            ),
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "memory.update"},
            actions=(
                DbPlannerAction(
                    action_id="commit",
                    kind=DbPlannerActionKind.COMMIT_MEMORY_UPDATE,
                    input={},
                    depends_on=("plan_previous_turn",),
                ),
            ),
        )

        compilation = DbAgentLoop(runtime, object()).compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.memory.commit_update"
    ]
    commit = compilation.task_specs[0]
    assert commit.input == {
        "proposal_evidence_id": "proposal-accepted",
        "proposal_fingerprint": "proposal-fingerprint",
    }
    assert commit.dependencies[0].evidence_kind == "db.memory.proposal"
    assert commit.dependencies[0].evidence_id == "proposal-accepted"
    assert commit.dependencies[0].payload_fingerprint == "payload-fingerprint"
    assert commit.metadata["dependency_recovery"] == "latest_accepted_memory_proposal"


async def test_memory_update_runtime_continuation_commits_without_planner_action(
    tmp_path,
):
    runtime = _runtime_with_memory_source(_memory(tmp_path))
    await runtime.setup()
    executed_capabilities = []
    original_execute_task = runtime.execute_task

    async def tracked_execute_task(task, operation, context=None):
        executed_capabilities.append(task.capability_id)
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = tracked_execute_task
    try:
        request = DbRequest(
            "Remember that revenue excludes tax.",
            mode="memory.update",
        )
        intent = runtime.classify_request(request)
        contract = runtime.build_contract(request, intent)
        operation = await runtime.kernel.create_operation(
            operation_type="db.run",
            request={
                "prompt": request.prompt,
                "mode": request.mode,
                "source_scope": list(request.source_scope),
                "requested_capabilities": list(request.requested_capabilities),
                "constraints": request.constraints,
                "metadata": request.metadata,
            },
            required_evidence=frozenset(),
            metadata={
                "mode": "memory.update",
                "resume_context": {
                    "request": {
                        "prompt": request.prompt,
                        "user_id": request.user_id,
                        "session_id": request.session_id,
                        "source_scope": list(request.source_scope),
                        "mode": request.mode,
                        "requested_capabilities": list(request.requested_capabilities),
                        "constraints": request.constraints,
                        "metadata": request.metadata,
                    },
                    "intent": {
                        "kind": intent.kind.value,
                        "confidence": intent.confidence,
                        "access": intent.access.value,
                        "evidence_mode": intent.evidence_mode,
                        "requested_outputs": list(intent.requested_outputs),
                        "constraints": intent.constraints,
                        "diagnostics": intent.diagnostics,
                    },
                    "contract": {
                        "operation_type": contract.operation_type,
                        "required_capabilities": list(contract.required_capabilities),
                        "required_evidence": list(contract.required_evidence),
                        "access": contract.access.value,
                        "limits": contract.limits.to_dict(),
                        "policy_ids": list(contract.policy_ids),
                        "metadata": contract.metadata,
                    },
                },
            },
            evaluate_governance=False,
        )
        planner = _ScriptedPlanner(
            DbPlannerDecision(
                status=DbPlannerDecisionStatus.CONTINUE,
                intent={"operation_type": "memory.update"},
                actions=(
                    DbPlannerAction(
                        action_id="plan_memory",
                        kind=DbPlannerActionKind.PLAN_MEMORY_UPDATE,
                        input={},
                    ),
                ),
            )
        )

        result = await DbAgentLoop(runtime, planner).run(
            operation,
            safety_frame={"max_access": "write"},
            max_turns=4,
        )
        executed_once = list(executed_capabilities)
        second_result = await DbAgentLoop(runtime, _ScriptedPlanner()).run(
            operation,
            safety_frame={"max_access": "write"},
            max_turns=2,
        )
        resumed = await runtime.resume_operation(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
        evidence = await runtime.store.list_evidence(operation.id)
    finally:
        await runtime.teardown()

    assert result.status == "finished"
    assert second_result.status == "finished"
    assert executed_capabilities[: len(executed_once)] == executed_once
    assert executed_capabilities.count("db.memory.commit_update") == 1
    assert executed_capabilities.count("memory.semantic.write") == 1
    assert resumed.completed_task_ids
    assert len(planner.states) == 1
    assert "db.memory.plan_update" in executed_capabilities
    assert "db.memory.commit_update" in executed_capabilities
    assert "memory.semantic.write" in executed_capabilities
    memory_task_ids = [
        task.capability_id
        for task in tasks
        if task.capability_id
        in {
            "db.memory.plan_update",
            "db.memory.commit_update",
            "memory.semantic.write",
        }
    ]
    assert memory_task_ids == [
        "db.memory.plan_update",
        "db.memory.commit_update",
        "memory.semantic.write",
    ]
    assert {item.kind for item in evidence if item.accepted} >= {
        "db.memory.proposal",
        "db.memory.definition",
        "memory.semantic.write",
    }
    commit = next(
        task for task in tasks if task.capability_id == "db.memory.commit_update"
    )
    assert commit.metadata["runtime_continuation"] is True
    assert commit.metadata["continuation_resolution"]["source"] == (
        "runtime_continuation"
    )


async def test_memory_commit_reuses_completed_write_task_without_replay(tmp_path):
    runtime = _runtime_with_memory_source(_memory(tmp_path))
    await runtime.setup()
    try:
        request = DbRequest(
            "Remember that revenue excludes tax.",
            mode="memory.update",
        )
        operation = await runtime.kernel.create_operation(
            operation_type="db.run",
            request={
                "prompt": request.prompt,
                "mode": request.mode,
                "source_scope": list(request.source_scope),
                "requested_capabilities": list(request.requested_capabilities),
                "constraints": request.constraints,
                "metadata": request.metadata,
            },
            required_evidence=frozenset(),
            metadata={"mode": "memory.update"},
            evaluate_governance=False,
        )
        proposal_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.memory.plan_update",
                    owner="db_runtime",
                    input={
                        "request": {
                            "prompt": request.prompt,
                            "mode": request.mode,
                            "source_scope": list(request.source_scope),
                            "requested_capabilities": list(
                                request.requested_capabilities
                            ),
                            "constraints": request.constraints,
                            "metadata": request.metadata,
                        }
                    },
                    reason="test_memory_plan_update",
                    deterministic_key="partial-resume-plan",
                ),
            ),
        )
        proposal_evidence = await runtime.execute_task(
            proposal_plan.tasks[0],
            operation,
        )
        proposal = next(
            item for item in proposal_evidence if item.kind == "db.memory.proposal"
        )
        proposal_fingerprint = proposal.payload["proposal_fingerprint"]
        record = dict(proposal.payload["record"])
        write_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="memory.semantic.write",
                    owner="memory",
                    input={
                        "db_memory_payload": record,
                        "db_memory_prompt": str(record.get("text") or ""),
                    },
                    reason="db_memory_commit_update",
                    sequence=1,
                    dependencies=(
                        TaskDependency(
                            kind="evidence",
                            evidence_kind="db.memory.proposal",
                            evidence_id=proposal.id,
                            evidence_owner="db_runtime",
                            producer_task_id=proposal.task_id,
                            evidence_payload={
                                "proposal_fingerprint": proposal_fingerprint,
                            },
                            evidence_accepted=True,
                            operation_id=operation.id,
                        ),
                    ),
                    metadata={
                        "proposal_evidence_id": proposal.id,
                        "proposal_fingerprint": proposal_fingerprint,
                        "source_identity": proposal.payload["source_identity"],
                    },
                    deterministic_key=proposal_fingerprint,
                ),
            ),
        )
        await runtime.execute_task(write_plan.tasks[0], operation)
        commit_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.memory.commit_update",
                    owner="db_runtime",
                    input={
                        "proposal_evidence_id": proposal.id,
                        "proposal_fingerprint": proposal_fingerprint,
                    },
                    reason="test_partial_resume_commit",
                    dependencies=(
                        TaskDependency(
                            kind="evidence",
                            evidence_kind="db.memory.proposal",
                            evidence_id=proposal.id,
                            evidence_owner="db_runtime",
                            producer_task_id=proposal.task_id,
                            evidence_accepted=True,
                            operation_id=operation.id,
                        ),
                    ),
                    deterministic_key=proposal_fingerprint,
                ),
            ),
        )
        original_execute_task = runtime.execute_task

        async def reject_write_replay(task, operation, context=None):
            if task.capability_id == "memory.semantic.write":
                raise AssertionError("completed memory write task was replayed")
            return await original_execute_task(task, operation, context=context)

        runtime.execute_task = reject_write_replay
        commit_evidence = await original_execute_task(commit_plan.tasks[0], operation)
        evidence = await runtime.store.list_evidence(operation.id)
    finally:
        await runtime.teardown()

    assert [item.kind for item in commit_evidence] == ["db.memory.definition"]
    assert sum(item.kind == "memory.semantic.write" for item in evidence) == 1
    definition = commit_evidence[0]
    assert definition.accepted is True
    assert definition.payload["write_evidence_ids"]


async def test_memory_update_finish_without_write_remains_blocked(tmp_path):
    runtime = _runtime_with_memory_source(_memory(tmp_path))
    await runtime.setup()
    try:
        request = DbRequest(
            "Remember that revenue excludes tax.",
            mode="memory.update",
        )
        intent = runtime.classify_request(request)
        contract = runtime.build_contract(request, intent)
        operation = await runtime.kernel.create_operation(
            operation_type="db.run",
            request={
                "prompt": request.prompt,
                "mode": request.mode,
                "source_scope": list(request.source_scope),
                "requested_capabilities": list(request.requested_capabilities),
                "constraints": request.constraints,
                "metadata": request.metadata,
            },
            required_evidence=frozenset(contract.required_evidence),
            metadata={
                "mode": "memory.update",
                "resume_context": {
                    "request": {
                        "prompt": request.prompt,
                        "mode": request.mode,
                        "source_scope": list(request.source_scope),
                        "requested_capabilities": list(request.requested_capabilities),
                        "constraints": request.constraints,
                        "metadata": request.metadata,
                    },
                    "intent": {
                        "kind": intent.kind.value,
                        "confidence": intent.confidence,
                        "access": intent.access.value,
                        "evidence_mode": intent.evidence_mode,
                        "requested_outputs": list(intent.requested_outputs),
                        "constraints": intent.constraints,
                        "diagnostics": intent.diagnostics,
                    },
                    "contract": {
                        "operation_type": contract.operation_type,
                        "required_capabilities": list(contract.required_capabilities),
                        "required_evidence": list(contract.required_evidence),
                        "access": contract.access.value,
                        "limits": contract.limits.to_dict(),
                        "policy_ids": list(contract.policy_ids),
                        "metadata": contract.metadata,
                    },
                },
            },
            evaluate_governance=False,
        )
        await runtime.store.save_evidence(
            Evidence(
                id="proposal-with-definition",
                kind="db.memory.proposal",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={"proposal_fingerprint": "proposal-fp"},
                metadata={"proposal_fingerprint": "proposal-fp"},
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="definition-without-write",
                kind="db.memory.definition",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "proposal_evidence_id": "proposal-with-definition",
                    "proposal_fingerprint": "proposal-fp",
                    "committed": True,
                },
                metadata={
                    "proposal_evidence_id": "proposal-with-definition",
                    "proposal_fingerprint": "proposal-fp",
                },
            )
        )
        planner = _ScriptedPlanner(
            DbPlannerDecision(
                status=DbPlannerDecisionStatus.FINISH,
                intent={"operation_type": "memory.update"},
                actions=(),
            )
        )

        result = await DbAgentLoop(runtime, planner).run(
            operation,
            safety_frame={"max_access": "write"},
            max_turns=1,
        )
    finally:
        await runtime.teardown()

    assert result.status == "blocked"
    assert "memory_update_not_committed" in result.warnings


@pytest.mark.parametrize(
    ("intent_kind", "prompt"),
    [
        ("schema.query", "What does the operations table mean?"),
        (
            "schema.relationship_query",
            "How are customers related to orders in business terms?",
        ),
    ],
)
def test_db_memory_planning_recall_allows_metadata_intents(intent_kind, prompt):
    decision = db_memory_planning_recall_decision(
        prompt=prompt,
        intent_kind=intent_kind,
        schema={
            "tables": [
                {"name": "operations", "columns": [{"name": "id"}]},
                {"name": "orders", "columns": [{"name": "customer_id"}]},
            ]
        },
        memory_config={
            "enabled": True,
            "recall": "auto",
            "limit": 3,
            "char_budget": 800,
        },
    )

    assert decision["recall"] is True
    assert decision["query"]


@pytest.mark.parametrize(
    "intent_kind",
    [
        "write.propose",
        "memory.update",
        "write.execute",
        "admin",
        "conversational",
        "lineage.trace",
    ],
)
def test_db_memory_planning_recall_excludes_non_allowlisted_metadata_intents(
    intent_kind,
):
    decision = db_memory_planning_recall_decision(
        prompt="What does operations mean?",
        intent_kind=intent_kind,
        schema={"tables": [{"name": "operations", "columns": [{"name": "id"}]}]},
        memory_config={
            "enabled": True,
            "recall": "auto",
            "limit": 3,
            "char_budget": 800,
        },
    )

    assert decision == {"recall": False, "reason": "intent_not_semantic_query"}


@pytest.mark.parametrize(
    ("memory_config", "reason"),
    [
        (
            {"enabled": False, "recall": "auto", "limit": 3, "char_budget": 800},
            "memory_disabled",
        ),
        (
            {"enabled": True, "recall": "off", "limit": 3, "char_budget": 800},
            "recall_disabled",
        ),
        (
            {"enabled": True, "recall": "auto", "limit": 0, "char_budget": 800},
            "limit_zero",
        ),
        (
            {"enabled": True, "recall": "auto", "limit": 3, "char_budget": 0},
            "char_budget_zero",
        ),
    ],
)
def test_db_memory_planning_recall_metadata_intents_keep_global_guards(
    memory_config,
    reason,
):
    decision = db_memory_planning_recall_decision(
        prompt="What does operations mean?",
        intent_kind="schema.query",
        schema={"tables": [{"name": "operations", "columns": [{"name": "id"}]}]},
        memory_config=memory_config,
    )

    assert decision == {"recall": False, "reason": reason}


async def test_memory_fact_query_executor_uses_memory_owned_fact_store(tmp_path):
    memory = _memory(tmp_path)
    await memory.backend.remember_batch(
        [{"content": "orders table has total column", "importance": 0.8}],
        extra_metadata_list=[
            {
                "extracted_facts": [
                    {
                        "entity": "orders",
                        "relation": "has column",
                        "value": "total",
                        "temporal_context": None,
                    }
                ]
            }
        ],
    )
    runtime = DbRuntime(plugins=(memory,))

    evidence = await runtime.execute_capability(
        "memory.fact.query",
        owner="memory",
        operation_type="memory.query",
        input={"entity": "orders", "relation": "has column"},
    )

    assert evidence[0].kind == "memory.fact.query"
    assert evidence[0].payload["results"][0]["entity"] == "orders"
    assert evidence[0].payload["results"][0]["value"] == "total"


async def test_db_runtime_renders_memory_context_through_context_provider(tmp_path):
    memory = _memory(tmp_path)
    await memory.backend.remember_batch(
        [
            {
                "content": "Orders revenue excludes tax",
                "importance": 0.9,
                "category": "db_semantic",
            }
        ]
    )
    runtime = DbRuntime(plugins=(memory,))

    blocks = await runtime.render_context(
        prompt="Orders revenue excludes tax",
        audience=ContextAudience.PRIMARY_MODEL,
        token_budget=1000,
    )

    assert len(blocks) == 1
    assert blocks[0].owner == "memory"
    assert "Orders revenue excludes tax" in blocks[0].content


async def test_db_memory_helpers_write_metric_definitions_and_business_rules():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(side_effect=lambda *args, **kwargs: kwargs)
    memory = MemoryPlugin()
    memory.backend = backend

    await write_db_memory_records(
        memory,
        [
            DBMemoryRecord(
                kind="metric_definition",
                key="metric:revenue",
                text="Revenue excludes refunded orders.",
                metadata={"metric": "revenue"},
            ),
            {
                "kind": "business_rule",
                "key": "rule:refunds",
                "text": "Refunded orders must be excluded from revenue.",
                "metadata": {"table": "orders"},
            },
        ],
    )

    assert backend.remember.await_count == 2
    first_call = backend.remember.await_args_list[0]
    second_call = backend.remember.await_args_list[1]
    assert first_call.kwargs["category"] == "db_semantics"
    assert first_call.kwargs["index_content"] == "Revenue excludes refunded orders."
    assert (
        first_call.kwargs["extra_metadata"]["db_memory"]["kind"] == "metric_definition"
    )
    assert second_call.kwargs["extra_metadata"]["db_memory"]["kind"] == "business_rule"


async def test_direct_db_memory_write_keeps_unvalidated_metric_memory_advisory():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(side_effect=lambda *args, **kwargs: kwargs)
    memory = MemoryPlugin()
    memory.backend = backend

    result = await memory._execute_semantic_write(
        {
            "db_memory_payload": {
                "kind": "metric_definition",
                "key": "metric:board_revenue",
                "text": (
                    "Board revenue SQL must return one aggregate: SUM(orders.total) "
                    "for orders whose status is complete minus "
                    "COALESCE(SUM(refunds.amount), 0). Join refunds on "
                    "refunds.order_id = orders.id."
                ),
                "metadata": {
                    "source_identity": "sqlite:from_db:source-a",
                    "workspace_scope": "source",
                    "active": True,
                    "confidence": 0.95,
                    "semantic_contract": {
                        "version": 1,
                        "contract_kind": "metric_definition",
                        "subject": {"key": "metric:board_revenue"},
                        "requirements": {
                            "refs": [{"kind": "column", "ref": "orders.total"}]
                        },
                    },
                    "schema_refs": [
                        "orders.total",
                        "orders.status",
                        "refunds.amount",
                    ],
                },
            },
            "db_memory_prompt": "Board revenue SQL must subtract refunds.",
        }
    )

    assert result["success"] is True
    stored = backend.remember.await_args.kwargs["extra_metadata"]["db_memory"]
    assert DB_MEMORY_SEMANTIC_CONTRACT_KEY not in stored["metadata"]
    assert (
        stored["metadata"]["semantic_contract_diagnostics"]["reason"]
        == "direct_write_unvalidated"
    )


async def test_db_memory_allows_catalog_cited_value_alias_without_observed_values():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend

    result = await write_db_memory_record(
        memory,
        DBMemoryRecord(
            kind="value_alias",
            key="value_alias:shipments.status:completed",
            text=(
                "When users say completed shipments, consult catalog profile "
                "shipments.status for the authoritative observed value."
            ),
            metadata={
                "table": "shipments",
                "column": "status",
                "alias": "completed shipments",
                "catalog_profile_ref": "shipments.status",
                "catalog_evidence_id": "evidence-1",
            },
        ),
    )

    assert result["success"] is True
    assert result["kind"] == "value_alias"
    assert result["category"] == "db_semantics"
    backend.remember.assert_awaited_once()
    stored = backend.remember.await_args.kwargs["extra_metadata"]["db_memory"]
    assert stored["metadata"]["catalog_profile_ref"] == "shipments.status"
    assert "observed_value" not in stored["metadata"]
    assert "top_values" not in stored["metadata"]


async def test_db_memory_rejects_value_alias_without_catalog_citation():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend

    result = await write_db_memory_record(
        memory,
        {
            "kind": "value_alias",
            "key": "value_alias:shipments.status:completed",
            "text": "completed shipments means the status users usually ask for",
            "metadata": {"table": "shipments", "column": "status"},
        },
    )

    assert result["success"] is False
    assert "catalog_profile_ref or catalog_evidence_id" in result["error"]
    backend.remember.assert_not_awaited()


async def test_db_memory_rejects_value_alias_observed_values_in_metadata():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend

    result = await write_db_memory_record(
        memory,
        {
            "kind": "value_alias",
            "key": "value_alias:shipments.status:completed",
            "text": "completed shipments should resolve through the catalog profile",
            "metadata": {
                "table": "shipments",
                "column": "status",
                "alias": "completed shipments",
                "catalog_profile_ref": "shipments.status",
                "observed_value": "complete",
            },
        },
    )

    assert result["success"] is False
    assert "cannot store observed value field 'observed_value'" in result["error"]
    backend.remember.assert_not_awaited()


async def test_db_memory_helpers_recall_single_category_and_filter_kind():
    backend = MagicMock()
    backend.recall = AsyncMock(
        return_value=[
            {
                "chunk_id": "1",
                "content": (
                    'DB memory record:\n{"kind": "unit_convention", '
                    '"text": "orders.total_cents is stored as cents"}'
                ),
            },
            {
                "chunk_id": "2",
                "content": (
                    'DB memory record:\n{"kind": "business_rule", '
                    '"text": "exclude refunds"}'
                ),
            },
        ]
    )
    memory = MemoryPlugin()
    memory.backend = backend

    results = await recall_db_memory_records(
        memory,
        "How much revenue?",
        kinds=["unit_convention"],
        limit=5,
    )

    backend.recall.assert_awaited_once()
    assert backend.recall.call_args.kwargs["category"] == "db_semantics"
    assert [item["chunk_id"] for item in results] == ["1"]


async def test_db_memory_helpers_recall_relevant_business_rules():
    backend = MagicMock()
    backend.recall = AsyncMock(
        return_value=[
            {
                "chunk_id": "rule-1",
                "content": (
                    'DB memory record:\n{"kind": "business_rule", '
                    '"text": "Revenue excludes refunded orders."}'
                ),
            },
            {
                "chunk_id": "metric-1",
                "content": (
                    'DB memory record:\n{"kind": "metric_definition", '
                    '"text": "Revenue is SUM(total_amount)."}'
                ),
            },
        ]
    )
    memory = MemoryPlugin()
    memory.backend = backend

    results = await recall_db_memory_records(
        memory,
        "How should revenue be calculated?",
        kinds=["business_rule"],
        limit=5,
    )

    assert [item["chunk_id"] for item in results] == ["rule-1"]
    backend.recall.assert_awaited_once()
    assert backend.recall.call_args.kwargs["category"] == "db_semantics"


async def test_structured_db_memory_ranks_key_alias_and_schema_matches(tmp_path):
    source_identity = "sqlite:from_db:structured-ranking"
    memory = MemoryPlugin(
        auto_curate="manual",
        db_memory_mode=True,
        db_memory_retrieval_mode="structured",
    )
    memory.backend = LocalMemoryBackend(
        workspace=source_identity,
        agent_id="memory-test",
        scope="project",
        base_dir=tmp_path,
        embedder=None,
        default_source_identity=source_identity,
    )

    await write_db_memory_record(
        memory,
        DBMemoryRecord(
            kind="metric_definition",
            key="metric:net_revenue",
            text="Net revenue excludes refunded orders.",
            metadata={
                "source_identity": source_identity,
                "workspace_scope": "source",
                "active": True,
                "aliases": ["revenue"],
                "schema_refs": ["orders.total"],
                "confidence": 0.95,
            },
            importance=0.9,
        ),
    )
    await write_db_memory_record(
        memory,
        DBMemoryRecord(
            kind="business_rule",
            key="business_rule:accounts",
            text="Revenue dashboards should exclude inactive accounts.",
            metadata={
                "source_identity": source_identity,
                "workspace_scope": "source",
                "active": True,
                "schema_refs": ["accounts.status"],
            },
            importance=0.8,
        ),
    )

    results = await recall_db_memory_records(
        memory,
        "calculate revenue from orders.total",
        limit=2,
        score_threshold=0.0,
    )

    assert [item["metadata"]["db_memory"]["key"] for item in results] == [
        "metric:net_revenue",
        "business_rule:accounts",
    ]
    top = results[0]
    assert top["source"] == "structured_db_memory"
    assert top["score_breakdown"]["fts_bm25"] is not None
    assert top["score_breakdown"]["fts_normalized"] > 0
    assert top["score_breakdown"]["alias_match"] is True
    assert top["score_breakdown"]["schema_ref_overlap"] > 0
    assert results[1]["score_breakdown"]["fts_bm25"] is not None


async def test_structured_db_memory_filters_before_scoring(tmp_path):
    source_identity = "sqlite:from_db:structured-filters"
    memory = MemoryPlugin(
        auto_curate="manual",
        db_memory_mode=True,
        db_memory_retrieval_mode="structured",
    )
    memory.backend = LocalMemoryBackend(
        workspace=source_identity,
        agent_id="memory-test",
        scope="project",
        base_dir=tmp_path,
        embedder=None,
        default_source_identity=source_identity,
    )

    async def write(key, **metadata):
        await write_db_memory_record(
            memory,
            DBMemoryRecord(
                kind=metadata.pop("kind", "metric_definition"),
                key=key,
                text="Revenue excludes refunded orders.",
                metadata={
                    "source_identity": metadata.pop("source_identity", source_identity),
                    "workspace_scope": "source",
                    "active": True,
                    "confidence": 0.95,
                    **metadata,
                },
                importance=0.9,
            ),
        )

    await write("metric:active")
    await write("metric:inactive", active=False)
    await write("metric:stale", stale=True)
    await write("metric:expired", expires_at="2000-01-01T00:00:00+00:00")
    await write("metric:other-source", source_identity="sqlite:from_db:other-source")
    await write("cache:marker", kind="cache_marker")

    results = await memory.backend.recall_db_records(
        "revenue refunded orders",
        source_identity=source_identity,
        kinds=["metric_definition"],
        score_threshold=0.0,
        limit=10,
    )

    assert [item["metadata"]["db_memory"]["key"] for item in results] == [
        "metric:active"
    ]


async def test_structured_db_memory_upsert_refreshes_only_one_record_index(tmp_path):
    source_identity = "sqlite:from_db:structured-upsert"
    backend = LocalMemoryBackend(
        workspace=source_identity,
        agent_id="memory-test",
        scope="project",
        base_dir=tmp_path,
        embedder=None,
        default_source_identity=source_identity,
    )

    await backend.upsert_db_record(
        {
            "kind": "metric_definition",
            "key": "metric:revenue",
            "text": "Revenue excludes refunds.",
            "metadata": {
                "aliases": ["old revenue"],
                "schema_refs": ["orders.old_total"],
            },
        }
    )
    other = await backend.upsert_db_record(
        {
            "kind": "metric_definition",
            "key": "metric:margin",
            "text": "Margin excludes tax.",
            "metadata": {
                "aliases": ["gross margin"],
                "schema_refs": ["orders.margin"],
            },
        }
    )
    updated = await backend.upsert_db_record(
        {
            "kind": "metric_definition",
            "key": "metric:revenue",
            "text": "Revenue excludes refunded orders.",
            "metadata": {
                "aliases": ["net revenue"],
                "schema_refs": ["orders.net_total"],
            },
        }
    )

    conn = sqlite3.connect(str(backend.db_memory_db))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT alias FROM db_memory_aliases WHERE record_id = ?",
        (updated["record_id"],),
    )
    revenue_aliases = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        "SELECT ref FROM db_memory_schema_refs WHERE record_id = ?",
        (updated["record_id"],),
    )
    revenue_refs = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        "SELECT COUNT(*) FROM db_memory_fts WHERE record_id = ?",
        (updated["record_id"],),
    )
    revenue_fts_count = cursor.fetchone()[0]
    cursor.execute(
        "SELECT alias FROM db_memory_aliases WHERE record_id = ?",
        (other["record_id"],),
    )
    other_aliases = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert updated["status"] == "updated"
    assert revenue_aliases == {"net revenue"}
    assert revenue_refs == {"orders.net_total"}
    assert revenue_fts_count == 1
    assert other_aliases == {"gross margin"}


async def test_structured_db_memory_backfills_phase31_rows(tmp_path):
    db_path = tmp_path / "db_semantics.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE db_memory_records (
            record_id TEXT PRIMARY KEY,
            source_identity_key TEXT NOT NULL,
            key TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            source_identity TEXT,
            workspace_scope TEXT NOT NULL,
            schema_refs_json TEXT NOT NULL,
            catalog_refs_json TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            importance REAL NOT NULL,
            active INTEGER NOT NULL,
            stale INTEGER NOT NULL,
            expires_at TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
    cursor.execute(
        """
        INSERT INTO db_memory_records VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            "legacy-record",
            "sqlite:from_db:phase31",
            "metric:revenue",
            "metric_definition",
            "Revenue excludes refunded orders.",
            "db_semantics",
            "sqlite:from_db:phase31",
            "source",
            json.dumps(["orders.total"]),
            json.dumps(["catalog:orders"]),
            json.dumps(["net revenue"]),
            0.95,
            0.9,
            1,
            0,
            None,
            json.dumps(
                {
                    "source_identity": "sqlite:from_db:phase31",
                    "workspace_scope": "source",
                }
            ),
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    store = DBSemanticMemoryStore(
        db_path,
        default_source_identity="sqlite:from_db:phase31",
    )

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM db_memory_fts WHERE record_id = ?", ("legacy-record",)
    )
    fts_count = cursor.fetchone()[0]
    cursor.execute(
        "SELECT alias FROM db_memory_aliases WHERE record_id = ?",
        ("legacy-record",),
    )
    aliases = {row[0] for row in cursor.fetchall()}
    conn.close()
    recalled = await store.recall_db_records(
        "calculate net revenue from orders.total",
        source_identity="sqlite:from_db:phase31",
        score_threshold=0.0,
    )

    assert fts_count == 1
    assert aliases == {"net revenue"}
    assert recalled[0]["metadata"]["db_memory"]["key"] == "metric:revenue"


async def test_local_backend_migrates_legacy_db_memory_chunks(tmp_path):
    workspace = "legacy-db-memory"
    workspace_dir = tmp_path / workspace
    workspace_dir.mkdir(parents=True)
    vector_db = workspace_dir / "vectors.db"
    conn = sqlite3.connect(str(vector_db))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            content TEXT NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    cursor.execute(
        "CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, embedding TEXT NOT NULL)"
    )
    metadata = {
        "category": "db_semantics",
        "db_memory": {
            "kind": "metric_definition",
            "key": "metric:revenue",
            "text": "Revenue excludes refunded orders.",
            "metadata": {
                "source_identity": "sqlite:from_db:legacy",
                "workspace_scope": "source",
                "aliases": ["net revenue"],
            },
            "importance": 0.9,
            "category": "db_semantics",
        },
    }
    cursor.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (
            "legacy-1",
            "memory://direct",
            "DB memory record",
            0,
            0,
            json.dumps(metadata),
        ),
    )
    cursor.execute("INSERT INTO embeddings VALUES (?, ?)", ("legacy-1", "[]"))
    conn.commit()
    conn.close()

    backend = LocalMemoryBackend(
        workspace=workspace,
        agent_id="memory-test",
        scope="project",
        base_dir=tmp_path,
        embedder=None,
    )
    records = await backend.list_db_records(category="db_semantics", limit=10)
    conn = sqlite3.connect(str(vector_db))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    embedding_count = cursor.fetchone()[0]
    conn.close()

    assert [item["metadata"]["db_memory"]["key"] for item in records] == [
        "metric:revenue"
    ]
    assert chunk_count == 0
    assert embedding_count == 0


def test_db_memory_planning_refs_omit_cross_source_irrelevant_and_stale_records():
    def result(chunk_id, text, metadata, *, score=0.9, kind="metric_definition"):
        return {
            "chunk_id": chunk_id,
            "content": "DB memory record:\n{}",
            "metadata": {
                "db_memory": {
                    "kind": kind,
                    "key": f"metric:{chunk_id}",
                    "text": text,
                    "metadata": metadata,
                    "importance": 0.7,
                    "category": "db_semantics",
                }
            },
            "score": score,
        }

    source_identity = "sqlite:from_db:source-a"
    evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                result(
                    "revenue",
                    "Revenue excludes refunded orders.",
                    {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "active": True,
                        "confidence": 0.9,
                    },
                ),
                result(
                    "other-source",
                    "Revenue includes tax in the other warehouse.",
                    {
                        "source_identity": "sqlite:from_db:source-b",
                        "workspace_scope": "source",
                        "active": True,
                        "confidence": 0.9,
                    },
                ),
                result(
                    "inventory",
                    "Inventory turns are counted weekly.",
                    {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "active": True,
                        "confidence": 0.9,
                    },
                ),
                result(
                    "stale",
                    "Revenue includes archived orders.",
                    {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "stale": True,
                        "active": True,
                        "confidence": 0.9,
                    },
                ),
                result(
                    "inactive",
                    "Revenue excludes wholesale orders.",
                    {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "active": False,
                        "confidence": 0.9,
                    },
                ),
                result(
                    "low-confidence",
                    "Revenue excludes shipping.",
                    {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "active": True,
                        "confidence": 0.2,
                    },
                ),
            ]
        },
    )

    refs, evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (evidence,),
        prompt="How should revenue be calculated?",
        schema={"tables": [{"name": "orders", "columns": [{"name": "total"}]}]},
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        limit=3,
        char_budget=800,
        score_threshold=0.45,
    )

    assert [ref["key"] for ref in refs] == ["metric:revenue"]
    assert evidence_refs == ("evidence-memory",)
    assert diagnostics["candidate_count"] == 6
    assert diagnostics["included_count"] == 1
    assert diagnostics["omitted_reasons"]["cross_source"] == 1
    assert diagnostics["omitted_reasons"]["irrelevant"] == 1
    assert diagnostics["omitted_reasons"]["stale"] == 1
    assert diagnostics["omitted_reasons"]["inactive"] == 1
    assert diagnostics["omitted_reasons"]["low_confidence"] == 1


def test_db_memory_selection_deduplicates_before_limits_and_budgets():
    source_identity = "sqlite:from_db:source-a"

    def result(
        *,
        key,
        text,
        score,
        evidence_ref,
        chunk_id=None,
    ):
        item = {
            "metadata": {
                "db_memory": {
                    "kind": "metric_definition",
                    "key": key,
                    "text": text,
                    "metadata": {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "active": True,
                        "confidence": 0.9,
                        "evidence_refs": [evidence_ref],
                    },
                    "importance": 0.8,
                    "category": "db_semantics",
                }
            },
            "score": score,
        }
        if chunk_id is not None:
            item["chunk_id"] = chunk_id
        return item

    recall_a = Evidence(
        id="recall-a",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                result(
                    chunk_id="record-revenue",
                    key="metric:old_revenue_key",
                    text="Revenue uses the lower-ranked definition.",
                    score=0.6,
                    evidence_ref="source-z",
                ),
                result(
                    key="metric:fallback_revenue",
                    text="Revenue uses the lower-ranked fallback.",
                    score=0.7,
                    evidence_ref="fallback-z",
                ),
            ]
        },
    )
    recall_b = Evidence(
        id="recall-b",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                result(
                    chunk_id="record-revenue",
                    key="metric:recognized_revenue",
                    text="Revenue uses the highest-ranked definition.",
                    score=0.95,
                    evidence_ref="source-a",
                ),
                result(
                    key="metric:fallback_revenue",
                    text="Revenue uses the highest-ranked fallback.",
                    score=0.9,
                    evidence_ref="fallback-a",
                ),
            ]
        },
    )

    refs, evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (recall_a, recall_b),
        prompt="How should revenue be calculated?",
        schema={"tables": []},
        source_identity=source_identity,
        schema_fingerprint=None,
        limit=2,
        char_budget=200,
        score_threshold=0.0,
    )

    assert [ref["key"] for ref in refs] == [
        "metric:recognized_revenue",
        "metric:fallback_revenue",
    ]
    assert [ref["text"] for ref in refs] == [
        "Revenue uses the highest-ranked definition.",
        "Revenue uses the highest-ranked fallback.",
    ]
    assert refs[0]["evidence_refs"] == ["source-a", "source-z"]
    assert refs[1]["evidence_refs"] == ["fallback-a", "fallback-z"]
    assert evidence_refs == (
        "recall-a",
        "recall-b",
        "source-a",
        "source-z",
        "fallback-a",
        "fallback-z",
    )
    assert diagnostics["candidate_count"] == 4
    assert diagnostics["deduplicated_candidate_count"] == 2
    assert diagnostics["included_count"] == 2
    assert diagnostics["omitted_reasons"] == {"duplicate": 2}
    assert diagnostics["used_chars"] == sum(
        len(f"- {ref['kind']} {ref['key']}: {ref['text']}") for ref in refs
    )


def test_db_memory_selection_artifact_reports_required_omission_reasons():
    source_identity = "sqlite:from_db:source-a"

    def result(chunk_id, text, metadata, *, score=0.9):
        return {
            "chunk_id": chunk_id,
            "content": "DB memory record:\n{}",
            "metadata": {
                "db_memory": {
                    "kind": "metric_definition",
                    "key": f"metric:{chunk_id}",
                    "text": text,
                    "metadata": {
                        "source_identity": source_identity,
                        "workspace_scope": "source",
                        "active": True,
                        "confidence": 0.95,
                        **metadata,
                    },
                    "importance": 0.7,
                    "category": "db_semantics",
                }
            },
            "score": score,
        }

    evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                result("active", "Revenue excludes refunded orders.", {}),
                result("stale", "Revenue includes archived orders.", {"stale": True}),
                result(
                    "expired",
                    "Revenue expires when the source profile expires.",
                    {"expires_at": "2000-01-01T00:00:00+00:00"},
                ),
                result(
                    "inactive", "Revenue excludes wholesale orders.", {"active": False}
                ),
                result(
                    "other-source",
                    "Revenue includes tax in the other warehouse.",
                    {"source_identity": "sqlite:from_db:source-b"},
                ),
                result(
                    "schema-mismatch",
                    "Revenue uses total from the prior schema.",
                    {"source_schema_fingerprint": "schema-b"},
                ),
                result(
                    "low-confidence", "Revenue excludes shipping.", {"confidence": 0.2}
                ),
                result(
                    "unsafe",
                    "Revenue owner email jane@example.com must be contacted.",
                    {},
                ),
                result("low-score", "Revenue excludes test orders.", {}, score=0.1),
                result("irrelevant", "Inventory turns are counted weekly.", {}),
            ]
        },
    )

    refs, evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (evidence,),
        prompt="How should revenue be calculated?",
        schema={"tables": [{"name": "orders", "columns": [{"name": "total"}]}]},
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        limit=20,
        char_budget=2000,
        score_threshold=0.45,
    )
    payload = db_memory_selection_artifact_payload(
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        recall_evidence_refs=("evidence-memory",),
        memory_evidence_refs=evidence_refs,
        included_refs=refs,
        diagnostics=diagnostics,
        limit=20,
        char_budget=2000,
        score_threshold=0.45,
    )

    assert [ref["key"] for ref in refs] == ["metric:active"]
    assert payload["raw_candidate_count"] == 10
    assert payload["included_count"] == 1
    assert payload["omitted_counts_by_reason"] == {
        "cross_source": 1,
        "expired": 1,
        "inactive": 1,
        "irrelevant": 1,
        "low_confidence": 1,
        "low_score": 1,
        "schema_mismatch": 1,
        "stale": 1,
        "unsafe": 1,
    }
    assert {
        item["reason"] for item in payload["safe_diagnostic_omission_summaries"]
    } == set(payload["omitted_counts_by_reason"])

    budget_refs, _budget_evidence_refs, budget_diagnostics = (
        db_memory_refs_from_recall_evidence(
            (
                Evidence(
                    id="evidence-budget",
                    kind="memory.semantic.recall",
                    owner="memory",
                    payload={
                        "results": [
                            result(
                                "budget-a",
                                "Revenue excludes refunds.",
                                {},
                            ),
                            result(
                                "budget-b",
                                "Revenue excludes a very long list of unusual "
                                "adjustments that should exceed the tiny budget.",
                                {},
                            ),
                        ]
                    },
                ),
            ),
            prompt="How should revenue be calculated?",
            schema={"tables": []},
            source_identity=source_identity,
            schema_fingerprint=None,
            limit=20,
            char_budget=70,
            score_threshold=0.0,
        )
    )
    assert [ref["key"] for ref in budget_refs] == ["metric:budget-a"]
    assert budget_diagnostics["omitted_reasons"]["budget"] == 1

    limit_refs, _limit_evidence_refs, limit_diagnostics = (
        db_memory_refs_from_recall_evidence(
            (
                Evidence(
                    id="evidence-limit",
                    kind="memory.semantic.recall",
                    owner="memory",
                    payload={
                        "results": [
                            result("limit-a", "Revenue excludes refunds.", {}),
                            result("limit-b", "Revenue excludes taxes.", {}),
                        ]
                    },
                ),
            ),
            prompt="How should revenue be calculated?",
            schema={"tables": []},
            source_identity=source_identity,
            schema_fingerprint=None,
            limit=1,
            char_budget=2000,
            score_threshold=0.0,
        )
    )
    assert [ref["key"] for ref in limit_refs] == ["metric:limit-a"]
    assert limit_diagnostics["omitted_reasons"]["limit"] == 1


def test_db_memory_schema_fingerprint_uses_structural_schema_shape():
    source_identity = "sqlite:from_db:source-a"
    write_schema = {
        "database_type": "sqlite",
        "database_name": "/tmp/write.sqlite",
        "tables": [
            {
                "name": "orders",
                "metadata": {"catalog_runtime_id": "write-only"},
                "columns": [
                    {"name": "status", "data_type": "TEXT", "nullable": False},
                    {"name": "total", "data_type": "REAL"},
                    {"name": "id", "data_type": "INTEGER", "is_primary_key": True},
                ],
            }
        ],
        "foreign_keys": [],
    }
    planning_schema = {
        "database_type": "sqlite",
        "database_name": "/tmp/planning.sqlite",
        "tables": [
            {
                "name": "orders",
                "metadata": {"catalog_runtime_id": "planning-only"},
                "columns": [
                    {"name": "id", "data_type": "INTEGER", "is_primary_key": True},
                    {"name": "total", "data_type": "REAL"},
                    {"name": "status", "data_type": "TEXT"},
                ],
            }
        ],
        "foreign_keys": [],
    }
    changed_schema = {
        **planning_schema,
        "tables": [
            {
                **planning_schema["tables"][0],
                "columns": [
                    *planning_schema["tables"][0]["columns"],
                    {"name": "discount", "data_type": "REAL"},
                ],
            }
        ],
    }

    write_fingerprint = structural_schema_fingerprint(write_schema)
    planning_fingerprint = structural_schema_fingerprint(planning_schema)

    assert write_fingerprint == planning_fingerprint
    assert structural_schema_fingerprint(changed_schema) != planning_fingerprint

    evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                {
                    "chunk_id": "mem-revenue",
                    "metadata": {
                        "db_memory": {
                            "kind": "metric_definition",
                            "key": "metric:revenue",
                            "text": "Revenue uses orders.total.",
                            "metadata": {
                                "source_identity": source_identity,
                                "workspace_scope": "source",
                                "active": True,
                                "confidence": 0.9,
                                "source_schema_fingerprint": write_fingerprint,
                            },
                            "importance": 0.7,
                            "category": "db_semantics",
                        }
                    },
                    "score": 0.9,
                }
            ]
        },
    )

    refs, _evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (evidence,),
        prompt="Calculate revenue",
        schema=planning_schema,
        source_identity=source_identity,
        schema_fingerprint=planning_fingerprint,
        limit=3,
        char_budget=800,
        score_threshold=0.0,
    )

    assert [ref["key"] for ref in refs] == ["metric:revenue"]
    assert diagnostics["omitted_reasons"] == {}


def test_board_revenue_metric_memory_projects_semantic_contract():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        evidence_refs=("evidence-contract",),
    )
    assert contract is not None
    record = normalize_db_memory_record(
        {
            **record.to_dict(),
            "metadata": {
                **record.metadata,
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            },
        }
    )
    evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                {
                    "chunk_id": "mem-board-revenue",
                    "metadata": {"db_memory": record.to_dict()},
                    "score": 0.99,
                }
            ]
        },
    )

    refs, _evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (evidence,),
        prompt="Calculate board revenue",
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        limit=3,
        char_budget=800,
        score_threshold=0.0,
    )
    semantics, contract_diagnostics = project_db_memory_semantic_contracts(
        refs,
        prompt="Calculate board revenue",
        schema=schema,
        policy_summary={},
    )

    assert diagnostics["included_count"] == 1
    assert semantics[0]["key"] == "metric:board_revenue"
    assert semantics[0]["contract_kind"] == "metric_definition"
    assert semantics[0]["required_refs"] == [
        "orders.total",
        "refunds.amount",
        "orders.status",
    ]
    assert semantics[0]["required_relationships"] == ["refunds.order_id -> orders.id"]
    assert semantics[0]["enforceable"] is True
    assert contract_diagnostics["candidate_count"] == 1
    assert contract_diagnostics["enforced_count"] == 1


def test_memory_contract_schema_refs_preserve_normalization_and_dedupe_order():
    refs = db_memory_contract_refs(
        {
            "requirements": {
                "refs": [
                    {"ref": "Orders.Customer_ID"},
                    {"ref": {"table": "orders", "column": "customer_id"}},
                    {"ref": "customers.id"},
                ],
                "relationships": [{"from": "customers.id", "to": "orders.customer_id"}],
            }
        }
    )

    assert refs == (
        {"table": "Orders", "column": "Customer_ID"},
        {"table": "customers", "column": "id"},
    )
    schema = {
        "tables": [
            {"name": "orders", "columns": [{"name": "customer_id"}]},
            {"name": "customers", "columns": [{"name": "id"}]},
        ]
    }
    assert schema_refs_known_schema(refs, schema) is True
    assert (
        schema_refs_known_schema((*refs, {"table": "missing", "column": "id"}), schema)
        is False
    )
    assert schema_refs_known_schema(refs, {"tables": []}) is True


def test_memory_contract_confidence_token_and_omission_helpers_are_golden():
    assert confidence_value(None, default=0.25) == 0.25
    assert confidence_value(" high ", default=0.0) == 0.9
    assert confidence_value("medium", default=0.0) == 0.7
    assert confidence_value("low", default=0.0) == 0.4
    assert confidence_value(2, default=0.0) == 1.0
    assert confidence_value(-1, default=1.0) == 0.0
    assert confidence_value("unknown", default=0.25) == 0.25
    assert meaningful_tokens("Show gross_revenue revenue2 with X") == [
        "gross",
        "revenue",
        "gross_revenue",
        "revenue2",
        "revenue2",
    ]
    assert safe_omission_summaries({"z": 2, "a": 1, "zero": 0, "neg": -1}) == [
        {"reason": "a", "count": 1},
        {"reason": "z", "count": 2},
    ]


def test_db_memory_contracts_artifact_records_enforceable_advisory_and_omissions():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    assert contract is not None

    semantics, diagnostics = project_db_memory_semantic_contracts(
        [
            {
                "key": "metric:board_revenue",
                "kind": "metric_definition",
                "text": record.text,
                "confidence": 0.95,
                "source_identity": source_identity,
                "semantic_contract_status": "validated",
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            },
            {
                "key": "metric:warehouse_revenue",
                "kind": "metric_definition",
                "text": record.text,
                "confidence": 0.95,
                "source_identity": "sqlite:from_db:source-b",
                "semantic_contract_status": "validated",
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            },
        ],
        prompt="Calculate board revenue",
        schema=schema,
        policy_summary={},
        source_identity=source_identity,
    )
    payload = db_memory_contracts_artifact_payload(
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        recall_evidence_refs=("evidence-memory",),
        selection_evidence_ref={
            "id": "selection-evidence",
            "kind": "db.memory.selection",
            "owner": "db_runtime",
        },
        contracts=semantics,
        diagnostics=diagnostics,
    )

    assert [item["enforceable"] for item in payload["contracts"]] == [True, False]
    assert payload["enforceable_contracts"][0]["key"] == "metric:board_revenue"
    assert payload["advisory_contracts"][0]["omission_reason"] == "cross_source"
    assert payload["contract_omission_reasons"] == {"cross_source": 1}
    assert payload["source_schema_applicability"] == {
        "source_identity": source_identity,
        "schema_fingerprint": "schema-a",
        "contract_candidate_count": 2,
        "enforced_count": 1,
        "advisory_count": 1,
        "omitted_count": 1,
    }
    assert payload["safe_diagnostic_summaries"] == [
        {"reason": "cross_source", "count": 1}
    ]


def test_planning_context_renders_memory_compatibility_fields_from_artifacts():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    assert contract is not None
    record = normalize_db_memory_record(
        {
            **record.to_dict(),
            "metadata": {
                **record.metadata,
                "confidence": 0.95,
                "semantic_contract_status": "validated",
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            },
        }
    )
    memory_evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        accepted=True,
        payload={
            "results": [
                {
                    "chunk_id": "mem-board-revenue",
                    "metadata": {"db_memory": record.to_dict()},
                    "score": 0.99,
                }
            ]
        },
    )

    context = DbPlanningContextBuilder(
        DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "source_identity": source_identity,
                        "score_threshold": 0.0,
                    }
                }
            }
        )
    ).build(
        request=DbRequest("Calculate board revenue."),
        intent=DbIntent(
            kind=DbIntentKind.DATA_QUERY,
            confidence=1.0,
            access=AccessMode.READ,
        ),
        operation=Operation(id="op-memory-artifacts", operation_type="data.query"),
        schema_evidence=Evidence(
            id="schema-memory-artifacts",
            kind="schema.asset_profile",
            owner="sqlite",
            accepted=True,
            payload=schema,
        ),
        memory_recall_evidence=(memory_evidence,),
    )
    raw_payload = context.to_payload()

    assert context.db_memory_selection_artifact["included_refs"][0]["key"] == (
        "metric:board_revenue"
    )
    assert context.db_memory_contracts_artifact["contracts"][0]["key"] == (
        "metric:board_revenue"
    )
    assert raw_payload["db_memory_refs"][0]["key"] == (
        context.db_memory_selection_artifact["included_refs"][0]["key"]
    )
    assert raw_payload["db_memory_semantics"][0]["key"] == (
        context.db_memory_contracts_artifact["contracts"][0]["key"]
    )
    assert raw_payload["db_memory_semantics"][0]["enforceable"] is True
    assert "Database memory:" in raw_payload["rendered_context"]
    assert "Board revenue is complete order total minus refunds." in (
        raw_payload["rendered_context"]
    )


def test_valid_contract_memory_ref_survives_schema_scope_mismatch_for_projection():
    source_identity = "sqlite:from_db:source-a"
    full_schema = _board_revenue_schema()
    narrowed_schema = {
        "tables": [
            {
                "name": "orders",
                "columns": [{"name": "id"}, {"name": "status"}, {"name": "total"}],
            },
            {"name": "refunds", "columns": [{"name": "order_id"}]},
        ]
    }
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=full_schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    record = normalize_db_memory_record(
        {
            **record.to_dict(),
            "metadata": {
                **record.metadata,
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            },
        }
    )
    evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                {
                    "chunk_id": "mem-board-revenue",
                    "metadata": {"db_memory": record.to_dict()},
                    "score": 0.99,
                }
            ]
        },
    )

    refs, _evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (evidence,),
        prompt="Calculate board revenue",
        schema=narrowed_schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
        limit=3,
        char_budget=800,
        score_threshold=0.0,
    )

    assert [ref["key"] for ref in refs] == ["metric:board_revenue"]
    assert diagnostics["omitted_reasons"] == {}


def test_contract_projection_downgrades_cross_source_stale_and_low_confidence():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    base = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        base,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    assert contract is not None
    refs = [
        {
            "key": "metric:board_revenue",
            "kind": "metric_definition",
            "text": base.text,
            "confidence": 0.7,
            "semantic_contract_status": "validated",
            DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
        },
        {
            "key": "metric:board_revenue",
            "kind": "metric_definition",
            "text": base.text,
            "confidence": 0.95,
            "semantic_contract_status": "validated",
            DB_MEMORY_SEMANTIC_CONTRACT_KEY: {
                **contract,
                "requirements": {
                    **contract["requirements"],
                    "refs": [{"kind": "column", "ref": "missing.total"}],
                },
            },
        },
    ]

    semantics, diagnostics = project_db_memory_semantic_contracts(
        refs,
        prompt="Calculate board revenue",
        schema=schema,
        policy_summary={},
    )

    assert [item["enforceable"] for item in semantics] == [False, False]
    assert diagnostics["enforced_count"] == 0
    assert diagnostics["advisory_count"] == 2
    assert diagnostics["omitted_reasons"]["low_confidence"] == 1
    assert diagnostics["omitted_reasons"]["schema_scope_mismatch"] == 1


def test_contract_projection_keeps_unvalidated_contract_metadata_advisory():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    assert contract is not None

    semantics, diagnostics = project_db_memory_semantic_contracts(
        [
            {
                "key": "metric:board_revenue",
                "kind": "metric_definition",
                "text": record.text,
                "confidence": 0.95,
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            }
        ],
        prompt="Calculate board revenue",
        schema=schema,
        policy_summary={},
    )

    assert semantics[0]["enforceable"] is False
    assert diagnostics["enforced_count"] == 0
    assert diagnostics["advisory_count"] == 1
    assert diagnostics["omitted_reasons"]["unvalidated_contract"] == 1


def test_contract_projection_blocks_policy_denied_refs():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    assert contract is not None

    semantics, diagnostics = project_db_memory_semantic_contracts(
        [
            {
                "key": "metric:board_revenue",
                "kind": "metric_definition",
                "text": record.text,
                "confidence": 0.95,
                "semantic_contract_status": "validated",
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            }
        ],
        prompt="Calculate board revenue",
        schema=schema,
        policy_summary={"blocked_columns": ["refunds.amount"]},
    )

    assert semantics[0]["enforceable"] is False
    assert diagnostics["enforced_count"] == 0
    assert diagnostics["omitted_reasons"]["blocked_by_policy"] == 1


def test_planning_context_projects_blocked_memory_refs_before_rendering():
    class BlockedSource:
        read_only = True
        allowed_tables = set()
        blocked_tables = set()
        blocked_columns = {"customers.loyalty_band"}

    source_identity = "sqlite:from_db:source-a"
    schema = {
        "database_type": "sqlite",
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "loyalty_band", "data_type": "TEXT"},
                    {"name": "revenue", "data_type": "REAL"},
                ],
            }
        ],
    }
    semantic_contract = {
        "version": 1,
        "contract_kind": "metric_definition",
        "subject": {
            "type": "metric",
            "key": "metric:loyalty_revenue",
            "aliases": ["loyalty revenue"],
        },
        "requirements": {
            "refs": [{"kind": "column", "ref": "customers.loyalty_band"}],
            "filters": [
                {
                    "ref": "customers.loyalty_band",
                    "operator": "=",
                    "value": "platinum",
                }
            ],
        },
        "grounding": {
            "source_identity": source_identity,
            "schema_fingerprint": "schema-a",
            "evidence_refs": [],
            "catalog_refs": [],
        },
        "enforcement": {
            "mode": "required_when_recalled",
            "min_confidence": 0.8,
        },
    }
    memory_evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        accepted=True,
        payload={
            "results": [
                {
                    "chunk_id": "mem-loyalty",
                    "metadata": {
                        "db_memory": {
                            "kind": "metric_definition",
                            "key": "metric:loyalty_revenue",
                            "text": (
                                "Loyalty revenue filters customers.loyalty_band "
                                "to platinum."
                            ),
                            "metadata": {
                                "source_identity": source_identity,
                                "workspace_scope": "source",
                                "active": True,
                                "confidence": 0.95,
                                "semantic_contract_status": "validated",
                                DB_MEMORY_SEMANTIC_CONTRACT_KEY: semantic_contract,
                                "schema_refs": ["customers.loyalty_band"],
                            },
                            "importance": 0.8,
                            "category": "db_semantics",
                        }
                    },
                    "score": 0.99,
                }
            ]
        },
    )

    context = DbPlanningContextBuilder(
        DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": True,
                        "source_identity": source_identity,
                        "score_threshold": 0.0,
                    }
                }
            }
        )
    ).build(
        request=DbRequest("Summarize loyalty revenue."),
        intent=DbIntent(
            kind=DbIntentKind.DATA_QUERY,
            confidence=1.0,
            access=AccessMode.READ,
        ),
        operation=Operation(id="op-memory-projection", operation_type="data.query"),
        schema_evidence=Evidence(
            id="schema-memory-projection",
            kind="schema.asset_profile",
            owner="sqlite",
            accepted=True,
            payload=schema,
        ),
        memory_recall_evidence=(memory_evidence,),
        source=BlockedSource(),
    )

    dumped = json.dumps(
        {
            "db_memory_refs": context.db_memory_refs,
            "db_memory_semantics": context.db_memory_semantics,
            "rendered_context": context.rendered_context,
        },
        sort_keys=True,
    )
    assert "customers.loyalty_band" not in dumped
    assert "platinum" not in dumped
    assert context.db_memory_refs[0]["projection"]["reason"] == "blocked_by_policy"
    assert context.db_memory_semantics[0]["enforceable"] is False
    raw_payload = context.to_payload()
    assert raw_payload["db_memory_semantics"][0]["enforceable"] is False
    assert (
        context.db_memory_contracts_artifact["source_schema_applicability"][
            "enforced_count"
        ]
        == 0
    )
    assert (
        context.db_memory_contracts_artifact["contract_omission_reasons"][
            "blocked_by_policy"
        ]
        == 1
    )


def test_mixed_contract_projection_keeps_enforceable_and_advisory_separate():
    source_identity = "sqlite:from_db:source-a"
    schema = _board_revenue_schema()
    record = DBMemoryRecord(
        kind="metric_definition",
        key="metric:board_revenue",
        text="Board revenue is complete order total minus refunds.",
        metadata=_board_revenue_contract_metadata(source_identity),
    )
    contract = extract_db_memory_semantic_contract(
        record,
        schema=schema,
        source_identity=source_identity,
        schema_fingerprint="schema-a",
    )
    assert contract is not None

    semantics, diagnostics = project_db_memory_semantic_contracts(
        [
            {
                "key": "metric:board_revenue",
                "kind": "metric_definition",
                "text": record.text,
                "confidence": 0.95,
                "semantic_contract_status": "validated",
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: contract,
            },
            {
                "key": "metric:legacy_revenue",
                "kind": "metric_definition",
                "text": "Legacy revenue is old advisory memory.",
                "confidence": 0.95,
            },
            {
                "key": "metric:unsupported_revenue",
                "kind": "metric_definition",
                "text": "Unsupported contract version.",
                "confidence": 0.95,
                DB_MEMORY_SEMANTIC_CONTRACT_KEY: {
                    **contract,
                    "version": 999,
                },
            },
        ],
        prompt="Calculate board revenue",
        schema=schema,
        policy_summary={},
    )

    assert [item["enforceable"] for item in semantics] == [True]
    assert diagnostics["candidate_count"] == 1
    assert diagnostics["enforced_count"] == 1
    assert diagnostics["advisory_count"] == 2
    assert diagnostics["omitted_reasons"]["invalid_contract"] == 1


def test_db_memory_planning_recall_skips_pii_row_lookup():
    decision = db_memory_planning_recall_decision(
        prompt="Look up the customer email for customer_id 123",
        intent_kind="data.query",
        schema={"tables": [{"name": "customers", "columns": [{"name": "email"}]}]},
        memory_config={
            "enabled": True,
            "recall": "auto",
            "limit": 3,
            "char_budget": 800,
        },
    )

    assert decision == {"recall": False, "reason": "row_level_or_pii_prompt"}


def test_db_memory_planning_recall_query_uses_prompt_matched_table_only():
    schema = {
        "tables": [
            {"name": "agents", "columns": [{"name": "agent_id"}]},
            {"name": "api_keys", "columns": [{"name": "api_key_id"}]},
            {"name": "operations", "columns": [{"name": "operation_id"}]},
        ]
    }

    query = db_memory_planning_recall_query(
        "Can you tell me what the operations table is?",
        schema,
        "schema.query",
    )

    assert "Matched schema terms: operations" in query
    assert "operations table" in query
    assert "agents" not in query
    assert "agents.agent_id" not in query
    assert "api_keys.api_key_id" not in query


def test_db_memory_planning_recall_query_includes_explicit_column_refs():
    schema = {
        "tables": [
            {"name": "orders", "columns": [{"name": "total"}, {"name": "status"}]},
            {"name": "refunds", "columns": [{"name": "amount"}]},
        ]
    }

    query = db_memory_planning_recall_query(
        "What does orders total mean?",
        schema,
        "metric.query",
    )

    assert "orders" in query
    assert "orders.total" in query
    assert "orders.status" not in query
    assert "refunds.amount" not in query


def test_db_memory_planning_recall_query_includes_relationship_tables():
    schema = {
        "tables": [
            {"name": "operations", "columns": [{"name": "deployment_id"}]},
            {"name": "deployments", "columns": [{"name": "deployment_id"}]},
        ]
    }

    query = db_memory_planning_recall_query(
        "How do operations relate to deployments?",
        schema,
        "schema.relationship_query",
    )

    assert "operations" in query
    assert "deployments" in query
    assert "relationship" in query


def test_db_memory_planning_recall_query_respects_matched_schema_terms_and_bounds():
    schema = {
        "tables": [{"name": f"table_{index}", "columns": []} for index in range(80)]
    }
    terms = [f"selected_{index}" for index in range(60)]

    query = db_memory_planning_recall_query(
        "Explain selected tables",
        schema,
        "schema.query",
        matched_schema_terms=terms,
    )

    assert "selected_0" in query
    assert "selected_23" in query
    assert "selected_24" not in query
    assert "table_0" not in query
    assert len(query) < 900


def test_db_memory_planning_recall_guards_disabled_zero_and_disallowed_intents():
    base_config = {
        "enabled": True,
        "recall": "auto",
        "limit": 3,
        "char_budget": 800,
    }
    schema = {"tables": [{"name": "operations", "columns": []}]}

    disabled = db_memory_planning_recall_decision(
        prompt="What is operations?",
        intent_kind="schema.query",
        schema=schema,
        memory_config={**base_config, "enabled": False},
    )
    recall_off = db_memory_planning_recall_decision(
        prompt="What is operations?",
        intent_kind="schema.query",
        schema=schema,
        memory_config={**base_config, "recall": "off"},
    )
    limit_zero = db_memory_planning_recall_decision(
        prompt="What is operations?",
        intent_kind="schema.query",
        schema=schema,
        memory_config={**base_config, "limit": 0},
    )
    char_budget_zero = db_memory_planning_recall_decision(
        prompt="What is operations?",
        intent_kind="schema.query",
        schema=schema,
        memory_config={**base_config, "char_budget": 0},
    )
    disallowed = db_memory_planning_recall_decision(
        prompt="Remember operations are agent runs.",
        intent_kind="memory.update",
        schema=schema,
        memory_config=base_config,
    )

    assert disabled["reason"] == "memory_disabled"
    assert recall_off["reason"] == "recall_disabled"
    assert limit_zero["reason"] == "limit_zero"
    assert char_budget_zero["reason"] == "char_budget_zero"
    assert disallowed["reason"] == "intent_not_semantic_query"


def test_db_memory_planning_context_remains_bounded():
    source_identity = "sqlite:from_db:source-a"
    evidence = Evidence(
        id="evidence-memory",
        kind="memory.semantic.recall",
        owner="memory",
        payload={
            "results": [
                {
                    "chunk_id": f"mem-{index}",
                    "metadata": {
                        "db_memory": {
                            "kind": "metric_definition",
                            "key": f"metric:revenue:{index}",
                            "text": f"Revenue rule {index} excludes refunds.",
                            "metadata": {
                                "source_identity": source_identity,
                                "workspace_scope": "source",
                                "active": True,
                                "confidence": 0.9,
                            },
                            "importance": 0.7,
                            "category": "db_semantics",
                        }
                    },
                    "score": 0.9,
                }
                for index in range(6)
            ]
        },
    )

    refs, _evidence_refs, diagnostics = db_memory_refs_from_recall_evidence(
        (evidence,),
        prompt="How should revenue be calculated?",
        schema={"tables": []},
        source_identity=source_identity,
        schema_fingerprint=None,
        limit=2,
        char_budget=95,
        score_threshold=0.0,
    )

    rendered = "\n".join(f"- {ref['kind']} {ref['key']}: {ref['text']}" for ref in refs)
    assert len(refs) <= 2
    assert len(rendered) <= 95
    assert diagnostics["omitted_reasons"]["budget"] >= 1


async def test_db_memory_helpers_marker_lookup_uses_exact_category_listing():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(
        return_value=[
            {"content": "DB exact cache marker: numeric_unit_calibration:abc"}
        ]
    )
    backend.recall = AsyncMock(return_value=[])
    memory = MemoryPlugin()
    memory.backend = backend

    assert await has_db_memory_marker(memory, "numeric_unit_calibration:abc") is True

    backend.list_by_category.assert_awaited_once_with(
        category="db_cache_marker",
        limit=1000,
    )
    backend.recall.assert_not_awaited()


async def test_runtime_memory_calibration_writes_through_capability_boundary():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = DbRuntime(plugins=(SchemaInspectPlugin(), memory))

    result = await calibrate_db_memory(
        runtime,
        source_owner="schema_probe",
        marker_key="numeric_unit_calibration:probe",
    )
    tasks = await runtime.store.list_tasks()

    assert result["calibrated"] is True
    assert result["record_count"] == 1
    assert backend.remember.await_count == 2
    assert sum(task.capability_id == "memory.semantic.write" for task in tasks) == 2
    assert any(task.capability_id == "db.schema.inspect" for task in tasks)


def test_db_memory_record_validation():
    record = normalize_db_memory_record(
        {
            "kind": "metric_definition",
            "key": "metric:revenue",
            "text": "Revenue excludes refunded orders.",
            "importance": 2,
        }
    )

    assert record.importance == 1.0
    assert record.category == "db_semantics"

    with pytest.raises(ValueError, match="Unsupported DB memory kind"):
        normalize_db_memory_record({"kind": "row", "key": "x", "text": "bad"})
