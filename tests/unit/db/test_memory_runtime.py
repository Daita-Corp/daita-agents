from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from unittest.mock import AsyncMock, MagicMock

from daita.db import (
    DbEvidenceStore,
    DbOperationExecutor,
    DbRequest,
    DbRuntime,
    DbRuntimeConfig,
)
from daita.db.analysis import structural_schema_fingerprint
from daita.db.llm_service import DbLLMConfig, DbLLMResponse, DbLLMService
from daita.db.llm_planner import _planner_messages
from daita.db.memory import (
    DBMemoryRecord,
    calibrate_db_memory,
    db_answer_memory_refs_from_recall_evidence,
    db_memory_planning_recall_decision,
    db_memory_planning_recall_query,
    db_memory_refs_from_recall_evidence,
    has_db_memory_marker,
    normalize_db_memory_record,
    recall_db_memory_records,
    write_db_memory_record,
    write_db_memory_records,
)
from daita.db.memory_contracts import (
    DB_MEMORY_SEMANTIC_CONTRACT_KEY,
    extract_db_memory_semantic_contract,
    project_db_memory_semantic_contracts,
)
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
                        "workspace_scope": "source",
                        "source_identity": "test:memory-runtime-source",
                    }
                }
            },
        )
    )


async def _plan_memory_update(
    runtime: DbRuntime,
    request: DbRequest,
    *,
    schema: dict | None = None,
) -> tuple[Operation, Evidence]:
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="memory.update",
        request={"prompt": request.prompt},
        required_evidence=("db.memory.proposal",),
        evaluate_governance=False,
    )
    task = Task(
        id=f"{operation.id}.memory-plan-update",
        operation_id=operation.id,
        capability_id="db.memory.plan_update",
        executor_id="db_runtime.memory.plan_update",
        input={
            "request": {
                "prompt": request.prompt,
                "mode": request.mode,
                "metadata": request.metadata,
                "constraints": request.constraints,
                "source_scope": list(request.source_scope),
                "requested_capabilities": list(request.requested_capabilities),
            },
            "schema": schema or {},
        },
        required_evidence=frozenset({"db.memory.proposal"}),
        metadata={"owner": "db_runtime"},
    )

    evidence = await runtime.execute_task(task, operation)
    proposal = next(item for item in evidence if item.kind == "db.memory.proposal")
    return operation, proposal


async def _commit_memory_update(
    runtime: DbRuntime,
    operation: Operation,
    proposal: Evidence,
) -> tuple[Evidence, ...]:
    task = Task(
        id=f"{operation.id}.memory-commit-update",
        operation_id=operation.id,
        capability_id="db.memory.commit_update",
        executor_id="db_runtime.memory.commit_update",
        input={
            "proposal_evidence_id": proposal.id,
            "proposal_fingerprint": proposal.payload["proposal_fingerprint"],
            "source_identity": "test:memory-runtime-source",
        },
        required_evidence=frozenset({"db.memory.definition"}),
        metadata={"owner": "db_runtime"},
    )
    return await runtime.execute_task(task, operation)


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


class _FakeMemorySynthesisLLMService(DbLLMService):
    def __init__(self) -> None:
        super().__init__(DbLLMConfig(provider="fake", model="memory-synthesis-test"))
        self.calls: list[list[dict[str, str]]] = []

    @property
    def available(self) -> bool:
        return True

    async def generate_json(self, messages: list[dict[str, str]]) -> DbLLMResponse:
        return await self.generate_synthesis_json(messages)

    async def generate_synthesis_json(
        self, messages: list[dict[str, str]]
    ) -> DbLLMResponse:
        self.calls.append(messages)
        return DbLLMResponse(
            content='{"answer": "LLM should not answer memory recall"}',
            diagnostics={"provider": "fake", "model": "memory-synthesis-test"},
        )


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

    def declare_capabilities(self) -> tuple[Capability, ...]:
        return (
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
            ),
        )

    def get_executors(self) -> tuple[SchemaInspectExecutor, ...]:
        return (self.executor,)

    def declare_evidence_schemas(self) -> tuple[EvidenceSchema, ...]:
        return (
            EvidenceSchema(
                kind="schema.asset_profile",
                owner="schema_probe",
                json_schema={"type": "object"},
            ),
        )


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
        "schema.relationships",
    } <= recall.operation_types
    planning_context = runtime.registry.get_capability(
        "db.planning.context.build",
        owner="db_runtime",
    )
    assert {
        "schema.query",
        "schema.relationships",
    } <= planning_context.operation_types
    answer_synthesis = runtime.registry.get_capability(
        "db.answer.synthesize",
        owner="db_runtime",
    )
    assert {
        "schema.query",
        "schema.relationships",
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
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            total REAL NOT NULL
        );
        INSERT INTO orders (total) VALUES (10.0), (20.0);
        """
    )
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
        safety_frame = runtime.build_safety_frame(request)
        contract = runtime.build_contract(request, safety_frame=safety_frame)
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
                "access": contract.access.value,
                "safety_frame": safety_frame.to_dict(),
                "granted_lanes": [lane.value for lane in safety_frame.granted_lanes],
                "forbidden_capabilities": list(safety_frame.forbidden_capabilities),
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
        evidence_store = DbEvidenceStore()
        tasks = []
        context = await DbOperationExecutor(runtime)._build_planning_context(
            request,
            operation,
            tasks,
            evidence_store,
            schema_evidence=replace(schema_evidence, id=None),
            catalog_evidence=(),
            relationship_evidence=(),
        )
    finally:
        await runtime.teardown()

    recall = next(
        item for item in evidence_store.list() if item.kind == "memory.semantic.recall"
    )
    assert [task.capability_id for task in tasks][-2:] == [
        "memory.semantic.recall",
        "db.planning.context.build",
    ]
    assert recall.owner == "memory"
    assert backend.recall_db_records.await_args.kwargs["category"] == "db_semantics"
    assert backend.recall.await_count == 0
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
    assert context.payload["db_memory_evidence_refs"] == [recall.id]
    assert "Database memory:" in context.payload["rendered_context"]
    assert "DB memory record:" not in context.payload["rendered_context"]


def test_llm_planner_includes_db_memory_advisory_contract():
    system_message = _planner_messages({"rendered_context": "Database memory:\n- x"})[
        0
    ]["content"]
    assert "DB memory is semantic business context" in system_message
    assert "must satisfy the memory contract" in system_message
    assert "Schema, catalog, policy, SQL validation" in system_message


@pytest.mark.parametrize(
    ("operation_type", "prompt"),
    [
        ("schema.query", "What does the operations table mean?"),
        (
            "schema.relationships",
            "How are customers related to orders in business terms?",
        ),
    ],
)
def test_db_memory_planning_recall_allows_metadata_operations(operation_type, prompt):
    decision = db_memory_planning_recall_decision(
        prompt=prompt,
        operation_type=operation_type,
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


def test_db_memory_planning_recall_does_not_match_sum_inside_summary():
    decision = db_memory_planning_recall_decision(
        prompt="Give me a brief summary of what the operations table is",
        operation_type="schema.query",
        schema={"tables": [{"name": "operations", "columns": [{"name": "id"}]}]},
        memory_config={
            "enabled": True,
            "recall": "auto",
            "limit": 3,
            "char_budget": 800,
        },
    )

    assert decision["recall"] is True
    assert decision["reason"] != "direct_schema_matched_query"


@pytest.mark.parametrize(
    "operation_type",
    [
        "write.propose",
        "memory.update",
        "write.execute",
        "admin",
        "conversational",
        "lineage.trace",
    ],
)
def test_db_memory_planning_recall_excludes_non_allowlisted_metadata_operations(
    operation_type,
):
    decision = db_memory_planning_recall_decision(
        prompt="What does operations mean?",
        operation_type=operation_type,
        schema={"tables": [{"name": "operations", "columns": [{"name": "id"}]}]},
        memory_config={
            "enabled": True,
            "recall": "auto",
            "limit": 3,
            "char_budget": 800,
        },
    )

    assert decision == {"recall": False, "reason": "operation_not_memory_eligible"}


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
def test_db_memory_planning_recall_metadata_operations_keep_global_guards(
    memory_config,
    reason,
):
    decision = db_memory_planning_recall_decision(
        prompt="What does operations mean?",
        operation_type="schema.query",
        schema={"tables": [{"name": "operations", "columns": [{"name": "id"}]}]},
        memory_config=memory_config,
    )

    assert decision == {"recall": False, "reason": reason}


async def test_schema_execution_with_memory_produces_planning_context(tmp_path):
    db_path = tmp_path / "metadata_memory.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE operations (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO operations (id, status) VALUES (1, 'complete');
        """
    )
    source_identity = "sqlite:from_db:metadata-memory"
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="table:operations",
                text="operations are agent runs",
                source_identity=source_identity,
                table="operations",
            )
        ]
    )
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:metadata-memory",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "source_identity": source_identity,
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run(
            DbRequest(
                "Can you tell me what the operations table is?",
                mode="schema.query",
            )
        )
    finally:
        await runtime.teardown()

    evidence_kinds = {item.kind for item in result.evidence}
    assert {
        "schema.search_result",
        "schema.asset_profile",
        "memory.semantic.recall",
        "planning.context",
        "answer.memory.context",
        "verification.result",
        "answer.synthesis",
    } <= evidence_kinds
    assert "query.result" not in evidence_kinds
    planning_context = next(
        item for item in result.evidence if item.kind == "planning.context"
    )
    assert planning_context.payload["db_memory_refs"][0]["text"] == (
        "operations are agent runs"
    )
    answer_context = next(
        item for item in result.evidence if item.kind == "answer.memory.context"
    )
    assert answer_context.payload["refs"][0]["text"] == "operations are agent runs"
    assert "Database memory:" in planning_context.payload["rendered_context"]
    assert result.answer is not None
    assert "operations: id, status" in result.answer
    assert "Semantic memory note: operations are agent runs" in result.answer


async def test_direct_memory_recall_answers_from_answer_memory_context(tmp_path):
    db_path = tmp_path / "direct_memory_recall.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE operations (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        """
    )
    source_identity = "sqlite:from_db:direct-memory"
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="business_rule:operations_are_agent_runs",
                text="operations are agent runs",
                source_identity=source_identity,
                table="operations",
            )
        ]
    )
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:direct-memory",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "source_identity": source_identity,
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run(
            DbRequest("recall what memories you have for the operations table")
        )
    finally:
        await runtime.teardown()

    assert result.operation_type == "memory.recall"
    assert result.answer is not None
    assert "operations are agent runs" in result.answer
    assert "operations: id, status" not in result.answer
    evidence_kinds = {item.kind for item in result.evidence}
    assert {"memory.semantic.recall", "answer.memory.context"} <= evidence_kinds
    answer_context = next(
        item for item in result.evidence if item.kind == "answer.memory.context"
    )
    assert answer_context.payload["refs"][0]["text"] == "operations are agent runs"


async def test_direct_memory_recall_uses_deterministic_synthesis_when_llm_available(
    tmp_path,
):
    db_path = tmp_path / "direct_memory_recall_llm.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE operations (
            id INTEGER PRIMARY KEY
        );
        """
    )
    source_identity = "sqlite:from_db:direct-memory-llm"
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="business_rule:operations_are_agent_runs",
                text="operations are agent runs",
                source_identity=source_identity,
                table="operations",
            )
        ]
    )
    llm_service = _FakeMemorySynthesisLLMService()
    runtime = DbRuntime(
        source=sqlite,
        db_llm_service=llm_service,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:direct-memory-llm",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "source_identity": source_identity,
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run(
            DbRequest("recall what memories you have for the operations table")
        )
    finally:
        await runtime.teardown()

    assert llm_service.calls == []
    assert result.answer is not None
    assert "operations are agent runs" in result.answer
    synthesis = next(
        item for item in result.evidence if item.kind == "answer.synthesis"
    )
    assert (
        synthesis.payload["diagnostics"]["fallback_reason"]
        == "deterministic_memory_synthesis"
    )


async def test_relationship_schema_execution_can_include_planning_context(tmp_path):
    db_path = tmp_path / "relationship_metadata_memory.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id)
        );
        """
    )
    source_identity = "sqlite:from_db:relationship-memory"
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="relationship:customers_orders",
                text="orders represent purchases made by customers",
                source_identity=source_identity,
                table="orders",
            )
        ]
    )
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:relationship-memory",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "source_identity": source_identity,
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run(
            DbRequest(
                "How are customers related to orders?",
                mode="schema.relationships",
            )
        )
    finally:
        await runtime.teardown()

    planning_context = next(
        item for item in result.evidence if item.kind == "planning.context"
    )
    assert planning_context.payload["db_memory_refs"][0]["text"] == (
        "orders represent purchases made by customers"
    )
    assert planning_context.payload["relationship_evidence_refs"]
    assert any(item.kind == "schema.relationship_path" for item in result.evidence)
    assert result.answer is not None
    assert "Semantic memory note:" in result.answer


async def test_schema_execution_memory_disabled_does_not_recall(tmp_path):
    db_path = tmp_path / "metadata_memory_disabled.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE operations (
            id INTEGER PRIMARY KEY
        );
        """
    )
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="table:operations",
                text="operations are agent runs",
                source_identity="sqlite:from_db:disabled",
                table="operations",
            )
        ]
    )
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:disabled",
                    "memory": {
                        "enabled": False,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "source_identity": "sqlite:from_db:disabled",
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run(
            DbRequest("What is the operations table?", mode="schema.query")
        )
    finally:
        await runtime.teardown()

    assert not [
        item for item in result.evidence if item.kind == "memory.semantic.recall"
    ]
    planning_context = next(
        item for item in result.evidence if item.kind == "planning.context"
    )
    assert planning_context.payload["db_memory_refs"] == []
    assert planning_context.payload["db_memory_diagnostics"]["decision"] == {
        "recall": False,
        "reason": "memory_disabled",
    }
    memory.backend.recall_db_records.assert_not_awaited()


async def test_schema_execution_filters_unrelated_or_cross_source_memory(tmp_path):
    db_path = tmp_path / "metadata_memory_filtered.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE operations (
            id INTEGER PRIMARY KEY
        );
        CREATE TABLE inventory (
            id INTEGER PRIMARY KEY
        );
        """
    )
    source_identity = "sqlite:from_db:source-a"
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend(
        [
            _db_memory_recall_result(
                key="table:operations",
                text="operations mean something else in another source",
                source_identity="sqlite:from_db:source-b",
                table="operations",
            ),
            _db_memory_recall_result(
                key="inventory-note",
                text="inventory tracks stock counts",
                source_identity=source_identity,
                table="inventory",
            ),
        ]
    )
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:source-a",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "score_threshold": 0.0,
                        "source_identity": source_identity,
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run(
            DbRequest("What is the operations table?", mode="schema.query")
        )
    finally:
        await runtime.teardown()

    planning_context = next(
        item for item in result.evidence if item.kind == "planning.context"
    )
    assert any(item.kind == "memory.semantic.recall" for item in result.evidence)
    assert planning_context.payload["db_memory_refs"] == []
    assert planning_context.payload["db_memory_diagnostics"]["omitted_reasons"] == {
        "cross_source": 1,
        "irrelevant": 1,
    }
    assert result.answer is not None
    assert "Semantic memory note:" not in result.answer


async def test_simple_data_query_fast_path_still_avoids_planning_context_with_memory(
    tmp_path,
):
    db_path = tmp_path / "simple_data_query_memory.sqlite"
    sqlite = SQLitePlugin(path=str(db_path))
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY
        );
        INSERT INTO orders (id) VALUES (1), (2);
        """
    )
    memory = MemoryPlugin()
    memory.backend = _mock_db_memory_backend([])
    runtime = DbRuntime(
        source=sqlite,
        config=DbRuntimeConfig(
            plugins=(CatalogPlugin(auto_persist=False), sqlite, memory),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "from_db:simple-fast-path",
                    "memory": {
                        "enabled": True,
                        "recall": "auto",
                        "limit": 3,
                        "char_budget": 800,
                        "source_identity": "sqlite:from_db:simple-fast-path",
                    },
                }
            },
        ),
    )

    try:
        await runtime.setup()
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert snapshot is not None
    assert result.answer == "The count is 2."
    assert not [item for item in snapshot.evidence if item.kind == "planning.context"]
    assert not [
        item for item in snapshot.evidence if item.kind == "memory.semantic.recall"
    ]
    memory.backend.recall_db_records.assert_not_awaited()


async def test_memory_fact_query_executor_uses_memory_owned_fact_store(tmp_path):
    memory = _memory(tmp_path)
    assert memory.backend is not None
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
    assert memory.backend is not None
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


async def test_db_runtime_executes_memory_update_with_typed_evidence(tmp_path):
    runtime = _runtime_with_memory_source(_memory(tmp_path))

    operation, proposal = await _plan_memory_update(
        runtime,
        DbRequest(
            "Remember that revenue excludes tax",
            mode="memory.update",
            metadata={"category": "db_semantic"},
        ),
    )
    evidence = await _commit_memory_update(runtime, operation, proposal)

    definition = next(item for item in evidence if item.kind == "db.memory.definition")
    write = next(
        item for item in evidence if item.kind == "memory.semantic.write"
    )
    assert proposal.accepted is True
    assert definition.accepted is True
    assert definition.payload["proposal_evidence_id"] == proposal.id
    assert write.owner == "memory"
    assert write.payload["success"] is True
    assert write.payload["status"] == "created"
    assert write.payload["stored"]["structured"] is True


async def test_db_runtime_memory_update_stores_db_semantic_record():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    operation, proposal = await _plan_memory_update(
        runtime,
        DbRequest(
            "Remember the revenue rule",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:revenue_refunds",
                "text": "Revenue excludes refunded orders.",
                "metadata": {"metric": "revenue"},
                "importance": 0.8,
            },
        ),
    )
    evidence = await _commit_memory_update(runtime, operation, proposal)

    write = next(
        item for item in evidence if item.kind == "memory.semantic.write"
    )
    assert proposal.accepted is True
    assert write.payload["success"] is True
    assert write.payload["kind"] == "business_rule"
    assert write.payload["category"] == "db_semantics"
    backend.remember.assert_awaited_once()
    assert backend.remember.await_args.kwargs["category"] == "db_semantics"
    assert backend.remember.await_args.kwargs["index_content"] == (
        "Revenue excludes refunded orders."
    )
    assert (
        backend.remember.await_args.kwargs["extra_metadata"]["db_memory"]["key"]
        == "business_rule:revenue_refunds"
    )


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
    cursor.execute(
        """
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
        """
    )
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
    cursor.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            content TEXT NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
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

    changed_fingerprint = structural_schema_fingerprint(changed_schema)
    stale_planner_refs, _planner_evidence_refs, stale_planner_diagnostics = (
        db_memory_refs_from_recall_evidence(
            (evidence,),
            prompt="Calculate revenue",
            schema=changed_schema,
            source_identity=source_identity,
            schema_fingerprint=changed_fingerprint,
            limit=3,
            char_budget=800,
            score_threshold=0.0,
        )
    )
    answer_refs, _answer_evidence_refs, answer_diagnostics = (
        db_answer_memory_refs_from_recall_evidence(
            (evidence,),
            prompt="Calculate revenue",
            schema=changed_schema,
            source_identity=source_identity,
            schema_fingerprint=changed_fingerprint,
            limit=3,
            char_budget=800,
            score_threshold=0.0,
        )
    )

    assert stale_planner_refs == ()
    assert stale_planner_diagnostics["omitted_reasons"] == {"stale_schema": 1}
    assert [ref["key"] for ref in answer_refs] == ["metric:revenue"]
    assert answer_refs[0]["caveats"] == ["schema_fingerprint_mismatch"]
    assert answer_diagnostics["caveat_reasons"] == {"schema_fingerprint_mismatch": 1}


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
        operation_type="data.query",
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
        "data.query",
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
        "schema.relationships",
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
        operation_type="schema.query",
        schema=schema,
        memory_config={**base_config, "enabled": False},
    )
    recall_off = db_memory_planning_recall_decision(
        prompt="What is operations?",
        operation_type="schema.query",
        schema=schema,
        memory_config={**base_config, "recall": "off"},
    )
    limit_zero = db_memory_planning_recall_decision(
        prompt="What is operations?",
        operation_type="schema.query",
        schema=schema,
        memory_config={**base_config, "limit": 0},
    )
    char_budget_zero = db_memory_planning_recall_decision(
        prompt="What is operations?",
        operation_type="schema.query",
        schema=schema,
        memory_config={**base_config, "char_budget": 0},
    )
    disallowed = db_memory_planning_recall_decision(
        prompt="Remember operations are agent runs.",
        operation_type="memory.update",
        schema=schema,
        memory_config=base_config,
    )

    assert disabled["reason"] == "memory_disabled"
    assert recall_off["reason"] == "recall_disabled"
    assert limit_zero["reason"] == "limit_zero"
    assert char_budget_zero["reason"] == "char_budget_zero"
    assert disallowed["reason"] == "operation_not_memory_eligible"


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

    assert result is not None
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


async def test_db_runtime_memory_update_replaces_existing_record_by_key():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(
        return_value=[
            {
                "chunk_id": "old-1",
                "content": "old content",
                "metadata": {
                    "db_memory": {
                        "kind": "metric_definition",
                        "key": "metric:revenue",
                    }
                },
            },
            {
                "chunk_id": "other-1",
                "content": "other content",
                "metadata": {
                    "db_memory": {
                        "kind": "metric_definition",
                        "key": "metric:margin",
                    }
                },
            },
        ]
    )
    backend.delete_chunks = AsyncMock()
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "new-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    operation, proposal = await _plan_memory_update(
        runtime,
        DbRequest(
            "Remember the revenue metric",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded orders.",
            },
        ),
    )
    evidence = await _commit_memory_update(runtime, operation, proposal)

    write = next(
        item for item in evidence if item.kind == "memory.semantic.write"
    )
    assert proposal.accepted is True
    assert write.payload["status"] == "updated"
    assert write.payload["updated"] == 1
    assert backend.list_by_category.await_count == 2
    backend.list_by_category.assert_awaited_with(category="db_semantics", limit=1000)
    backend.delete_chunks.assert_awaited_once_with(["old-1"])


async def test_db_runtime_memory_update_appends_when_backend_cannot_delete_existing_key():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(
        return_value=[
            {
                "chunk_id": "old-1",
                "content": 'DB memory record:\n{"key": "business_rule:refunds"}',
                "metadata": {},
            }
        ]
    )
    del backend.delete_chunks
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "new-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    operation, proposal = await _plan_memory_update(
        runtime,
        DbRequest(
            "Remember the refunds rule",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:refunds",
                "text": "Refunded orders are excluded from revenue.",
            },
        ),
    )
    evidence = await _commit_memory_update(runtime, operation, proposal)

    write = next(
        item for item in evidence if item.kind == "memory.semantic.write"
    )
    assert proposal.accepted is True
    assert write.payload["status"] == "stored"
    assert write.payload["updated"] == 0
    assert write.payload["stored"]["upsert_fallback"] == "append"
    backend.remember.assert_awaited_once()


async def test_db_runtime_memory_update_rejects_pii_values():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    result = await runtime.run(
        DbRequest(
            "Remember VIP customer email",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:vip_customer",
                "text": "VIP customer email is jane@example.com.",
            },
        )
    )

    assert result.status is OperationStatus.FAILED
    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert proposal.accepted is False
    assert (
        "pii_or_row_level_memory_rejected" in proposal.payload["validation"]["reasons"]
    )
    assert "memory_proposal_not_accepted" in result.warnings
    assert not any(item.kind == "memory.semantic.write" for item in result.evidence)
    backend.remember.assert_not_awaited()


async def test_db_runtime_memory_update_rejects_sensitive_metadata_keys():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    result = await runtime.run(
        DbRequest(
            "Remember users contract",
            mode="memory.update",
            metadata={
                "kind": "data_contract_note",
                "key": "contract:users",
                "text": "Users must have verified contact info.",
                "metadata": {"email": "jane@example.com"},
            },
        )
    )

    assert result.status is OperationStatus.FAILED
    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert proposal.accepted is False
    assert (
        "pii_or_row_level_memory_rejected" in proposal.payload["validation"]["reasons"]
    )
    assert "memory_proposal_not_accepted" in result.warnings
    assert not any(item.kind == "memory.semantic.write" for item in result.evidence)
    backend.remember.assert_not_awaited()


async def test_db_runtime_memory_update_allows_schema_level_pii_column_mentions():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    operation, proposal = await _plan_memory_update(
        runtime,
        DbRequest(
            "Remember users email schema interpretation",
            mode="memory.update",
            metadata={
                "kind": "schema_interpretation",
                "key": "schema:users.email",
                "text": (
                    "users.email is a contact column and should not be used "
                    "as an entity key."
                ),
                "metadata": {"table": "users", "column": "email"},
            },
        ),
    )
    evidence = await _commit_memory_update(runtime, operation, proposal)

    write = next(
        item for item in evidence if item.kind == "memory.semantic.write"
    )
    assert proposal.accepted is True
    assert write.payload["success"] is True
    assert write.payload["kind"] == "schema_interpretation"
    backend.remember.assert_awaited_once()


async def test_db_runtime_memory_update_rejects_unsupported_kind():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=[])
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-1"}
    )
    memory = MemoryPlugin()
    memory.backend = backend
    runtime = _runtime_with_memory_source(memory)

    result = await runtime.run(
        DbRequest(
            "Remember generic knowledge",
            mode="memory.update",
            metadata={"kind": "knowledge", "key": "x", "text": "too vague"},
        )
    )

    assert result.status is OperationStatus.FAILED
    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert proposal.accepted is False
    assert (
        "unsupported_or_ambiguous_memory_kind"
        in proposal.payload["validation"]["reasons"]
    )
    assert "memory_proposal_not_accepted" in result.warnings
    assert not any(item.kind == "memory.semantic.write" for item in result.evidence)
    backend.remember.assert_not_awaited()
