from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from daita.db import (
    DbEvidenceStore,
    DbOperationExecutor,
    DbRequest,
    DbRuntime,
    DbRuntimeConfig,
)
from daita.db.llm_planner import _planner_messages
from daita.db.memory import (
    DBMemoryRecord,
    calibrate_db_memory,
    db_memory_planning_recall_decision,
    db_memory_refs_from_recall_evidence,
    has_db_memory_marker,
    normalize_db_memory_record,
    recall_db_memory_records,
    write_db_memory_record,
    write_db_memory_records,
)
from daita.embeddings.mock import MockEmbeddingProvider
from daita.plugins.catalog import CatalogPlugin
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
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
    backend.recall = AsyncMock(
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
    assert backend.recall.await_args.kwargs["category"] == "db_semantics"
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
    assert "DB memory is advisory semantic context only" in system_message
    assert "Schema, catalog, policy, SQL validation" in system_message


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


async def test_db_runtime_executes_memory_update_with_typed_evidence(tmp_path):
    runtime = _runtime_with_memory_source(_memory(tmp_path))

    result = await runtime.run(
        DbRequest(
            "Remember that revenue excludes tax",
            mode="memory.update",
            metadata={"category": "db_semantic"},
        )
    )

    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["verification"]["passed"] is True
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    assert evidence.owner == "memory"
    assert evidence.payload["result"]["status"] == "success"


async def test_db_runtime_memory_update_stores_db_semantic_record():
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
            "Remember the revenue rule",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:revenue_refunds",
                "text": "Revenue excludes refunded orders.",
                "metadata": {"metric": "revenue"},
                "importance": 0.8,
            },
        )
    )

    assert result.status is OperationStatus.SUCCEEDED
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    assert evidence.payload["success"] is True
    assert evidence.payload["kind"] == "business_rule"
    assert evidence.payload["category"] == "db_semantics"
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

    result = await runtime.run(
        DbRequest(
            "Remember the revenue metric",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded orders.",
            },
        )
    )

    assert result.status is OperationStatus.SUCCEEDED
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    assert evidence.payload["status"] == "updated"
    assert evidence.payload["updated"] == 1
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

    result = await runtime.run(
        DbRequest(
            "Remember the refunds rule",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:refunds",
                "text": "Refunded orders are excluded from revenue.",
            },
        )
    )

    assert result.status is OperationStatus.SUCCEEDED
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    assert evidence.payload["status"] == "stored"
    assert evidence.payload["updated"] == 0
    assert evidence.payload["stored"]["upsert_fallback"] == "append"
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

    result = await runtime.run(
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
        )
    )

    assert result.status is OperationStatus.SUCCEEDED
    evidence = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    assert evidence.payload["success"] is True
    assert evidence.payload["kind"] == "schema_interpretation"
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
