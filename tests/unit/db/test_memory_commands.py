from unittest.mock import AsyncMock, MagicMock

import pytest

from daita.db import DbRequest, DbRuntime, DbRuntimeConfig
from daita.db.analysis import stable_fingerprint
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    OperationStatus,
    RiskLevel,
    Task,
)

TEST_SOURCE_IDENTITY = "test:memory-command-source"


def _memory_runtime(*, backend=None, plugins=()) -> tuple[DbRuntime, MagicMock]:
    if backend is None:
        backend = MagicMock()
        backend.list_by_category = AsyncMock(return_value=[])
        backend.remember = AsyncMock(
            return_value={"status": "success", "chunk_id": "mem-1"}
        )
    memory = MemoryPlugin()
    memory.backend = backend
    return (
        DbRuntime(
            config=DbRuntimeConfig(
                plugins=(*plugins, memory),
                metadata={
                    "from_db_options": {
                        "memory": {
                            "enabled": True,
                            "workspace_scope": "source",
                            "source_identity": TEST_SOURCE_IDENTITY,
                        }
                    }
                },
            )
        ),
        backend,
    )


class SchemaPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="schema_probe",
        display_name="Schema Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
        provides=frozenset({"schema"}),
    )

    def declare_capabilities(self):
        return (
            Capability(
                id="db.schema.inspect",
                owner="schema_probe",
                description="Inspect test schema.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"source.profile", "memory.update"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"schema.asset_profile"}),
                executor="schema_probe.inspect",
                runtime_only=True,
                side_effecting=False,
            ),
        )

    def get_executors(self):
        return (SchemaExecutor(),)


class SchemaExecutor:
    id = "schema_probe.inspect"
    capability_ids = frozenset({"db.schema.inspect"})

    async def execute(self, task: Task, operation, context):
        return [
            Evidence(
                kind="schema.asset_profile",
                owner="schema_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={
                    "tables": [
                        {
                            "name": "orders",
                            "columns": [
                                {"name": "total_cents", "type": "integer"},
                                {"name": "status", "type": "text"},
                            ],
                        }
                    ]
                },
            )
        ]


async def test_explicit_remember_metric_definition_proposes_commits_and_writes():
    runtime, backend = _memory_runtime()

    result = await runtime.run(
        DbRequest(
            "Remember the revenue metric definition",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded orders.",
            },
        )
    )

    assert result.status is OperationStatus.SUCCEEDED
    assert [item.kind for item in result.evidence[:3]] == [
        "db.memory.proposal",
        "db.memory.definition",
        "memory.semantic.write",
    ]
    proposal, definition, write = result.evidence[:3]
    assert proposal.accepted is True
    assert definition.payload["proposal_evidence_id"] == proposal.id
    assert write.payload["success"] is True
    tasks = await runtime.store.list_tasks(result.operation_id)
    proposal_task = next(
        task for task in tasks if task.capability_id == "db.memory.plan_update"
    )
    write_task = next(
        task for task in tasks if task.capability_id == "memory.semantic.write"
    )
    proposal_dependency = next(
        dependency
        for dependency in write_task.dependencies
        if dependency.kind.value == "evidence"
    )
    assert proposal_dependency.evidence_kind == "db.memory.proposal"
    assert proposal_dependency.evidence_id == proposal.id
    assert proposal_dependency.producer_task_id == proposal_task.id
    assert write_task.metadata["reason"] == "db_memory_commit_update"
    stored = backend.remember.await_args.kwargs["extra_metadata"]["db_memory"]
    assert stored["metadata"]["creation_path"] == "explicit_intent"
    assert stored["metadata"]["source_identity"] == TEST_SOURCE_IDENTITY


async def test_explicit_update_replaces_existing_record_by_key():
    backend = MagicMock()
    backend.list_by_category = AsyncMock(
        return_value=[
            {
                "chunk_id": "old-1",
                "content": 'DB memory record:\n{"key": "metric:revenue"}',
                "metadata": {
                    "db_memory": {"kind": "metric_definition", "key": "metric:revenue"}
                },
            }
        ]
    )
    backend.delete_chunks = AsyncMock(return_value=True)
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "new-1"}
    )
    runtime, _ = _memory_runtime(backend=backend)

    result = await runtime.run(
        DbRequest(
            "Update the revenue definition",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunded and cancelled orders.",
            },
        )
    )

    write = next(
        item for item in result.evidence if item.kind == "memory.semantic.write"
    )
    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert result.status is OperationStatus.SUCCEEDED
    assert proposal.payload["commit_behavior"] == "update"
    assert proposal.payload["existing_chunk_ids"] == ["old-1"]
    assert write.payload["status"] == "updated"
    backend.delete_chunks.assert_awaited_once_with(["old-1"])


async def test_unsupported_ambiguous_remember_rejected_without_write():
    runtime, backend = _memory_runtime()

    result = await runtime.run(DbRequest("Remember this", mode="memory.update"))

    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert result.status is OperationStatus.FAILED
    assert proposal.accepted is False
    assert "memory_key_required" in proposal.payload["validation"]["reasons"]
    assert not any(item.kind == "memory.semantic.write" for item in result.evidence)
    backend.remember.assert_not_awaited()


async def test_pii_or_row_level_memory_rejected_without_write():
    runtime, backend = _memory_runtime()

    result = await runtime.run(
        DbRequest(
            "Remember customer email",
            mode="memory.update",
            metadata={
                "kind": "business_rule",
                "key": "business_rule:vip_email",
                "text": "VIP email is jane@example.com.",
            },
        )
    )

    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert proposal.accepted is False
    assert (
        "pii_or_row_level_memory_rejected" in proposal.payload["validation"]["reasons"]
    )
    assert not any(item.kind == "memory.semantic.write" for item in result.evidence)
    backend.remember.assert_not_awaited()


async def test_missing_or_invalid_schema_refs_rejected_for_schema_memory():
    runtime, backend = _memory_runtime(plugins=(SchemaPlugin(),))

    result = await runtime.run(
        DbRequest(
            "Remember schema interpretation",
            mode="memory.update",
            metadata={
                "kind": "schema_interpretation",
                "key": "schema:customers.email",
                "text": "customers.email is the contact column.",
                "schema_refs": [{"table": "customers", "column": "email"}],
            },
        )
    )

    proposal = next(
        item for item in result.evidence if item.kind == "db.memory.proposal"
    )
    assert result.status is OperationStatus.FAILED
    assert proposal.accepted is False
    assert "schema_refs_not_found" in proposal.payload["validation"]["reasons"]
    backend.remember.assert_not_awaited()


async def test_cross_source_proposal_commit_rejected():
    runtime, _backend = _memory_runtime()
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="memory.update",
        request={"prompt": "commit"},
        required_evidence=("db.memory.definition",),
        evaluate_governance=False,
    )
    proposal = _proposal_evidence(operation.id, source_identity="other-source")
    await runtime.store.save_evidence(proposal)
    task = Task(
        id="commit-cross-source",
        operation_id=operation.id,
        capability_id="db.memory.commit_update",
        executor_id="db_runtime.memory.commit_update",
        input={
            "proposal_evidence_id": proposal.id,
            "proposal_fingerprint": proposal.payload["proposal_fingerprint"],
            "source_identity": "other-source",
        },
        required_evidence=frozenset({"db.memory.definition"}),
        metadata={"owner": "db_runtime"},
    )

    with pytest.raises(RuntimeError, match="source identity mismatch"):
        await runtime.execute_task(task, operation)


async def test_proposal_fingerprint_mismatch_rejected():
    runtime, _backend = _memory_runtime()
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="memory.update",
        request={"prompt": "commit"},
        required_evidence=("db.memory.definition",),
        evaluate_governance=False,
    )
    proposal = _proposal_evidence(operation.id, source_identity=TEST_SOURCE_IDENTITY)
    await runtime.store.save_evidence(proposal)
    task = Task(
        id="commit-bad-fingerprint",
        operation_id=operation.id,
        capability_id="db.memory.commit_update",
        executor_id="db_runtime.memory.commit_update",
        input={
            "proposal_evidence_id": proposal.id,
            "proposal_fingerprint": "not-the-fingerprint",
            "source_identity": TEST_SOURCE_IDENTITY,
        },
        required_evidence=frozenset({"db.memory.definition"}),
        metadata={"owner": "db_runtime"},
    )

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        await runtime.execute_task(task, operation)


async def test_memory_update_runs_through_runtime_tasks_and_no_learners():
    runtime, _backend = _memory_runtime()

    result = await runtime.run(
        DbRequest(
            "Remember revenue definition",
            mode="memory.update",
            metadata={
                "kind": "metric_definition",
                "key": "metric:revenue",
                "text": "Revenue excludes refunds.",
            },
        )
    )
    tasks = await runtime.store.list_tasks(result.operation_id)
    capability_ids = [task.capability_id for task in tasks]

    assert "db.memory.plan_update" in capability_ids
    assert "db.memory.commit_update" in capability_ids
    assert "memory.semantic.write" in capability_ids
    assert capability_ids.index("memory.semantic.write") > capability_ids.index(
        "db.memory.commit_update"
    )
    assert "memory.semantic.write" not in result.contract.required_capabilities
    assert not any("learner" in item or "learning" in item for item in capability_ids)


def _proposal_evidence(operation_id: str, *, source_identity: str) -> Evidence:
    record = {
        "kind": "metric_definition",
        "key": "metric:revenue",
        "text": "Revenue excludes refunds.",
        "metadata": {
            "source_identity": source_identity,
            "workspace_scope": "source",
            "active": True,
            "confidence": 1.0,
            "creation_path": "explicit_intent",
        },
        "importance": 0.7,
        "category": "db_semantics",
    }
    payload = {
        "kind": "db.memory.proposal",
        "action": "remember",
        "source_identity": source_identity,
        "workspace_scope": "source",
        "record": record,
        "validation": {"accepted": True, "status": "accepted", "reasons": []},
    }
    payload["proposal_fingerprint"] = stable_fingerprint(payload)
    return Evidence(
        id=f"proposal-{source_identity}",
        kind="db.memory.proposal",
        owner="db_runtime",
        operation_id=operation_id,
        accepted=True,
        payload=payload,
        metadata={"payload_fingerprint": payload["proposal_fingerprint"]},
    )
