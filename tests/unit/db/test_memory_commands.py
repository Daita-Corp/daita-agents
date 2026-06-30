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
