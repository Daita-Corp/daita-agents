from unittest.mock import AsyncMock, MagicMock

from daita.db import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
    DbRuntime,
    DbRuntimeConfig,
)
from daita.db.analysis import stable_fingerprint
from daita.plugins.memory.memory_plugin import MemoryPlugin
from daita.runtime import (
    AccessMode,
    Evidence,
    Operation,
    OperationStatus,
    TaskStatus,
    WorkerRuntime,
    WorkerRuntimeOptions,
)

SOURCE_IDENTITY = "sqlite:from_db:learning-source"


def _memory_backend(*, existing=()):
    backend = MagicMock()
    backend.list_by_category = AsyncMock(return_value=list(existing))
    backend.remember = AsyncMock(
        return_value={"status": "success", "chunk_id": "mem-learned"}
    )
    return backend


def _memory_plugin(backend=None):
    memory = MemoryPlugin(auto_curate="manual")
    memory.backend = backend or _memory_backend()
    return memory


def _runtime(
    *,
    backend=None,
    memory_enabled=True,
    learning="safe",
    source_identity=SOURCE_IDENTITY,
):
    plugins = (_memory_plugin(backend),) if memory_enabled else ()
    return DbRuntime(
        config=DbRuntimeConfig(
            plugins=plugins,
            metadata={
                "from_db_options": {
                    "memory": {
                        "enabled": memory_enabled,
                        "recall": "auto" if memory_enabled else "off",
                        "learning": learning,
                        "workspace_scope": "source",
                        "source_identity": source_identity,
                    }
                }
            },
        )
    )


def _schema_evidence(
    operation_id,
    *,
    source_identity=SOURCE_IDENTITY,
    table="orders",
    column="total_cents",
    evidence_id="evidence-schema",
):
    schema = {
        "database_type": "sqlite",
        "tables": [
            {
                "name": table,
                "columns": [
                    {"name": column, "data_type": "real"},
                    {"name": "status", "data_type": "text"},
                ],
            }
        ],
    }
    return Evidence(
        id=evidence_id,
        kind="schema.asset_profile",
        owner="sqlite",
        operation_id=operation_id,
        accepted=True,
        payload=schema,
        metadata={
            "payload_fingerprint": stable_fingerprint(schema),
            **({"source_identity": source_identity} if source_identity else {}),
        },
    )


def _planning_context_evidence(operation_id):
    payload = {
        "schema": {
            "database_type": "sqlite",
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {"name": "status", "data_type": "text"},
                    ],
                }
            ],
        },
        "schema_fingerprint": "schema-value-alias",
        "column_value_hints": [
            {
                "table": "orders",
                "column": "status",
                "profile_ref": "catalog:orders.status",
                "distinct_count": 3,
                "observed_values": [{"value": "complete"}],
                "profile_status": "profiled",
                "sampled": False,
                "truncated": False,
                "redacted": False,
                "stale": False,
                "candidate_mapping": {
                    "prompt_term": "completed",
                    "confidence": 1.0,
                    "reason": "exact_match",
                },
            }
        ],
    }
    return Evidence(
        id="evidence-planning-context",
        kind="planning.context",
        owner="db_runtime",
        operation_id=operation_id,
        accepted=True,
        payload=payload,
        metadata={"source_identity": SOURCE_IDENTITY},
    )


def _verification_evidence(operation_id, *, passed=True):
    payload = {
        "passed": passed,
        "evidence_refs": ["evidence-schema"],
        "evidence_details": [],
        "warnings": [],
        "missing_evidence": [],
        "diagnostics": {},
    }
    return Evidence(
        id="evidence-verification",
        kind="verification.result",
        owner="db_runtime",
        operation_id=operation_id,
        accepted=passed,
        payload=payload,
    )


async def _source_operation(runtime, *, operation_type="data.query"):
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type=operation_type,
        request={"prompt": "How much revenue is in orders?"},
        required_evidence=frozenset({"schema.asset_profile", "verification.result"}),
        metadata={"intent_kind": operation_type},
        evaluate_governance=False,
    )
    return operation


def _result(operation, evidence, *, status=OperationStatus.SUCCEEDED, verified=True):
    intent_kind = (
        DbIntentKind.MEMORY_UPDATE
        if operation.operation_type == "memory.update"
        else DbIntentKind.DATA_QUERY
    )
    return DbOperationResult(
        operation_id=operation.id,
        request=DbRequest("How much revenue is in orders?"),
        intent=DbIntent(kind=intent_kind, confidence=1.0),
        contract=DbOperationContract(
            operation_type=operation.operation_type,
            required_capabilities=(),
            required_evidence=("schema.asset_profile", "verification.result"),
            access=AccessMode.READ,
        ),
        status=status,
        evidence=tuple(evidence),
        diagnostics={"verification": {"passed": verified}},
    )


async def _record_successful_source(runtime, *, schema_evidence=None):
    operation = await _source_operation(runtime)
    schema = schema_evidence or _schema_evidence(operation.id)
    verification = _verification_evidence(operation.id)
    await runtime.store.save_evidence(schema)
    await runtime.store.save_evidence(verification)
    result = _result(operation, (schema, verification))
    await runtime._record_operation_result(result, operation=operation)
    return operation, result


async def test_successful_eligible_operation_enqueues_child_learning_operation():
    runtime = _runtime()
    source_operation, _ = await _record_successful_source(runtime)

    source_snapshot = await runtime.inspect_operation(source_operation.id)
    operations = await runtime.store.list_operations()
    child = next(
        item for item in operations if item.operation_type == "db.memory.learning"
    )
    child_tasks = await runtime.store.list_tasks(child.id)
    enqueue_evidence = await runtime.store.list_evidence(child.id)

    assert source_snapshot.operation.status is OperationStatus.SUCCEEDED
    assert not [
        task
        for task in source_snapshot.tasks
        if task.capability_id == "db.memory.learning.run"
        and task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
    ]
    assert child.metadata["source_operation_id"] == source_operation.id
    assert child.metadata["source_identity"] == SOURCE_IDENTITY
    assert child.metadata["learning_mode"] == "safe"
    assert [task.capability_id for task in child_tasks] == [
        "db.memory.learning.enqueue",
        "db.memory.learning.run",
    ]
    assert child_tasks[1].metadata["queue"] == "memory_learning"
    assert child_tasks[1].status is TaskStatus.PENDING
    assert enqueue_evidence[0].kind == "db.memory.learning.enqueue"


async def test_learning_enqueue_gates_skip_ineligible_operations():
    cases = [
        {"memory_enabled": False, "learning": "off"},
        {"memory_enabled": True, "learning": "off"},
        {"status": OperationStatus.FAILED},
        {"status": OperationStatus.BLOCKED},
        {"verified": False},
        {"operation_type": "memory.update"},
    ]
    for case in cases:
        runtime = _runtime(
            memory_enabled=case.get("memory_enabled", True),
            learning=case.get("learning", "safe"),
        )
        operation = await _source_operation(
            runtime,
            operation_type=case.get("operation_type", "data.query"),
        )
        schema = _schema_evidence(operation.id)
        verification = _verification_evidence(
            operation.id,
            passed=case.get("verified", True),
        )
        await runtime.store.save_evidence(schema)
        await runtime.store.save_evidence(verification)

        result = _result(
            operation,
            (schema, verification),
            status=case.get("status", OperationStatus.SUCCEEDED),
            verified=case.get("verified", True),
        )
        await runtime._record_operation_result(result, operation=operation)

        operations = await runtime.store.list_operations()
        assert [item.operation_type for item in operations] == [
            operation.operation_type
        ]


async def test_learner_promotes_safe_unit_candidate_through_memory_write():
    backend = _memory_backend()
    runtime = _runtime(backend=backend)
    source_operation, _ = await _record_successful_source(runtime)
    worker = WorkerRuntime(
        kernel=runtime.kernel,
        options=WorkerRuntimeOptions(
            worker_id="db.memory.learner",
            owner="db_runtime",
            queues=("memory_learning",),
        ),
    )

    run = await worker.run_once()
    child_snapshot = await runtime.inspect_operation(run.handoff.operation_id)
    evidence = await runtime.store.list_evidence(run.handoff.operation_id)

    assert child_snapshot.operation.status is OperationStatus.SUCCEEDED
    assert {item.kind for item in evidence} >= {
        "db.memory.learning.enqueue",
        "db.memory.candidate",
        "db.memory.promotion",
        "memory.semantic.write",
    }
    write = next(item for item in evidence if item.kind == "memory.semantic.write")
    promotion = next(item for item in evidence if item.kind == "db.memory.promotion")
    assert promotion.payload["promoted"] is True
    assert write.payload["success"] is True
    assert write.payload["kind"] == "unit_convention"
    backend.remember.assert_awaited_once()


async def test_learner_promotes_catalog_cited_value_alias_without_observed_values():
    backend = _memory_backend()
    runtime = _runtime(backend=backend)
    operation = await _source_operation(runtime)
    planning = _planning_context_evidence(operation.id)
    verification = _verification_evidence(operation.id)
    await runtime.store.save_evidence(planning)
    await runtime.store.save_evidence(verification)
    await runtime._record_operation_result(
        _result(operation, (planning, verification)),
        operation=operation,
    )
    worker = WorkerRuntime(
        kernel=runtime.kernel,
        options=WorkerRuntimeOptions(
            worker_id="db.memory.learner",
            owner="db_runtime",
            queues=("memory_learning",),
        ),
    )

    run = await worker.run_once()
    evidence = await runtime.store.list_evidence(run.handoff.operation_id)
    candidate = next(item for item in evidence if item.kind == "db.memory.candidate")
    write = next(item for item in evidence if item.kind == "memory.semantic.write")
    stored_record = backend.remember.await_args.kwargs["extra_metadata"]["db_memory"]

    assert write.payload["kind"] == "value_alias"
    assert "observed_values" not in str(candidate.payload)
    assert "closest_value" not in str(candidate.payload)
    assert stored_record["metadata"]["catalog_profile_ref"] == "catalog:orders.status"
    assert "observed_values" not in str(stored_record)
    assert "closest_value" not in str(stored_record)


async def test_learner_rejects_duplicate_missing_source_cross_source_and_pii_candidates():
    duplicate_record = {
        "chunk_id": "existing",
        "metadata": {
            "db_memory": {
                "kind": "unit_convention",
                "key": "unit_convention:orders.total_cents",
                "text": "orders.total_cents is stored as cents.",
                "metadata": {"source_identity": SOURCE_IDENTITY},
            }
        },
    }
    cases = [
        {
            "name": "duplicate",
            "runtime": _runtime(backend=_memory_backend(existing=[duplicate_record])),
            "schema": _schema_evidence("placeholder"),
            "expected": "duplicate",
        },
        {
            "name": "missing-source",
            "runtime": _runtime(backend=_memory_backend(), source_identity=""),
            "schema": _schema_evidence("placeholder", source_identity=""),
            "expected": "missing_source_identity",
        },
        {
            "name": "cross-source",
            "runtime": _runtime(
                backend=_memory_backend(), source_identity=SOURCE_IDENTITY
            ),
            "schema": _schema_evidence("placeholder", source_identity="sqlite:other"),
            "expected": "cross_source_candidate",
        },
    ]
    for case in cases:
        runtime = case["runtime"]
        operation = await _source_operation(runtime)
        schema = Evidence(
            **{
                **case["schema"].to_dict(),
                "operation_id": operation.id,
            }
        )
        verification = _verification_evidence(operation.id)
        await runtime.store.save_evidence(schema)
        await runtime.store.save_evidence(verification)
        await runtime._record_operation_result(
            _result(operation, (schema, verification)),
            operation=operation,
        )
        worker = WorkerRuntime(
            kernel=runtime.kernel,
            options=WorkerRuntimeOptions(
                worker_id="db.memory.learner",
                owner="db_runtime",
                queues=("memory_learning",),
            ),
        )

        run = await worker.run_once()
        evidence = await runtime.store.list_evidence(run.handoff.operation_id)
        rejection = next(
            item for item in evidence if item.kind == "db.memory.rejection"
        )

        assert rejection.payload["reason"] == case["expected"]
        assert not any(item.kind == "memory.semantic.write" for item in evidence)

    pii_backend = _memory_backend()
    pii_runtime = _runtime(backend=pii_backend)
    operation = await _source_operation(pii_runtime)
    pii_hint = _planning_context_evidence(operation.id)
    pii_payload = dict(pii_hint.payload)
    pii_payload["column_value_hints"][0]["candidate_mapping"] = {
        "prompt_term": "jane@example.com",
        "confidence": 1.0,
        "reason": "exact_match",
    }
    pii_hint = Evidence(**{**pii_hint.to_dict(), "payload": pii_payload})
    verification = _verification_evidence(operation.id)
    await pii_runtime.store.save_evidence(pii_hint)
    await pii_runtime.store.save_evidence(verification)
    await pii_runtime._record_operation_result(
        _result(operation, (pii_hint, verification)),
        operation=operation,
    )
    worker = WorkerRuntime(
        kernel=pii_runtime.kernel,
        options=WorkerRuntimeOptions(
            worker_id="db.memory.learner",
            owner="db_runtime",
            queues=("memory_learning",),
        ),
    )

    run = await worker.run_once()
    evidence = await pii_runtime.store.list_evidence(run.handoff.operation_id)
    rejection = next(item for item in evidence if item.kind == "db.memory.rejection")

    assert rejection.payload["reason"] == "pii_or_sensitive_candidate"
    assert not any(item.kind == "memory.semantic.write" for item in evidence)
    pii_backend.remember.assert_not_awaited()


async def test_runtime_inspection_exposes_memory_learning_worker_and_capabilities():
    runtime = _runtime()

    inspection = await runtime.inspect()

    assert "db_runtime:db.memory.learning.enqueue" in inspection.capability_ids
    assert "db_runtime:db.memory.learning.run" in inspection.capability_ids
    assert "db_runtime:db.memory.learner" in inspection.worker_ids
    assert "db_runtime:db.memory.learning.enqueue" in inspection.evidence_schema_kinds
    assert "db_runtime.memory.learning.run" in inspection.executor_ids
