import asyncio
import sqlite3
import threading

from daita.runtime import (
    ApprovalRequest,
    ApprovalStatus,
    Evidence,
    GovernanceAuditRecord,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    PolicyDecision,
    PolicyDecisionTrace,
    RiskLevel,
    RuntimeEvent,
    RuntimeEventType,
    SQLiteRuntimeStore,
    Task,
    TaskDependency,
    TaskStatus,
)


async def test_sqlite_runtime_store_does_not_block_the_event_loop(
    tmp_path, monkeypatch
):
    store = SQLiteRuntimeStore(tmp_path / "runtime-responsive.sqlite")
    operation = Operation(id="responsive-op", operation_type="data.query")
    helper_started = threading.Event()
    helper_released = threading.Event()
    helper_finished = threading.Event()
    release_timed_out = threading.Event()
    save_operation_sync = store._save_operation_sync

    def blocking_save_operation_sync(value):
        helper_started.set()
        if not helper_released.wait(timeout=5):
            release_timed_out.set()
        try:
            return save_operation_sync(value)
        finally:
            helper_finished.set()

    monkeypatch.setattr(store, "_save_operation_sync", blocking_save_operation_sync)
    store_task = asyncio.create_task(store.save_operation(operation))
    loop_advanced = asyncio.Event()

    async def advance_loop():
        loop_advanced.set()

    try:
        assert await asyncio.to_thread(helper_started.wait, 5)
        progress_task = asyncio.create_task(advance_loop())
        await progress_task

        assert loop_advanced.is_set()
        assert not helper_released.is_set()
        assert not helper_finished.is_set()
        assert not release_timed_out.is_set()
    finally:
        helper_released.set()
        await store_task

    assert await store.load_operation(operation.id) == operation


async def test_sqlite_runtime_store_closes_read_connection_before_return(
    tmp_path, monkeypatch
):
    store = SQLiteRuntimeStore(tmp_path / "runtime-read-close.sqlite")
    operation = Operation(id="read-close-op", operation_type="data.query")
    await store.save_operation(operation)
    connections = []

    class ConnectionProbe:
        def __init__(self):
            self.connection = sqlite3.connect(store.path)
            self.connection.row_factory = sqlite3.Row
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return self.connection.__exit__(exc_type, exc, tb)

        def execute(self, *args, **kwargs):
            return self.connection.execute(*args, **kwargs)

        def close(self):
            self.closed = True
            self.connection.close()

    def connect():
        connection = ConnectionProbe()
        connections.append(connection)
        return connection

    monkeypatch.setattr(store, "_connect", connect)

    assert await store.load_operation(operation.id) == operation
    assert len(connections) == 1
    assert connections[0].closed is True


async def test_in_memory_runtime_store_persists_operation_state_for_inspection():
    store = InMemoryRuntimeStore()
    operation = Operation(
        id="op-1",
        operation_type="data.query",
        status=OperationStatus.BLOCKED,
        request={"prompt": "count orders"},
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="db.sql.execute_read",
        executor_id="sqlite.sql.execute_read",
        status=TaskStatus.SUCCEEDED,
    )
    evidence = Evidence(
        id="evidence-1",
        kind="query.result",
        owner="sqlite",
        operation_id=operation.id,
        task_id=task.id,
        payload={"rows": [{"count": 3}]},
    )
    decision = PolicyDecision(
        policy_id="governance.require_approval",
        owner="governance",
        effect="require_approval",
        reason="Writes require approval.",
        severity=RiskLevel.HIGH,
        operation_id=operation.id,
    )
    trace = PolicyDecisionTrace(
        trace_id="trace-1",
        operation_id=operation.id,
        policy_id=decision.policy_id,
        owner=decision.owner,
        policy_version=decision.policy_version,
        policy_identity=decision.policy_identity,
        effect=decision.effect,
        reason=decision.reason,
        stage="operation",
    )
    audit = GovernanceAuditRecord(
        audit_id="audit-1",
        operation_id=operation.id,
        stage="operation",
        allowed=False,
        blocked=False,
        pending_approval=True,
        policy_decisions=(decision,),
        traces=(trace,),
    )
    approval = ApprovalRequest(
        approval_id="approval-1",
        operation_id=operation.id,
        reason="Approve write.",
        proposed_action={"operation_type": "write.execute"},
        risk=RiskLevel.HIGH,
        requested_by_policy_id=decision.policy_id,
        owner=decision.owner,
    )
    event = RuntimeEvent(
        type=RuntimeEventType.POLICY_DECISION,
        operation_id=operation.id,
        message="Policy decision recorded.",
    )

    await store.save_operation(operation)
    await store.save_task(task)
    await store.save_evidence(evidence)
    await store.save_policy_decision(decision)
    await store.save_governance_audit_record(audit)
    await store.save_approval_request(approval)
    await store.append_event(event)

    snapshot = await store.inspect_operation(operation.id)

    assert await store.load_operation(operation.id) == operation
    assert await store.load_task(task.id) == task
    assert snapshot.operation == operation
    assert snapshot.tasks == (task,)
    assert snapshot.evidence == (evidence,)
    assert snapshot.policy_decisions == (decision,)
    assert snapshot.governance_audit_records == (audit,)
    assert snapshot.approval_requests == (approval,)
    assert snapshot.events == (event,)
    assert snapshot.completed_task_ids == ("task-1",)
    assert snapshot.resumable_task_ids == ()
    assert snapshot.metadata["resumable"] is True
    assert snapshot.to_dict()["operation"]["status"] == "blocked"
    assert snapshot.to_dict()["governance_audit_records"][0]["audit_id"] == "audit-1"


async def test_runtime_store_governance_audit_records_are_append_only():
    store = InMemoryRuntimeStore()
    operation = Operation(id="audit-op", operation_type="data.query")
    decision = PolicyDecision(
        policy_id="governance.allow",
        owner="governance",
        effect="allow",
        reason="Allowed.",
        severity=RiskLevel.LOW,
        operation_id=operation.id,
    )
    first = GovernanceAuditRecord(
        audit_id="audit-1",
        operation_id=operation.id,
        stage="operation",
        allowed=True,
        blocked=False,
        pending_approval=False,
        policy_decisions=(decision,),
    )
    second = GovernanceAuditRecord(
        audit_id="audit-2",
        operation_id=operation.id,
        stage="task",
        allowed=True,
        blocked=False,
        pending_approval=False,
        policy_decisions=(decision,),
    )

    await store.save_operation(operation)
    await store.save_governance_audit_record(first)
    await store.save_governance_audit_record(second)

    assert await store.list_governance_audit_records(operation.id) == [first, second]


async def test_in_memory_runtime_store_returns_copy_safe_audit_records():
    store = InMemoryRuntimeStore()
    operation = Operation(id="audit-copy-op", operation_type="data.query")
    audit = GovernanceAuditRecord(
        audit_id="audit-copy-1",
        operation_id=operation.id,
        stage="operation",
        allowed=True,
        blocked=False,
        pending_approval=False,
        evaluation_trace={"effect": "allow", "nested": {"value": "original"}},
    )

    await store.save_operation(operation)
    await store.save_governance_audit_record(audit)
    listed = await store.list_governance_audit_records(operation.id)
    listed[0].evaluation_trace["nested"]["value"] = "mutated"

    relisted = await store.list_governance_audit_records(operation.id)

    assert relisted[0].evaluation_trace["nested"]["value"] == "original"


async def test_in_memory_runtime_store_commits_governance_blocked_atomically():
    store = InMemoryRuntimeStore()
    operation = Operation(
        id="blocked-op",
        operation_type="write.execute",
        status=OperationStatus.BLOCKED,
    )
    task = Task(
        id="blocked-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="sqlite.sql.execute_write",
        status=TaskStatus.BLOCKED,
    )
    decision = PolicyDecision(
        policy_id="governance.require_approval",
        owner="governance",
        effect="require_approval",
        reason="Writes require approval.",
        severity=RiskLevel.HIGH,
        operation_id=operation.id,
    )
    approval = ApprovalRequest(
        approval_id="blocked-op:governance.require_approval:human",
        operation_id=operation.id,
        reason=decision.reason,
        proposed_action={"operation_type": "write.execute", "approval": "human"},
        risk=RiskLevel.HIGH,
        requested_by_policy_id=decision.policy_id,
        owner=decision.owner,
    )
    audit = GovernanceAuditRecord(
        audit_id="blocked-audit",
        operation_id=operation.id,
        stage="task",
        allowed=False,
        blocked=False,
        pending_approval=True,
        policy_decisions=(decision,),
        evaluation_trace={"effect": "require_approval"},
    )
    event = RuntimeEvent(
        type=RuntimeEventType.OPERATION_UPDATED,
        operation_id=operation.id,
        task_id=task.id,
        message="Blocked by governance.",
    )

    await store.commit_governance_blocked(
        operation=operation,
        task=task,
        decisions=(decision,),
        audit_record=audit,
        approval_requests=(approval,),
        events=(event,),
    )
    snapshot = await store.inspect_operation(operation.id)

    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert snapshot.tasks == (task,)
    assert snapshot.policy_decisions == (decision,)
    assert snapshot.governance_audit_records == (audit,)
    assert snapshot.approval_requests == (approval,)
    assert snapshot.events == (event,)


async def test_in_memory_runtime_store_identifies_resumable_tasks():
    store = InMemoryRuntimeStore()
    operation = Operation(
        id="op-2",
        operation_type="data.query",
        status=OperationStatus.BLOCKED,
    )
    pending = Task(
        id="task-pending",
        operation_id=operation.id,
        capability_id="db.sql.execute_read",
        executor_id="sqlite.sql.execute_read",
        status=TaskStatus.PENDING,
    )
    completed = Task(
        id="task-completed",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="sqlite.sql.validate",
        status=TaskStatus.SUCCEEDED,
    )

    await store.save_operation(operation)
    await store.save_task(pending)
    await store.save_task(completed)

    snapshot = await store.inspect_operation(operation.id)

    assert snapshot.resumable_task_ids == ("task-pending",)
    assert snapshot.completed_task_ids == ("task-completed",)


async def test_in_memory_runtime_store_claims_task_once():
    store = InMemoryRuntimeStore()
    task = Task(
        id="claim-task",
        operation_id="claim-op",
        capability_id="db.sql.execute_write",
        executor_id="sqlite.sql.execute_write",
        status=TaskStatus.BLOCKED,
    )

    await store.save_task(task)

    claimed = await store.claim_task("claim-task", lease_owner="worker-1")
    second_claim = await store.claim_task("claim-task", lease_owner="worker-2")
    stored = await store.load_task("claim-task")

    assert claimed is not None
    assert claimed.status is TaskStatus.RUNNING
    assert claimed.metadata["lease_owner"] == "worker-1"
    assert claimed.metadata["attempt_count"] == 1
    assert second_claim is None
    assert stored == claimed


async def test_in_memory_runtime_store_commits_task_success_as_one_transition():
    store = InMemoryRuntimeStore()
    operation = Operation(
        id="commit-op",
        operation_type="data.query",
        status=OperationStatus.RUNNING,
    )
    task = Task(
        id="commit-task",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="sqlite.sql.validate",
        status=TaskStatus.SUCCEEDED,
    )
    evidence = Evidence(
        id="commit-evidence",
        kind="sql.validation",
        owner="sqlite",
        operation_id=task.operation_id,
        task_id=task.id,
        payload={"valid": True},
    )
    event = RuntimeEvent(
        type=RuntimeEventType.TASK_UPDATED,
        operation_id=task.operation_id,
        task_id=task.id,
        message="Task succeeded.",
    )

    await store.save_operation(operation)
    await store.commit_task_succeeded(task, (evidence,), event)
    snapshot = await store.inspect_operation(task.operation_id)

    assert snapshot.tasks == (task,)
    assert snapshot.evidence == (evidence,)
    assert snapshot.events == (event,)


async def test_sqlite_runtime_store_persists_task_dependencies_across_restart(
    tmp_path,
):
    path = tmp_path / "runtime.sqlite"
    operation = Operation(
        id="sqlite-op",
        operation_type="write.execute",
        status=OperationStatus.BLOCKED,
    )
    task = Task(
        id="sqlite-write-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="sqlite.sql.execute_write",
        status=TaskStatus.BLOCKED,
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                evidence_payload={"valid": True},
                operation_id=operation.id,
            ),
            TaskDependency(
                kind="approval",
                approval_status=ApprovalStatus.APPROVED,
                operation_id=operation.id,
            ),
        ),
    )

    store = SQLiteRuntimeStore(path)
    await store.save_operation(operation)
    await store.save_task(task)
    reopened = SQLiteRuntimeStore(path)

    loaded = await reopened.load_task(task.id)
    snapshot = await reopened.inspect_operation(operation.id)

    assert loaded == task
    assert snapshot.tasks == (task,)
    assert snapshot.resumable_task_ids == (task.id,)
    assert snapshot.to_dict()["tasks"][0]["dependencies"][0]["evidence_kind"] == (
        "sql.validation"
    )


async def test_sqlite_runtime_store_commits_governance_audit_across_restart(tmp_path):
    path = tmp_path / "runtime.sqlite"
    operation = Operation(id="sqlite-audit-op", operation_type="write.execute")
    decision = PolicyDecision(
        policy_id="governance.require_approval",
        owner="governance",
        policy_version="v3",
        effect="require_approval",
        reason="Writes require approval.",
        severity=RiskLevel.HIGH,
        operation_id=operation.id,
    )
    approval = ApprovalRequest(
        approval_id="sqlite-audit-op:governance.require_approval:human",
        operation_id=operation.id,
        reason=decision.reason,
        proposed_action={"operation_type": "write.execute", "approval": "human"},
        risk=RiskLevel.HIGH,
        requested_by_policy_id=decision.policy_id,
        owner=decision.owner,
    )
    audit = GovernanceAuditRecord(
        audit_id="sqlite-audit-1",
        operation_id=operation.id,
        stage="operation",
        allowed=False,
        blocked=False,
        pending_approval=True,
        policy_decisions=(decision,),
        approval_context={"new_request_ids": [approval.approval_id]},
    )
    event = RuntimeEvent(
        type=RuntimeEventType.POLICY_DECISION,
        operation_id=operation.id,
        message="Policy decision recorded.",
    )

    store = SQLiteRuntimeStore(path)
    await store.save_operation(operation)
    await store.commit_governance_evaluation(
        decisions=(decision,),
        audit_record=audit,
        approval_requests=(approval,),
        events=(event,),
    )
    reopened = SQLiteRuntimeStore(path)
    snapshot = await reopened.inspect_operation(operation.id)

    assert snapshot.policy_decisions == (decision,)
    assert snapshot.governance_audit_records == (audit,)
    assert snapshot.approval_requests == (approval,)
    assert snapshot.events == (event,)


async def test_sqlite_runtime_store_commits_governance_blocked_across_restart(
    tmp_path,
):
    path = tmp_path / "runtime.sqlite"
    operation = Operation(
        id="sqlite-blocked-op",
        operation_type="write.execute",
        status=OperationStatus.BLOCKED,
    )
    task = Task(
        id="sqlite-blocked-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="sqlite.sql.execute_write",
        status=TaskStatus.BLOCKED,
    )
    decision = PolicyDecision(
        policy_id="governance.require_approval",
        owner="governance",
        effect="require_approval",
        reason="Writes require approval.",
        severity=RiskLevel.HIGH,
        operation_id=operation.id,
    )
    approval = ApprovalRequest(
        approval_id="sqlite-blocked-op:governance.require_approval:human",
        operation_id=operation.id,
        reason=decision.reason,
        proposed_action={"operation_type": "write.execute", "approval": "human"},
        risk=RiskLevel.HIGH,
        requested_by_policy_id=decision.policy_id,
        owner=decision.owner,
    )
    audit = GovernanceAuditRecord(
        audit_id="sqlite-blocked-audit",
        operation_id=operation.id,
        stage="task",
        allowed=False,
        blocked=False,
        pending_approval=True,
        policy_decisions=(decision,),
        evaluation_trace={"effect": "require_approval"},
    )
    event = RuntimeEvent(
        type=RuntimeEventType.OPERATION_UPDATED,
        operation_id=operation.id,
        task_id=task.id,
        message="Blocked by governance.",
    )

    store = SQLiteRuntimeStore(path)
    await store.commit_governance_blocked(
        operation=operation,
        task=task,
        decisions=(decision,),
        audit_record=audit,
        approval_requests=(approval,),
        events=(event,),
    )
    reopened = SQLiteRuntimeStore(path)
    snapshot = await reopened.inspect_operation(operation.id)

    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert snapshot.tasks == (task,)
    assert snapshot.policy_decisions == (decision,)
    assert snapshot.governance_audit_records == (audit,)
    assert snapshot.approval_requests == (approval,)
    assert snapshot.events == (event,)


async def test_sqlite_runtime_store_approval_wait_survives_restart(tmp_path):
    path = tmp_path / "runtime.sqlite"
    operation = Operation(
        id="sqlite-approval-op",
        operation_type="write.execute",
        status=OperationStatus.BLOCKED,
    )
    task = Task(
        id="sqlite-approval-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="sqlite.sql.execute_write",
        status=TaskStatus.BLOCKED,
    )
    approval = ApprovalRequest(
        approval_id="sqlite-approval",
        operation_id=operation.id,
        reason="Approve write.",
        proposed_action={"operation_type": "write.execute"},
        risk=RiskLevel.HIGH,
    )

    store = SQLiteRuntimeStore(path)
    await store.save_operation(operation)
    await store.save_task(task)
    await store.save_approval_request(approval)
    await store.append_event(
        RuntimeEvent(
            type=RuntimeEventType.APPROVAL_REQUESTED,
            operation_id=operation.id,
            message="Approval requested.",
        )
    )

    reopened = SQLiteRuntimeStore(path)
    await reopened.save_approval_request(
        ApprovalRequest.from_dict(
            {
                **approval.to_dict(),
                "status": ApprovalStatus.APPROVED.value,
            }
        )
    )
    restarted = SQLiteRuntimeStore(path)
    snapshot = await restarted.inspect_operation(operation.id)

    assert snapshot.operation == operation
    assert snapshot.tasks == (task,)
    assert snapshot.approval_requests[0].status is ApprovalStatus.APPROVED
    assert snapshot.events[0].type is RuntimeEventType.APPROVAL_REQUESTED


async def test_sqlite_runtime_store_commits_task_success_across_restart(tmp_path):
    path = tmp_path / "runtime.sqlite"
    store = SQLiteRuntimeStore(path)
    operation = Operation(
        id="sqlite-commit-op",
        operation_type="data.query",
        status=OperationStatus.RUNNING,
    )
    task = Task(
        id="sqlite-commit-task",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="sqlite.sql.validate",
        status=TaskStatus.SUCCEEDED,
    )
    evidence = Evidence(
        id="sqlite-commit-evidence",
        kind="sql.validation",
        owner="sqlite",
        operation_id=operation.id,
        task_id=task.id,
        payload={"valid": True},
    )
    event = RuntimeEvent(
        type=RuntimeEventType.TASK_UPDATED,
        operation_id=operation.id,
        task_id=task.id,
        message="Task succeeded.",
    )

    await store.save_operation(operation)
    await store.commit_task_succeeded(task, (evidence,), event)
    reopened = SQLiteRuntimeStore(path)
    snapshot = await reopened.inspect_operation(operation.id)

    assert snapshot.tasks == (task,)
    assert snapshot.evidence == (evidence,)
    assert snapshot.events == (event,)
