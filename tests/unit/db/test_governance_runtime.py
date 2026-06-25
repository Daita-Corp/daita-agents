import asyncio

import pytest

from daita.db import DbRequest, DbRuntime
from daita.db.query_sql_validation import validate_sql_against_schema
from daita.db.runtime import DbRuntimeTaskNotRunnable
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import (
    AccessMode,
    ApprovalStatus,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    OperationStatus,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    RuntimeEventType,
    SQLiteRuntimeStore,
    Task,
    TaskDependency,
    TaskStatus,
    ApprovalRequest,
)
from daita.skills import Skill, SkillRuntimeEffects


class WriteProbeExecutor:
    def __init__(self, executor_id: str, evidence_kind: str):
        self.id = executor_id
        self.capability_ids = frozenset({"db.sql.validate", "db.sql.execute_write"})
        self.evidence_kind = evidence_kind
        self.calls = 0
        self.delay_seconds = 0.0

    async def execute(self, task: Task, operation: Operation, context):
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        self.calls += 1
        if self.evidence_kind == "sql.validation":
            payload = {
                "calls": self.calls,
                "valid": True,
                "sql": task.input.get("sql"),
                "statement_facts": _statement_facts_for_probe_sql(
                    task.input.get("sql")
                ),
            }
        else:
            payload = {"calls": self.calls, "sql": task.input.get("sql")}
        return [
            Evidence(
                kind=self.evidence_kind,
                owner="write_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload=payload,
            )
        ]


def _statement_facts_for_probe_sql(sql):
    normalized = str(sql or "").strip().lower()
    statement_type = normalized.split(maxsplit=1)[0] if normalized else ""
    mutating = []
    destructive = []
    admin = []
    if statement_type in {"insert", "update", "delete", "create", "drop", "alter"}:
        mutating.append(statement_type.upper())
    if statement_type in {"delete", "drop", "alter", "truncate"}:
        destructive.append(statement_type.upper())
    if statement_type in {"create", "drop", "alter", "truncate"}:
        admin.append(statement_type.upper())
    return {
        "statement_type": statement_type,
        "statement_count": 1 if statement_type else 0,
        "is_read": statement_type in {"select", "show", "describe"},
        "mutating_statement_classes": mutating,
        "destructive_statement_classes": destructive,
        "admin_statement_classes": admin,
        "target_resources": ["orders"] if "orders" in normalized else [],
        "guardrail_result": "passed",
        "sql_fingerprint": f"probe-{abs(hash(normalized))}",
    }


class WriteProbePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="write_probe",
        display_name="Write Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.validate_executor = WriteProbeExecutor(
            "write_probe.sql.validate", "sql.validation"
        )
        self.write_executor = WriteProbeExecutor(
            "write_probe.sql.execute_write", "write.execution"
        )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.validate",
                owner="write_probe",
                description="Validate write SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"write.propose", "write.execute"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"sql.validation"}),
                executor="write_probe.sql.validate",
                runtime_only=True,
                side_effecting=False,
            ),
            Capability(
                id="db.sql.execute_write",
                owner="write_probe",
                description="Execute write SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"write.execute"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                input_schema={"type": "object"},
                output_evidence=frozenset({"write.execution"}),
                executor="write_probe.sql.execute_write",
                runtime_only=True,
                side_effecting=True,
            ),
        ]

    def get_executors(self):
        return [self.validate_executor, self.write_executor]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="sql.validation",
                owner="write_probe",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="write.execution",
                owner="write_probe",
                json_schema={"type": "object"},
            ),
        ]


class ReadProbeExecutor:
    id = "read_probe.sql.execute_read"
    capability_ids = frozenset({"db.sql.execute_read"})

    async def execute(self, task: Task, operation: Operation, context):
        return [
            Evidence(
                kind="query.result",
                owner="read_probe",
                operation_id=operation.id,
                task_id=task.id,
                payload={"rows": []},
            )
        ]


class ReadProbePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="read_probe",
        display_name="Read Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.executor = ReadProbeExecutor()

    def declare_capabilities(self):
        return [
            Capability(
                id="db.sql.execute_read",
                owner="read_probe",
                description="Execute read SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="read_probe.sql.execute_read",
                runtime_only=True,
                side_effecting=False,
            )
        ]

    def get_executors(self):
        return [self.executor]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="query.result",
                owner="read_probe",
                json_schema={"type": "object"},
            )
        ]


class RecordingPolicy:
    def __init__(self, *, policy_id: str, owner: str, effect: PolicyEffect | None):
        self.id = policy_id
        self.owner = owner
        self.effect = effect
        self.calls = 0

    def applies_to(self, request, operation_type):
        return True

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation):
        self.calls += 1
        if self.effect is None:
            return None
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=self.effect,
            reason=f"{self.owner}:{self.id} evaluated",
            severity=RiskLevel.LOW,
        )


class GovernancePolicyPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="governance_probe",
        display_name="Governance Probe",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self, policy: RecordingPolicy):
        self.policy = policy

    def declare_policies(self):
        return (self.policy,)


async def test_unselected_skill_owned_policy_does_not_evaluate():
    skill_policy = RecordingPolicy(
        policy_id="aggregate_only",
        owner="skill_finance",
        effect=PolicyEffect.WARN,
    )
    skill = Skill(name="finance", policies=(skill_policy,), context_mode="runtime_only")
    runtime = DbRuntime(plugins=(skill,))

    result = await runtime.run(DbRequest("Hello there"))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert skill_policy.calls == 0
    assert result.contract.policy_ids == ()
    assert snapshot.policy_decisions == ()


async def test_selected_skill_owned_policy_evaluates_when_contract_references_it():
    skill_policy = RecordingPolicy(
        policy_id="aggregate_only",
        owner="skill_finance",
        effect=PolicyEffect.WARN,
    )
    skill = Skill(
        name="finance",
        policies=(skill_policy,),
        runtime_effects=SkillRuntimeEffects(
            skill_id="skill_finance",
            policy_ids=("skill_finance:aggregate_only",),
        ),
        context_mode="runtime_only",
    )
    runtime = DbRuntime(plugins=(skill,))

    result = await runtime.run(DbRequest("Hello there"))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert skill_policy.calls == 1
    assert result.contract.policy_ids == ("skill_finance:aggregate_only",)
    assert [decision.policy_identity for decision in snapshot.policy_decisions] == [
        "skill_finance:aggregate_only@1"
    ]


async def test_non_skill_registry_policy_still_evaluates_without_contract_reference():
    policy = RecordingPolicy(
        policy_id="warn_all",
        owner="governance_probe",
        effect=PolicyEffect.WARN,
    )
    runtime = DbRuntime(plugins=(GovernancePolicyPlugin(policy),))

    result = await runtime.run(DbRequest("Hello there"))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert policy.calls == 1
    assert result.contract.policy_ids == ()
    assert [decision.policy_identity for decision in snapshot.policy_decisions] == [
        "governance_probe:warn_all@1"
    ]


async def test_db_runtime_requires_approval_for_write_before_tasks_run():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    result = await runtime.run(
        DbRequest(
            "execute insert into orders values (1)",
            user_id="analyst-1",
            session_id="session-1",
            source_scope=("warehouse.orders",),
            mode="write_execute",
            metadata={"tenant_id": "tenant-1"},
        )
    )
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_runtime_approval_required",)
    assert plugin.validate_executor.calls == 0
    assert plugin.write_executor.calls == 0
    assert [task.status for task in snapshot.tasks] == [
        TaskStatus.PENDING,
        TaskStatus.PENDING,
    ]
    assert [task.metadata["sequence"] for task in snapshot.tasks] == [1, 2]
    assert [task.metadata["owner"] for task in snapshot.tasks] == [
        "write_probe",
        "write_probe",
    ]
    assert snapshot.tasks[1].input["sql_ref"] == "sql.validation"
    assert snapshot.tasks[1].input["input_hash"]
    assert [dependency.kind.value for dependency in snapshot.tasks[1].dependencies] == [
        "evidence",
        "approval",
    ]
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.REQUIRE_APPROVAL
    ]
    assert len(snapshot.approval_requests) == 1
    assert snapshot.approval_requests[0].status is ApprovalStatus.PENDING
    assert snapshot.approval_requests[0].requested_by_policy_id == (
        "approval_required_for_writes"
    )
    assert RuntimeEventType.APPROVAL_REQUESTED in {
        event.type for event in snapshot.events
    }
    assert result.diagnostics["governance"]["pending_approval"] is True
    assert len(snapshot.governance_audit_records) == 1
    audit = snapshot.governance_audit_records[0]
    assert audit.stage == "operation"
    assert audit.pending_approval is True
    assert audit.runtime_facts["governance_facts"]["version"] == "15.10"
    assert audit.runtime_facts["governance_facts"]["fact_source"]["planned"] is True
    assert audit.actor["user_id"] == "analyst-1"
    assert audit.actor["session_id"] == "session-1"
    assert audit.tenant["tenant_id"] == "tenant-1"
    assert audit.source_scope == ("warehouse.orders",)
    assert audit.traces[0].policy_identity == ("runtime:approval_required_for_writes@1")
    assert audit.traces[0].approval_ids == (snapshot.approval_requests[0].approval_id,)
    assert audit.approval_context["new_request_ids"] == [
        snapshot.approval_requests[0].approval_id
    ]
    assert audit.evaluation_trace["effect"] == "require_approval"
    assert audit.operation_context["request"]["prompt_hash"]
    assert "execute insert into orders values (1)" not in str(audit.to_dict())

    approved = await runtime.approval_channel.approve(
        snapshot.approval_requests[0].approval_id
    )
    approvals = await runtime.store.list_approval_requests(result.operation_id)

    assert approved.status is ApprovalStatus.APPROVED
    assert approvals[0].status is ApprovalStatus.APPROVED
    assert approvals[0].approval_id == snapshot.approval_requests[0].approval_id


async def test_allowed_direct_capability_records_evaluation_trace_without_policy_decision():
    plugin = ReadProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    evidence = await runtime.execute_capability(
        "db.sql.execute_read",
        owner="read_probe",
        operation_type="data.query",
        input={"prompt": "show orders"},
        operation_id="direct-read-allow",
    )
    snapshot = await runtime.inspect_operation("direct-read-allow")

    assert evidence[0].kind == "query.result"
    assert snapshot.policy_decisions == ()
    assert len(snapshot.governance_audit_records) >= 1
    first_audit = snapshot.governance_audit_records[0]
    assert first_audit.allowed is True
    assert first_audit.traces == ()
    assert first_audit.evaluation_trace["effect"] == "allow"
    assert first_audit.evaluation_trace["reason"] == (
        "No applicable policy denied, modified, warned, or required approval."
    )
    assert first_audit.evaluation_trace["evaluated_policy_identities"]


async def test_db_runtime_denies_destructive_operations_without_running_tasks():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    result = await runtime.run(
        DbRequest("execute drop table orders", mode="write_execute")
    )
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_runtime_governance_denied",)
    assert plugin.validate_executor.calls == 0
    assert plugin.write_executor.calls == 0
    assert [task.status for task in snapshot.tasks] == [
        TaskStatus.PENDING,
        TaskStatus.PENDING,
    ]
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.DENY
    ]
    decision = snapshot.policy_decisions[0]
    assert decision.policy_identity == "runtime:deny_destructive_operations@2"
    assert decision.metadata["decision_source"] == "narrow_prompt_fallback"
    assert decision.metadata["fact_source"] == "prompt_fallback"
    audit = snapshot.governance_audit_records[0]
    assert audit.runtime_facts["governance_facts"]["version"] == "15.10"
    assert snapshot.approval_requests == ()
    assert RuntimeEventType.POLICY_DECISION in {event.type for event in snapshot.events}
    assert result.diagnostics["governance"]["blocked"] is True


async def test_read_prompt_destructive_terms_do_not_trigger_destructive_policy():
    plugin = ReadProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    evidence = await runtime.execute_capability(
        "db.sql.execute_read",
        owner="read_probe",
        operation_type="data.query",
        input={"prompt": "show the customer drop rate by month"},
        operation_id="read-drop-rate",
    )
    snapshot = await runtime.inspect_operation("read-drop-rate")

    assert evidence[0].kind == "query.result"
    assert snapshot.policy_decisions == ()
    assert snapshot.operation.status is OperationStatus.SUCCEEDED
    audit = snapshot.governance_audit_records[0]
    assert audit.allowed is True
    assert audit.runtime_facts["governance_facts"]["version"] == "15.10"
    assert audit.runtime_facts["governance_facts"]["operation"]["access"] == "read"
    event_types = [event.type for event in snapshot.events]
    assert RuntimeEventType.EXECUTOR_STARTED in event_types
    assert RuntimeEventType.EVIDENCE_ACCEPTED in event_types
    assert RuntimeEventType.EXECUTOR_COMPLETED in event_types
    evidence_event = next(
        event
        for event in snapshot.events
        if event.type is RuntimeEventType.EVIDENCE_ACCEPTED
    )
    assert evidence_event.runtime_id == runtime.runtime_id
    assert evidence_event.runtime_kind == "db"
    assert evidence_event.operation_id == "read-drop-rate"
    assert evidence_event.task_id == evidence[0].task_id
    assert evidence_event.capability_id == "db.sql.execute_read"
    assert evidence_event.executor_id == "read_probe.sql.execute_read"
    assert evidence_event.plugin_id == "read_probe"
    assert evidence_event.evidence_id == evidence[0].id


async def test_db_runtime_requires_approval_for_admin_operations():
    runtime = DbRuntime()

    result = await runtime.run(DbRequest("run database maintenance", mode="admin"))
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_runtime_approval_required",)
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.REQUIRE_APPROVAL
    ]
    assert len(snapshot.approval_requests) == 1
    assert snapshot.tasks == ()


async def test_direct_capability_execution_requires_governance_approval():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_capability(
            "db.sql.execute_write",
            owner="write_probe",
            operation_type="write.execute",
            input={"sql": "insert into orders values (1)"},
            operation_id="direct-write",
        )
    snapshot = await runtime.inspect_operation("direct-write")

    assert plugin.write_executor.calls == 0
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert len(snapshot.tasks) == 2
    assert [task.capability_id for task in snapshot.tasks] == [
        "db.sql.validate",
        "db.sql.execute_write",
    ]
    assert [task.status for task in snapshot.tasks] == [
        TaskStatus.PENDING,
        TaskStatus.BLOCKED,
    ]
    assert snapshot.tasks[1].metadata["reason"] == "direct"
    assert snapshot.tasks[1].dependencies[0].producer_task_id == snapshot.tasks[0].id
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.REQUIRE_APPROVAL
    ]
    assert len(snapshot.approval_requests) == 1
    assert len(snapshot.governance_audit_records) == 1
    audit = snapshot.governance_audit_records[0]
    assert audit.stage == "operation"
    assert audit.capability_id == "db.sql.execute_write"
    assert audit.capability_context["owner"] == "write_probe"
    assert audit.resource["capability"]["risk"] == "high"
    assert audit.traces[0].capability_id == "db.sql.execute_write"


async def test_direct_write_capability_cannot_hide_behind_read_operation_type():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_capability(
            "db.sql.execute_write",
            owner="write_probe",
            operation_type="data.query",
            input={"sql": "insert into orders values (1)"},
            operation_id="direct-disguised-write",
        )
    snapshot = await runtime.inspect_operation("direct-disguised-write")

    assert plugin.write_executor.calls == 0
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert snapshot.tasks[1].status is TaskStatus.BLOCKED
    assert snapshot.approval_requests[0].status is ApprovalStatus.PENDING


async def test_execute_task_is_governance_choke_point_before_executor_runs():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="task-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    await runtime.store.save_operation(operation)
    task = Task(
        id="task-write-1",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql": "insert into orders values (1)"},
        metadata={"owner": "write_probe"},
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(task, operation)
    snapshot = await runtime.inspect_operation(operation.id)

    assert plugin.write_executor.calls == 0
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert snapshot.tasks[0].status is TaskStatus.BLOCKED
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.REQUIRE_APPROVAL
    ]
    assert len(snapshot.approval_requests) == 1
    assert len(snapshot.governance_audit_records) == 1
    audit = snapshot.governance_audit_records[0]
    assert audit.stage == "task"
    assert audit.task_id == "task-write-1"
    assert audit.task_context["capability_id"] == "db.sql.execute_write"
    assert audit.traces[0].task_id == "task-write-1"
    assert audit.runtime_facts["governance_facts"]["version"] == "15.10"


async def test_approved_request_is_not_reset_when_task_governance_rechecks():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="task-approved-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    await runtime.store.save_operation(operation)
    task = Task(
        id="task-approved-write-1",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql": "insert into orders values (1)"},
        metadata={"owner": "write_probe"},
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(task, operation)
    first_snapshot = await runtime.inspect_operation(operation.id)
    await runtime.approval_channel.approve(
        first_snapshot.approval_requests[0].approval_id
    )
    await runtime.store.save_evidence(
        Evidence(
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id="task-approved-validate",
            payload={"valid": True, "sql": "insert into orders values (1)"},
        )
    )

    evidence = await runtime.execute_task(task, operation)
    snapshot = await runtime.inspect_operation(operation.id)

    assert plugin.write_executor.calls == 1
    assert evidence[0].kind == "write.execution"
    assert snapshot.approval_requests[0].status is ApprovalStatus.APPROVED
    assert await runtime.approval_channel.pending(operation.id) == ()
    assert (
        snapshot.governance_audit_records[-1].runtime_facts["governance_facts"][
            "validation"
        ]["destructive_statement_classes"]
        == []
    )


async def test_validated_destructive_sql_denies_even_after_approval():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="approved-destructive-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "apply the approved maintenance change"},
    )
    validate_task = Task(
        id="approved-destructive-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "drop table orders"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe", "sequence": 1},
    )
    write_task = Task(
        id="approved-destructive-write-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql_ref": "sql.validation"},
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                producer_task_id=validate_task.id,
                evidence_payload={"valid": True},
                operation_id=operation.id,
            ),
            TaskDependency(
                kind="approval",
                approval_status=ApprovalStatus.APPROVED,
                approval_policy_id="approval_required_for_writes",
                approval_name="human",
                operation_id=operation.id,
            ),
        ),
        metadata={"owner": "write_probe", "sequence": 2},
    )
    validation_evidence = Evidence(
        id="approved-destructive-validation-evidence",
        kind="sql.validation",
        owner="write_probe",
        operation_id=operation.id,
        task_id=validate_task.id,
        payload={
            "valid": True,
            "sql": "drop table orders",
            "statement_facts": _statement_facts_for_probe_sql("drop table orders"),
        },
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(validate_task)
    await runtime.store.save_task(write_task)
    await runtime.store.save_evidence(validation_evidence)
    await runtime.store.save_approval_request(
        ApprovalRequest(
            approval_id="approved-destructive-write:approval_required_for_writes:human",
            operation_id=operation.id,
            reason="Approve write.",
            proposed_action={"approval": "human", "operation_type": "write.execute"},
            risk=RiskLevel.HIGH,
            status=ApprovalStatus.APPROVED,
            requested_by_policy_id="approval_required_for_writes",
            owner="runtime",
        )
    )

    with pytest.raises(PermissionError, match="denied"):
        await runtime.execute_task(write_task, operation)
    snapshot = await runtime.inspect_operation(operation.id)

    assert plugin.write_executor.calls == 0
    assert snapshot.operation.status is OperationStatus.BLOCKED
    assert snapshot.tasks[1].status is TaskStatus.BLOCKED
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.DENY
    ]
    decision = snapshot.policy_decisions[0]
    assert decision.policy_identity == "runtime:deny_destructive_operations@2"
    assert decision.metadata["decision_source"] == "validated_statement_facts"
    assert decision.metadata["validation_evidence_ids"] == (validation_evidence.id,)
    audit = snapshot.governance_audit_records[-1]
    assert audit.traces[0].evidence_ids == (validation_evidence.id,)
    assert audit.runtime_facts["governance_facts"]["validation"][
        "destructive_statement_classes"
    ] == ["DROP"]
    assert snapshot.approval_requests[0].status is ApprovalStatus.APPROVED


async def test_task_governance_uses_dependency_bound_validation_facts():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="dependency-scoped-validation",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "apply the approved order insert"},
    )
    stale_validate_task = Task(
        id="dependency-scoped-stale-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "drop table orders"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe", "sequence": 1},
    )
    benign_validate_task = Task(
        id="dependency-scoped-benign-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "insert into orders values (1)"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe", "sequence": 2},
    )
    write_task = Task(
        id="dependency-scoped-write",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql_ref": "sql.validation"},
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                producer_task_id=benign_validate_task.id,
                evidence_payload={"valid": True},
                operation_id=operation.id,
            ),
            TaskDependency(
                kind="approval",
                approval_status=ApprovalStatus.APPROVED,
                approval_policy_id="approval_required_for_writes",
                approval_name="human",
                operation_id=operation.id,
            ),
        ),
        metadata={"owner": "write_probe", "sequence": 3},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(stale_validate_task)
    await runtime.store.save_task(benign_validate_task)
    await runtime.store.save_task(write_task)
    await runtime.store.save_evidence(
        Evidence(
            id="dependency-scoped-stale-evidence",
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id=stale_validate_task.id,
            payload={
                "valid": True,
                "sql": "drop table orders",
                "statement_facts": _statement_facts_for_probe_sql("drop table orders"),
            },
        )
    )
    benign_evidence = Evidence(
        id="dependency-scoped-benign-evidence",
        kind="sql.validation",
        owner="write_probe",
        operation_id=operation.id,
        task_id=benign_validate_task.id,
        payload={
            "valid": True,
            "sql": "insert into orders values (1)",
            "statement_facts": _statement_facts_for_probe_sql(
                "insert into orders values (1)"
            ),
        },
    )
    await runtime.store.save_evidence(benign_evidence)
    await runtime.store.save_approval_request(
        ApprovalRequest(
            approval_id=f"{operation.id}:approval_required_for_writes:human",
            operation_id=operation.id,
            reason="Approve write.",
            proposed_action={"approval": "human", "operation_type": "write.execute"},
            risk=RiskLevel.HIGH,
            status=ApprovalStatus.APPROVED,
            requested_by_policy_id="approval_required_for_writes",
            owner="runtime",
        )
    )

    evidence = await runtime.execute_task(write_task, operation)
    snapshot = await runtime.inspect_operation(operation.id)
    audit = snapshot.governance_audit_records[-1]
    facts = audit.runtime_facts["governance_facts"]

    assert plugin.write_executor.calls == 1
    assert evidence[0].payload["sql"] == "insert into orders values (1)"
    assert audit.allowed is True
    assert facts["authoritative"]["source"] == "task_dependency"
    assert facts["validation"]["evidence_ids"] == [benign_evidence.id]
    assert facts["validation"]["destructive_statement_classes"] == []
    assert facts["context"]["validation"]["operation_wide"][
        "destructive_statement_classes"
    ] == ["DROP"]


async def test_non_destructive_validated_write_requires_approval_not_denial():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="validated-insert-approval",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "apply the approved order insert"},
    )
    validate_task = Task(
        id="validated-insert-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "insert into orders values (1)"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe", "sequence": 1},
    )
    write_task = Task(
        id="validated-insert-write",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql_ref": "sql.validation"},
        metadata={"owner": "write_probe", "sequence": 2},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(validate_task)
    await runtime.store.save_evidence(
        Evidence(
            id="validated-insert-evidence",
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id=validate_task.id,
            payload={
                "valid": True,
                "sql": "insert into orders values (1)",
                "statement_facts": _statement_facts_for_probe_sql(
                    "insert into orders values (1)"
                ),
            },
        )
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(write_task, operation)
    snapshot = await runtime.inspect_operation(operation.id)

    assert plugin.write_executor.calls == 0
    assert [decision.effect for decision in snapshot.policy_decisions] == [
        PolicyEffect.REQUIRE_APPROVAL
    ]
    assert snapshot.policy_decisions[0].policy_identity == (
        "runtime:approval_required_for_writes@1"
    )
    assert snapshot.governance_audit_records[-1].runtime_facts["governance_facts"][
        "validation"
    ]["mutating_statement_classes"] == ["INSERT"]


async def test_schema_empty_sql_validation_still_emits_statement_facts():
    validation = validate_sql_against_schema("DROP TABLE orders", {})

    assert validation["ok"] is True
    assert validation["statement_facts"]["statement_type"] == "drop"
    assert validation["statement_facts"]["destructive_statement_classes"] == ["DROP"]
    assert validation["statement_facts"]["target_resources"] == ["orders"]


async def test_resume_operation_executes_approved_blocked_task_once_and_skips_completed():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="resume-task-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    await runtime.store.save_operation(operation)
    completed = Task(
        id="resume-task-complete",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "insert into orders values (1)"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe", "sequence": 1},
    )
    blocked = Task(
        id="resume-task-blocked",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql": "insert into orders values (1)"},
        metadata={"owner": "write_probe", "sequence": 2},
    )
    await runtime.store.save_task(completed)
    await runtime.store.save_evidence(
        Evidence(
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id=completed.id,
            payload={"valid": True, "sql": "insert into orders values (1)"},
        )
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(blocked, operation)
    first_snapshot = await runtime.inspect_operation(operation.id)
    first_audit_ids = tuple(
        record.audit_id for record in first_snapshot.governance_audit_records
    )
    await runtime.approval_channel.approve(
        first_snapshot.approval_requests[0].approval_id
    )

    resumed = await runtime.resume_operation(operation.id)
    resumed_again = await runtime.resume_operation(operation.id)

    assert plugin.validate_executor.calls == 0
    assert plugin.write_executor.calls == 1
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert resumed_again.operation.status is OperationStatus.SUCCEEDED
    assert plugin.write_executor.calls == 1
    assert {task.id: task.status for task in resumed.tasks} == {
        completed.id: TaskStatus.SUCCEEDED,
        blocked.id: TaskStatus.SUCCEEDED,
    }
    assert (
        tuple(
            record.audit_id
            for record in resumed.governance_audit_records[: len(first_audit_ids)]
        )
        == first_audit_ids
    )
    assert len(resumed.governance_audit_records) == len(first_audit_ids) + 1
    assert resumed.governance_audit_records[0].pending_approval is True
    assert resumed.governance_audit_records[-1].allowed is True
    assert resumed_again.governance_audit_records == resumed.governance_audit_records
    assert RuntimeEventType.TASK_SKIPPED in {event.type for event in resumed.events}


async def test_resume_direct_capability_requires_original_operation_approval_scope():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    for operation_id in ("direct-scope-one", "direct-scope-two"):
        with pytest.raises(PermissionError, match="requires approval"):
            await runtime.execute_capability(
                "db.sql.execute_write",
                owner="write_probe",
                operation_type="write.execute",
                input={"sql": "insert into orders values (1)"},
                operation_id=operation_id,
            )

    first = await runtime.inspect_operation("direct-scope-one")
    second = await runtime.inspect_operation("direct-scope-two")
    await runtime.approval_channel.approve(first.approval_requests[0].approval_id)

    second_after_resume = await runtime.resume_operation("direct-scope-two")
    first_after_resume = await runtime.resume_operation("direct-scope-one")

    assert plugin.write_executor.calls == 1
    assert second_after_resume.operation.status is OperationStatus.BLOCKED
    assert second_after_resume.tasks[1].status is TaskStatus.BLOCKED
    assert second_after_resume.approval_requests[0].status is ApprovalStatus.PENDING
    assert first_after_resume.operation.status is OperationStatus.SUCCEEDED
    assert first_after_resume.approval_requests[0].status is ApprovalStatus.APPROVED


async def test_resume_run_operation_uses_persisted_plan_context_for_completion():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    result = await runtime.run(
        DbRequest("execute insert into orders values (1)", mode="write_execute")
    )
    snapshot = await runtime.inspect_operation(result.operation_id)
    await runtime.approval_channel.approve(snapshot.approval_requests[0].approval_id)

    resumed = await runtime.resume_operation(result.operation_id)

    assert plugin.validate_executor.calls == 1
    assert plugin.write_executor.calls == 1
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert [task.status for task in resumed.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    assert {item.kind for item in resumed.evidence} == {
        "sql.validation",
        "write.execution",
        "verification.result",
        "answer.synthesis",
    }
    write_task = next(
        task for task in resumed.tasks if task.capability_id == "db.sql.execute_write"
    )
    validate_task = next(
        task for task in resumed.tasks if task.capability_id == "db.sql.validate"
    )
    assert write_task.dependencies[0].producer_task_id == validate_task.id
    assert write_task.input["validated_task_id"] == validate_task.id
    assert runtime.operation_results[-1].operation_id == result.operation_id
    assert runtime.operation_results[-1].status is OperationStatus.SUCCEEDED
    assert runtime.operation_results[-1].answer == (
        "The DB operation completed with verified evidence."
    )


async def test_rejected_expired_and_cancelled_approvals_remain_inspectable():
    for transition, expected_approval, expected_status in (
        ("reject", ApprovalStatus.REJECTED, OperationStatus.FAILED),
        ("expire", ApprovalStatus.EXPIRED, OperationStatus.BLOCKED),
        ("cancel", ApprovalStatus.CANCELLED, OperationStatus.CANCELLED),
    ):
        plugin = WriteProbePlugin()
        runtime = DbRuntime(plugins=(plugin,))
        operation_id = f"direct-{transition}"
        with pytest.raises(PermissionError, match="requires approval"):
            await runtime.execute_capability(
                "db.sql.execute_write",
                owner="write_probe",
                operation_type="write.execute",
                input={"sql": "insert into orders values (1)"},
                operation_id=operation_id,
            )
        snapshot = await runtime.inspect_operation(operation_id)
        approval = getattr(runtime.approval_channel, transition)(
            snapshot.approval_requests[0].approval_id
        )
        updated_approval = await approval

        resumed = await runtime.resume_operation(operation_id)

        assert updated_approval.status is expected_approval
        assert resumed.operation.status is expected_status
        assert resumed.approval_requests[0].status is updated_approval.status
        assert any(task.status is TaskStatus.BLOCKED for task in resumed.tasks)
        assert RuntimeEventType.APPROVAL_UPDATED in {
            event.type for event in resumed.events
        }
        assert plugin.write_executor.calls == 0


async def test_approved_write_task_without_validation_evidence_is_not_runnable():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="dependency-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    await runtime.store.save_operation(operation)
    task = Task(
        id="dependency-write-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql": "insert into orders values (1)"},
        metadata={"owner": "write_probe"},
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(task, operation)
    snapshot = await runtime.inspect_operation(operation.id)
    await runtime.approval_channel.approve(snapshot.approval_requests[0].approval_id)

    with pytest.raises(DbRuntimeTaskNotRunnable):
        await runtime.execute_task(task, operation)
    blocked = await runtime.inspect_operation(operation.id)

    assert plugin.write_executor.calls == 0
    assert blocked.tasks[0].status is TaskStatus.BLOCKED
    assert blocked.tasks[0].dependencies


async def test_write_dependency_requires_declared_validation_producer():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="producer-bound-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    validate_task = Task(
        id="producer-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "insert into orders values (1)"},
        metadata={"owner": "write_probe", "input_hash": "expected-input"},
    )
    write_task = Task(
        id="producer-write",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql_ref": "sql.validation"},
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                producer_task_id=validate_task.id,
                evidence_payload={"valid": True},
                operation_id=operation.id,
            ),
            TaskDependency(
                kind="approval",
                approval_status=ApprovalStatus.APPROVED,
                approval_policy_id="approval_required_for_writes",
                approval_name="human",
                operation_id=operation.id,
            ),
        ),
        metadata={"owner": "write_probe"},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(validate_task)
    await runtime.store.save_task(write_task)
    await runtime.store.save_evidence(
        Evidence(
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id="other-validation-task",
            payload={"valid": True, "sql": "insert into orders values (999)"},
        )
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(write_task, operation)
    snapshot = await runtime.inspect_operation(operation.id)
    await runtime.approval_channel.approve(snapshot.approval_requests[0].approval_id)

    with pytest.raises(DbRuntimeTaskNotRunnable):
        await runtime.execute_task(write_task, operation)

    assert plugin.write_executor.calls == 0


async def test_completed_task_cannot_be_rerun_through_direct_execute_task():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="completed-rerun",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    task = Task(
        id="completed-rerun-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql": "insert into orders values (1)"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe"},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(task)

    with pytest.raises(DbRuntimeTaskNotRunnable, match="already succeeded"):
        await runtime.execute_task(task, operation)

    assert plugin.write_executor.calls == 0


async def test_terminal_approval_state_cannot_later_be_approved():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_capability(
            "db.sql.execute_write",
            owner="write_probe",
            operation_type="write.execute",
            input={"sql": "insert into orders values (1)"},
            operation_id="terminal-approval",
        )
    snapshot = await runtime.inspect_operation("terminal-approval")
    approval_id = snapshot.approval_requests[0].approval_id
    await runtime.approval_channel.reject(approval_id)

    with pytest.raises(ValueError, match="create a new approval request"):
        await runtime.approval_channel.approve(approval_id)

    approvals = await runtime.store.list_approval_requests("terminal-approval")
    assert approvals[0].status is ApprovalStatus.REJECTED


async def test_concurrent_resume_claims_side_effecting_task_once():
    plugin = WriteProbePlugin()
    plugin.write_executor.delay_seconds = 0.05
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="concurrent-resume",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    completed = Task(
        id="concurrent-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "insert into orders values (1)"},
        status=TaskStatus.SUCCEEDED,
        metadata={"owner": "write_probe", "sequence": 1},
    )
    blocked = Task(
        id="concurrent-write",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql_ref": "sql.validation"},
        status=TaskStatus.BLOCKED,
        metadata={"owner": "write_probe", "sequence": 2},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(completed)
    await runtime.store.save_task(blocked)
    await runtime.store.save_evidence(
        Evidence(
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id=completed.id,
            payload={"valid": True, "sql": "insert into orders values (1)"},
        )
    )

    with pytest.raises(PermissionError, match="requires approval"):
        await runtime.execute_task(blocked, operation)
    snapshot = await runtime.inspect_operation(operation.id)
    await runtime.approval_channel.approve(snapshot.approval_requests[0].approval_id)

    first, second = await asyncio.gather(
        runtime.resume_operation(operation.id),
        runtime.resume_operation(operation.id),
    )

    assert plugin.write_executor.calls == 1
    assert {first.operation.status, second.operation.status} <= {
        OperationStatus.RUNNING,
        OperationStatus.SUCCEEDED,
    }
    final = await runtime.inspect_operation(operation.id)
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert {task.id: task.status for task in final.tasks}[
        "concurrent-write"
    ] is TaskStatus.SUCCEEDED


async def test_expired_idempotent_lease_is_reclaimed_on_resume():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="expired-idempotent",
        operation_type="data.query",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    task = Task(
        id="expired-validate",
        operation_id=operation.id,
        capability_id="db.sql.validate",
        executor_id="write_probe.sql.validate",
        input={"sql": "insert into orders values (1)"},
        status=TaskStatus.RUNNING,
        metadata={
            "owner": "write_probe",
            "sequence": 1,
            "lease_expires_at": 1.0,
        },
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(task)

    resumed = await runtime.resume_operation(operation.id)

    assert plugin.validate_executor.calls == 1
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert resumed.tasks[0].status is TaskStatus.SUCCEEDED


async def test_expired_side_effecting_lease_requires_manual_recovery():
    plugin = WriteProbePlugin()
    runtime = DbRuntime(plugins=(plugin,))
    operation = Operation(
        id="expired-write",
        operation_type="write.execute",
        status=OperationStatus.RUNNING,
        request={"prompt": "insert into orders values (1)"},
    )
    task = Task(
        id="expired-write-task",
        operation_id=operation.id,
        capability_id="db.sql.execute_write",
        executor_id="write_probe.sql.execute_write",
        input={"sql_ref": "sql.validation"},
        status=TaskStatus.RUNNING,
        metadata={
            "owner": "write_probe",
            "sequence": 1,
            "lease_expires_at": 1.0,
        },
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(task)
    await runtime.store.save_evidence(
        Evidence(
            kind="sql.validation",
            owner="write_probe",
            operation_id=operation.id,
            task_id="expired-write-validation",
            payload={"valid": True, "sql": "insert into orders values (1)"},
        )
    )
    await runtime.store.save_approval_request(
        ApprovalRequest(
            approval_id="expired-write:approval_required_for_writes:human",
            operation_id=operation.id,
            reason="Approve write.",
            proposed_action={"approval": "human", "operation_type": "write.execute"},
            risk=RiskLevel.HIGH,
            status=ApprovalStatus.APPROVED,
            requested_by_policy_id="approval_required_for_writes",
            owner="runtime",
        )
    )

    resumed = await runtime.resume_operation(operation.id)

    assert plugin.write_executor.calls == 0
    assert resumed.operation.status is OperationStatus.BLOCKED
    assert resumed.tasks[0].status is TaskStatus.BLOCKED
    assert resumed.tasks[0].metadata["manual_recovery_required"] is True


async def test_approved_write_resume_survives_sqlite_store_restart(tmp_path):
    path = tmp_path / "operations.sqlite"
    first_plugin = WriteProbePlugin()
    first_runtime = DbRuntime(
        plugins=(first_plugin,),
        store=SQLiteRuntimeStore(path),
    )

    result = await first_runtime.run(
        DbRequest("execute insert into orders values (1)", mode="write_execute")
    )
    blocked = await first_runtime.inspect_operation(result.operation_id)
    blocked_audit_ids = tuple(
        record.audit_id for record in blocked.governance_audit_records
    )

    second_plugin = WriteProbePlugin()
    second_runtime = DbRuntime(
        plugins=(second_plugin,),
        store=SQLiteRuntimeStore(path),
    )
    await second_runtime.approval_channel.approve(
        blocked.approval_requests[0].approval_id
    )
    resumed = await second_runtime.resume_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert first_plugin.validate_executor.calls == 0
    assert first_plugin.write_executor.calls == 0
    assert second_plugin.validate_executor.calls == 1
    assert second_plugin.write_executor.calls == 1
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert [task.status for task in resumed.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    assert {item.kind for item in resumed.evidence} == {
        "sql.validation",
        "write.execution",
        "verification.result",
        "answer.synthesis",
    }
    assert resumed.approval_requests[0].status is ApprovalStatus.APPROVED
    assert (
        tuple(
            record.audit_id
            for record in resumed.governance_audit_records[: len(blocked_audit_ids)]
        )
        == blocked_audit_ids
    )
    assert len(resumed.governance_audit_records) == len(blocked_audit_ids) + 3
    assert resumed.governance_audit_records[0].pending_approval is True
    assert resumed.governance_audit_records[-1].allowed is True
