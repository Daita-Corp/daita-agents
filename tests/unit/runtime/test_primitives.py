import json

import pytest

from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    ContextAudience,
    ContextBlock,
    Evidence,
    EvidenceSchema,
    GovernanceAuditRecord,
    GovernanceResult,
    Operation,
    OperationStatus,
    RiskLevel,
    PolicyDecision,
    PolicyDecisionTrace,
    PolicyEffect,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStreamEvent,
    Task,
    TaskDependency,
    TaskStatus,
    ToolView,
    Worker,
)


def test_capability_is_serializable_and_round_trips():
    capability = Capability(
        id="db.sql.execute_read",
        owner="sqlite",
        description="Execute a read-only SQL query.",
        domains=frozenset({"db"}),
        operation_types=frozenset({"data.query"}),
        access=AccessMode.READ,
        risk=RiskLevel.MEDIUM,
        input_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
        output_evidence=frozenset({"query.result"}),
        executor="sqlite.sql.execute_read",
        model_visible=False,
        runtime_only=True,
        side_effecting=False,
        timeout_seconds=10,
    )

    payload = capability.to_dict()

    assert payload["domains"] == ["db"]
    assert payload["operation_types"] == ["data.query"]
    assert payload["access"] == "read"
    assert payload["risk"] == "medium"
    assert json.loads(json.dumps(payload)) == payload
    assert Capability.from_dict(payload) == capability


@pytest.mark.parametrize(
    "capability_id",
    ["DB.SQL", "db sql execute", "db..sql", "1db.sql"],
)
def test_capability_rejects_invalid_ids(capability_id):
    with pytest.raises(ValueError):
        Capability(
            id=capability_id,
            owner="sqlite",
            description="Invalid capability.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.LOW,
            input_schema={"type": "object"},
            output_evidence=frozenset({"query.result"}),
            executor="sqlite.sql.execute_read",
        )


def test_capability_rejects_non_json_serializable_schemas():
    with pytest.raises(TypeError):
        Capability(
            id="db.sql.execute_read",
            owner="sqlite",
            description="Bad schema.",
            domains=frozenset({"db"}),
            operation_types=frozenset({"data.query"}),
            access=AccessMode.READ,
            risk=RiskLevel.LOW,
            input_schema={"bad": object()},
            output_evidence=frozenset({"query.result"}),
            executor="sqlite.sql.execute_read",
        )


def test_evidence_schema_and_evidence_round_trip():
    schema = EvidenceSchema(
        kind="query.result",
        owner="sqlite",
        json_schema={"type": "object", "required": ["rows"]},
        description="Rows returned by a read query.",
    )
    evidence = Evidence(
        kind=schema.kind,
        owner=schema.owner,
        operation_id="op-1",
        task_id="task-1",
        payload={"rows": [{"count": 3}]},
        metadata={"source": "unit"},
    )

    assert EvidenceSchema.from_dict(schema.to_dict()) == schema
    assert Evidence.from_dict(evidence.to_dict()) == evidence
    assert json.loads(json.dumps(evidence.to_dict())) == evidence.to_dict()


def test_context_block_round_trips_with_audience_enum():
    block = ContextBlock(
        id="catalog.summary",
        owner="catalog",
        audience=ContextAudience.PRIMARY_MODEL,
        content="Tables: orders, customers",
        priority=10,
    )

    payload = block.to_dict()

    assert payload["audience"] == "primary_model"
    assert ContextBlock.from_dict(payload) == block


def test_tool_view_validates_model_tool_name_and_serializes_parameters():
    view = ToolView(
        name="catalog_find_join_paths",
        capability_id="catalog.relationship_paths.find",
        description="Find join paths.",
        parameters={"type": "object", "properties": {"from": {"type": "string"}}},
    )

    assert ToolView.from_dict(view.to_dict()) == view

    with pytest.raises(ValueError):
        ToolView(
            name="catalog-find-join-paths",
            capability_id="catalog.relationship_paths.find",
            description="Find join paths.",
            parameters={"type": "object"},
        )


def test_worker_validates_capabilities_and_concurrency():
    worker = Worker(
        id="catalog.refresh.worker",
        owner="catalog",
        role="catalog_refresh",
        capability_ids=frozenset({"catalog.schema.search"}),
        input_schema={"type": "object"},
        output_evidence=frozenset({"schema.search_result"}),
        max_concurrency=2,
    )

    assert Worker.from_dict(worker.to_dict()) == worker

    with pytest.raises(ValueError):
        Worker(
            id="catalog.refresh.worker",
            owner="catalog",
            role="catalog_refresh",
            capability_ids=frozenset({"catalog.schema.search"}),
            input_schema={"type": "object"},
            output_evidence=frozenset({"schema.search_result"}),
            max_concurrency=0,
        )


def test_operation_and_task_statuses_are_explicit_and_serializable():
    operation = Operation(
        id="op-1",
        operation_type="data.query",
        status=OperationStatus.RUNNING,
        request={"prompt": "count orders"},
        required_evidence=frozenset({"query.result"}),
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        capability_id="db.sql.execute_read",
        executor_id="sqlite.sql.execute_read",
        input={"sql": "select count(*) from orders"},
        status=TaskStatus.RUNNING,
        required_evidence=frozenset({"query.result"}),
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                evidence_payload={"valid": True},
            ),
        ),
    )

    assert Operation.from_dict(operation.to_dict()) == operation
    assert Task.from_dict(task.to_dict()) == task
    assert operation.to_dict()["status"] == "running"
    assert task.to_dict()["status"] == "running"
    assert task.to_dict()["dependencies"][0]["evidence_kind"] == "sql.validation"


def test_runtime_event_round_trips():
    event = RuntimeEvent(
        id="event-1",
        type=RuntimeEventType.EVIDENCE_ACCEPTED,
        operation_id="op-1",
        runtime_id="runtime-1",
        runtime_kind="db",
        task_id="task-1",
        capability_id="db.sql.execute_read",
        executor_id="sqlite.sql.execute_read",
        plugin_id="sqlite",
        evidence_id="evidence-1",
        trace_id="trace-1",
        span_id="span-1",
        message="Accepted query result evidence.",
        payload={"kind": "query.result"},
        timestamp=1.5,
    )

    assert RuntimeEvent.from_dict(event.to_dict()) == event
    assert json.loads(json.dumps(event.to_dict())) == event.to_dict()


def test_runtime_stream_event_projects_runtime_event_correlation():
    event = RuntimeEvent(
        type=RuntimeEventType.EVIDENCE_ACCEPTED,
        runtime_id="runtime-1",
        runtime_kind="db",
        operation_id="op-1",
        task_id="task-1",
        capability_id="db.sql.execute_read",
        executor_id="sqlite.sql.execute_read",
        plugin_id="sqlite",
        evidence_id="evidence-1",
        trace_id="trace-1",
        span_id="span-1",
        message="Evidence accepted.",
        payload={"kind": "query.result"},
    )

    stream_event = RuntimeStreamEvent.from_runtime_event(event)

    assert stream_event.runtime_id == event.runtime_id
    assert stream_event.runtime_kind == event.runtime_kind
    assert stream_event.operation_id == event.operation_id
    assert stream_event.task_id == event.task_id
    assert stream_event.capability_id == event.capability_id
    assert stream_event.evidence_id == event.evidence_id
    assert stream_event.to_dict()["trace_id"] == "trace-1"


def test_policy_decision_round_trips_with_evidence_and_modifications():
    evidence = Evidence(
        id="evidence-1",
        kind="governance.decision",
        owner="governance",
        payload={"matched": True},
    )
    decision = PolicyDecision(
        policy_id="governance.require_limit",
        owner="governance",
        effect=PolicyEffect.MODIFY,
        reason="Apply a smaller row limit.",
        severity=RiskLevel.MEDIUM,
        operation_id="op-1",
        modifications={"limit": 100},
        required_approvals=("data_owner",),
        evidence=(evidence,),
        metadata={"source": "unit"},
    )

    payload = decision.to_dict()

    assert payload["effect"] == "modify"
    assert payload["severity"] == "medium"
    assert json.loads(json.dumps(payload)) == payload
    assert PolicyDecision.from_dict(payload) == decision


def test_approval_request_and_governance_result_round_trip():
    decision = PolicyDecision(
        policy_id="governance.require_write_approval",
        owner="governance",
        effect="require_approval",
        reason="Writes require approval.",
        severity=RiskLevel.HIGH,
    )
    request = ApprovalRequest(
        approval_id="approval-1",
        operation_id="op-1",
        reason=decision.reason,
        proposed_action={"operation_type": "data.write"},
        risk=RiskLevel.HIGH,
        evidence_ids=("evidence-1",),
        status=ApprovalStatus.PENDING,
        requested_by_policy_id=decision.policy_id,
        owner=decision.owner,
    )
    result = GovernanceResult(
        allowed=False,
        blocked=False,
        pending_approval=True,
        decisions=(decision,),
        approval_requests=(request,),
        modified_contract={"capabilities": ["db.sql.execute_write"]},
    )

    payload = result.to_dict()

    assert payload["approval_requests"][0]["status"] == "pending"
    assert json.loads(json.dumps(payload)) == payload
    assert GovernanceResult.from_dict(payload) == result


def test_governance_audit_record_round_trips_with_decision_trace_context():
    decision = PolicyDecision(
        policy_id="governance.require_write_approval",
        owner="governance",
        policy_version="v2",
        effect="require_approval",
        reason="Writes require approval.",
        severity=RiskLevel.HIGH,
        operation_id="op-1",
    )
    trace = PolicyDecisionTrace(
        trace_id="trace-1",
        operation_id="op-1",
        policy_id=decision.policy_id,
        owner=decision.owner,
        policy_version=decision.policy_version,
        policy_identity=decision.policy_identity,
        effect=decision.effect,
        reason=decision.reason,
        stage="task",
        task_id="task-1",
        capability_id="db.sql.execute_write",
        approval_ids=("approval-1",),
        evidence_ids=("evidence-1",),
        actor={"user_id": "user-1"},
        tenant={"tenant_id": "tenant-1"},
        source_scope=("orders",),
        resource={"source_type": "sqlite"},
        runtime_facts={"capability": {"risk": "high"}},
    )
    audit = GovernanceAuditRecord(
        audit_id="audit-1",
        operation_id="op-1",
        stage="task",
        allowed=False,
        blocked=False,
        pending_approval=True,
        policy_decisions=(decision,),
        traces=(trace,),
        task_id="task-1",
        capability_id="db.sql.execute_write",
        actor=trace.actor,
        tenant=trace.tenant,
        source_scope=trace.source_scope,
        resource=trace.resource,
        approval_context={"pending_request_ids": ["approval-1"]},
        evidence_context={"evidence": [{"id": "evidence-1"}]},
        runtime_facts={"result": {"pending_approval": True}},
        timestamp=10.0,
    )

    payload = audit.to_dict()

    assert payload["traces"][0]["policy_version"] == "v2"
    assert payload["actor"]["user_id"] == "user-1"
    assert payload["source_scope"] == ["orders"]
    assert json.loads(json.dumps(payload)) == payload
    assert GovernanceAuditRecord.from_dict(payload) == audit


def test_policy_decision_rejects_non_decision_evidence_items():
    with pytest.raises(TypeError, match="Evidence"):
        PolicyDecision(
            policy_id="governance.bad",
            owner="governance",
            effect="warn",
            reason="Bad evidence.",
            severity=RiskLevel.LOW,
            evidence=(object(),),
        )
