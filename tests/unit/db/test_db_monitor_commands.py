import json
from pathlib import Path

import pytest

from daita.db import DbAgent, DbMonitor, DbRuntime
from daita.db.llm_agent_planner import _action_input_hints
from daita.db.models import DbRequest
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.runtime import (
    ApprovalRequest,
    ApprovalStatus,
    OperationStatus,
    RiskLevel,
    TaskDependency,
    TaskStatus,
)


class FakeMonitorPlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.decisions.pop(0)


class MonitorInventorySpyRuntime(DbRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.monitor_inventory_loads = 0

    async def list_monitors(self, *, status=None):
        self.monitor_inventory_loads += 1
        return await super().list_monitors(status=status)


async def test_prompt_list_monitors_enters_runtime_run_and_agent_loop():
    planner = FakeMonitorPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "monitor.list"},
            actions=(
                DbPlannerAction(
                    action_id="list_monitors",
                    kind=DbPlannerActionKind.READ_MONITOR_STATE,
                    input={"read_kind": "list"},
                ),
            ),
        )
    )
    runtime = DbRuntime(host_services={"db_agent_planner": planner})
    await runtime.create_monitor(_monitor("orders_watch", "Orders Watch"))
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed("list monitors")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.SUCCEEDED
    assert "orders_watch" in result.answer
    assert len(planner.states) == 1
    assert planner.states[0].normalized_user_request["prompt"] == "list monitors"
    assert snapshot.operation.operation_type == "db.run"
    assert [task.capability_id for task in snapshot.tasks] == [
        "db.monitor.read",
        "db.answer.synthesize",
    ]
    assert any(item.kind == "monitor.listing" for item in snapshot.evidence)
    synthesis = next(
        item for item in snapshot.evidence if item.kind == "answer.synthesis"
    )
    assert any(
        ref["kind"] == "monitor.listing"
        for ref in synthesis.payload["cited_evidence_refs"]
    )


async def test_prompt_monitor_create_persists_loop_and_monitor_evidence():
    planner = FakeMonitorPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "monitor.create"},
            actions=(
                DbPlannerAction(
                    action_id="plan_create",
                    kind=DbPlannerActionKind.PLAN_MONITOR_CREATE,
                    input={"proposal": _proposal("orders_watch", "Orders Watch")},
                ),
                DbPlannerAction(
                    action_id="commit_create",
                    kind=DbPlannerActionKind.COMMIT_MONITOR_CREATE,
                    input={},
                    depends_on=("plan_create",),
                ),
            ),
        )
    )
    runtime = DbRuntime(host_services={"db_agent_planner": planner})
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed("create a monitor for orders")
    snapshot = await runtime.inspect_operation(result.operation_id)
    committed = await runtime.inspect_monitor("orders_watch")

    assert result.status is OperationStatus.SUCCEEDED
    assert "Created monitor Orders Watch (orders_watch)" in result.answer
    assert committed is not None
    kinds = [item.kind for item in snapshot.evidence]
    assert "planner.decision" in kinds
    assert "planner.compilation" in kinds
    assert "monitor.proposal" in kinds
    assert "monitor.definition" in kinds
    assert "answer.synthesis" in kinds
    assert [task.capability_id for task in snapshot.tasks] == [
        "db.monitor.plan_create",
        "db.monitor.commit_create",
        "db.answer.synthesize",
    ]


async def test_prompt_monitor_create_commits_each_proposal_dependency():
    planner = FakeMonitorPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "monitor.create"},
            actions=(
                DbPlannerAction(
                    action_id="plan_orders",
                    kind=DbPlannerActionKind.PLAN_MONITOR_CREATE,
                    input={
                        "proposal": _proposal(
                            "orders_watch",
                            "Orders Watch",
                            target="orders",
                        )
                    },
                ),
                DbPlannerAction(
                    action_id="plan_users",
                    kind=DbPlannerActionKind.PLAN_MONITOR_CREATE,
                    input={
                        "proposal": _proposal(
                            "users_watch",
                            "Users Watch",
                            target="users",
                        )
                    },
                ),
                DbPlannerAction(
                    action_id="commit_orders",
                    kind=DbPlannerActionKind.COMMIT_MONITOR_CREATE,
                    input={},
                    depends_on=("plan_orders",),
                ),
                DbPlannerAction(
                    action_id="commit_users",
                    kind=DbPlannerActionKind.COMMIT_MONITOR_CREATE,
                    input={},
                    depends_on=("plan_users",),
                ),
            ),
        )
    )
    runtime = DbRuntime(host_services={"db_agent_planner": planner})
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed("create order and user monitors")
    snapshot = await runtime.inspect_operation(result.operation_id)
    monitors = await runtime.list_monitors()
    definitions = [
        item for item in snapshot.evidence if item.kind == "monitor.definition"
    ]
    proposals = [item for item in snapshot.evidence if item.kind == "monitor.proposal"]
    task_ids_by_action = {
        task.metadata.get("planner_action_id"): task.id for task in snapshot.tasks
    }

    assert result.status is OperationStatus.SUCCEEDED
    assert sorted(monitor.id for monitor in monitors) == [
        "orders_watch",
        "users_watch",
    ]
    assert sorted(item.payload["monitor"]["id"] for item in definitions) == [
        "orders_watch",
        "users_watch",
    ]
    assert len({item.payload["proposal_evidence_id"] for item in definitions}) == 2
    for proposal in proposals:
        assert proposal.task_id in {
            task_ids_by_action["plan_orders"],
            task_ids_by_action["plan_users"],
        }
    for task in snapshot.tasks:
        if task.metadata.get("planner_action_id") == "commit_orders":
            assert (
                task.dependencies[0].producer_task_id
                == task_ids_by_action["plan_orders"]
            )
        if task.metadata.get("planner_action_id") == "commit_users":
            assert (
                task.dependencies[0].producer_task_id
                == task_ids_by_action["plan_users"]
            )


async def test_prompt_monitor_lifecycle_compiles_to_task_specs_and_execute_task():
    planner = FakeMonitorPlanner(_lifecycle_decision("orders_watch", action="pause"))
    runtime = MonitorInventorySpyRuntime(host_services={"db_agent_planner": planner})
    await runtime.create_monitor(_monitor("orders_watch", "Orders Watch"))
    original_execute_task = runtime.execute_task
    executed_capabilities = []

    async def execute_task_spy(task, operation, context=None):
        executed_capabilities.append(task.capability_id)
        persisted = await runtime.store.load_task(task.id)
        if task.capability_id != "db.answer.synthesize":
            assert persisted is not None
        return await original_execute_task(task, operation, context=context)

    runtime.execute_task = execute_task_spy

    result = await runtime.run(DbRequest(prompt="pause the orders monitor"))
    snapshot = await runtime.inspect_operation(result.operation_id)
    inspection = await runtime.inspect_monitor("orders_watch")

    assert snapshot is not None
    assert result.status is OperationStatus.SUCCEEDED
    assert "Paused monitor Orders Watch (orders_watch)" in result.answer
    assert inspection is not None
    assert inspection.monitor.status == "paused"
    assert executed_capabilities[:2] == [
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
    ]
    assert [task.capability_id for task in snapshot.tasks] == [
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.answer.synthesize",
    ]
    assert all(task.status is TaskStatus.SUCCEEDED for task in snapshot.tasks)
    assert any(item.kind == "monitor.paused" for item in snapshot.evidence)
    proposal = next(
        item for item in snapshot.evidence if item.kind == "monitor.proposal"
    )
    assert proposal.accepted is True
    assert proposal.payload["monitor_id"] == "orders_watch"
    assert proposal.payload["monitor_ref"] == "orders_watch"
    assert proposal.payload["resolution_source"] == "canonical_id"
    assert runtime.monitor_inventory_loads == 0

    plan_task, commit_task = snapshot.tasks[:2]
    dependency = commit_task.dependencies[0]
    assert dependency.evidence_kind == "monitor.proposal"
    assert dependency.producer_task_id == plan_task.id
    assert dependency.producer_capability_id == "db.monitor.plan_lifecycle"
    assert dependency.producer_executor_id == "db_runtime.monitor.plan_lifecycle"
    assert dependency.evidence_accepted is True


async def test_prompt_monitor_lifecycle_resolves_unique_human_reference():
    planner = FakeMonitorPlanner(_lifecycle_decision("Orders Watch", action="pause"))
    runtime = MonitorInventorySpyRuntime(host_services={"db_agent_planner": planner})
    await runtime.create_monitor(_monitor("orders_watch", "Orders Watch"))

    result = await runtime.run(DbRequest(prompt="pause Orders Watch"))
    snapshot = await runtime.inspect_operation(result.operation_id)
    inspection = await runtime.inspect_monitor("orders_watch")
    assert snapshot is not None
    proposal = next(
        item for item in snapshot.evidence if item.kind == "monitor.proposal"
    )

    assert result.status is OperationStatus.SUCCEEDED
    assert inspection is not None
    assert inspection.monitor.status == "paused"
    assert proposal.accepted is True
    assert proposal.payload["monitor_id"] == "orders_watch"
    assert proposal.payload["monitor_ref"] == "Orders Watch"
    assert proposal.payload["resolution_source"] == "resolver"
    assert proposal.payload["validation"]["warnings"] == []
    assert runtime.monitor_inventory_loads == 1


async def test_ambiguous_lifecycle_reference_persists_bounded_candidates_without_mutation():
    secret = "AMBIGUOUS_MONITOR_SECRET"
    planner = FakeMonitorPlanner(
        _lifecycle_decision("orders backlog", action="pause"),
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CLARIFY,
            clarification_question="Which orders backlog monitor should I pause?",
        ),
    )
    runtime = MonitorInventorySpyRuntime(host_services={"db_agent_planner": planner})
    for index in reversed(range(12)):
        await runtime.create_monitor(
            DbMonitor(
                id=f"orders_backlog_{index:02d}",
                name=f"Orders Backlog {index:02d}",
                description=secret,
                status="active",
                observation_plan={
                    "kind": "plugin_source",
                    "source_kind": "test_source",
                    "value_path": "rows",
                    "sql": secret,
                },
                action_plan={"delivery_target": secret},
                metadata={"raw_prompt": secret},
            )
        )
    mutations = []
    original_commit_monitor_mutation = runtime.commit_monitor_mutation

    async def commit_monitor_mutation_spy(mutation):
        mutations.append(mutation)
        return await original_commit_monitor_mutation(mutation)

    runtime.commit_monitor_mutation = commit_monitor_mutation_spy

    result = await runtime.run(DbRequest(prompt="pause the orders backlog monitor"))
    snapshot = await runtime.inspect_operation(result.operation_id)
    assert snapshot is not None
    proposal = next(
        item for item in snapshot.evidence if item.kind == "monitor.proposal"
    )
    plan_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "db.monitor.plan_lifecycle"
    )
    commit_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "db.monitor.commit_lifecycle"
    )

    assert result.status is OperationStatus.BLOCKED
    assert proposal.accepted is False
    assert proposal.payload["monitor_id"] is None
    assert proposal.payload["monitor_ref"] == "orders backlog"
    assert proposal.payload["resolution_source"] == "resolver"
    assert proposal.payload["validation"]["errors"] == ["monitor_reference_ambiguous"]
    assert proposal.payload["candidate_count"] == 12
    assert proposal.payload["included_candidate_count"] == 10
    assert proposal.payload["candidates_truncated"] is True
    assert proposal.payload["candidates"] == [
        {
            "id": f"orders_backlog_{index:02d}",
            "name": f"Orders Backlog {index:02d}",
        }
        for index in range(10)
    ]
    assert all(
        set(candidate) <= {"id", "name", "truncated_fields"}
        for candidate in proposal.payload["candidates"]
    )
    assert secret not in json.dumps(proposal.payload, sort_keys=True)
    assert mutations == []
    assert commit_task.status is TaskStatus.BLOCKED
    dependency = commit_task.dependencies[0]
    assert dependency.evidence_kind == "monitor.proposal"
    assert dependency.producer_task_id == plan_task.id
    assert dependency.producer_capability_id == "db.monitor.plan_lifecycle"
    assert dependency.producer_executor_id == "db_runtime.monitor.plan_lifecycle"
    assert dependency.evidence_accepted is True
    for index in range(12):
        inspection = await runtime.inspect_monitor(f"orders_backlog_{index:02d}")
        assert inspection is not None
        assert inspection.monitor.status == "active"

    summary = next(
        item
        for item in planner.states[1].rejected_evidence_summaries
        if item["kind"] == "monitor.proposal"
    )
    assert summary["monitor_ref"] == "orders backlog"
    assert summary["resolution_source"] == "resolver"
    assert summary["validation_errors"] == ["monitor_reference_ambiguous"]
    assert summary["candidate_count"] == 12
    assert summary["included_candidate_count"] == 10
    assert summary["candidates_truncated"] is True
    assert summary["candidates"] == proposal.payload["candidates"]


async def test_missing_lifecycle_reference_rejects_before_commit_mutation():
    runtime = MonitorInventorySpyRuntime(runtime_id="missing-monitor-reference")
    mutations = []
    original_commit_monitor_mutation = runtime.commit_monitor_mutation

    async def commit_monitor_mutation_spy(mutation):
        mutations.append(mutation)
        return await original_commit_monitor_mutation(mutation)

    runtime.commit_monitor_mutation = commit_monitor_mutation_spy

    result = await runtime.execute_monitor_lifecycle_operation(
        {
            "operation_type": "monitor.pause",
            "monitor_id": "missing monitor",
        }
    )
    snapshot = await runtime.inspect_operation(result.operation_id)
    assert snapshot is not None
    proposal = next(
        item for item in snapshot.evidence if item.kind == "monitor.proposal"
    )

    assert result.status is OperationStatus.BLOCKED
    assert proposal.accepted is False
    assert proposal.payload["monitor_id"] is None
    assert proposal.payload["monitor_ref"] == "missing monitor"
    assert proposal.payload["validation"]["errors"] == ["monitor_not_found"]
    assert proposal.payload["candidate_count"] == 0
    assert proposal.payload["candidates"] == []
    assert [task.capability_id for task in snapshot.tasks] == [
        "db.monitor.plan_lifecycle"
    ]
    assert mutations == []
    assert runtime.monitor_inventory_loads == 1


@pytest.mark.parametrize(
    "monitor_ref",
    (None, "", "   ", {"name": "orders"}, ["orders"]),
)
async def test_empty_or_malformed_lifecycle_reference_rejects_safely(monitor_ref):
    runtime = MonitorInventorySpyRuntime(runtime_id="malformed-monitor-reference")

    task, proposal = await _execute_lifecycle_plan(runtime, monitor_ref)

    assert task.status is TaskStatus.SUCCEEDED
    assert proposal.accepted is False
    assert proposal.payload["monitor_id"] is None
    assert proposal.payload["monitor_ref"] is None
    assert proposal.payload["validation"]["errors"] == ["monitor_reference_required"]
    assert proposal.payload["candidates"] == []
    assert runtime.monitor_inventory_loads == 1


async def test_explicit_null_lifecycle_reference_does_not_fall_back_to_metadata():
    runtime = MonitorInventorySpyRuntime(runtime_id="null-monitor-reference")
    await runtime.create_monitor(_monitor("orders_watch", "Orders Watch"))

    _, proposal = await _execute_lifecycle_plan(
        runtime,
        None,
        operation_monitor_id="orders_watch",
    )

    inspection = await runtime.inspect_monitor("orders_watch")
    assert proposal.accepted is False
    assert proposal.payload["monitor_id"] is None
    assert proposal.payload["monitor_ref"] is None
    assert proposal.payload["validation"]["errors"] == ["monitor_reference_required"]
    assert inspection is not None
    assert inspection.monitor.status == "active"
    assert [task.capability_id for task in await runtime.store.list_tasks()] == [
        "db.monitor.plan_lifecycle"
    ]


async def test_ambiguous_lifecycle_candidate_text_is_bounded():
    runtime = MonitorInventorySpyRuntime(runtime_id="bounded-monitor-candidates")
    for suffix in ("a", "b"):
        await runtime.create_monitor(
            _monitor(
                f"orders_{suffix}_{'x' * 80}",
                f"Orders {suffix.upper()} {'N' * 80}",
            )
        )

    _, proposal = await _execute_lifecycle_plan(runtime, "orders")

    assert proposal.accepted is False
    assert proposal.payload["validation"]["errors"] == ["monitor_reference_ambiguous"]
    assert proposal.payload["candidate_count"] == 2
    assert proposal.payload["included_candidate_count"] == 2
    assert proposal.payload["candidates_truncated"] is False
    for candidate in proposal.payload["candidates"]:
        assert len(candidate["id"]) == 48
        assert len(candidate["name"]) == 64
        assert candidate["truncated_fields"] == ["id", "name"]


async def test_prompt_monitor_request_without_llm_does_not_fall_back_to_regex_router():
    runtime = DbRuntime()
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed("list monitors")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert "DB LLM service is required" in result.answer
    assert snapshot.tasks == ()
    assert not [item for item in snapshot.evidence if item.kind == "planner.decision"]


def test_monitor_create_action_hint_exposes_catalog_evidence_ids():
    target = _action_input_hints()["monitor_action_inputs"]["plan_monitor_create"][
        "intent"
    ]["target"]

    assert target["evidence"] == ["supporting_catalog_evidence_id"]


def test_no_production_prompt_monitor_router_or_service_sources_remain():
    blocked = (
        "DbMonitorCommandService",
        "DbCommandRouter",
        "monitor_commands.router",
        "DeterministicMonitorIntentExtractor",
        "_looks_like_monitor",
        "_actions_from_prompt",
        "_schedule_from_prompt",
        "record_monitor_command_result",
        "monitor.command",
    )
    matches = []
    for path in (Path(__file__).parents[3] / "daita" / "db").rglob("*.py"):
        text = path.read_text()
        for token in blocked:
            if token in text:
                matches.append(f"{path.relative_to(Path(__file__).parents[3])}:{token}")

    assert matches == []


async def test_monitor_approval_read_and_ambiguous_resolution_use_bounded_inbox_evidence(
    monkeypatch,
):
    secret = "MONITOR_APPROVAL_SECRET"
    runtime = DbRuntime(runtime_id="monitor-approval-bounded-inbox")
    calls = []
    approvals = tuple(
        {
            "approval_id": f"approval_{index:02d}",
            "operation_id": f"target_operation_{index:02d}",
            "status": "pending",
            "requested_by_policy_id": "monitor-policy",
            "context": {
                "monitor_id": "pending_orders",
                "governance": {"raw_prompt": secret, "sql": secret},
            },
            "reason": secret,
            "delivery_target": secret,
            "metadata": {"secret": secret},
        }
        for index in reversed(range(25))
    )

    async def list_monitor_approvals_spy(
        *, monitor_id=None, monitor_run_id=None, pending_only=True
    ):
        calls.append(
            {
                "monitor_id": monitor_id,
                "monitor_run_id": monitor_run_id,
                "pending_only": pending_only,
            }
        )
        return approvals

    mutations = []
    resume_calls = []

    async def approve_spy(approval_id):
        mutations.append(approval_id)
        raise AssertionError("ambiguous resolution must not approve a candidate")

    async def resume_spy(operation_id):
        resume_calls.append(operation_id)
        raise AssertionError("approval resolution must not resume an operation")

    monkeypatch.setattr(runtime, "list_monitor_approvals", list_monitor_approvals_spy)
    monkeypatch.setattr(runtime, "approve_monitor_approval", approve_spy)
    monkeypatch.setattr(runtime, "resume_operation", resume_spy)
    operation = await _monitor_control_operation(runtime, "bounded-approval-inbox")
    read_task, read_evidence = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.read",
        task_input={
            "read_kind": "approvals",
            "monitor_id": "pending_orders",
            "monitor_run_id": "run-1",
            "pending_only": True,
        },
    )

    assert calls == [
        {
            "monitor_id": "pending_orders",
            "monitor_run_id": "run-1",
            "pending_only": True,
        }
    ]
    assert read_evidence.kind == "monitor.approval_state"
    assert read_evidence.payload["read_kind"] == "approvals"
    assert read_evidence.payload["monitor_id"] == "pending_orders"
    assert read_evidence.payload["pending_only"] is True
    assert read_evidence.payload["approval_count"] == 25
    assert read_evidence.payload["included_approval_count"] == 20
    assert read_evidence.payload["approvals_truncated"] is True
    assert read_evidence.payload["approvals"][0] == {
        "approval_id": "approval_00",
        "target_operation_id": "target_operation_00",
        "monitor_id": "pending_orders",
        "policy_id": "monitor-policy",
        "status": "pending",
    }
    serialized = json.dumps(read_evidence.payload, sort_keys=True)
    assert secret not in serialized
    for forbidden in (
        "raw_prompt",
        "sql",
        "delivery_target",
        "governance",
        "metadata",
    ):
        assert forbidden not in serialized

    dependency = _approval_state_dependency(
        read_task,
        read_evidence,
        operation_id=operation.id,
    )
    _, resolution = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.resolve_approval",
        task_input={"approval_action": "approve"},
        dependencies=(dependency,),
    )

    assert calls == [
        {
            "monitor_id": "pending_orders",
            "monitor_run_id": "run-1",
            "pending_only": True,
        }
    ]
    assert resolution.payload["status"] == "ambiguous"
    assert resolution.payload["matched_approval_count"] == 25
    assert resolution.payload["included_matched_approval_count"] == 20
    assert resolution.payload["matched_approvals_truncated"] is True
    assert resolution.payload["matched_approvals"] == read_evidence.payload["approvals"]
    assert secret not in json.dumps(resolution.payload, sort_keys=True)
    assert mutations == []
    assert resume_calls == []


async def test_monitor_approval_resolution_from_empty_inbox_is_not_found(
    monkeypatch,
):
    runtime = DbRuntime(runtime_id="monitor-approval-empty-inbox")
    mutations = []
    resume_calls = []

    async def empty_inbox(**kwargs):
        return ()

    async def approve_spy(approval_id):
        mutations.append(approval_id)
        raise AssertionError("missing approval must not be mutated")

    async def resume_spy(operation_id):
        resume_calls.append(operation_id)
        raise AssertionError("missing approval must not resume an operation")

    monkeypatch.setattr(runtime, "list_monitor_approvals", empty_inbox)
    monkeypatch.setattr(runtime, "approve_monitor_approval", approve_spy)
    monkeypatch.setattr(runtime, "resume_operation", resume_spy)
    operation = await _monitor_control_operation(runtime, "empty-approval-inbox")
    read_task, read_evidence = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.read",
        task_input={"read_kind": "approvals", "pending_only": True},
    )
    dependency = _approval_state_dependency(
        read_task,
        read_evidence,
        operation_id=operation.id,
    )

    _, resolution = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.resolve_approval",
        task_input={"approval_action": "approve"},
        dependencies=(dependency,),
    )

    assert resolution.payload == {
        "status": "not_found",
        "approval_action": "approve",
        "matched_approvals": [],
        "matched_approval_count": 0,
        "included_matched_approval_count": 0,
        "matched_approvals_truncated": False,
    }
    assert mutations == []
    assert resume_calls == []


async def test_ungrounded_monitor_approval_resolution_mutates_exactly_one_inbox_candidate(
    monkeypatch,
):
    runtime = DbRuntime(runtime_id="monitor-approval-single-inbox-candidate")
    inbox_calls = []
    mutations = []
    resume_calls = []
    candidate = {
        "approval_id": "approval-1",
        "operation_id": "target-operation",
        "status": "pending",
        "requested_by_policy_id": "monitor-policy",
        "context": {"monitor_id": "pending_orders"},
    }

    async def list_monitor_approvals_spy(**kwargs):
        inbox_calls.append(kwargs)
        return (candidate,)

    async def approve_spy(approval_id):
        mutations.append(approval_id)
        return ApprovalRequest(
            approval_id=approval_id,
            operation_id="target-operation",
            reason="Approve monitor update.",
            proposed_action={"approval": "human"},
            risk=RiskLevel.MEDIUM,
            requested_by_policy_id="monitor.policy",
            status=ApprovalStatus.APPROVED,
        )

    async def resume_spy(operation_id):
        resume_calls.append(operation_id)
        raise AssertionError("approval resolution must not resume an operation")

    monkeypatch.setattr(runtime, "list_monitor_approvals", list_monitor_approvals_spy)
    monkeypatch.setattr(runtime, "approve_monitor_approval", approve_spy)
    monkeypatch.setattr(runtime, "resume_operation", resume_spy)
    operation = await _monitor_control_operation(runtime, "single-approval-candidate")
    read_task, read_evidence = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.read",
        task_input={
            "read_kind": "approvals",
            "monitor_id": "pending_orders",
            "pending_only": True,
        },
    )
    dependency = _approval_state_dependency(
        read_task,
        read_evidence,
        operation_id=operation.id,
    )

    _, resolution = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.resolve_approval",
        task_input={
            "approval_action": "approve",
            "monitor_id": "pending_orders",
        },
        dependencies=(dependency,),
    )

    assert inbox_calls == [
        {
            "monitor_id": "pending_orders",
            "monitor_run_id": None,
            "pending_only": True,
        }
    ]
    assert mutations == ["approval-1"]
    assert resume_calls == []
    assert resolution.payload["status"] == "resolved"
    assert resolution.payload["approval_id"] == "approval-1"
    assert resolution.payload["approval_status"] == "approved"
    assert resolution.payload["operation_id"] == "target-operation"


async def test_ungrounded_monitor_approval_resolution_requires_inbox_evidence(
    monkeypatch,
):
    runtime = DbRuntime(runtime_id="monitor-approval-read-required")

    async def unexpected_inbox_read(**kwargs):
        raise AssertionError("resolve executor must not bypass db.monitor.read")

    monkeypatch.setattr(runtime, "list_monitor_approvals", unexpected_inbox_read)
    operation = await _monitor_control_operation(runtime, "approval-read-required")

    _, resolution = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.resolve_approval",
        task_input={"approval_action": "approve"},
    )

    assert resolution.payload["status"] == "inbox_required"
    assert resolution.payload["matched_approval_count"] == 0


@pytest.mark.parametrize(
    (
        "requested_id",
        "approval_action",
        "expected_status",
        "expected_mutations",
        "expected_approval_status",
    ),
    (
        (
            "approval-1",
            "approve",
            "resolved",
            [("approve", "approval-1")],
            "approved",
        ),
        (
            "approval-1",
            "reject",
            "resolved",
            [("reject", "approval-1")],
            "rejected",
        ),
        (
            "approval-1",
            "cancel",
            "resolved",
            [("cancel", "approval-1")],
            "cancelled",
        ),
        ("approval", "approve", "not_found", [], None),
        ("Approval-1", "approve", "not_found", [], None),
        ("approval-1 ", "approve", "not_found", [], None),
    ),
)
async def test_grounded_monitor_approval_id_uses_exact_equality_and_action_channel(
    monkeypatch,
    requested_id,
    approval_action,
    expected_status,
    expected_mutations,
    expected_approval_status,
):
    runtime = DbRuntime(
        runtime_id=f"monitor-approval-exact-{approval_action}-{expected_status}"
    )
    mutations = []
    resume_calls = []
    candidate = {
        "approval_id": "approval-1",
        "operation_id": "target-operation",
        "status": "pending",
        "requested_by_policy_id": "monitor-policy",
        "context": {"monitor_id": "pending_orders"},
    }

    async def list_monitor_approvals_spy(**kwargs):
        assert kwargs == {"monitor_id": "pending_orders", "pending_only": True}
        return (candidate,)

    async def mutation_spy(approval_id, *, action, status):
        mutations.append((action, approval_id))
        return ApprovalRequest(
            approval_id=approval_id,
            operation_id="target-operation",
            reason="Approve monitor update.",
            proposed_action={"approval": "human"},
            risk=RiskLevel.MEDIUM,
            requested_by_policy_id="monitor.policy",
            status=status,
        )

    async def approve_spy(approval_id):
        return await mutation_spy(
            approval_id,
            action="approve",
            status=ApprovalStatus.APPROVED,
        )

    async def reject_spy(approval_id):
        return await mutation_spy(
            approval_id,
            action="reject",
            status=ApprovalStatus.REJECTED,
        )

    async def cancel_spy(approval_id):
        return await mutation_spy(
            approval_id,
            action="cancel",
            status=ApprovalStatus.CANCELLED,
        )

    async def resume_spy(operation_id):
        resume_calls.append(operation_id)
        raise AssertionError("approval resolution must not resume an operation")

    monkeypatch.setattr(runtime, "list_monitor_approvals", list_monitor_approvals_spy)
    monkeypatch.setattr(runtime, "approve_monitor_approval", approve_spy)
    monkeypatch.setattr(runtime, "reject_monitor_approval", reject_spy)
    monkeypatch.setattr(runtime, "cancel_monitor_approval", cancel_spy)
    monkeypatch.setattr(runtime, "resume_operation", resume_spy)
    operation = await _monitor_control_operation(
        runtime,
        f"exact-approval-{requested_id}",
    )

    _, resolution = await _execute_monitor_task(
        runtime,
        operation,
        capability_id="db.monitor.resolve_approval",
        task_input={
            "approval_action": approval_action,
            "approval_id": requested_id,
            "monitor_id": "pending_orders",
        },
    )

    assert resolution.payload["status"] == expected_status
    assert resolution.payload["approval_id"] == requested_id
    assert mutations == expected_mutations
    assert resume_calls == []
    if expected_status == "resolved":
        assert resolution.payload["approval_id"] == "approval-1"
        assert resolution.payload["approval_status"] == expected_approval_status
        assert resolution.payload["operation_id"] == "target-operation"


def _lifecycle_decision(monitor_ref, *, action):
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": f"monitor.{action}"},
        actions=(
            DbPlannerAction(
                action_id=f"plan_{action}",
                kind=DbPlannerActionKind.PLAN_MONITOR_LIFECYCLE,
                input={"action": action, "monitor_id": monitor_ref},
            ),
            DbPlannerAction(
                action_id=f"commit_{action}",
                kind=DbPlannerActionKind.COMMIT_MONITOR_LIFECYCLE,
                input={"action": action},
                depends_on=(f"plan_{action}",),
            ),
        ),
    )


async def _monitor_control_operation(runtime, operation_id):
    await runtime.setup()
    return await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="monitor.approval",
        request={"kind": "monitor.approval"},
        required_evidence=frozenset(),
        metadata={
            "runtime_id": runtime.runtime_id,
            "runtime_kind": runtime.runtime_kind,
        },
        evaluate_governance=False,
    )


async def _execute_monitor_task(
    runtime,
    operation,
    *,
    capability_id,
    task_input,
    dependencies=(),
):
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id=capability_id,
                owner="db_runtime",
                input=task_input,
                reason=f"test:{capability_id}",
                dependencies=dependencies,
            ),
        ),
    )
    task = plan.tasks[0]
    evidence = await runtime.execute_task(task, operation)
    assert len(evidence) == 1
    return task, evidence[0]


def _approval_state_dependency(read_task, read_evidence, *, operation_id):
    return TaskDependency(
        kind="evidence",
        evidence_kind="monitor.approval_state",
        evidence_id=read_evidence.id,
        evidence_owner=read_evidence.owner,
        producer_task_id=read_task.id,
        producer_capability_id=read_task.capability_id,
        producer_executor_id=read_task.executor_id,
        evidence_accepted=True,
        operation_id=operation_id,
    )


async def _execute_lifecycle_plan(
    runtime,
    monitor_ref,
    *,
    operation_monitor_id=None,
):
    await runtime.setup()
    operation = await runtime.kernel.create_operation(
        operation_type="monitor.pause",
        request={"kind": "monitor.pause"},
        required_evidence=frozenset({"monitor.proposal"}),
        metadata={
            "runtime_id": runtime.runtime_id,
            "runtime_kind": runtime.runtime_kind,
            "control_plane": "db.monitor",
            "monitor_id": operation_monitor_id,
        },
        evaluate_governance=False,
    )
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="db.monitor.plan_lifecycle",
                owner="db_runtime",
                input={
                    "action": "pause",
                    "monitor_id": monitor_ref,
                },
                reason="test_monitor_reference_resolution",
            ),
        ),
    )
    task = plan.tasks[0]
    evidence = await runtime.execute_task(task, operation)
    persisted_task = await runtime.store.load_task(task.id)
    assert persisted_task is not None
    return persisted_task, evidence[0]


def _monitor(monitor_id: str, name: str) -> DbMonitor:
    return DbMonitor(
        id=monitor_id,
        name=name,
        description=f"{name} test monitor",
        status="active",
        source_scope=("orders",),
        schedule={"interval_seconds": 300},
        trigger={"type": "rows_present", "truthy": True},
        observation_plan={
            "kind": "plugin_source",
            "source_kind": "test_source",
            "value_path": "rows",
        },
        action_plan={"kind": "none", "steps": []},
    )


def _proposal(monitor_id: str, name: str, *, target: str = "orders") -> dict:
    return {
        "monitor_id": monitor_id,
        "name": name,
        "description": f"{name} test monitor",
        "status": "active",
        "target_type": "table",
        "target_name": target,
        "source_scope": [target],
        "schedule": {"interval_seconds": 300},
        "trigger": {"type": "rows_present", "truthy": True},
        "observation_plan": {
            "kind": "plugin_source",
            "source_kind": "test_source",
            "value_path": "rows",
        },
        "action_plan": {"kind": "none", "steps": []},
        "initial_state": {},
        "policy": {},
        "budgets": {},
        "owner": {},
        "governance": {
            "approval_required": False,
            "risk": "medium",
            "side_effect_summary": "Creates a test monitor.",
        },
    }
