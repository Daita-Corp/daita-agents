from pathlib import Path

from daita.db import DbAgent, DbMonitor, DbRuntime
from daita.db.llm_agent_planner import _action_input_hints
from daita.db.models import DbRequest
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.runtime import OperationStatus, TaskStatus


class FakeMonitorPlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.decisions.pop(0)


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
    assert sorted(
        item.payload["monitor"]["id"] for item in definitions
    ) == [
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
            assert task.dependencies[0].producer_task_id == task_ids_by_action[
                "plan_orders"
            ]
        if task.metadata.get("planner_action_id") == "commit_users":
            assert task.dependencies[0].producer_task_id == task_ids_by_action[
                "plan_users"
            ]


async def test_prompt_monitor_lifecycle_compiles_to_task_specs_and_execute_task():
    planner = FakeMonitorPlanner(
        DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "monitor.pause"},
            actions=(
                DbPlannerAction(
                    action_id="plan_pause",
                    kind=DbPlannerActionKind.PLAN_MONITOR_LIFECYCLE,
                    input={"action": "pause", "monitor_id": "orders_watch"},
                ),
                DbPlannerAction(
                    action_id="commit_pause",
                    kind=DbPlannerActionKind.COMMIT_MONITOR_LIFECYCLE,
                    input={"action": "pause"},
                    depends_on=("plan_pause",),
                ),
            ),
        )
    )
    runtime = DbRuntime(host_services={"db_agent_planner": planner})
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
