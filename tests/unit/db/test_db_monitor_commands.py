import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from daita.db import (
    DbAgent,
    DbLimits,
    DbMonitor,
    DbRuntime,
    DbRuntimeConfig,
    HostedInAppMonitorDeliveryPlugin,
)
from daita.db.llm_agent_planner import _action_input_hints
from daita.db.loop import DbAgentLoop
from daita.db.loop.summaries import _capability_summary
from daita.db.models import DbRequest
from daita.db.monitor_commands import (
    DbMonitorCommand,
    DbMonitorPlanner,
    monitor_create_intent_from_dict,
)
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
    DbLoopState,
)
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.db.runtime import DbRuntimeTaskNotRunnable
from daita.plugins.catalog import CatalogPlugin
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    Evidence,
    OperationStatus,
    RiskLevel,
    Task,
    TaskDependency,
    TaskStatus,
)
from daita.runtime.monitors import monitor_trigger_matches


class FakeMonitorPlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.decisions.pop(0)


class CatalogCreatePlanner:
    def __init__(self, targets, *, filtered=False, delivery=False):
        self.targets = tuple(targets)
        self.filtered = filtered
        self.delivery = delivery
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        if not state.catalog_context:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.CONTINUE,
                intent={"operation_type": "monitor.create"},
                actions=(
                    DbPlannerAction(
                        action_id="build_monitor_context",
                        kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                        input={
                            "assets": list(self.targets),
                            "query": " ".join(self.targets),
                        },
                    ),
                ),
            )

        actions = []
        hint_evidence_ids = [
            str(item["id"])
            for item in state.accepted_evidence_summaries
            if item.get("kind") == "schema.column_value_hint" and item.get("id")
        ]
        assets = {str(item["name"]): item for item in state.catalog_context["assets"]}
        for target in self.targets:
            asset = assets[target]
            plan_id = f"plan_{target}"
            observation = {"filters": [], "value_path": "rows"}
            condition = {"kind": "new_rows", "operator": ">", "value": 0}
            if self.filtered:
                observation["filters"] = [
                    {
                        "column": "status",
                        "operator": "eq",
                        "value": "pending",
                        "evidence_ids": hint_evidence_ids,
                    }
                ]
                condition = {"kind": "threshold", "operator": ">", "value": 500}
            intent = {
                "target": {
                    "target_type": "table",
                    "name": target,
                    "source_scope": [target],
                    "evidence": asset["evidence_ids"][:1],
                },
                "condition": condition,
                "observation": observation,
                "schedule": {"kind": "interval", "interval_seconds": 900},
                "delivery": ({"delivery_kind": "in_app"} if self.delivery else None),
                "display": {
                    "explicit_name": f"{target.title()} Watch",
                    "description": f"Watch {target}.",
                },
            }
            actions.extend(
                (
                    DbPlannerAction(
                        action_id=plan_id,
                        kind=DbPlannerActionKind.PLAN_MONITOR_CREATE,
                        input={"intent": intent},
                    ),
                    DbPlannerAction(
                        action_id=f"commit_{target}",
                        kind=DbPlannerActionKind.COMMIT_MONITOR_CREATE,
                        input={},
                        depends_on=(plan_id,),
                    ),
                )
            )
        return DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "monitor.create"},
            actions=tuple(actions),
        )


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
    catalog = await _monitor_catalog("orders")
    planner = CatalogCreatePlanner(
        ("orders",),
        filtered=True,
        delivery=True,
    )
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            limits=DbLimits(max_rows=1000),
            metadata={
                "catalog_store_id": "store:monitor-tests",
                "host_runtime": {"delivery_defaults": ["in_app"]},
            },
        ),
        plugins=(catalog, HostedInAppMonitorDeliveryPlugin()),
        host_services={"db_agent_planner": planner},
    )
    agent = DbAgent(runtime=runtime)

    result = await agent.run_detailed(
        "create a monitor for pending orders over 500 every 15 minutes"
    )
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
    assert committed.monitor.source_scope == ("orders",)
    assert committed.monitor.schedule == {"interval_seconds": 900}
    assert committed.monitor.trigger == {
        "type": "threshold",
        "operator": "count_gt",
        "path": "rows",
        "value": 500,
    }
    assert "status = ?" in committed.monitor.observation_plan["sql"]
    assert "'pending'" not in committed.monitor.observation_plan["sql"]
    assert committed.monitor.observation_plan["parameters"][0]["value"] == "pending"
    assert committed.monitor.observation_plan["value_path"] == "rows"
    assert committed.monitor.action_plan["delivery_intent"]["target"] == {
        "type": "requesting_user"
    }
    monitor_tasks = [
        task
        for task in snapshot.tasks
        if task.capability_id.startswith("db.monitor")
        or (
            task.capability_id == "catalog.asset.inspect"
            and task.metadata.get("planner_action_id") == "plan_orders"
        )
    ]
    assert [task.capability_id for task in monitor_tasks] == [
        "catalog.asset.inspect",
        "db.monitor.plan_create",
        "db.monitor.commit_create",
    ]
    assert not any(task.capability_id == "db.schema.inspect" for task in snapshot.tasks)
    assert len(planner.states) == 2


async def test_prompt_monitor_create_commits_each_proposal_dependency():
    catalog = await _monitor_catalog("orders", "users")
    planner = CatalogCreatePlanner(("orders", "users"))
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            limits=DbLimits(max_rows=1000),
            metadata={"catalog_store_id": "store:monitor-tests"},
        ),
        plugins=(catalog,),
        host_services={"db_agent_planner": planner},
    )
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
        if task.capability_id == "db.monitor.plan_create":
            dependency = next(
                item
                for item in task.dependencies
                if item.evidence_kind == "schema.asset_profile"
            )
            assert dependency.producer_capability_id == "catalog.asset.inspect"
            assert dependency.producer_executor_id == "catalog.inspect_asset"
            assert dependency.input_hash


def test_monitor_create_compiler_materializes_exact_catalog_dependency_chain():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    state = _monitor_compiler_state(runtime)
    decision = _monitor_create_decision(_monitor_create_intent())

    compilation = DbAgentLoop(runtime, object()).compile_actions(decision, state)

    assert compilation.rejected_action_summaries == ()
    inspect, plan, commit = compilation.task_specs
    assert [item.capability_id for item in compilation.task_specs] == [
        "catalog.asset.inspect",
        "db.monitor.plan_create",
        "db.monitor.commit_create",
    ]
    assert inspect.input == {
        "store_id": "store:monitor-tests",
        "asset_ref": "orders",
        "limit": 100,
        "include_relationships": True,
    }
    catalog_dependency = plan.dependencies[0]
    assert catalog_dependency.evidence_kind == "schema.asset_profile"
    assert catalog_dependency.evidence_owner == "catalog"
    assert catalog_dependency.producer_task_id == inspect.task_id
    assert catalog_dependency.producer_capability_id == "catalog.asset.inspect"
    assert catalog_dependency.producer_executor_id == "catalog.inspect_asset"
    assert catalog_dependency.input_hash
    proposal_dependency = commit.dependencies[0]
    assert proposal_dependency.evidence_kind == "monitor.proposal"
    assert proposal_dependency.producer_task_id == plan.task_id
    assert proposal_dependency.producer_capability_id == "db.monitor.plan_create"
    assert proposal_dependency.producer_executor_id == (
        "db_runtime.monitor.plan_create"
    )


def test_monitor_create_compiler_owns_canonical_catalog_facts():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    state = _monitor_compiler_state(
        runtime,
        catalog_override={
            "assets": [
                {
                    "name": "orders",
                    "asset_ref": "catalog://warehouse/orders",
                    "evidence_ids": ["catalog-orders"],
                    "catalog_store_id": "store:monitor-tests",
                    "catalog_store_ids": ["store:monitor-tests"],
                }
            ]
        },
    )
    intent = _monitor_create_intent()
    intent["target"]["name"] = "catalog://warehouse/orders"
    intent["target"]["source_scope"] = ["catalog://warehouse/orders"]
    decision = _monitor_create_decision(
        intent,
        include_commit=False,
        plan_input={
            "catalog_asset_name": "model_orders",
            "catalog_asset_ref": "catalog://unrelated",
            "catalog_store_id": "store:unrelated",
            "catalog_selection_evidence_ids": ["catalog-unrelated"],
            "source_scope": ["users"],
        },
    )

    compilation = DbAgentLoop(runtime, object()).compile_actions(decision, state)

    inspect, plan = compilation.task_specs
    assert inspect.input["asset_ref"] == "catalog://warehouse/orders"
    assert plan.input["catalog_asset_name"] == "orders"
    assert plan.input["catalog_asset_ref"] == "catalog://warehouse/orders"
    assert plan.input["catalog_store_id"] == "store:monitor-tests"
    assert plan.input["catalog_selection_evidence_ids"] == ["catalog-orders"]
    assert plan.input["source_scope"] == ["orders"]
    assert plan.input["intent"]["target"]["name"] == "orders"
    assert plan.input["intent"]["target"]["source_scope"] == ["orders"]


@pytest.mark.parametrize(
    ("intent_override", "catalog_override", "expected_error"),
    (
        ({"target": {"name": None}}, {}, "monitor_target_name_required"),
        (
            {"target": {"name": "pending orders"}},
            {},
            "monitor_target_not_canonical",
        ),
        (
            {"target": {"evidence": []}},
            {},
            "monitor_target_catalog_evidence_required",
        ),
        (
            {"target": {"evidence": ["catalog-unrelated"]}},
            {},
            "monitor_target_catalog_evidence_unrelated",
        ),
        (
            {},
            {
                "assets": [
                    {
                        "name": "orders",
                        "asset_ref": "orders",
                        "evidence_ids": ["catalog-orders"],
                        "catalog_store_ids": ["store:a", "store:b"],
                    }
                ]
            },
            "monitor_target_catalog_store_ambiguous",
        ),
        (
            {},
            {
                "assets": [
                    {
                        "name": "orders",
                        "asset_ref": "orders",
                        "evidence_ids": ["catalog-orders"],
                        "catalog_store_ids": ["store:a"],
                        "catalog_store_count": 2,
                        "catalog_stores_truncated": True,
                    }
                ]
            },
            "monitor_target_catalog_store_ambiguous",
        ),
    ),
)
def test_monitor_create_compiler_rejects_unproven_catalog_selection(
    intent_override,
    catalog_override,
    expected_error,
):
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    state = _monitor_compiler_state(runtime, catalog_override=catalog_override)
    intent = _monitor_create_intent()
    for key, value in intent_override.items():
        if isinstance(value, dict):
            intent.setdefault(key, {}).update(value)
        else:
            intent[key] = value

    compilation = DbAgentLoop(runtime, object()).compile_actions(
        _monitor_create_decision(intent, include_commit=False),
        state,
    )

    assert compilation.task_specs == ()
    rejection = compilation.rejected_action_summaries[0]
    assert rejection["error"] == expected_error
    assert rejection["catalog_candidates"]
    assert rejection["candidate_count"] == 1
    assert rejection["included_candidate_count"] == 1


def test_monitor_create_compiler_bounds_catalog_candidate_text():
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False),))
    long_value = "x" * 500
    state = _monitor_compiler_state(
        runtime,
        catalog_override={
            "assets": [
                {
                    "name": long_value,
                    "asset_ref": long_value,
                    "evidence_ids": [long_value],
                    "catalog_store_ids": [long_value],
                }
            ]
        },
    )

    compilation = DbAgentLoop(runtime, object()).compile_actions(
        _monitor_create_decision(_monitor_create_intent(), include_commit=False),
        state,
    )

    candidate = compilation.rejected_action_summaries[0]["catalog_candidates"][0]
    assert len(candidate["name"]) == 256
    assert len(candidate["asset_ref"]) == 256
    assert len(candidate["evidence_ids"][0]) == 256
    assert len(candidate["catalog_store_ids"][0]) == 256


@pytest.mark.parametrize(
    (
        "evidence_state",
        "asset_ref",
        "target",
        "source_scope",
        "outcome",
        "expected_error",
    ),
    (
        (
            "missing",
            "orders",
            "orders",
            ["orders"],
            "blocked",
            None,
        ),
        (
            "rejected",
            "orders",
            "orders",
            ["orders"],
            "blocked",
            None,
        ),
        (
            "accepted",
            "orders",
            "users",
            ["users"],
            "rejected",
            "monitor.catalog_dependency_selection_mismatch",
        ),
        (
            "accepted",
            "orders",
            "orders",
            ["users"],
            "rejected",
            "monitor.catalog_dependency_source_scope_mismatch",
        ),
        (
            "accepted",
            "catalog://warehouse/orders",
            "orders",
            ["orders"],
            "accepted",
            None,
        ),
    ),
)
async def test_monitor_plan_dependency_outcomes_execute_through_kernel(
    evidence_state,
    asset_ref,
    target,
    source_scope,
    outcome,
    expected_error,
):
    catalog = await _monitor_catalog("orders")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            limits=DbLimits(max_rows=1000),
            metadata={"catalog_store_id": "store:monitor-tests"},
        ),
        plugins=(catalog,),
    )
    await runtime.setup()
    try:
        operation = await runtime.kernel.create_operation(
            operation_type="monitor.create",
            request={"prompt": "create monitor"},
            required_evidence=frozenset(),
            evaluate_governance=False,
        )
        producer_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="catalog.asset.inspect",
                    owner="catalog",
                    input={
                        "store_id": "store:monitor-tests",
                        "asset_ref": asset_ref,
                        "limit": 100,
                        "include_relationships": True,
                    },
                    reason="test_catalog_dependency",
                ),
            ),
        )
        producer = producer_plan.tasks[0]
        if evidence_state != "missing":
            accepted = evidence_state == "accepted"
            await runtime.store.save_evidence(
                Evidence(
                    id="catalog-profile-orders",
                    kind="schema.asset_profile",
                    owner="catalog",
                    operation_id=operation.id,
                    task_id=producer.id,
                    accepted=accepted,
                    payload={
                        "success": accepted,
                        "store_id": "store:monitor-tests",
                        "asset": {
                            "store_id": "store:monitor-tests",
                            "name": "orders",
                            "asset_ref": asset_ref,
                        },
                        "fields": [
                            {"name": "created_at", "type": "TIMESTAMP"},
                            {"name": "status", "type": "TEXT"},
                        ],
                    },
                    metadata={"task_input_hash": producer.metadata["input_hash"]},
                )
            )
        dependency = TaskDependency(
            kind="evidence",
            evidence_kind="schema.asset_profile",
            evidence_owner="catalog",
            producer_task_id=producer.id,
            producer_capability_id="catalog.asset.inspect",
            producer_executor_id="catalog.inspect_asset",
            evidence_accepted=True,
            input_hash=producer.metadata["input_hash"],
            operation_id=operation.id,
        )
        intent = _monitor_create_intent()
        intent["target"]["name"] = target
        intent["target"]["source_scope"] = source_scope
        intent["observation"] = {"filters": [], "value_path": "rows"}
        intent["condition"] = {"kind": "new_rows", "operator": ">", "value": 0}
        plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.monitor.plan_create",
                    owner="db_runtime",
                    input={
                        "intent": intent,
                        "catalog_asset_name": "orders",
                        "catalog_asset_ref": asset_ref,
                        "catalog_store_id": "store:monitor-tests",
                        "source_scope": source_scope,
                    },
                    reason="test_monitor_catalog_dependency",
                    dependencies=(dependency,),
                ),
            ),
        )

        if outcome == "blocked":
            with pytest.raises(DbRuntimeTaskNotRunnable):
                await runtime.execute_task(plan.tasks[0], operation)
            blocked = await runtime.store.load_task(plan.tasks[0].id)
            assert blocked is not None
            assert blocked.status is TaskStatus.BLOCKED
            assert not [
                item
                for item in await runtime.store.list_evidence(operation.id)
                if item.kind == "monitor.proposal"
            ]
            return

        evidence = await runtime.execute_task(plan.tasks[0], operation)
    finally:
        await runtime.teardown()

    assert evidence[0].kind == "monitor.proposal"
    assert evidence[0].accepted is (outcome == "accepted")
    if expected_error is not None:
        assert expected_error in evidence[0].payload["validation"]["errors"]
    else:
        assert evidence[0].payload["metadata"]["schema_evidence_id"] == (
            "catalog-profile-orders"
        )
        assert "catalog_asset_profile_evidence_id" not in evidence[0].payload
        assert "catalog_asset_profile_evidence_id" not in (
            evidence[0].payload["metadata"]
        )


async def test_monitor_proposal_delivery_kind_defaults_preserve_explicit_semantics():
    parsed = monitor_create_intent_from_dict({"delivery": {"target": {}}})
    assert parsed.delivery is not None
    assert parsed.delivery.delivery_kind == ""
    assert parsed.delivery.to_action_plan_intent()["target"] == {}
    omitted_target = monitor_create_intent_from_dict({"delivery": {}})
    assert omitted_target.delivery is not None
    assert "target" not in omitted_target.delivery.to_action_plan_intent()

    runtime = DbRuntime(plugins=(HostedInAppMonitorDeliveryPlugin(),))
    await runtime.setup()
    try:
        hosted, hosted_validation = DbMonitorPlanner(
            registry=runtime.registry,
            delivery_default="in_app",
        ).create_structured_proposal(
            _proposal_with_delivery({"payload_source": {"type": "monitor.report"}})
        )
        explicit, explicit_validation = DbMonitorPlanner(
            registry=runtime.registry,
            delivery_default="in_app",
        ).create_structured_proposal(
            _proposal_with_delivery(
                {
                    "delivery_kind": "local",
                    "target": {"type": "runtime_console"},
                    "payload_source": {"type": "monitor.report"},
                }
            )
        )
        empty_target, empty_target_validation = DbMonitorPlanner(
            registry=runtime.registry,
            delivery_default="in_app",
        ).create_structured_proposal(
            _proposal_with_delivery(
                {
                    "target": {},
                    "payload_source": {"type": "monitor.report"},
                }
            )
        )
        missing, missing_validation = DbMonitorPlanner(
            registry=runtime.registry,
        ).create_structured_proposal(
            _proposal_with_delivery({"payload_source": {"type": "monitor.report"}})
        )
        scheduled, scheduled_validation = DbMonitorPlanner(
            registry=runtime.registry,
            delivery_default="in_app",
        ).create_structured_proposal(
            _proposal_with_delivery(
                {"payload_source": {"type": "monitor.report"}},
                action_kind="scheduled_report",
            )
        )
        scheduled_missing, scheduled_missing_validation = DbMonitorPlanner(
            registry=runtime.registry,
        ).create_structured_proposal(
            _proposal_with_delivery(
                {"payload_source": {"type": "monitor.report"}},
                action_kind="scheduled_report",
            )
        )
        scheduled_without_delivery_input = _proposal(
            "scheduled_without_delivery",
            "Scheduled Without Delivery",
        )
        scheduled_without_delivery_input["action_plan"] = {
            "kind": "scheduled_report",
            "steps": [],
        }
        scheduled_without_delivery, scheduled_without_delivery_validation = (
            DbMonitorPlanner(
                registry=runtime.registry,
                delivery_default="in_app",
            ).create_structured_proposal(scheduled_without_delivery_input)
        )
        malformed_request = monitor_create_intent_from_dict(
            {
                "delivery": {
                    "delivery_kind": "in_app",
                    "target": "requesting_user",
                }
            }
        )
        assert malformed_request.delivery is not None
        malformed, malformed_validation = DbMonitorPlanner(
            registry=runtime.registry,
        ).create_structured_proposal(
            _proposal_with_delivery(malformed_request.delivery.to_action_plan_intent())
        )
        no_delivery, no_delivery_validation = DbMonitorPlanner(
            registry=runtime.registry,
            delivery_default="in_app",
        ).create_structured_proposal(_proposal("no_delivery", "No Delivery"))
    finally:
        await runtime.teardown()

    hosted_intent = hosted["action_plan"]["delivery_intent"]
    assert hosted_validation.accepted is True
    assert hosted_intent["delivery_kind"] == "in_app"
    assert hosted_intent["target"] == {"type": "requesting_user"}
    assert hosted_intent["capability_id"] == "monitor.delivery.in_app"
    assert hosted_intent["capability_owner"] == "hosted_monitor_delivery"

    explicit_intent = explicit["action_plan"]["delivery_intent"]
    assert explicit_validation.accepted is True
    assert explicit_intent["delivery_kind"] == "local"
    assert explicit_intent["target"] == {"type": "runtime_console"}
    assert explicit_intent["capability_id"] == "monitor.delivery.local"

    assert empty_target_validation.accepted is False
    assert empty_target["action_plan"]["delivery_intent"]["target"] == {}
    assert empty_target_validation.diagnostics["delivery_validation"]["reason"] == (
        "missing_delivery_target"
    )

    assert missing_validation.accepted is False
    assert "monitor.proposal_incomplete:delivery" in missing_validation.errors
    assert missing["action_plan"]["delivery_intent"].get("delivery_kind") is None

    scheduled_intent = scheduled["action_plan"]["delivery_intent"]
    assert scheduled_validation.accepted is True
    assert scheduled_intent["delivery_kind"] == "in_app"
    assert scheduled_intent["target"] == {"type": "requesting_user"}
    assert scheduled_intent["capability_id"] == "monitor.delivery.in_app"

    assert scheduled_missing_validation.accepted is False
    assert "monitor.proposal_incomplete:delivery" in scheduled_missing_validation.errors
    assert (
        scheduled_missing["action_plan"]["delivery_intent"].get("delivery_kind") is None
    )

    assert scheduled_without_delivery_validation.accepted is True
    assert "delivery_intent" not in scheduled_without_delivery["action_plan"]

    assert malformed_validation.accepted is False
    assert malformed["action_plan"]["delivery_intent"]["target"] == {}
    assert malformed_validation.diagnostics["delivery_validation"]["reason"] == (
        "missing_delivery_target"
    )

    assert no_delivery_validation.accepted is True
    assert no_delivery["action_plan"] == {"kind": "none", "steps": []}


def test_monitor_proposal_rejects_delivery_inputs_runtime_cannot_construct():
    capability = Capability(
        id="monitor.delivery.test",
        owner="test.delivery",
        description="Test delivery",
        domains=frozenset({"monitor", "test"}),
        operation_types=frozenset({"monitor.delivery"}),
        access=AccessMode.NONE,
        risk=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "required": ["delivery_kind", "target", "payload_source", "tenant"],
            "properties": {
                "delivery_kind": {"type": "string"},
                "target": {"type": "object"},
                "payload_source": {"type": "object"},
                "tenant": {"type": "string"},
            },
        },
        output_evidence=frozenset({"test.delivery.result"}),
        executor="test.delivery.execute",
        runtime_only=True,
        metadata={
            "monitor_roles": ["delivery"],
            "delivery_kind": "test",
            "accepted_payload_kinds": ["monitor.report"],
            "default_target": {"type": "test_inbox"},
        },
    )
    planner = DbMonitorPlanner(
        registry=SimpleNamespace(capabilities=(capability,)),  # type: ignore[arg-type]
    )

    proposal, validation = planner.create_structured_proposal(
        _proposal_with_delivery(
            {
                "delivery_kind": "test",
                "payload_source": {"type": "monitor.report"},
            }
        )
    )

    assert validation.accepted is False
    assert "monitor.delivery_unsupported:invalid_capability_input" in validation.errors
    diagnostics = validation.diagnostics["delivery_validation"]
    assert diagnostics["capability_id"] == capability.id
    assert diagnostics["capability_owner"] == capability.owner
    assert diagnostics["details"]["errors"] == ("$.tenant:missing_required",)
    assert proposal["action_plan"]["delivery_intent"].get("target") is None


def test_monitor_validation_normalizes_a_copy_of_the_action_plan():
    capabilities = HostedInAppMonitorDeliveryPlugin().declare_capabilities()
    planner = DbMonitorPlanner(
        registry=SimpleNamespace(capabilities=capabilities),  # type: ignore[arg-type]
        delivery_default="in_app",
    )
    action_plan = {
        "kind": "scheduled_report",
        "delivery_intent": {"payload_source": {"type": "monitor.report"}},
    }

    validation = planner.validate(
        action_steps=(),
        actions=(),
        source_scope=(),
        policy={},
        budgets={},
        action_plan=action_plan,
    )

    assert action_plan == {
        "kind": "scheduled_report",
        "delivery_intent": {"payload_source": {"type": "monitor.report"}},
    }
    delivery_diagnostics = validation.diagnostics["delivery_validation"]
    assert delivery_diagnostics["accepted"] is True
    assert delivery_diagnostics["delivery_kind"] == "in_app"


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
    intent = _action_input_hints()["monitor_action_inputs"]["plan_monitor_create"][
        "intent"
    ]
    target = intent["target"]

    assert target["evidence"] == ["supporting_catalog_evidence_id"]
    assert intent["observation"] == {
        "filters": [
            {
                "column": "canonical column name",
                "operator": "eq",
                "value": "grounded filter value",
                "evidence_ids": ["supporting catalog value-hint evidence id"],
            }
        ],
        "value_path": "rows",
    }


def test_monitor_create_observation_round_trips_as_a_defensive_bounded_copy():
    raw = {
        "filters": [
            {
                "column": "status",
                "operator": "eq",
                "value": "pending",
                "evidence_ids": ["hint-orders-status"],
            }
        ],
        "value_path": "rows",
    }

    intent = monitor_create_intent_from_dict({"observation": raw})
    raw["filters"][0]["value"] = "mutated"

    assert intent.observation == {
        "filters": [
            {
                "column": "status",
                "operator": "eq",
                "value": "pending",
                "evidence_ids": ["hint-orders-status"],
            }
        ],
        "value_path": "rows",
    }
    assert intent.to_dict()["observation"] == intent.observation
    serialized = intent.to_dict()
    serialized["observation"]["filters"][0]["value"] = "serialized mutation"
    assert intent.observation["filters"][0]["value"] == "pending"


@pytest.mark.parametrize(
    ("dialect", "expected_filter", "expected_cursor"),
    (
        ("sqlite", "status = ?", "created_at > ?"),
        ("mysql", "status = %s", "created_at > %s"),
        ("postgresql", "status = $1", "created_at > $2"),
    ),
)
def test_monitor_planner_parameterizes_grounded_filters_for_each_placeholder_style(
    dialect,
    expected_filter,
    expected_cursor,
):
    planner = DbMonitorPlanner(
        registry=_monitor_sql_registry(dialect),
        limits={"max_rows": 1000},
    )
    command = _monitor_create_command(_monitor_create_intent())

    proposal, validation = planner.create_proposal(
        command,
        schema=_monitor_schema(dialect),
        schema_evidence_id="catalog-profile-orders",
        grounding_evidence_by_id=_monitor_grounding_evidence(),
    )

    assert validation.accepted is True
    plan = proposal["observation_plan"]
    assert expected_filter in plan["sql"]
    assert expected_cursor in plan["sql"]
    assert "'pending'" not in plan["sql"]
    assert plan["parameters"][0] == {
        "value": "pending",
        "source": "catalog_value_hint",
        "table": "orders",
        "column": "status",
        "evidence_ids": ["hint-orders-status"],
        "db_type": "TEXT",
        "native_type": "string",
        "dialect": dialect,
    }
    assert plan["parameters"][1]["ref"] == ("monitor.state.cursor.last_created_at")
    assert proposal["trigger"] == {
        "type": "threshold",
        "operator": "count_gt",
        "path": "rows",
        "value": 500,
    }
    assert plan["max_rows"] == 501


def test_monitor_planner_uses_executable_scalar_predicate_after_output_is_known():
    intent = _monitor_create_intent()
    intent["observation"] = {"filters": [], "value_path": "rows.0.total"}
    intent["condition"] = {
        "kind": "threshold",
        "operator": ">",
        "value": 100,
        "path": "rows.0.total",
    }
    proposal, validation = DbMonitorPlanner(
        registry=_monitor_sql_registry("sqlite"),
        limits={"max_rows": 1000},
    ).create_proposal(
        _monitor_create_command(intent),
        schema=_monitor_schema("sqlite"),
    )

    assert validation.accepted is True
    assert proposal["trigger"] == {
        "type": "threshold",
        "gt": 100,
    }
    assert monitor_trigger_matches(101, proposal["trigger"]) is True
    assert monitor_trigger_matches(100, proposal["trigger"]) is False


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        (
            lambda intent: intent["observation"]["filters"][0].update(
                {"column": "status;drop"}
            ),
            "monitor.observation_invalid:unsafe_filter_identifier",
        ),
        (
            lambda intent: intent["observation"]["filters"][0].update(
                {"operator": "contains"}
            ),
            "monitor.observation_invalid:unsupported_filter_operator",
        ),
        (
            lambda intent: intent["observation"]["filters"][0].update(
                {"column": "missing_column"}
            ),
            "monitor.observation_invalid:unknown_filter_column",
        ),
        (
            lambda intent: intent["observation"]["filters"][0].update(
                {"evidence_ids": []}
            ),
            "monitor.observation_invalid:ambiguous_ungrounded_value",
        ),
        (
            lambda intent: intent["observation"]["filters"][0].update(
                {"evidence_ids": ["hint-unrelated"]}
            ),
            "monitor.observation_invalid:unrelated_grounding_evidence",
        ),
        (
            lambda intent: intent["observation"]["filters"][0].pop("value"),
            "monitor.observation_invalid:filter_value",
        ),
        (
            lambda intent: intent["observation"]["filters"][0].update(
                {"evidence_ids": "hint-orders-status"}
            ),
            "monitor.observation_invalid:filter_evidence_ids",
        ),
        (
            lambda intent: intent["condition"].update({"path": "rows.0.total"}),
            "monitor.trigger_invalid:path_mismatch",
        ),
        (
            lambda intent: intent["observation"].update(
                {"sql": "select * from orders"}
            ),
            "monitor.observation_invalid:unsupported_fields",
        ),
    ),
)
def test_monitor_planner_blocks_unsafe_or_ungrounded_observation_semantics(
    mutate,
    expected_error,
):
    intent = _monitor_create_intent()
    mutate(intent)

    _, validation = DbMonitorPlanner(
        registry=_monitor_sql_registry("sqlite"),
        limits={"max_rows": 1000},
    ).create_proposal(
        _monitor_create_command(intent),
        schema=_monitor_schema("sqlite"),
        grounding_evidence_by_id={
            **_monitor_grounding_evidence(),
            "hint-unrelated": {
                "id": "hint-unrelated",
                "kind": "schema.column_value_hint",
                "owner": "catalog",
                "accepted": True,
                "payload": {
                    "hints": [
                        {
                            "table": "users",
                            "column": "status",
                            "profile_status": "profiled",
                            "observed_values": [{"value": "pending"}],
                        }
                    ]
                },
            },
        },
    )

    assert validation.accepted is False
    assert expected_error in validation.errors


def test_monitor_planner_blocks_row_threshold_beyond_runtime_limit():
    _, validation = DbMonitorPlanner(
        registry=_monitor_sql_registry("sqlite"),
        limits={"max_rows": 500},
    ).create_proposal(
        _monitor_create_command(_monitor_create_intent()),
        schema=_monitor_schema("sqlite"),
        grounding_evidence_by_id=_monitor_grounding_evidence(),
    )

    assert validation.accepted is False
    assert "monitor.observation_invalid:threshold_exceeds_row_limit" in (
        validation.errors
    )


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


def _proposal_with_delivery(
    delivery_intent: dict,
    *,
    action_kind: str = "notification",
) -> dict:
    proposal = _proposal("delivery_watch", "Delivery Watch")
    proposal["action_plan"] = {
        "kind": action_kind,
        "delivery_intent": delivery_intent,
    }
    return proposal


def _monitor_create_intent():
    return {
        "target": {
            "target_type": "table",
            "name": "orders",
            "source_scope": ["orders"],
            "evidence": ["catalog-orders"],
        },
        "condition": {
            "kind": "threshold",
            "operator": ">",
            "value": 500,
        },
        "observation": {
            "filters": [
                {
                    "column": "status",
                    "operator": "eq",
                    "value": "pending",
                    "evidence_ids": ["hint-orders-status"],
                }
            ],
            "value_path": "rows",
        },
        "schedule": {"kind": "interval", "interval_seconds": 900},
        "delivery": None,
        "display": {"explicit_name": "Orders Watch"},
    }


def _monitor_create_decision(intent, *, include_commit=True, plan_input=None):
    actions = [
        DbPlannerAction(
            action_id="plan_orders",
            kind=DbPlannerActionKind.PLAN_MONITOR_CREATE,
            input={"intent": intent, **dict(plan_input or {})},
        )
    ]
    if include_commit:
        actions.append(
            DbPlannerAction(
                action_id="commit_orders",
                kind=DbPlannerActionKind.COMMIT_MONITOR_CREATE,
                input={},
                depends_on=("plan_orders",),
            )
        )
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "monitor.create"},
        actions=tuple(actions),
    )


def _monitor_compiler_state(runtime, *, catalog_override=None):
    catalog_context = {
        "assets": [
            {
                "name": "orders",
                "asset_ref": "orders",
                "evidence_ids": ["catalog-orders"],
                "catalog_store_id": "store:monitor-tests",
                "catalog_store_ids": ["store:monitor-tests"],
            }
        ],
        "candidate_count": 1,
        "included_candidate_count": 1,
        "truncated": False,
    }
    catalog_context.update(catalog_override or {})
    return DbLoopState(
        operation_id="monitor-create-compiler",
        normalized_user_request={"operation_type": "monitor.create"},
        safety_frame={"max_access": "write"},
        capability_summaries=tuple(
            _capability_summary(item) for item in runtime.registry.capabilities
        ),
        catalog_context=catalog_context,
    )


def _monitor_create_command(intent):
    return DbMonitorCommand(
        kind="create",
        monitor_id="orders_watch",
        prompt="Create a monitor for pending orders.",
        diagnostics={"intent": intent},
    )


def _monitor_schema(dialect):
    return {
        "database_type": dialect,
        "sql_dialect": dialect,
        "tables": [
            {
                "name": "orders",
                "columns": [
                    {"name": "created_at", "data_type": "TIMESTAMP"},
                    {"name": "status", "data_type": "TEXT"},
                    {"name": "total", "data_type": "INTEGER"},
                ],
            }
        ],
    }


def _monitor_sql_registry(dialect):
    common = {
        "owner": dialect,
        "description": f"{dialect} monitor test capability",
        "domains": frozenset({"db"}),
        "operation_types": frozenset({"monitor.tick"}),
        "access": AccessMode.READ,
        "risk": RiskLevel.LOW,
        "input_schema": {"type": "object"},
        "runtime_only": True,
        "side_effecting": False,
    }
    return SimpleNamespace(
        capabilities=(
            Capability(
                id="db.sql.validate",
                output_evidence=frozenset({"sql.validation"}),
                executor=f"{dialect}.sql.validate",
                **common,
            ),
            Capability(
                id="db.sql.execute_read",
                output_evidence=frozenset({"query.result"}),
                executor=f"{dialect}.sql.execute_read",
                **common,
            ),
        )
    )


def _monitor_grounding_evidence():
    return {
        "hint-orders-status": {
            "id": "hint-orders-status",
            "kind": "schema.column_value_hint",
            "owner": "catalog",
            "accepted": True,
            "payload": {
                "hints": [
                    {
                        "table": "orders",
                        "column": "status",
                        "profile_status": "profiled",
                        "observed_values": [
                            {"value": "pending"},
                            {"value": "complete"},
                        ],
                        "sampled": False,
                        "truncated": False,
                        "redacted": False,
                        "stale": False,
                    }
                ]
            },
        }
    }


async def _monitor_catalog(*targets):
    catalog = CatalogPlugin(auto_persist=False)
    await catalog.register_schema(
        {
            "database_type": "sqlite",
            "database_name": "monitor_tests",
            "tables": [
                {
                    "name": target,
                    "columns": [
                        {
                            "name": "created_at",
                            "type": "TIMESTAMP",
                            "is_primary_key": False,
                        },
                        {"name": "status", "type": "TEXT"},
                    ],
                }
                for target in targets
            ],
        },
        store_type="sqlite",
        store_id="store:monitor-tests",
        persist=False,
    )
    await catalog.register_column_value_profiles(
        "store:monitor-tests",
        [
            {
                "table": target,
                "column": "status",
                "distinct_count": 2,
                "top_values": [
                    {"value": "pending", "count": 10},
                    {"value": "complete", "count": 5},
                ],
            }
            for target in targets
        ],
        persist=False,
    )
    return catalog
