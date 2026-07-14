"""Live-LLM integration gates for prompt-driven ``from_db`` monitors.

Run:
    DAITA_RUN_LIVE_LLM=1 OPENAI_API_KEY=sk-... pytest \
        tests/integration/from_db/test_from_db_live_prompt_monitors.py \
        -m "requires_llm and integration" -v -s

These tests keep interactive monitor creation and lifecycle work inside
``DbRuntime.run()`` and ``DbAgentLoop``. Durable scheduler ticks, monitor
actions, delivery execution, and restart recovery remain in the Bucket 5
monitor suite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from daita.db import DbMonitor
from daita.db.runtime.extensions import HostedInAppMonitorDeliveryPlugin
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import (
    ApprovalStatus,
    HostRuntimeContext,
    OperationStatus,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    TaskStatus,
    host_runtime_context,
)

from tests.integration.from_db.live_production_helpers import (
    assert_successful_prompt_run,
    assert_synthesized_answer,
    create_live_sqlite_from_db_agent,
    evidence_kinds,
    latest_evidence,
    seed_rich_sqlite_schema,
    task_capabilities,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_llm,
]


async def test_live_prompt_monitor_create_uses_llm_planner_loop(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    delivered_notifications: list[dict[str, Any]] = []

    async def deterministic_notification_service(payload):
        delivered_notifications.append(dict(payload))
        return {"notification_id": "bucket4-in-app-notification"}

    host_context = HostRuntimeContext(
        surface="web_app",
        delivery_defaults=("in_app",),
        services={
            "hosted_in_app_notification_service": deterministic_notification_service,
        },
        runtime_extensions=(HostedInAppMonitorDeliveryPlugin(),),
    )
    with host_runtime_context(host_context):
        agent = await create_live_sqlite_from_db_agent(
            db_path,
            runtime_path=tmp_path / "runtime.sqlite",
            name="LivePromptMonitorCreate",
            stateful=False,
        )

    try:
        result = await agent.run_detailed(
            "Create a monitor for pending orders every 15 minutes. If pending "
            "orders exceed 500, notify me in app."
        )
        snapshot = await agent.runtime.inspect_operation(result.operation_id)

        assert snapshot is not None
        _assert_successful_monitor_prompt(
            agent.runtime,
            result,
            snapshot,
            required_capabilities={
                "db.monitor.plan_create",
                "db.monitor.commit_create",
            },
            outcome_evidence_kind="monitor.definition",
        )
        _assert_planner_action_dependency(
            snapshot,
            plan_kind="plan_monitor_create",
            commit_kind="commit_monitor_create",
        )
        plan_task, commit_task, proposal = _assert_monitor_proposal_dependency(
            agent.runtime,
            snapshot,
            plan_capability="db.monitor.plan_create",
            commit_capability="db.monitor.commit_create",
            commit_status=TaskStatus.SUCCEEDED,
        )

        assert plan_task.status is TaskStatus.SUCCEEDED
        assert commit_task.status is TaskStatus.SUCCEEDED
        validation = dict(proposal.payload["validation"])
        assert validation["accepted"] is True
        assert validation["missing_capabilities"] == []
        assert "monitor.delivery.in_app" in validation["required_capabilities"]
        assert validation["diagnostics"]["delivery_validation"]["accepted"] is True

        definition = latest_evidence(snapshot, "monitor.definition")
        assert definition is not None
        assert definition.payload["proposal_evidence_id"] == proposal.id
        assert definition.payload["proposal_fingerprint"] == (
            proposal.payload["proposal_fingerprint"]
        )

        schema_evidence_id = proposal.payload["metadata"]["schema_evidence_id"]
        structural_evidence = next(
            item
            for item in snapshot.evidence
            if item.id == schema_evidence_id and item.accepted
        )
        structural_truth_owners = {
            manifest.id
            for manifest in agent.runtime.registry.manifests
            if {"schema", "relationships"} <= set(manifest.provides)
        }
        assert structural_truth_owners
        assert structural_evidence.owner in structural_truth_owners
        assert "orders" in str(structural_evidence.payload).lower()

        assert proposal.payload["target_name"] == "orders"
        assert proposal.payload["source_scope"] == ["orders"]
        monitor_id = str(definition.payload["monitor"]["id"])
        inspection = await agent.inspect_monitor(monitor_id)
        assert inspection is not None
        monitor = inspection.monitor
        assert monitor.status == "active"
        assert monitor.schedule == {"interval_seconds": 900}
        assert monitor.source_scope == ("orders",)
        assert monitor.observation_plan["kind"] == "planned_read"
        assert monitor.observation_plan["target_name"] == "orders"
        assert "orders" in monitor.observation_plan["sql"].lower()
        assert monitor.observation_plan["cursor"]
        assert monitor.observation_plan["cursor_update"]
        assert monitor.observation_plan["value_path"] == "rows"
        assert monitor.trigger["type"] == "threshold"
        assert monitor.trigger["operator"] in {"gt", "count_gt"}
        assert monitor.trigger["value"] == 500
        assert monitor.action_plan["kind"] == "notification"
        delivery_intent = monitor.action_plan["delivery_intent"]
        assert delivery_intent["delivery_kind"] == "in_app"
        assert delivery_intent["target"] == {"type": "requesting_user"}
        assert delivered_notifications == []
    finally:
        await agent.stop()


async def test_live_prompt_monitor_lifecycle_via_loop(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LivePromptMonitorLifecycle",
        stateful=False,
    )

    try:
        await agent.create_monitor(_pending_orders_monitor())

        listed = await agent.run_detailed("List active monitors.")
        listed_snapshot = await agent.runtime.inspect_operation(listed.operation_id)
        assert listed_snapshot is not None
        _assert_successful_monitor_prompt(
            agent.runtime,
            listed,
            listed_snapshot,
            required_capabilities={"db.monitor.read"},
            outcome_evidence_kind="monitor.listing",
        )
        listing = latest_evidence(listed_snapshot, "monitor.listing")
        assert listing is not None
        assert any(
            item["id"] == "pending_orders" and item["status"] == "active"
            for item in listing.payload["monitors"]
        )

        inspected = await agent.run_detailed(
            "Inspect the monitor with ID pending_orders."
        )
        inspected_snapshot = await agent.runtime.inspect_operation(
            inspected.operation_id
        )
        assert inspected_snapshot is not None
        _assert_successful_monitor_prompt(
            agent.runtime,
            inspected,
            inspected_snapshot,
            required_capabilities={"db.monitor.read"},
            outcome_evidence_kind="monitor.inspection",
        )
        inspection_evidence = latest_evidence(
            inspected_snapshot,
            "monitor.inspection",
        )
        assert inspection_evidence is not None
        assert (
            inspection_evidence.payload["inspection"]["monitor"]["id"]
            == "pending_orders"
        )

        lifecycle_cases = (
            (
                "Change the monitor with ID pending_orders to run every 30 minutes.",
                "monitor.state_update",
                "active",
                {"interval_seconds": 1800},
            ),
            (
                "Pause the monitor with ID pending_orders.",
                "monitor.paused",
                "paused",
                {"interval_seconds": 1800},
            ),
            (
                "Resume the monitor with ID pending_orders.",
                "monitor.resumed",
                "active",
                {"interval_seconds": 1800},
            ),
            (
                "Disable the monitor with ID pending_orders.",
                "monitor.disabled",
                "disabled",
                {"interval_seconds": 1800},
            ),
            (
                "Permanently delete the monitor with ID pending_orders. I confirm "
                "this deletion.",
                "monitor.deleted",
                None,
                None,
            ),
        )
        for (
            prompt,
            outcome_kind,
            expected_status,
            expected_schedule,
        ) in lifecycle_cases:
            result = await agent.run_detailed(prompt)
            snapshot = await agent.runtime.inspect_operation(result.operation_id)

            assert snapshot is not None
            _assert_successful_monitor_prompt(
                agent.runtime,
                result,
                snapshot,
                required_capabilities={
                    "db.monitor.plan_lifecycle",
                    "db.monitor.commit_lifecycle",
                },
                outcome_evidence_kind=outcome_kind,
            )
            _assert_planner_action_dependency(
                snapshot,
                plan_kind="plan_monitor_lifecycle",
                commit_kind="commit_monitor_lifecycle",
            )
            _assert_monitor_proposal_dependency(
                agent.runtime,
                snapshot,
                plan_capability="db.monitor.plan_lifecycle",
                commit_capability="db.monitor.commit_lifecycle",
                commit_status=TaskStatus.SUCCEEDED,
            )
            lifecycle_evidence = latest_evidence(snapshot, outcome_kind)
            assert lifecycle_evidence is not None
            assert lifecycle_evidence.payload["monitor_id"] == "pending_orders"

            current = await agent.inspect_monitor("pending_orders")
            if expected_status is None:
                assert current is None
            else:
                assert current is not None
                assert current.monitor.status == expected_status
                assert current.monitor.schedule == expected_schedule
    finally:
        await agent.stop()


async def test_live_prompt_monitor_approval_resolution_then_runtime_resume(
    tmp_path,
):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    with host_runtime_context(
        HostRuntimeContext(
            surface="test",
            runtime_extensions=(_MonitorLifecycleApprovalPlugin(),),
        )
    ):
        agent = await create_live_sqlite_from_db_agent(
            db_path,
            runtime_path=tmp_path / "runtime.sqlite",
            name="LivePromptMonitorApproval",
            stateful=False,
        )

    try:
        await agent.create_monitor(_pending_orders_monitor())
        original = await agent.inspect_monitor("pending_orders")
        assert original is not None

        blocked_result = await agent.runtime.execute_monitor_lifecycle_operation(
            {
                "operation_type": "monitor.update",
                "monitor_id": "pending_orders",
                "patch": {"schedule": {"interval_seconds": 1800}},
            }
        )
        target_operation_id = blocked_result.operation_id
        blocked_snapshot = await agent.runtime.inspect_operation(target_operation_id)
        unchanged_before_approval = await agent.inspect_monitor("pending_orders")

        assert blocked_snapshot is not None
        assert blocked_result.status is OperationStatus.BLOCKED
        assert blocked_snapshot.operation.status is OperationStatus.BLOCKED
        assert len(blocked_snapshot.approval_requests) == 1
        approval = blocked_snapshot.approval_requests[0]
        approval_id = approval.approval_id
        assert approval.status is ApprovalStatus.PENDING
        plan_task, commit_task, _ = _assert_monitor_proposal_dependency(
            agent.runtime,
            blocked_snapshot,
            plan_capability="db.monitor.plan_lifecycle",
            commit_capability="db.monitor.commit_lifecycle",
            commit_status=TaskStatus.BLOCKED,
        )
        assert plan_task.status is TaskStatus.SUCCEEDED
        assert commit_task.status is TaskStatus.BLOCKED
        assert unchanged_before_approval is not None
        assert (
            unchanged_before_approval.monitor.to_dict()
            == original.monitor.to_dict()
        )
        assert unchanged_before_approval.state.to_dict() == original.state.to_dict()

        approval_result = await agent.run_detailed(
            "Approve the pending approval for the monitor with ID pending_orders."
        )
        approval_snapshot = await agent.runtime.inspect_operation(
            approval_result.operation_id
        )
        assert approval_snapshot is not None
        _assert_successful_monitor_prompt(
            agent.runtime,
            approval_result,
            approval_snapshot,
            required_capabilities={"db.monitor.resolve_approval"},
            outcome_evidence_kind="monitor.approval_resolution",
        )
        assert _planner_has_action(
            approval_snapshot,
            "resolve_monitor_approval",
        )
        resolution = latest_evidence(
            approval_snapshot,
            "monitor.approval_resolution",
        )
        assert resolution is not None
        assert resolution.payload["status"] == "resolved"
        assert resolution.payload["approval_id"] == approval_id
        assert resolution.payload["approval_status"] == "approved"
        assert resolution.payload["operation_id"] == target_operation_id

        resolved_target = await agent.runtime.inspect_operation(target_operation_id)
        unchanged_after_approval = await agent.inspect_monitor("pending_orders")
        assert resolved_target is not None
        assert resolved_target.operation.status is OperationStatus.BLOCKED
        assert next(
            task
            for task in resolved_target.tasks
            if task.id == commit_task.id
        ).status is TaskStatus.BLOCKED
        assert (
            resolved_target.approval_requests[0].status
            is ApprovalStatus.APPROVED
        )
        assert unchanged_after_approval is not None
        assert (
            unchanged_after_approval.monitor.to_dict()
            == original.monitor.to_dict()
        )
        assert unchanged_after_approval.state.to_dict() == original.state.to_dict()

        resumed = await agent.runtime.resume_operation(target_operation_id)
        updated = await agent.inspect_monitor("pending_orders")
        assert resumed.operation.status is OperationStatus.SUCCEEDED
        assert next(
            task for task in resumed.tasks if task.id == commit_task.id
        ).status is TaskStatus.SUCCEEDED
        assert updated is not None
        assert updated.monitor.schedule == {"interval_seconds": 1800}
        lifecycle_evidence_before_second_resume = tuple(
            item
            for item in resumed.evidence
            if item.kind == "monitor.state_update" and item.accepted
        )
        assert len(lifecycle_evidence_before_second_resume) == 1

        resumed_again = await agent.runtime.resume_operation(target_operation_id)
        updated_again = await agent.inspect_monitor("pending_orders")
        lifecycle_evidence_after_second_resume = tuple(
            item
            for item in resumed_again.evidence
            if item.kind == "monitor.state_update" and item.accepted
        )
        assert resumed_again.operation.status is OperationStatus.SUCCEEDED
        assert [item.id for item in lifecycle_evidence_after_second_resume] == [
            item.id for item in lifecycle_evidence_before_second_resume
        ]
        assert updated_again is not None
        assert updated_again.monitor.to_dict() == updated.monitor.to_dict()
        assert updated_again.state.to_dict() == updated.state.to_dict()
    finally:
        await agent.stop()


async def test_live_prompt_monitor_ambiguous_reference_clarifies(tmp_path):
    db_path = await seed_rich_sqlite_schema(tmp_path / "rich.sqlite")
    agent = await create_live_sqlite_from_db_agent(
        db_path,
        runtime_path=tmp_path / "runtime.sqlite",
        name="LivePromptMonitorAmbiguity",
        stateful=False,
    )

    try:
        await agent.create_monitor(
            _pending_orders_monitor(
                monitor_id="pending_orders_na",
                name="Pending Orders North America",
            )
        )
        await agent.create_monitor(
            _pending_orders_monitor(
                monitor_id="pending_orders_eu",
                name="Pending Orders Europe",
            )
        )
        before = {
            monitor_id: (await agent.inspect_monitor(monitor_id)).to_dict()
            for monitor_id in ("pending_orders_na", "pending_orders_eu")
        }

        result = await agent.run_detailed("Pause the pending orders monitor.")
        snapshot = await agent.runtime.inspect_operation(result.operation_id)
        after = {
            monitor_id: (await agent.inspect_monitor(monitor_id)).to_dict()
            for monitor_id in ("pending_orders_na", "pending_orders_eu")
        }

        assert snapshot is not None
        assert result.status is OperationStatus.BLOCKED
        assert snapshot.operation.operation_type == "db.run"
        assert snapshot.operation.status is OperationStatus.BLOCKED
        assert result.diagnostics["planner"]["status"] == "clarification_required"
        decision_evidence = latest_evidence(snapshot, "planner.decision")
        assert decision_evidence is not None
        decision = decision_evidence.payload["decision"]
        assert decision["status"] == "clarify"
        question = str(decision["clarification_question"] or "").strip()
        assert question
        lowered_question = question.lower()
        assert any(
            label in lowered_question
            for label in ("pending_orders_na", "north america")
        )
        assert any(
            label in lowered_question
            for label in ("pending_orders_eu", "europe")
        )
        assert result.answer == question
        assert after == before

        assert not any(
            task.capability_id == "db.monitor.commit_lifecycle"
            and task.status is TaskStatus.SUCCEEDED
            for task in snapshot.tasks
        )
        assert not any(
            item.accepted
            and item.kind
            in {
                "monitor.paused",
                "monitor.state_update",
                "monitor.disabled",
                "monitor.deleted",
            }
            for item in snapshot.evidence
        )
        _assert_registered_monitor_task_bindings(agent.runtime, snapshot)
    finally:
        await agent.stop()


class _MonitorLifecycleApprovalPolicy:
    id = "monitor.lifecycle.approval"
    owner = "bucket4_monitor_lifecycle_approval"
    version = "1"

    def applies_to(self, request, operation_type):
        capability = request.get("capability") if isinstance(request, Mapping) else None
        return (
            operation_type == "monitor.update"
            and isinstance(capability, Mapping)
            and capability.get("id") == "db.monitor.commit_lifecycle"
        )

    def modify_contract(self, contract):
        return contract

    def evaluate_operation(self, operation):
        return PolicyDecision(
            policy_id=self.id,
            owner=self.owner,
            effect=PolicyEffect.REQUIRE_APPROVAL,
            reason="Bucket 4 monitor lifecycle updates require human approval.",
            severity=RiskLevel.MEDIUM,
            operation_id=operation.id,
            required_approvals=("human",),
            metadata={"operation_type": operation.operation_type},
        )


class _MonitorLifecycleApprovalPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="bucket4_monitor_lifecycle_approval",
        display_name="Bucket 4 Monitor Lifecycle Approval",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db", "monitor"}),
    )

    def declare_policies(self):
        return (_MonitorLifecycleApprovalPolicy(),)


def _pending_orders_monitor(
    *,
    monitor_id: str = "pending_orders",
    name: str = "Pending Orders",
) -> DbMonitor:
    return DbMonitor(
        id=monitor_id,
        name=name,
        description="Monitor pending orders for the Bucket 4 prompt-loop tests.",
        status="active",
        source_scope=("orders",),
        schedule={"interval_seconds": 900},
        trigger={
            "type": "threshold",
            "path": "rows.0.pending_count",
            "operator": "gt",
            "value": 500,
        },
        observation_plan={
            "kind": "metric_sql",
            "metric": "pending_count",
            "sql": (
                "select count(*) as pending_count from orders "
                "where status = 'pending'"
            ),
            "value_path": "rows.0.pending_count",
            "source_scope": ["orders"],
        },
        action_plan={"kind": "none", "steps": []},
        metadata={"suite": "from_db_live_prompt_monitors"},
    )


def _assert_successful_monitor_prompt(
    runtime,
    result,
    snapshot,
    *,
    required_capabilities: set[str],
    outcome_evidence_kind: str,
) -> None:
    assert_successful_prompt_run(result, snapshot=snapshot)
    assert snapshot.operation.operation_type == "db.run"
    assert required_capabilities <= set(task_capabilities(snapshot))
    assert "db.answer.synthesize" in task_capabilities(snapshot)
    for capability_id in (*required_capabilities, "db.answer.synthesize"):
        assert any(
            task.capability_id == capability_id
            and task.status is TaskStatus.SUCCEEDED
            for task in snapshot.tasks
        )

    assert {
        "planner.decision",
        "planner.compilation",
        "verification.result",
        "answer.synthesis",
        outcome_evidence_kind,
    } <= evidence_kinds(snapshot)
    verification = latest_evidence(snapshot, "verification.result")
    synthesis = latest_evidence(snapshot, "answer.synthesis")
    assert verification is not None
    assert verification.accepted is True
    assert verification.payload["passed"] is True
    assert synthesis is not None
    assert synthesis.accepted is True
    assert outcome_evidence_kind in {
        citation["kind"] for citation in synthesis.payload["cited_evidence_refs"]
    }
    assert_synthesized_answer(snapshot, public_result=result)
    _assert_registered_monitor_task_bindings(runtime, snapshot)


def _assert_registered_monitor_task_bindings(runtime, snapshot) -> None:
    for task in snapshot.tasks:
        if not task.capability_id.startswith("db.monitor."):
            continue
        owner = str(task.metadata.get("owner") or "")
        assert owner
        capability = runtime.registry.get_capability(
            task.capability_id,
            owner=owner,
        )
        assert task.metadata["owner"] == capability.owner
        assert task.executor_id == capability.executor
        assert runtime.registry.get_executor(capability.executor) is not None


def _assert_monitor_proposal_dependency(
    runtime,
    snapshot,
    *,
    plan_capability: str,
    commit_capability: str,
    commit_status: TaskStatus,
):
    commit_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == commit_capability and task.status is commit_status
    )
    dependency = next(
        item
        for item in commit_task.dependencies
        if item.kind.value == "evidence" and item.evidence_kind == "monitor.proposal"
    )
    plan_task = next(
        task for task in snapshot.tasks if task.id == dependency.producer_task_id
    )
    proposal = next(
        item
        for item in snapshot.evidence
        if item.kind == "monitor.proposal"
        and item.accepted
        and item.task_id == plan_task.id
    )

    assert plan_task.capability_id == plan_capability
    assert dependency.evidence_accepted is True
    assert dependency.evidence_owner == proposal.owner
    assert dependency.producer_task_id == plan_task.id
    assert dependency.producer_capability_id == plan_task.capability_id
    assert dependency.producer_executor_id == plan_task.executor_id
    assert dependency.operation_id == snapshot.operation.id
    _assert_registered_monitor_task_bindings(runtime, snapshot)
    return plan_task, commit_task, proposal


def _assert_planner_action_dependency(
    snapshot,
    *,
    plan_kind: str,
    commit_kind: str,
) -> None:
    for item in snapshot.evidence:
        if item.kind != "planner.decision" or not item.accepted:
            continue
        actions = item.payload["decision"]["actions"]
        plan_ids = {
            action["action_id"] for action in actions if action["kind"] == plan_kind
        }
        if not plan_ids:
            continue
        if any(
            action["kind"] == commit_kind
            and plan_ids.intersection(action["depends_on"])
            for action in actions
        ):
            return
    raise AssertionError(
        f"No planner decision linked {commit_kind} to {plan_kind}."
    )


def _planner_has_action(snapshot, action_kind: str) -> bool:
    return any(
        action["kind"] == action_kind
        for item in snapshot.evidence
        if item.kind == "planner.decision" and item.accepted
        for action in item.payload["decision"]["actions"]
    )
