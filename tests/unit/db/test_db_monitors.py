from collections.abc import Mapping

from daita.db import (
    DbAgent,
    DbMonitor,
    DbRuntime,
    DbRuntimeConfig,
    SQLiteDbMonitorStore,
)
from daita.db.runtime.tasks.runtime import DbTaskSpec
from daita.runtime import (
    ApprovalStatus,
    OperationStatus,
    PolicyDecision,
    PolicyEffect,
    RiskLevel,
    SQLiteRuntimeStore,
    TaskStatus,
)


class MonitorLifecycleApprovalPolicy:
    id = "monitor_lifecycle_approval"
    owner = "test"
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
            reason="Monitor updates require approval.",
            severity=RiskLevel.MEDIUM,
            operation_id=operation.id,
            required_approvals=("human",),
            metadata={"operation_type": operation.operation_type},
        )


class SpecSpyRuntime(DbRuntime):
    def __init__(self) -> None:
        super().__init__(runtime_id="db-monitor-spec-spy")
        self.spec_batches: list[tuple[DbTaskSpec, ...]] = []

    async def plan_task_specs(self, operation, specs, *, contract=None):
        materialized = tuple(specs)
        self.spec_batches.append(materialized)
        return await super().plan_task_specs(
            operation,
            materialized,
            contract=contract,
        )


async def test_db_agent_typed_monitor_crud_records_runtime_operations():
    runtime = DbRuntime(runtime_id="db-monitor-runtime")
    agent = DbAgent(runtime=runtime, name="monitor-test")

    monitor = await agent.monitor(
        name="Orders Backlog",
        schedule="*/15 * * * *",
        watch="pending orders",
        observation_plan={
            "kind": "metric_sql",
            "metric": "pending_count",
            "sql": "select count(*) as pending_count from orders where status = 'pending'",
            "value_path": "rows.0.pending_count",
        },
        trigger="pending_count > 500 for 2 consecutive checks",
        then=("inspect freshness", "notify #ops"),
        budgets={"max_rows_per_tick": 500},
    )

    assert monitor.id == "orders_backlog"
    assert monitor.schedule == {"expression": "*/15 * * * *"}
    assert monitor.observation_plan["kind"] == "metric_sql"
    assert await agent.list_monitors() == (monitor,)

    inspected = await agent.inspect_monitor("orders_backlog")
    assert inspected is not None
    assert inspected.monitor == monitor
    assert inspected.state.monitor_id == "orders_backlog"
    assert inspected.runs == ()

    try:
        await agent.create_monitor(monitor)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate monitor create should fail")
    assert [
        operation.operation_type for operation in await runtime.store.list_operations()
    ] == ["monitor.create"]

    updated = await agent.update_monitor(
        "orders_backlog",
        {"budgets": {"max_rows_per_tick": 250}},
    )
    assert updated.budgets == {"max_rows_per_tick": 250}
    try:
        await agent.update_monitor("orders_backlog", {"id": "renamed_monitor"})
    except ValueError as exc:
        assert "monitor id cannot be changed" in str(exc)
    else:
        raise AssertionError("monitor id update should fail")

    paused = await agent.pause_monitor(
        "orders_backlog",
        paused_until="2026-06-15T09:00:00+00:00",
    )
    assert paused.status == "paused"
    assert await agent.list_monitors(status="active") == ()
    paused_inspection = await agent.inspect_monitor("orders_backlog")
    assert paused_inspection.state.paused_until == "2026-06-15T09:00:00+00:00"

    resumed = await agent.resume_monitor("orders_backlog")
    assert resumed.status == "active"
    resumed_inspection = await agent.inspect_monitor("orders_backlog")
    assert resumed_inspection.state.paused_until is None

    deleted = await agent.delete_monitor("orders_backlog")
    assert deleted.id == "orders_backlog"
    assert await agent.list_monitors() == ()

    operations = await runtime.store.list_operations()
    assert [operation.operation_type for operation in operations] == [
        "monitor.create",
        "monitor.update",
        "monitor.update",
        "monitor.pause",
        "monitor.resume",
        "monitor.delete",
    ]
    assert [operation.status for operation in operations] == [
        OperationStatus.SUCCEEDED,
        OperationStatus.SUCCEEDED,
        OperationStatus.BLOCKED,
        OperationStatus.SUCCEEDED,
        OperationStatus.SUCCEEDED,
        OperationStatus.SUCCEEDED,
    ]
    tasks = await runtime.store.list_tasks()
    assert [task.capability_id for task in tasks] == [
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
    ]
    create_evidence = await runtime.store.list_evidence(operations[0].id)
    assert create_evidence[0].kind == "monitor.definition"
    assert create_evidence[0].payload["monitor"]["id"] == "orders_backlog"
    update_evidence = await runtime.store.list_evidence(operations[1].id)
    assert [item.kind for item in update_evidence] == [
        "monitor.proposal",
        "monitor.state_update",
    ]
    rejected_evidence = await runtime.store.list_evidence(operations[2].id)
    assert rejected_evidence[0].kind == "monitor.proposal"
    assert rejected_evidence[0].accepted is False
    delete_evidence = await runtime.store.list_evidence(operations[-1].id)
    assert [item.kind for item in delete_evidence] == [
        "monitor.proposal",
        "monitor.deleted",
    ]


async def test_monitor_lifecycle_tasks_materialize_from_task_specs():
    runtime = SpecSpyRuntime()
    agent = DbAgent(runtime=runtime, name="monitor-spec-test")
    monitor = await agent.monitor(
        name="Spec Monitor",
        schedule="*/10 * * * *",
        watch="orders",
        observation_plan={
            "kind": "metric_sql",
            "sql": "select 1 as value",
            "value_path": "rows.0.value",
        },
        trigger="value > 0",
        then="notify",
    )

    await agent.update_monitor(monitor.id, {"description": "updated"})
    lifecycle_specs = [spec for batch in runtime.spec_batches for spec in batch]
    tasks = await runtime.store.list_tasks()
    plan_task = next(
        task for task in tasks if task.capability_id == "db.monitor.plan_lifecycle"
    )
    commit_task = next(
        task for task in tasks if task.capability_id == "db.monitor.commit_lifecycle"
    )

    assert [spec.capability_id for spec in lifecycle_specs] == [
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
    ]
    assert all(isinstance(spec, DbTaskSpec) for spec in lifecycle_specs)
    assert plan_task.metadata["reason"] == "monitor_update_planning"
    assert commit_task.metadata["reason"] == "monitor_update_commit"
    assert (
        commit_task.metadata["idempotency_key"]
        == commit_task.input["proposal_fingerprint"]
    )
    dependency = next(
        dependency
        for dependency in commit_task.dependencies
        if dependency.kind.value == "evidence"
    )
    assert dependency.evidence_kind == "monitor.proposal"
    assert dependency.producer_task_id == plan_task.id


async def test_sqlite_db_monitor_store_shares_runtime_store_database(tmp_path):
    path = tmp_path / "runtime.sqlite"
    runtime = DbRuntime(
        runtime_id="db-monitor-sqlite-runtime",
        store=SQLiteRuntimeStore(path),
    )
    agent = DbAgent(runtime=runtime)

    monitor = await agent.create_monitor(
        DbMonitor(
            id="weekday_revops_report",
            name="Weekday RevOps Report",
            schedule={"expression": "0 9 * * 1-5 America/Chicago"},
            trigger={"type": "schedule", "expression": "0 9 * * 1-5"},
            observation_plan={
                "kind": "metric_sql",
                "metric": "revenue",
                "sql": "select sum(revenue) as revenue from renewals",
                "value_path": "rows.0.revenue",
            },
            action_plan={"steps": [{"kind": "report_generate"}]},
        )
    )

    reopened_monitor_store = SQLiteDbMonitorStore(path)
    reopened_runtime_store = SQLiteRuntimeStore(path)

    assert await reopened_monitor_store.load_monitor(monitor.id) == monitor
    state = await reopened_monitor_store.load_monitor_state(monitor.id)
    assert state.monitor_id == monitor.id
    assert state.last_operation_id is not None

    try:
        await agent.create_monitor(monitor)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate monitor create should fail")

    operations = await reopened_runtime_store.list_operations()
    assert [operation.operation_type for operation in operations] == ["monitor.create"]
    evidence = await reopened_runtime_store.list_evidence(operations[0].id)
    assert evidence[0].kind == "monitor.definition"
    assert evidence[0].payload["monitor"]["name"] == "Weekday RevOps Report"


async def test_monitor_update_requires_approval_then_resumes_from_proposal():
    runtime = DbRuntime(
        runtime_id="db-monitor-lifecycle-approval",
        config=DbRuntimeConfig(policies=(MonitorLifecycleApprovalPolicy(),)),
    )
    agent = DbAgent(runtime=runtime)
    monitor = await agent.create_monitor(
        DbMonitor(
            id="orders_backlog",
            name="Orders Backlog",
            trigger={"type": "threshold", "path": "rows.0.pending_count", "gt": 500},
            observation_plan={
                "kind": "metric_sql",
                "sql": "select count(*) as pending_count from orders",
                "value_path": "rows.0.pending_count",
            },
        )
    )

    result = await runtime.execute_monitor_lifecycle_operation(
        {
            "operation_type": "monitor.update",
            "monitor_id": monitor.id,
            "patch": {"budgets": {"max_rows_per_tick": 250}},
        }
    )
    snapshot = await runtime.inspect_operation(result.operation_id)
    unchanged = await agent.inspect_monitor(monitor.id)

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_runtime_approval_required",)
    assert unchanged.monitor.budgets == {}
    assert [task.capability_id for task in snapshot.tasks] == [
        "db.monitor.plan_lifecycle",
        "db.monitor.commit_lifecycle",
    ]
    assert [task.status for task in snapshot.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.BLOCKED,
    ]
    assert [item.kind for item in snapshot.evidence] == ["monitor.proposal"]
    assert snapshot.evidence[0].payload["after"]["budgets"] == {
        "max_rows_per_tick": 250
    }
    assert len(snapshot.approval_requests) == 1
    assert snapshot.approval_requests[0].status is ApprovalStatus.PENDING

    await runtime.approval_channel.approve(snapshot.approval_requests[0].approval_id)
    resumed = await runtime.resume_operation(result.operation_id)
    updated = await agent.inspect_monitor(monitor.id)

    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert updated.monitor.budgets == {"max_rows_per_tick": 250}
    assert [task.status for task in resumed.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    assert [item.kind for item in resumed.evidence] == [
        "monitor.proposal",
        "monitor.state_update",
    ]
    resumed_again = await runtime.resume_operation(result.operation_id)
    assert [task.status for task in resumed_again.tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]
    assert updated.monitor == (await agent.inspect_monitor(monitor.id)).monitor


async def test_active_watch_only_monitor_create_is_rejected():
    runtime = DbRuntime(runtime_id="db-monitor-watch-reject-runtime")
    agent = DbAgent(runtime=runtime)

    try:
        await agent.monitor(name="Orders Backlog", watch="pending orders")
    except ValueError as exc:
        assert "requires an executable observation_plan" in str(exc)
    else:
        raise AssertionError("watch-only monitor helper should fail")

    try:
        await agent.create_monitor(
            DbMonitor(
                id="watch_only",
                name="Watch Only",
                observation_plan={"watch": ["pending orders"]},
            )
        )
    except ValueError as exc:
        assert "active monitors require an executable observation_plan" in str(exc)
    else:
        raise AssertionError("active watch-only monitor create should fail")

    assert await agent.list_monitors() == ()
