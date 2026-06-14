from daita.db import DbAgent, DbMonitor, DbRuntime, SQLiteDbMonitorStore
from daita.runtime import OperationStatus, SQLiteRuntimeStore


async def test_db_agent_typed_monitor_crud_records_runtime_operations():
    runtime = DbRuntime(runtime_id="db-monitor-runtime")
    agent = DbAgent(runtime=runtime, name="monitor-test")

    monitor = await agent.monitor(
        name="Orders Backlog",
        schedule="*/15 * * * *",
        watch="pending orders",
        trigger="pending_count > 500 for 2 consecutive checks",
        then=("inspect freshness", "notify #ops"),
        budgets={"max_rows_per_tick": 500},
    )

    assert monitor.id == "orders_backlog"
    assert monitor.schedule == {"expression": "*/15 * * * *"}
    assert monitor.observation_plan == {"watch": ["pending orders"]}
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
        "monitor.pause",
        "monitor.resume",
        "monitor.delete",
    ]
    assert all(
        operation.status is OperationStatus.SUCCEEDED for operation in operations
    )
    assert await runtime.store.list_tasks() == []
    create_evidence = await runtime.store.list_evidence(operations[0].id)
    assert create_evidence[0].kind == "monitor.definition"
    assert create_evidence[0].payload["monitor"]["id"] == "orders_backlog"


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
            observation_plan={"metrics": ["revenue", "renewals"]},
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
