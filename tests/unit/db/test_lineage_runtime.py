from daita.db import DbRequest, DbRuntime
from daita.plugins.base import PluginContext
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.lineage import LineagePlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script("""
        CREATE TABLE raw_orders (
            id INTEGER PRIMARY KEY,
            total REAL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            total REAL
        );
        INSERT INTO raw_orders (id, total) VALUES (1, 42.5);
        INSERT INTO orders (id, total) VALUES (1, 42.5);
        """)


async def _runtime() -> tuple[DbRuntime, SQLitePlugin, LineagePlugin]:
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    lineage = LineagePlugin(risk_thresholds={"HIGH": 3, "MEDIUM": 1})
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), sqlite, lineage))
    await runtime.setup(agent_id="phase-12-lineage-test")
    await _seed(sqlite)
    await lineage.register_flow(
        "table:raw_orders",
        "table:orders",
        flow_type="transforms",
        transformation="Load cleaned orders",
    )
    return runtime, sqlite, lineage


async def test_lineage_registers_domain_service_capabilities():
    runtime = DbRuntime(plugins=(LineagePlugin(),))

    inspection = await runtime.inspect()

    assert inspection.plugin_ids == ("lineage",)
    assert "lineage:lineage.trace" in inspection.capability_ids
    assert "lineage:lineage.impact.analyze" in inspection.capability_ids
    assert "lineage:lineage.flow.register" in inspection.capability_ids
    assert "lineage:lineage.path.find" in inspection.capability_ids
    assert "lineage.trace" in inspection.executor_ids
    assert "lineage:lineage.trace" in inspection.evidence_schema_kinds


def test_db_runtime_can_require_lineage_evidence_when_service_registered():
    runtime = DbRuntime(
        plugins=(
            SQLitePlugin(path=":memory:"),
            LineagePlugin(),
        )
    )

    contract = runtime.build_contract(
        DbRequest("Trace lineage for orders", mode="lineage.trace")
    )

    assert contract.required_capabilities == ("lineage.trace",)
    assert contract.required_evidence == ("lineage.trace",)
    assert contract.metadata["missing_capabilities"] == []
    selected = contract.metadata["selected_capabilities"][0]
    assert selected["owner"] == "lineage"
    assert selected["executor"] == "lineage.trace"


async def test_lineage_setup_uses_plugin_context_and_auto_selects_backend(mocker):
    lineage = LineagePlugin()
    mock_backend = mocker.MagicMock()
    mocker.patch(
        "daita.core.graph.backend.auto_select_backend",
        return_value=mock_backend,
    )

    await lineage.setup(
        PluginContext(
            runtime_id="runtime-lineage",
            runtime_kind="agent",
            agent_id="agent-lineage",
        )
    )

    assert lineage._agent_id == "agent-lineage"
    assert lineage._graph_backend is mock_backend


async def test_lineage_executors_return_typed_evidence_without_graph_backend():
    lineage = LineagePlugin(risk_thresholds={"HIGH": 3, "MEDIUM": 1})
    runtime = DbRuntime(plugins=(lineage,))

    registered = await runtime.execute_capability(
        "lineage.flow.register",
        owner="lineage",
        operation_type="lineage.register",
        input={
            "source_id": "table:raw_orders",
            "target_id": "table:orders",
            "flow_type": "transforms",
        },
    )
    trace = await runtime.execute_capability(
        "lineage.trace",
        owner="lineage",
        operation_type="lineage.trace",
        input={"entity_id": "table:orders", "direction": "upstream"},
    )
    impact = await runtime.execute_capability(
        "lineage.impact.analyze",
        owner="lineage",
        operation_type="lineage.trace",
        input={"entity_id": "table:raw_orders"},
    )

    assert registered[0].kind == "lineage.flow_registered"
    assert registered[0].owner == "lineage"
    assert trace[0].kind == "lineage.trace"
    assert trace[0].payload["upstream_count"] == 1
    assert trace[0].payload["lineage"]["upstream"][0]["entity_id"] == (
        "table:raw_orders"
    )
    assert impact[0].kind == "lineage.impact"
    assert impact[0].payload["total_affected_count"] == 1


async def test_db_runtime_executes_lineage_trace_with_typed_evidence():
    runtime, _, _ = await _runtime()

    try:
        result = await runtime.run(
            DbRequest("Trace lineage for orders", mode="lineage.trace")
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["verification"]["passed"] is True
    lineage_evidence = next(
        item for item in result.evidence if item.kind == "lineage.trace"
    )
    assert lineage_evidence.owner == "lineage"
    assert lineage_evidence.payload["upstream_count"] == 1
    assert lineage_evidence.payload["lineage"]["entity_id"] == "table:orders"
