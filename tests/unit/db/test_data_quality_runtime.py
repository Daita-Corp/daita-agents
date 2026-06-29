from daita.db import DbRequest, DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.data_quality import DataQualityPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total REAL,
            status TEXT
        );
        INSERT INTO orders (id, customer_id, total, status) VALUES
            (1, 10, 42.5, 'paid'),
            (2, 11, NULL, 'pending'),
            (3, 12, 99.0, 'paid');
        """
    )


async def _runtime() -> tuple[DbRuntime, SQLitePlugin]:
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    quality = DataQualityPlugin(db=sqlite)
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), sqlite, quality))
    await runtime.setup(agent_id="phase-12-quality-test")
    await _seed(sqlite)
    return runtime, sqlite


async def test_data_quality_registers_domain_service_capabilities():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(DataQualityPlugin(db=sqlite),))

    inspection = await runtime.inspect()

    assert inspection.plugin_ids == ("data_quality",)
    assert "data_quality:quality.profile" in inspection.capability_ids
    assert "data_quality:quality.anomaly.detect" in inspection.capability_ids
    assert "data_quality:quality.freshness.check" in inspection.capability_ids
    assert "data_quality:quality.report.generate" in inspection.capability_ids
    assert "data_quality.profile" in inspection.executor_ids
    assert "data_quality:quality.profile" in inspection.evidence_schema_kinds


def test_db_runtime_can_require_quality_evidence_when_service_registered():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite, DataQualityPlugin(db=sqlite)))

    contract = runtime.build_contract(
        DbRequest(
            "Check data quality for the orders table",
            mode="quality.check",
            requested_capabilities=("quality.profile",),
        )
    )

    assert contract.required_capabilities == ("quality.profile",)
    assert contract.required_evidence == ("quality.profile",)
    assert contract.metadata["missing_capabilities"] == []
    selected = contract.metadata["selected_capabilities"][0]
    assert selected["owner"] == "data_quality"
    assert selected["executor"] == "data_quality.profile"


async def test_db_runtime_executes_quality_check_with_typed_evidence():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            DbRequest(
                "Check data quality for the orders table",
                mode="quality.check",
                requested_capabilities=("quality.profile",),
                metadata={"table": "orders"},
            )
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.diagnostics["verification"]["passed"] is True
    quality = next(item for item in result.evidence if item.kind == "quality.profile")
    assert quality.owner == "data_quality"
    assert quality.payload["success"] is True
    assert quality.payload["table"] == "orders"
    assert quality.payload["columns_profiled"] == 4
    assert quality.payload["profile"]["total"]["null_count"] == 1
