from daita.db import DbRequest, DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL NOT NULL
        );
        INSERT INTO customers (id, email, status) VALUES
            (1, 'ada@example.com', 'active'),
            (2, 'grace@example.com', 'inactive');
        INSERT INTO orders (id, customer_id, total) VALUES
            (10, 1, 42.5),
            (11, 2, 12.0);
        """)


async def _runtime() -> tuple[DbRuntime, SQLitePlugin]:
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), sqlite))
    await runtime.setup(agent_id="phase-8-test")
    await _seed(sqlite)
    return runtime, sqlite


async def test_run_executes_schema_query_with_typed_evidence():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            DbRequest("What columns are in the customers table?", mode="schema.query")
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert {"schema.asset_profile", "schema.search_result"} <= {
        item.kind for item in result.evidence
    }
    assert result.diagnostics["verification"]["passed"] is True
    assert "customers: id, email, status" in result.answer


async def test_run_executes_simple_count_query_with_validation_and_result_evidence():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run("How many orders are there?")
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.answer == "The count is 2."
    assert {"sql.validation", "query.result"} <= {item.kind for item in result.evidence}
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"count": 2}]
    assert (
        "SELECT COUNT(*) AS count FROM"
        in result.diagnostics["execution"]["planned_sql"]
    )


async def test_run_executes_filtered_data_query():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run("List orders where total > 40")
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"id": 10, "customer_id": 1, "total": 42.5}]
    assert 'WHERE "total" > 40' in query_result.payload["sql"]


async def test_run_executes_catalog_assisted_join_query():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            "Join orders to customers using their relationship",
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert {"schema.search_result", "schema.relationship_path", "query.result"} <= {
        item.kind for item in result.evidence
    }
    relationship = next(
        item for item in result.evidence if item.kind == "schema.relationship_path"
    )
    assert relationship.payload["reachable"] is True
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"][0]["customers_email"] == "ada@example.com"
    assert query_result.payload["rows"][0]["orders_total"] == 42.5
