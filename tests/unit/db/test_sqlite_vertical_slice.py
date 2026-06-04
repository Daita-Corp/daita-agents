from daita.db import DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL
        );
        INSERT INTO customers (id, email) VALUES (1, 'ada@example.com');
        INSERT INTO orders (id, customer_id, total) VALUES (10, 1, 42.5);
        """)


async def test_sqlite_registers_provider_neutral_db_capabilities():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))

    inspection = await runtime.inspect()

    assert inspection.plugin_ids == ("sqlite",)
    assert "sqlite:db.schema.inspect" in inspection.capability_ids
    assert "sqlite:db.sql.validate" in inspection.capability_ids
    assert "sqlite:db.sql.execute_read" in inspection.capability_ids
    assert "sqlite:db.sql.execute_write" in inspection.capability_ids
    assert "sqlite:db.sql.explain" in inspection.capability_ids
    assert "sqlite.sql.execute_read" in inspection.executor_ids
    assert "sqlite:query.result" in inspection.evidence_schema_kinds


async def test_sqlite_schema_inspect_returns_typed_evidence_through_runtime():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup()
    await _seed(sqlite)

    try:
        evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
    finally:
        await runtime.teardown()

    assert evidence[0].kind == "schema.asset_profile"
    assert evidence[0].owner == "sqlite"
    assert evidence[0].payload["database_type"] == "sqlite"
    assert evidence[0].payload["table_count"] == 2
    assert {table["name"] for table in evidence[0].payload["tables"]} == {
        "customers",
        "orders",
    }
    assert evidence[0].payload["foreign_keys"][0]["source_table"] == "orders"


async def test_sqlite_read_query_returns_typed_query_result_through_runtime():
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup()
    await _seed(sqlite)

    try:
        validation = await runtime.execute_capability(
            "db.sql.validate",
            owner="sqlite",
            operation_type="data.query",
            input={"sql": "SELECT COUNT(*) AS order_count FROM orders"},
        )
        result = await runtime.execute_capability(
            "db.sql.execute_read",
            owner="sqlite",
            operation_type="data.query",
            input={"sql": "SELECT COUNT(*) AS order_count FROM orders"},
        )
    finally:
        await runtime.teardown()

    assert validation[0].kind == "sql.validation"
    assert validation[0].payload["valid"] is True
    assert validation[0].payload["tables"] == ["orders"]
    assert result[0].kind == "query.result"
    assert result[0].owner == "sqlite"
    assert result[0].payload["rows"] == [{"order_count": 1}]
    assert result[0].payload["sql"].endswith("LIMIT 10")


async def test_sqlite_and_catalog_vertical_slice_through_db_runtime():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(catalog, sqlite))
    await runtime.setup(agent_id="db-runtime-test")
    await _seed(sqlite)

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        registered = await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": schema_evidence[0].payload,
                "store_type": "sqlite",
                "store_id": "store:sqlite",
                "persist": False,
            },
        )
        search = await runtime.execute_capability(
            "catalog.schema.search",
            owner="catalog",
            operation_type="schema.query",
            input={"store_id": "store:sqlite", "query": "customer email"},
        )
        query = await runtime.execute_capability(
            "db.sql.execute_read",
            owner="sqlite",
            operation_type="data.query",
            input={"sql": "SELECT email FROM customers"},
        )
    finally:
        await runtime.teardown()

    assert registered[0].kind == "catalog.source_registered"
    assert registered[0].payload["store_id"] == "store:sqlite"
    assert search[0].kind == "schema.search_result"
    assert search[0].payload["tables"][0]["name"] == "customers"
    assert query[0].kind == "query.result"
    assert query[0].payload["rows"] == [{"email": "ada@example.com"}]


async def test_sqlite_explain_and_write_executors_return_typed_evidence():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup()
    await _seed(sqlite)

    try:
        plan = await runtime.execute_capability(
            "db.sql.explain",
            owner="sqlite",
            operation_type="data.query",
            input={"sql": "SELECT email FROM customers"},
        )
        write = await runtime.execute_capability(
            "db.sql.execute_write",
            owner="sqlite",
            operation_type="write.execute",
            input={
                "sql": "UPDATE orders SET total = ? WHERE id = ?",
                "params": [50, 10],
            },
        )
    finally:
        await runtime.teardown()

    assert plan[0].kind == "query.plan"
    assert plan[0].payload["plan"]
    assert write[0].kind == "write.execution"
    assert write[0].payload["affected_rows"] == 1
