from daita.db import DbRequest, DbRuntime
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import OperationStatus


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script(
        """
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
        """
    )


async def _runtime() -> tuple[DbRuntime, SQLitePlugin]:
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), sqlite))
    await runtime.setup(agent_id="phase-8-test")
    await _seed(sqlite)
    return runtime, sqlite


def _schema_with_legacy_status(schema: dict) -> dict:
    return {
        **schema,
        "tables": [
            {
                **table,
                "columns": [
                    *table.get("columns", []),
                    *(
                        [{"name": "legacy_status", "data_type": "TEXT"}]
                        if table.get("name") == "customers"
                        else []
                    ),
                ],
            }
            for table in schema.get("tables", [])
        ],
    }


async def test_run_executes_schema_query_with_typed_evidence():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            DbRequest("What columns are in the customers table?", mode="schema.query")
        )
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.BLOCKED
    assert {item.kind for item in result.evidence} >= {
        "schema.asset_profile",
        "verification.result",
    }
    assert "schema.search_result" not in {item.kind for item in result.evidence}
    assert result.diagnostics["verification"]["passed"] is False
    assert result.warnings == ("missing_evidence:schema.search_result",)
    assert result.answer == "Schema inspection completed."


async def test_run_executes_database_wide_schema_query_when_prompt_is_broad():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(DbRequest("What tables exist?", mode="schema.query"))
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.BLOCKED
    assert "schema.asset_profile" in {item.kind for item in result.evidence}
    assert "schema.search_result" not in {item.kind for item in result.evidence}
    assert result.answer == "Schema inspection completed."


async def test_run_executes_simple_count_query_with_validation_and_result_evidence():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    assert result.answer == "The count is 2."
    assert {"sql.validation", "query.result"} <= {item.kind for item in result.evidence}
    query_result = next(item for item in result.evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"count": 2}]
    assert (
        "SELECT COUNT(*) AS count FROM"
        in result.diagnostics["execution"]["planned_sql"]
    )
    assert not [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
    ]


async def test_run_simple_count_query_has_no_eager_value_profile_tasks():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    reasons = {task.metadata.get("reason") for task in snapshot.tasks}
    assert "catalog_column_value_profile_search" not in reasons
    assert "minimal_fallback_read_prepare" in reasons
    assert not [reason for reason in reasons if reason and "eager" in reason]
    assert not [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
    ]
    assert not [item for item in snapshot.evidence if item.kind == "planning.context"]
    assert {
        "query.plan.proposal",
        "query.plan.validation",
        "sql.validation",
    } <= {item.kind for item in snapshot.evidence}


async def test_run_repeated_simple_count_queries_do_not_profile_values():
    runtime, _ = await _runtime()

    try:
        first = await runtime.run("How many orders are there?")
        second = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(second.operation_id)
    finally:
        await runtime.teardown()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    assert not [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
    ]
    assert not [
        task
        for task in snapshot.tasks
        if task.capability_id == "catalog.column_values.search"
    ]


async def test_run_repeated_simple_count_does_not_register_catalog_sources():
    runtime, _ = await _runtime()

    try:
        first = await runtime.run("How many orders are there?")
        first_snapshot = await runtime.inspect_operation(first.operation_id)
        second = await runtime.run("How many orders are there?")
        second_snapshot = await runtime.inspect_operation(second.operation_id)
    finally:
        await runtime.teardown()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert first_snapshot is not None
    assert second_snapshot is not None
    assert not [
        task
        for task in first_snapshot.tasks
        if task.capability_id == "catalog.source.register"
    ]
    assert not [
        task
        for task in second_snapshot.tasks
        if task.capability_id == "catalog.source.register"
    ]
    assert "catalog.source_registered" not in {
        item.kind for item in second_snapshot.evidence
    }


async def test_run_simple_count_query_does_not_refresh_stale_value_profiles():
    runtime, _ = await _runtime()
    catalog = runtime.registry.get_plugin("catalog")

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        stale_schema = _schema_with_legacy_status(schema_evidence[0].payload)
        await catalog.register_schema(
            stale_schema,
            store_type="sqlite",
            store_id="runtime_source",
            persist=False,
        )
        await catalog.register_column_value_profiles(
            "runtime_source",
            [
                {
                    "table": "customers",
                    "column": "status",
                    "distinct_count": 2,
                    "source_fingerprint": "stale-fingerprint",
                    "top_values": [
                        {"value": "active", "count": 1},
                        {"value": "inactive", "count": 1},
                    ],
                }
            ],
        )

        result = await runtime.run("How many orders are there?")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    reasons = {task.metadata.get("reason") for task in snapshot.tasks}
    assert "column_value_source_fingerprint_check" not in reasons
    assert "stale_column_value_profile_refresh" not in reasons
    assert not [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
    ]
    stored = catalog.get_schema("runtime_source").metadata["column_value_profiles"]
    assert stored["customers.status"]["source_fingerprint"] == "stale-fingerprint"


async def test_run_filtered_data_query_requires_model_planner():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run("List orders where total > 40")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.BLOCKED
    assert result.answer == "A model planner is required for this DB read request."
    assert "query.result" not in {item.kind for item in result.evidence}
    assert snapshot is not None
    assert [task.capability_id for task in snapshot.tasks] == ["db.schema.inspect"]


async def test_run_catalog_join_query_requires_model_planner():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            "Join orders to customers using their relationship and return records",
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.BLOCKED
    assert result.answer == "A model planner is required for this DB read request."
    assert {"schema.relationship_path", "query.result"}.isdisjoint(
        {item.kind for item in result.evidence}
    )
    assert snapshot is not None
    assert [task.capability_id for task in snapshot.tasks] == ["db.schema.inspect"]


async def test_run_relationship_inspection_requires_model_planner_for_catalog_paths():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            "What relationships do I need to join customers to orders?",
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.BLOCKED
    assert result.operation_type == "schema.query"
    assert "schema.asset_profile" in {item.kind for item in result.evidence}
    assert "schema.search_result" not in {item.kind for item in result.evidence}
    assert "schema.relationship_path" not in {item.kind for item in result.evidence}
    assert "query.result" not in {item.kind for item in result.evidence}
    assert "sql.validation" not in {item.kind for item in result.evidence}
    assert snapshot is not None
    assert "db.sql.validate" not in {task.capability_id for task in snapshot.tasks}
    assert "db.sql.execute_read" not in {task.capability_id for task in snapshot.tasks}
