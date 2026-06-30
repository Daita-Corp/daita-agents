from daita.db import DbIntentKind, DbRequest, DbRuntime
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

    assert result.status is OperationStatus.SUCCEEDED
    assert {"schema.asset_profile", "schema.search_result"} <= {
        item.kind for item in result.evidence
    }
    assert result.diagnostics["verification"]["passed"] is True
    assert "customers: id, email, status" in result.answer
    assert "orders:" not in result.answer


async def test_run_executes_database_wide_schema_query_when_prompt_is_broad():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(DbRequest("What tables exist?", mode="schema.query"))
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert "Found 2 tables" in result.answer
    assert "customers: id, email, status" in result.answer
    assert "orders: id, customer_id, total" in result.answer


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
    validation = next(item for item in result.evidence if item.kind == "sql.validation")
    assert "SELECT COUNT(*) AS count FROM" in validation.payload["sql"]
    assert result.diagnostics["execution"]["planned_sql"] == validation.payload["sql"]
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
    assert any(str(reason).startswith("planner:") for reason in reasons if reason)
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


async def test_run_repeated_simple_count_reuses_catalog_source_registration():
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
    assert [
        task
        for task in first_snapshot.tasks
        if task.capability_id == "catalog.source.register"
    ]
    assert not [
        task
        for task in second_snapshot.tasks
        if task.capability_id == "catalog.source.register"
    ]
    reused = [
        item
        for item in second_snapshot.evidence
        if item.kind == "catalog.source_registered"
    ]
    assert reused
    assert reused[-1].metadata["catalog_source_cache"] == "hit"
    assert reused[-1].metadata["reused_evidence_id"]


async def test_run_natural_language_count_uses_catalog_value_hints_before_planning():
    runtime, sqlite = await _runtime()
    catalog = runtime.registry.get_plugin("catalog")

    try:
        await sqlite.execute_script("""
            CREATE TABLE support_tickets (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER REFERENCES customers(id),
                status TEXT NOT NULL,
                severity TEXT NOT NULL
            );
            INSERT INTO support_tickets (id, customer_id, status, severity) VALUES
                (100, 1, 'open', 'high'),
                (101, 1, 'open', 'low'),
                (102, 2, 'closed', 'high');
            """)
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await catalog.register_schema(
            schema_evidence[0].payload,
            store_type="sqlite",
            store_id="runtime_source",
            persist=False,
        )
        await catalog.register_column_value_profiles(
            "runtime_source",
            [
                {
                    "table": "support_tickets",
                    "column": "status",
                    "distinct_count": 2,
                    "top_values": [
                        {"value": "open", "count": 2},
                        {"value": "closed", "count": 1},
                    ],
                },
                {
                    "table": "support_tickets",
                    "column": "severity",
                    "distinct_count": 2,
                    "top_values": [
                        {"value": "high", "count": 2},
                        {"value": "low", "count": 1},
                    ],
                },
            ],
        )

        result = await runtime.run("Count open high severity support tickets.")
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.answer == "The count is 1."
    assert snapshot is not None
    validation = next(item for item in result.evidence if item.kind == "sql.validation")
    sql = validation.payload["sql"]
    assert result.diagnostics["execution"]["planned_sql"] == sql
    assert '"support_tickets"' in sql
    assert "\"status\" = 'open'" in sql
    assert "\"severity\" = 'high'" in sql
    assert any(
        task.capability_id == "catalog.column_values.search"
        and task.metadata.get("reason") == "catalog_column_value_profile_search"
        for task in snapshot.tasks
    )


async def test_run_catalog_source_reregisters_when_schema_fingerprint_changes():
    runtime, sqlite = await _runtime()

    try:
        first = await runtime.run("How many orders are there?")
        await sqlite.execute_script("ALTER TABLE orders ADD COLUMN note TEXT")
        refreshed = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        runtime.remember_schema_evidence(refreshed[0])
        second = await runtime.run("How many orders are there?")
        second_snapshot = await runtime.inspect_operation(second.operation_id)
    finally:
        await runtime.teardown()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert second_snapshot is not None
    assert [
        task
        for task in second_snapshot.tasks
        if task.capability_id == "catalog.source.register"
    ]


async def test_run_cache_ttl_zero_keeps_catalog_source_cold_across_operations():
    runtime, _ = await _runtime()
    runtime.config.metadata.setdefault("from_db_options", {})["cache_ttl"] = 0

    try:
        first = await runtime.run("How many orders are there?")
        second = await runtime.run("How many orders are there?")
        second_snapshot = await runtime.inspect_operation(second.operation_id)
    finally:
        await runtime.teardown()

    assert first.status is OperationStatus.SUCCEEDED
    assert second.status is OperationStatus.SUCCEEDED
    assert second_snapshot is not None
    assert [
        task
        for task in second_snapshot.tasks
        if task.capability_id == "db.schema.inspect"
    ]
    assert [
        task
        for task in second_snapshot.tasks
        if task.capability_id == "catalog.source.register"
    ]


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


async def test_run_predicate_query_profiles_only_needed_columns():
    runtime, _ = await _runtime()
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    try:
        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    profile_tasks = [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
    ]
    assert [(task.input["table"], task.input["column"]) for task in profile_tasks] == [
        ("customers", "status")
    ]
    assert profile_tasks[0].metadata.get("reason") == "column_value_predicate_profile"
    assert any(
        task.capability_id == "catalog.column_values.register"
        and task.metadata.get("reason") == "catalog_column_value_predicate_registration"
        for task in snapshot.tasks
    )
    planning_contexts = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ]
    assert len(planning_contexts) == 2
    assert (
        "customers.status: active (1), inactive (1)"
        in planning_contexts[-1].payload["rendered_context"]
    )


async def test_run_predicate_query_reuses_fresh_catalog_profile():
    runtime, _ = await _runtime()
    catalog = runtime.registry.get_plugin("catalog")
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await catalog.register_schema(
            schema_evidence[0].payload,
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
                    "source_fingerprint": "fresh-fingerprint",
                    "top_values": [
                        {"value": "active", "count": 1},
                        {"value": "inactive", "count": 1},
                    ],
                }
            ],
        )

        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    assert not [
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
    ]
    planning_contexts = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ]
    planning_context = planning_contexts[-1]
    assert (
        "customers.status: active (1), inactive (1)"
        in planning_context.payload["rendered_context"]
    )


async def test_run_operation_local_profile_is_fresh_when_cache_ttl_is_zero():
    runtime, _ = await _runtime()
    runtime.config.metadata.setdefault("from_db_options", {})["cache_ttl"] = 0
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    try:
        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    reasons = [task.metadata.get("reason") for task in snapshot.tasks]
    assert reasons.count("column_value_predicate_profile") == 1
    assert "catalog_column_value_predicate_registration" in reasons
    assert "column_value_source_fingerprint_check" not in reasons
    assert "stale_column_value_profile_refresh" not in reasons
    planning_contexts = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ]
    assert len(planning_contexts) == 2
    assert planning_contexts[-1].payload["diagnostics"]["column_value_hint_count"] == 1


async def test_run_refreshes_catalog_profile_when_source_fingerprint_changes():
    runtime, sqlite = await _runtime()
    catalog = runtime.registry.get_plugin("catalog")
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": _schema_with_legacy_status(schema_evidence[0].payload),
                "store_type": "sqlite",
                "store_id": "runtime_source",
                "persist": False,
            },
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

        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    reasons = [task.metadata.get("reason") for task in snapshot.tasks]
    assert "catalog_column_value_profile_search" in reasons
    assert "catalog_column_value_existing_profile_search" not in reasons
    assert "catalog_column_value_source_fingerprint_search" not in reasons
    assert "column_value_source_fingerprint_check" in reasons
    assert "stale_column_value_profile_refresh" in reasons
    assert "stale_catalog_column_value_registration" in reasons
    search_tasks = [
        task
        for task in snapshot.tasks
        if task.capability_id == "catalog.column_values.search"
    ]
    assert len(search_tasks) == 1
    assert search_tasks[0].input == {
        "store_id": "runtime_source",
        "query": "List active customers",
        "limit": 12,
    }
    assert not any(
        task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "column_value_profile"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
        for task in snapshot.tasks
    )
    search_evidence = [
        item
        for item in snapshot.evidence
        if item.kind == "schema.column_value_search_result"
    ]
    assert len(search_evidence) == 1
    planning_context = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ][-1]
    assert (
        search_evidence[0].id in planning_context.payload["column_value_evidence_refs"]
    )
    assert (
        "customers.status: active (1), inactive (1)"
        in planning_context.payload["rendered_context"]
    )

    stored = catalog.get_schema("runtime_source").metadata["column_value_profiles"]
    assert stored["customers.status"]["source_fingerprint"] != "stale-fingerprint"
    assert stored["customers.status"]["source_fingerprint_status"] == "best_effort"
    assert "data_version:" in stored["customers.status"]["source_revision"]


async def test_run_authoritative_changed_fingerprint_triggers_full_refresh(tmp_path):
    sqlite = SQLitePlugin(path=str(tmp_path / "shop.db"), query_default_limit=10)
    runtime = DbRuntime(plugins=(CatalogPlugin(auto_persist=False), sqlite))
    await runtime.setup(agent_id="phase-8-test")
    await _seed(sqlite)
    catalog = runtime.registry.get_plugin("catalog")
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": _schema_with_legacy_status(schema_evidence[0].payload),
                "store_type": "sqlite",
                "store_id": "runtime_source",
                "persist": False,
            },
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

        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    fingerprint_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "column_value_source_fingerprint_check"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
    )
    fingerprint_evidence = next(
        item for item in snapshot.evidence if item.task_id == fingerprint_task.id
    )
    assert fingerprint_evidence.payload["source_fingerprint_status"] == "authoritative"
    assert "file:" in fingerprint_evidence.payload["source_revision"]
    assert any(
        task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "stale_column_value_profile_refresh"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
        for task in snapshot.tasks
    )
    stored = catalog.get_schema("runtime_source").metadata["column_value_profiles"]
    assert stored["customers.status"]["source_fingerprint"] != "stale-fingerprint"
    assert stored["customers.status"]["source_fingerprint_status"] == "authoritative"


async def test_run_best_effort_matching_fingerprint_respects_profile_ttl():
    runtime, _ = await _runtime()
    runtime.config.metadata.setdefault("from_db_options", {})["cache_ttl"] = 0
    catalog = runtime.registry.get_plugin("catalog")
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": schema_evidence[0].payload,
                "store_type": "sqlite",
                "store_id": "runtime_source",
                "persist": False,
            },
        )
        raw_profile = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "customers",
                "column": "status",
                "max_values": 25,
                "include_source_revision": True,
            },
        )
        await catalog.register_column_value_profiles(
            "runtime_source",
            [raw_profile[0].payload],
            source_evidence_id=raw_profile[0].id,
        )

        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    search_evidence = next(
        item
        for item in snapshot.evidence
        if item.kind == "schema.column_value_search_result"
    )
    assert (
        search_evidence.payload["profiles"][0]["stale_reason"] == "profile_ttl_expired"
    )
    fingerprint_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "column_value_source_fingerprint_check"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
    )
    fingerprint_evidence = next(
        item for item in snapshot.evidence if item.task_id == fingerprint_task.id
    )
    assert fingerprint_evidence.payload["source_fingerprint_status"] == "best_effort"
    assert not any(
        task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "stale_column_value_profile_refresh"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
        for task in snapshot.tasks
    )
    assert any(
        task.capability_id == "catalog.column_values.register"
        and task.metadata.get("reason") == "stale_catalog_column_value_registration"
        for task in snapshot.tasks
    )
    planning_contexts = [
        item for item in snapshot.evidence if item.kind == "planning.context"
    ]
    assert len(planning_contexts) == 2
    assert (
        "customers.status: active (1), inactive (1)"
        in planning_contexts[-1].payload["rendered_context"]
    )


async def test_run_unavailable_fingerprint_does_not_preserve_profile_freshness():
    runtime, sqlite = await _runtime()
    catalog = runtime.registry.get_plugin("catalog")
    original_query = sqlite.query
    request = DbRequest(
        "List active customers",
        metadata={"sql": "SELECT * FROM customers WHERE status = 'active'"},
    )

    async def query_with_unavailable_revision(sql, params=None):
        if str(sql).strip().upper().startswith("PRAGMA DATA_VERSION"):
            raise RuntimeError("data version unavailable")
        return await original_query(sql, params)

    try:
        schema_evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": _schema_with_legacy_status(schema_evidence[0].payload),
                "store_type": "sqlite",
                "store_id": "runtime_source",
                "persist": False,
            },
        )
        await catalog.register_column_value_profiles(
            "runtime_source",
            [
                {
                    "table": "customers",
                    "column": "status",
                    "distinct_count": 2,
                    "source_fingerprint": "stored-fingerprint",
                    "source_fingerprint_status": "best_effort",
                    "top_values": [
                        {"value": "active", "count": 1},
                        {"value": "inactive", "count": 1},
                    ],
                }
            ],
        )
        sqlite.query = query_with_unavailable_revision

        result = await runtime.run(request)
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        sqlite.query = original_query
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert snapshot is not None
    fingerprint_task = next(
        task
        for task in snapshot.tasks
        if task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "column_value_source_fingerprint_check"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
    )
    fingerprint_evidence = next(
        item for item in snapshot.evidence if item.task_id == fingerprint_task.id
    )
    assert fingerprint_evidence.payload["source_fingerprint_status"] == "unavailable"
    assert "source_fingerprint" not in fingerprint_evidence.payload
    assert any(
        task.capability_id == "db.column_values.profile"
        and task.metadata.get("reason") == "stale_column_value_profile_refresh"
        and task.input["table"] == "customers"
        and task.input["column"] == "status"
        for task in snapshot.tasks
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
            "Join orders to customers using their relationship and return records",
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


async def test_run_executes_relationship_query_without_sql_execution():
    runtime, _ = await _runtime()

    try:
        result = await runtime.run(
            "What relationships do I need to join customers to orders?",
        )
        snapshot = await runtime.inspect_operation(result.operation_id)
    finally:
        await runtime.teardown()

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.SCHEMA_RELATIONSHIP_QUERY
    assert {
        "schema.asset_profile",
        "schema.search_result",
        "schema.relationship_path",
    } <= {item.kind for item in result.evidence}
    assert "query.result" not in {item.kind for item in result.evidence}
    assert "sql.validation" not in {item.kind for item in result.evidence}
    assert snapshot is not None
    assert "db.sql.validate" not in {task.capability_id for task in snapshot.tasks}
    assert "db.sql.execute_read" not in {task.capability_id for task in snapshot.tasks}
    relationship = next(
        item for item in result.evidence if item.kind == "schema.relationship_path"
    )
    assert relationship.payload["reachable"] is True
