from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.agent_loop import DbAgentLoop
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.db.runtime.tasks import DbTaskSpec
from daita.db.runtime import DbRuntimeGovernanceBlocked
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode, Capability, RiskLevel, TaskStatus
import pytest


class _ProfileOnlyExecutor:
    def __init__(self, owner: str) -> None:
        self.id = f"{owner}.column_values.profile"
        self.owner = owner
        self.capability_ids = frozenset({"db.column_values.profile"})

    async def execute(self, task, operation, context):
        raise AssertionError("ambiguous source-owned profile task should not execute")


class _ProfileOnlySourcePlugin(RuntimeExtensionPlugin):
    def __init__(self, owner: str) -> None:
        self.manifest = PluginManifest(
            id=owner,
            display_name=owner,
            version="1.0.0",
            kind=PluginKind.RUNTIME_EXTENSION,
            domains=frozenset({"db"}),
        )

    def declare_capabilities(self):
        owner = self.manifest.id
        return (
            Capability(
                id="db.column_values.profile",
                owner=owner,
                description="Profile source column values.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"source.profile", "data.query"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"column_values.profile"}),
                executor=f"{owner}.column_values.profile",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
        )

    def get_executors(self):
        return (_ProfileOnlyExecutor(self.manifest.id),)


async def _compile_column_value_search(
    runtime: DbRuntime,
    operation,
    *,
    target: str,
    query: str,
):
    loop = DbAgentLoop(runtime, object())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "read"},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query.catalog_assisted"},
        actions=(
            DbPlannerAction(
                action_id="search_values",
                kind=DbPlannerActionKind.SEARCH_COLUMN_VALUES,
                input={"owner": "catalog"},
                metadata={"target": target, "query": query},
            ),
        ),
    )
    compilation = loop.compile_actions(decision, state)
    assert compilation.rejected_action_summaries == ()
    operation = await loop._persist_compiled_contract(operation, compilation)
    return operation, compilation.task_specs, compilation.compiled_contract_snapshot


async def _seed(plugin: SQLitePlugin) -> None:
    await plugin.execute_script(
        """
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
        """
    )


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
    assert "sqlite:db.column_values.profile" in inspection.capability_ids
    assert "sqlite.sql.execute_read" in inspection.executor_ids
    assert "sqlite:query.result" in inspection.evidence_schema_kinds
    assert "sqlite:column_values.profile" in inspection.evidence_schema_kinds
    profile_capability = runtime.registry.get_capability(
        "db.column_values.profile",
        owner="sqlite",
    )
    assert profile_capability.metadata["profile_policy"]["bounded_aggregate"] is True
    assert (
        profile_capability.metadata["profile_policy"]["fingerprint_only_supported"]
        is True
    )


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
    query_result = next(item for item in result if item.kind == "query.result")
    assert query_result.owner == "sqlite"
    assert query_result.payload["rows"] == [{"order_count": 1}]
    assert query_result.payload["sql"].endswith("LIMIT 10")


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
    query_result = next(item for item in query if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"email": "ada@example.com"}]


async def test_catalog_relationship_task_prepares_source_from_schema_evidence():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
    )
    await runtime.setup(agent_id="db-runtime-test")
    await _seed(sqlite)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Join orders to customers using their relationship",
            },
            evaluate_governance=False,
        )
        schema_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.schema.inspect",
                    owner="sqlite",
                    reason="test:schema",
                ),
            ),
        )
        await runtime.execute_task(schema_plan.tasks[0], operation)

        context_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.planning.context.build",
                    owner="db_runtime",
                    input={},
                    reason="test:planning_context",
                ),
            ),
        )
        context = await runtime.execute_task(context_plan.tasks[0], operation)

        relationship_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="catalog.relationship_paths.find",
                    owner="catalog",
                    input={},
                    metadata={"from": "orders", "to": "customers"},
                    reason="test:relationship",
                ),
            ),
        )
        relationship = await runtime.execute_task(
            relationship_plan.tasks[0],
            operation,
        )
    finally:
        await runtime.teardown()

    registered = [
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "catalog.source_registered"
    ]
    hydrated_task = await runtime.store.load_task(relationship_plan.tasks[0].id)

    assert registered[-1].payload["store_id"] == "store:sqlite"
    assert {table["name"] for table in context[0].payload["schema"]["tables"]} == {
        "customers",
        "orders",
    }
    assert relationship[0].kind == "schema.relationship_path"
    assert relationship[0].payload["reachable"] is True
    assert hydrated_task.input["store_id"] == "store:sqlite"
    assert hydrated_task.input["from_assets"] == ["orders"]
    assert hydrated_task.input["to_assets"] == ["customers"]


async def test_catalog_column_value_search_profiles_dotted_target_before_search():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
    )
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status)
        VALUES (1, 'complete'), (2, 'complete'), (3, 'pending');
        """
    )

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Search observed status values for completed orders",
            },
            evaluate_governance=False,
        )
        operation, search_specs, contract = await _compile_column_value_search(
            runtime,
            operation,
            target="orders.status",
            query="complete orders",
        )
        search_plan = await runtime.plan_task_specs(
            operation,
            search_specs,
            contract=contract,
        )
        search = await runtime.execute_task(search_plan.tasks[0], operation)
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
        hydrated_task = await runtime.store.load_task(search_plan.tasks[0].id)
    finally:
        await runtime.teardown()

    prerequisite_tasks = {
        task.capability_id: task
        for task in tasks
        if task.metadata.get("prerequisite_for") == "catalog.column_values.search"
    }
    kinds = {item.kind for item in evidence}
    assert {
        "schema.asset_profile",
        "catalog.source_registered",
        "column_values.profile",
        "schema.column_value_profile",
        "schema.column_value_search_result",
    } <= kinds
    assert hydrated_task.input["store_id"] == "store:sqlite"
    assert hydrated_task.input["tables"] == ["orders"]
    assert hydrated_task.input["columns"] == ["status"]
    assert {
        "db.schema.inspect",
        "catalog.source.register",
        "db.column_values.profile",
        "catalog.column_values.register",
    } <= set(prerequisite_tasks)
    for capability_id, task in prerequisite_tasks.items():
        assert task.metadata["declared_by_contract"] is True
        assert task.metadata["prerequisite_for"] == "catalog.column_values.search"
        assert task.metadata["prerequisite_reason"] == "catalog_column_value_grounding"
        assert task.metadata["contract_prerequisite"]["capability_id"] == capability_id
    assert search[0].kind == "schema.column_value_search_result"
    assert search[0].payload["profiles"][0]["profile_ref"] == "orders.status"
    assert search[0].payload["profiles"][0]["top_values"][0]["value"] == "complete"


async def test_catalog_column_value_search_profiles_dotted_input_target_before_search():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
    )
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status)
        VALUES (1, 'complete'), (2, 'complete'), (3, 'pending');
        """
    )

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Search observed status values for completed orders",
            },
            evaluate_governance=False,
        )
        operation, _search_specs, contract = await _compile_column_value_search(
            runtime,
            operation,
            target="orders.status",
            query="complete orders",
        )
        search_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="catalog.column_values.search",
                    owner="catalog",
                    input={"target": "orders.status", "query": "complete orders"},
                    metadata={},
                    reason="test:column_values_search",
                ),
            ),
            contract=contract,
        )
        search = await runtime.execute_task(search_plan.tasks[0], operation)
        evidence = await runtime.store.list_evidence(operation.id)
        hydrated_task = await runtime.store.load_task(search_plan.tasks[0].id)
    finally:
        await runtime.teardown()

    kinds = {item.kind for item in evidence}
    assert "column_values.profile" in kinds
    assert "schema.column_value_profile" in kinds
    assert hydrated_task.input["tables"] == ["orders"]
    assert hydrated_task.input["columns"] == ["status"]
    assert "target" not in hydrated_task.input
    assert search[0].kind == "schema.column_value_search_result"
    assert search[0].payload["profiles"][0]["profile_ref"] == "orders.status"
    assert search[0].payload["profiles"][0]["top_values"][0]["value"] == "complete"


async def test_catalog_column_value_profile_prerequisite_requires_contract_declaration():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
    )
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status)
        VALUES (1, 'complete'), (2, 'pending');
        """
    )

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Search observed status values for completed orders",
            },
            evaluate_governance=False,
        )
        search_plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="catalog.column_values.search",
                    owner="catalog",
                    input={},
                    metadata={"target": "orders.status", "query": "complete orders"},
                    reason="test:column_values_search",
                ),
            ),
        )
        with pytest.raises(
            RuntimeError,
            match="undeclared_runtime_prerequisite:db\\.column_values\\.profile",
        ):
            await runtime.execute_task(search_plan.tasks[0], operation)
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert "column_values.profile" not in {item.kind for item in evidence}
    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)
    blocked_search = next(
        task for task in tasks if task.capability_id == "catalog.column_values.search"
    )
    assert blocked_search.status is TaskStatus.BLOCKED
    assert (
        blocked_search.metadata["runtime_prerequisite_blocked"]["error"]
        == "undeclared_runtime_prerequisite:db.column_values.profile"
    )


async def test_catalog_column_value_profile_ambiguous_source_owner_blocks_prepare():
    catalog = CatalogPlugin(auto_persist=False)
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:test",
                }
            },
        ),
        plugins=(
            catalog,
            _ProfileOnlySourcePlugin("source_alpha"),
            _ProfileOnlySourcePlugin("source_beta"),
        ),
    )
    await runtime.setup(agent_id="db-runtime-test")

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Search observed status values for completed orders",
            },
            evaluate_governance=False,
        )
        operation, search_specs, contract = await _compile_column_value_search(
            runtime,
            operation,
            target="orders.status",
            query="complete orders",
        )
        search_plan = await runtime.plan_task_specs(
            operation,
            search_specs,
            contract=contract,
        )
        with pytest.raises(
            RuntimeError,
            match="ambiguous_source_owner:db\\.column_values\\.profile",
        ):
            await runtime.execute_task(search_plan.tasks[0], operation)
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert "column_values.profile" not in {item.kind for item in evidence}
    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)
    blocked_search = next(
        task for task in tasks if task.capability_id == "catalog.column_values.search"
    )
    diagnostic = blocked_search.metadata["runtime_prerequisite_blocked"]
    assert blocked_search.status is TaskStatus.BLOCKED
    assert diagnostic["error"] == "ambiguous_source_owner"
    assert diagnostic["capability_id"] == "db.column_values.profile"
    assert diagnostic["candidate_owners"] == ["source_alpha", "source_beta"]


async def test_sqlite_column_value_profile_registers_with_catalog():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(catalog, sqlite))
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status)
        VALUES ('complete'), ('complete'), ('pending');
        """
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
                "store_id": "store:sqlite",
                "persist": False,
            },
        )
        raw_profile = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={"table": "orders", "column": "status", "max_values": 25},
        )
        registered = await runtime.execute_capability(
            "catalog.column_values.register",
            owner="catalog",
            operation_type="source.profile",
            input={
                "store_id": "store:sqlite",
                "profiles": [raw_profile[0].payload],
                "source_evidence_id": raw_profile[0].id,
            },
        )
    finally:
        await runtime.teardown()

    assert raw_profile[0].kind == "column_values.profile"
    assert raw_profile[0].payload["top_values"] == [
        {"value": "complete", "count": 2},
        {"value": "pending", "count": 1},
    ]
    assert raw_profile[0].payload["source_fingerprint"]
    assert raw_profile[0].payload["policy"]["policy_owner"] == "sqlite"
    assert raw_profile[0].payload["policy"]["bounded_aggregate"] is True
    assert "max_profile_rows" in raw_profile[0].payload["policy"]["eligibility_checks"]
    assert registered[0].kind == "schema.column_value_profile"
    stored = catalog.get_schema("store:sqlite").metadata["column_value_profiles"]
    assert stored["orders.status"]["distinct_count"] == 2
    assert (
        stored["orders.status"]["source_fingerprint"]
        == raw_profile[0].payload["source_fingerprint"]
    )


async def test_sqlite_column_value_profile_fingerprint_only_uses_live_revision():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status) VALUES ('complete'), ('pending');
        """
    )

    try:
        fingerprint = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "orders",
                "column": "status",
                "fingerprint_only": True,
            },
        )
    finally:
        await runtime.teardown()

    assert fingerprint[0].kind == "column_values.profile"
    assert fingerprint[0].payload["profile_status"] == "fingerprint"
    assert fingerprint[0].payload["profile_kind"] == "source_fingerprint"
    assert fingerprint[0].payload["source_fingerprint"]
    assert "data_version:" in fingerprint[0].payload["source_revision"]
    assert fingerprint[0].payload["source_fingerprint_status"] == "best_effort"
    assert (
        fingerprint[0].payload["source_fingerprint_reason"]
        == "sqlite_in_memory_data_version"
    )
    assert fingerprint[0].payload["top_values"] == []


async def test_sqlite_column_value_profile_fingerprint_only_respects_blocked_table():
    sqlite = SQLitePlugin(path=":memory:", blocked_tables=["orders"])
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup(agent_id="db-runtime-test")

    async def fail_query(sql, params=None):
        raise AssertionError("blocked table should not be queried")

    sqlite.query = fail_query
    try:
        full_profile = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={"table": "orders", "column": "status"},
        )
        fingerprint = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "orders",
                "column": "status",
                "fingerprint_only": True,
            },
        )
    finally:
        await runtime.teardown()

    for evidence in (full_profile[0], fingerprint[0]):
        assert evidence.payload["profile_status"] == "skipped"
        assert evidence.payload["skipped_reason"] == "blocked_table"
        assert evidence.payload["top_values"] == []
        assert "source_revision" not in evidence.payload
        assert "source_fingerprint" not in evidence.payload
        assert "row_count" not in evidence.payload


async def test_sqlite_column_value_profile_fingerprint_only_respects_sensitive_column():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup(agent_id="db-runtime-test")

    async def fail_query(sql, params=None):
        raise AssertionError("sensitive column should not be queried")

    sqlite.query = fail_query
    try:
        full_profile = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={"table": "customers", "column": "email"},
        )
        fingerprint = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "customers",
                "column": "email",
                "fingerprint_only": True,
            },
        )
    finally:
        await runtime.teardown()

    for evidence in (full_profile[0], fingerprint[0]):
        assert evidence.payload["profile_status"] == "skipped"
        assert evidence.payload["redacted"] is True
        assert evidence.payload["skipped_reason"] == "sensitive_or_blocked_column"
        assert evidence.payload["top_values"] == []
        assert "source_revision" not in evidence.payload
        assert "source_fingerprint" not in evidence.payload
        assert "row_count" not in evidence.payload


async def test_sqlite_column_value_profile_skips_when_row_count_exceeds_limit():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status)
        VALUES ('complete'), ('complete'), ('pending');
        """
    )

    try:
        raw_profile = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "orders",
                "column": "status",
                "max_profile_rows": 2,
            },
        )
    finally:
        await runtime.teardown()

    assert raw_profile[0].payload["profile_status"] == "skipped"
    assert raw_profile[0].payload["skipped_reason"] == "row_count_exceeds_profile_limit"
    assert raw_profile[0].payload["row_count"] == 3
    assert raw_profile[0].payload["top_values"] == []


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
        try:
            await runtime.execute_capability(
                "db.sql.execute_write",
                owner="sqlite",
                operation_type="write.execute",
                input={
                    "sql": "UPDATE orders SET total = ? WHERE id = ?",
                    "params": [50, 10],
                },
            )
        except DbRuntimeGovernanceBlocked as exc:
            snapshot = await runtime.inspect_operation(exc.operation.id)
            await runtime.approval_channel.approve(
                snapshot.approval_requests[0].approval_id
            )
            resumed = await runtime.resume_operation(exc.operation.id)
            write = tuple(
                item for item in resumed.evidence if item.kind == "write.execution"
            )
    finally:
        await runtime.teardown()

    assert plan[0].kind == "sql.explain.plan"
    assert plan[0].payload["plan"]
    assert write[0].kind == "write.execution"
    assert write[0].payload["affected_rows"] == 1
