import asyncio
import json

from daita.core.exceptions import ValidationError
from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.loop.legacy import DbLegacyAgentLoop as DbAgentLoop
from daita.db.llm_service import DbLLMResponse
from daita.db.plan_validation import DbQueryPlanValidator
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.db.query_plan import DbQueryPlan
from daita.db.runtime.tasks.models import DbTaskSpec
from daita.db.runtime import DbRuntimeGovernanceBlocked
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode, Capability, Evidence, RiskLevel, TaskStatus
import pytest

from tests.db_evidence_helpers import assert_no_invalid_accepted_query_plans


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


def _runtime_with_ready_catalog(
    sqlite: SQLitePlugin,
    *,
    catalog: CatalogPlugin | None = None,
) -> DbRuntime:
    catalog = catalog or CatalogPlugin(auto_persist=False)

    async def prepare(runtime, *, source_owner, store_id, profile_key, **_kwargs):
        catalog._runtime_source_binding = {
            "runtime": runtime,
            "source_owner": source_owner,
            "store_id": store_id,
            "profile_key": profile_key,
        }
        catalog._runtime_source_state = {"status": "ready", "freshness": "fresh"}
        return dict(catalog._runtime_source_state)

    catalog.prepare_runtime_source = prepare
    return DbRuntime(
        config=DbRuntimeConfig(
            plugins=(catalog, sqlite),
            metadata={
                "from_db_options": {
                    "catalog_store_id": "sqlite-vertical:test",
                    "catalog_profile_key": "sqlite-vertical:test",
                    "source_options": {"include_sample_values": False},
                }
            },
        )
    )


class _SequentialLLMService:
    available = True
    safe_metadata = {"provider": "test", "model": "sequential"}

    def __init__(self, *payloads: dict) -> None:
        self.payloads = [json.dumps(payload) for payload in payloads]
        self.messages = []

    async def generate_json(self, messages):
        self.messages.append(messages)
        return DbLLMResponse(
            content=self.payloads.pop(0),
            diagnostics={"provider": "test", "model": "sequential"},
        )


class _ValidationRepairPlanner:
    def __init__(self) -> None:
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        plans = [
            item
            for item in state.accepted_evidence_summaries
            if item.get("kind") == "query.plan.proposal"
        ]
        has_context = any(
            item.get("kind") == "planning.context"
            for item in state.accepted_evidence_summaries
        )
        has_value_hint = any(
            item.get("kind") == "schema.column_value_hint" and item.get("hints")
            for item in state.accepted_evidence_summaries
        )
        failed_value_validation = next(
            (
                item
                for item in state.validation_summaries
                if item.get("kind") == "query.plan.validation"
                and item.get("valid") is False
                and item.get("validation_facts")
            ),
            None,
        )
        if not has_context:
            return _planner_decision(
                DbPlannerAction(
                    action_id="context_initial",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={"source_owner": "sqlite"},
                )
            )
        if failed_value_validation is not None and not has_value_hint:
            return _planner_decision(
                DbPlannerAction(
                    action_id="context_validation_repair",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={"source_owner": "sqlite"},
                )
            )
        if failed_value_validation is not None and has_value_hint and len(plans) < 2:
            return _planner_decision(
                DbPlannerAction(
                    action_id="repair_plan",
                    kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
                    input={"owner": "db_runtime"},
                    depends_on=("a3",),
                )
            )
        if not plans:
            return _planner_decision(
                DbPlannerAction(
                    action_id="plan_initial",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={"owner": "db_runtime"},
                )
            )
        return _planner_decision(
            DbPlannerAction(
                action_id=f"execute_plan_{len(plans)}",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "sqlite",
                    "query_plan_ref": "latest_accepted_query_plan",
                },
            )
        )


class _ValidationRepairSameTurnExecutePlanner(_ValidationRepairPlanner):
    async def plan(self, state):
        self.states.append(state)
        plans = [
            item
            for item in state.accepted_evidence_summaries
            if item.get("kind") == "query.plan.proposal"
        ]
        has_context = any(
            item.get("kind") == "planning.context"
            for item in state.accepted_evidence_summaries
        )
        has_value_hint = any(
            item.get("kind") == "schema.column_value_hint" and item.get("hints")
            for item in state.accepted_evidence_summaries
        )
        failed_value_validation = next(
            (
                item
                for item in state.validation_summaries
                if item.get("kind") == "query.plan.validation"
                and item.get("valid") is False
                and item.get("validation_facts")
            ),
            None,
        )
        if not has_context:
            return _planner_decision(
                DbPlannerAction(
                    action_id="context_initial",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={"source_owner": "sqlite"},
                )
            )
        if failed_value_validation is not None and not has_value_hint:
            return _planner_decision(
                DbPlannerAction(
                    action_id="context_validation_repair",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={"source_owner": "sqlite"},
                )
            )
        if failed_value_validation is not None and has_value_hint and len(plans) < 2:
            return _planner_decision(
                DbPlannerAction(
                    action_id="repair_plan",
                    kind=DbPlannerActionKind.REPAIR_QUERY_PLAN,
                    input={"owner": "db_runtime"},
                    depends_on=("failed_validation",),
                ),
                DbPlannerAction(
                    action_id="execute_same_turn_repair",
                    kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                    input={
                        "owner": "sqlite",
                        "query_plan_ref": "latest_accepted_query_plan",
                    },
                    depends_on=("repair_plan",),
                ),
            )
        if not plans:
            return _planner_decision(
                DbPlannerAction(
                    action_id="plan_initial",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={"owner": "db_runtime"},
                )
            )
        return _planner_decision(
            DbPlannerAction(
                action_id=f"execute_plan_{len(plans)}",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "sqlite",
                    "query_plan_ref": "latest_accepted_query_plan",
                },
            )
        )


class _ProposeThenExecutePlanner:
    def __init__(self, *, initial_input: dict | None = None) -> None:
        self.initial_input = dict(initial_input or {})
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        has_result = any(
            item.get("kind") == "query.result"
            for item in state.accepted_evidence_summaries
        )
        if has_result:
            return DbPlannerDecision(
                status=DbPlannerDecisionStatus.FINISH,
                intent={"operation_type": "data.query"},
                actions=(),
            )
        plans = [
            item
            for item in state.accepted_evidence_summaries
            if item.get("kind") == "query.plan.proposal"
        ]
        if not plans:
            return _planner_decision(
                DbPlannerAction(
                    action_id="plan_initial",
                    kind=DbPlannerActionKind.PROPOSE_SQL_READ,
                    input={"owner": "db_runtime", **self.initial_input},
                )
            )
        return _planner_decision(
            DbPlannerAction(
                action_id=f"execute_plan_{len(plans)}",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "sqlite",
                    "query_plan_ref": "latest_accepted_query_plan",
                },
            )
        )


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


def _planner_decision(*actions):
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=actions,
    )


def test_query_plan_validation_requires_grounding_for_unprofiled_filter_literal():
    plan = DbQueryPlan.from_mapping(
        {
            "operation": "read",
            "selected_sql": (
                "SELECT COUNT(*) AS order_count FROM orders "
                "WHERE orders.status = 'completed'"
            ),
            "selected_tables": ["orders"],
            "confidence": 0.91,
        }
    )
    context = {
        "dialect": "sqlite",
        "schema": {
            "database_type": "sqlite",
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "status", "data_type": "TEXT"},
                    ],
                }
            ],
        },
        "column_value_hints": [],
    }

    validation = DbQueryPlanValidator().validate(plan, context)

    assert validation.valid is False
    assert validation.accepted_sql is None
    assert validation.errors == (
        "filter_literal_requires_grounding:orders.status=completed",
    )
    assert validation.validation_facts == (
        {
            "kind": "filter_literal_requires_grounding",
            "table": "orders",
            "column": "status",
            "operator": "=",
            "literal": "completed",
            "source": "query.plan.validation",
            "reason": ("proposed_sql_filter_literal_without_accepted_value_evidence"),
        },
    )


def test_query_plan_validation_does_not_ground_numeric_filter_literal():
    plan = DbQueryPlan.from_mapping(
        {
            "operation": "read",
            "selected_sql": (
                "SELECT COUNT(*) AS order_count FROM orders " "WHERE orders.total = 120"
            ),
            "selected_tables": ["orders"],
            "confidence": 0.91,
        }
    )
    context = {
        "dialect": "sqlite",
        "schema": {
            "database_type": "sqlite",
            "tables": [
                {
                    "name": "orders",
                    "columns": [
                        {"name": "id", "data_type": "INTEGER"},
                        {"name": "total", "data_type": "REAL"},
                    ],
                }
            ],
        },
        "column_value_hints": [],
    }

    validation = DbQueryPlanValidator().validate(plan, context)

    assert validation.valid is True
    assert validation.validation_facts == ()


async def test_blocked_column_sql_validation_is_terminal_blocked():
    class _BlockedColumnPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def plan(self, state):
            self.calls += 1
            return _planner_decision(
                DbPlannerAction(
                    action_id=f"blocked_read_{self.calls}",
                    kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                    input={
                        "owner": "sqlite",
                        "sql": "SELECT SUM(refunds.amount) AS total FROM refunds",
                    },
                )
            )

    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:", blocked_columns=["refunds.amount"])
    runtime = _runtime_with_ready_catalog(sqlite, catalog=catalog)
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE refunds (
            id INTEGER PRIMARY KEY,
            amount REAL NOT NULL
        );
        INSERT INTO refunds (id, amount) VALUES (1, 35.0);
        """)
    planner = _BlockedColumnPlanner()

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Calculate refunds amount.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        result = await DbAgentLoop(runtime, planner).run(
            operation,
            safety_frame={"max_access": "read"},
            max_turns=4,
        )
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert result.status == "blocked"
    assert planner.calls == 1
    assert "db_agent_loop_blocked_resource_validation" in result.warnings
    assert [task.capability_id for task in tasks].count("db.sql.validate") == 1
    assert not any(
        task.capability_id == "db.sql.execute_read"
        and task.status is TaskStatus.SUCCEEDED
        for task in tasks
    )


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
    assert "sqlite:db.column_values.profile" in inspection.capability_ids
    assert "sqlite.sql.execute_read" in inspection.executor_ids
    assert "sqlite:query.result" in inspection.evidence_schema_kinds
    assert "sqlite:column_values.profile" in inspection.evidence_schema_kinds
    assert {
        capability.id
        for capability in runtime.registry.capabilities
        if capability.owner == "sqlite" and capability.concurrent_safe
    } == {"db.sql.execute_read"}
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
    runtime = _runtime_with_ready_catalog(sqlite)
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
    runtime = _runtime_with_ready_catalog(sqlite)
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


async def test_sqlite_runtime_supports_two_safe_reads_through_one_plugin():
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    runtime = _runtime_with_ready_catalog(sqlite)
    await runtime.setup()
    await _seed(sqlite)
    original_query = sqlite.query
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0

    async def synchronized_query(sql, params=None):
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await release.wait()
        try:
            return await original_query(sql, params)
        finally:
            active -= 1

    sqlite.query = synchronized_query
    first = asyncio.create_task(
        runtime.execute_capability(
            "db.sql.execute_read",
            owner="sqlite",
            operation_type="data.query",
            input={"sql": "SELECT COUNT(*) AS value FROM customers"},
        )
    )
    second = asyncio.create_task(
        runtime.execute_capability(
            "db.sql.execute_read",
            owner="sqlite",
            operation_type="data.query",
            input={"sql": "SELECT COUNT(*) AS value FROM orders"},
        )
    )
    try:
        await both_started.wait()
        assert active == 2
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        with pytest.raises(ValidationError):
            await runtime.execute_capability(
                "db.sql.execute_read",
                owner="sqlite",
                operation_type="data.query",
                input={"sql": "DELETE FROM orders"},
            )
    finally:
        release.set()
        await runtime.teardown()

    assert next(item for item in first_result if item.kind == "query.result").payload[
        "rows"
    ] == [{"value": 1}]
    assert next(item for item in second_result if item.kind == "query.result").payload[
        "rows"
    ] == [{"value": 1}]


async def test_sqlite_and_catalog_vertical_slice_through_db_runtime():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = _runtime_with_ready_catalog(sqlite, catalog=catalog)
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
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status)
        VALUES (1, 'complete'), (2, 'complete'), (3, 'pending');
        """)
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

    assert_no_invalid_accepted_query_plans(evidence)
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
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status)
        VALUES (1, 'complete'), (2, 'complete'), (3, 'pending');
        """)

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

    assert_no_invalid_accepted_query_plans(evidence)
    kinds = {item.kind for item in evidence}
    assert "column_values.profile" in kinds
    assert "schema.column_value_profile" in kinds
    assert hydrated_task.input["tables"] == ["orders"]
    assert hydrated_task.input["columns"] == ["status"]
    assert "target" not in hydrated_task.input
    assert search[0].kind == "schema.column_value_search_result"
    assert search[0].payload["profiles"][0]["profile_ref"] == "orders.status"
    assert search[0].payload["profiles"][0]["top_values"][0]["value"] == "complete"


async def test_planning_context_general_prompt_does_not_profile_values_without_filter_literal():
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
    await sqlite.execute_script("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            lifecycle_state TEXT NOT NULL,
            customer_segment TEXT NOT NULL,
            revenue REAL NOT NULL
        );
        INSERT INTO customers (id, lifecycle_state, customer_segment, revenue)
        VALUES (1, 'active', 'enterprise', 1200.0), (2, 'active', 'startup', 800.0);
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "summarize customer revenue",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        loop = DbAgentLoop(runtime, object())
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "read"},
            turn=1,
            remaining_turns=1,
        )
        decision = DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=(
                DbPlannerAction(
                    action_id="context",
                    kind=DbPlannerActionKind.BUILD_PLANNING_CONTEXT,
                    input={"source_owner": "sqlite"},
                ),
            ),
        )
        compilation = loop.compile_actions(decision, state)
        operation = await loop._persist_compiled_contract(operation, compilation)
        plan = await runtime.plan_task_specs(
            operation,
            compilation.task_specs,
            contract=compilation.compiled_contract_snapshot,
        )
        for task in plan.tasks:
            await runtime.execute_task(task, operation)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)
    assert not any(
        task.metadata.get("prerequisite_for") == "catalog.column_value_hints.resolve"
        and task.capability_id == "db.column_values.profile"
        for task in tasks
    )


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
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status)
        VALUES (1, 'complete'), (2, 'pending');
        """)

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
            match="undeclared_runtime_prerequisite:catalog\\.value_grounding\\.plan",
        ):
            await runtime.execute_task(search_plan.tasks[0], operation)
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert "column_values.profile" not in {item.kind for item in evidence}
    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)
    blocked_search = next(
        task for task in tasks if task.capability_id == "catalog.column_values.search"
    )
    assert blocked_search.status is TaskStatus.BLOCKED
    assert (
        blocked_search.metadata["runtime_prerequisite_blocked"]["error"]
        == "undeclared_runtime_prerequisite:catalog.value_grounding.plan"
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
        await runtime.store.save_evidence(
            Evidence(
                id="schema-for-ambiguous-profile-owner",
                kind="schema.asset_profile",
                owner="source_alpha",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "database_type": "sqlite",
                    "tables": [
                        {
                            "name": "orders",
                            "columns": [
                                {"name": "id", "data_type": "INTEGER"},
                                {"name": "status", "data_type": "TEXT"},
                            ],
                        }
                    ],
                },
            )
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

    assert_no_invalid_accepted_query_plans(evidence)
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
    runtime = _runtime_with_ready_catalog(sqlite, catalog=catalog)
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status)
        VALUES ('complete'), ('complete'), ('pending');
        """)

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
    runtime = _runtime_with_ready_catalog(sqlite)
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status) VALUES ('complete'), ('pending');
        """)

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
    runtime = _runtime_with_ready_catalog(sqlite)
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
    runtime = _runtime_with_ready_catalog(sqlite)
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


async def test_sqlite_column_value_profile_honors_source_value_policy():
    sqlite = SQLitePlugin(path=":memory:", include_sample_values=False)
    runtime = _runtime_with_ready_catalog(sqlite)
    await runtime.setup(agent_id="db-runtime-test")

    async def fail_query(sql, params=None):
        raise AssertionError("disabled sample values should not be queried")

    sqlite.query = fail_query
    try:
        disabled = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={"table": "orders", "column": "status"},
        )
    finally:
        await runtime.teardown()

    assert disabled[0].payload["profile_status"] == "skipped"
    assert disabled[0].payload["skipped_reason"] == "sample_values_disabled"
    assert disabled[0].payload["policy"]["include_sample_values"] is False

    sqlite = SQLitePlugin(path=":memory:", redact_pii_columns=False)
    runtime = _runtime_with_ready_catalog(sqlite)
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE customers (email TEXT NOT NULL);
        INSERT INTO customers (email) VALUES ('user@example.com');
        """)
    try:
        profiled = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={"table": "customers", "column": "email"},
        )
    finally:
        await runtime.teardown()

    assert profiled[0].payload["profile_status"] == "profiled"
    assert profiled[0].payload["redacted"] is False
    assert profiled[0].payload["top_values"] == [
        {"value": "user@example.com", "count": 1}
    ]
    assert profiled[0].payload["policy"]["redact_pii_columns"] is False


async def test_sqlite_column_value_profile_skips_when_row_count_exceeds_limit():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = _runtime_with_ready_catalog(sqlite)
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status)
        VALUES ('complete'), ('complete'), ('pending');
        """)

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
    runtime = _runtime_with_ready_catalog(sqlite)
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
