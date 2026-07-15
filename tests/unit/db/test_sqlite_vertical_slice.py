import asyncio
import json

from daita.core.exceptions import ValidationError
from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.fingerprints import persisted_fingerprint
from daita.db.loop import DbAgentLoop
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
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    RiskLevel,
    TaskDependency,
    TaskStatus,
)
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
    runtime = DbRuntime(plugins=(catalog, sqlite))
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


async def test_propose_sql_read_builds_context_prerequisite_and_query_plan_succeeds():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    llm_service = _SequentialLLMService(
        {
            "operation": "read",
            "selected_sql": (
                "SELECT COUNT(*) AS order_count FROM orders "
                "WHERE orders.status = 'complete'"
            ),
            "selected_tables": ["orders"],
            "filters": [
                {
                    "column": "orders.status",
                    "operator": "=",
                    "value": "complete",
                }
            ],
            "confidence": 0.94,
        }
    )
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
        db_llm_service=llm_service,
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
                "prompt": "How many completed orders are there?",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        planner = _ProposeThenExecutePlanner(
            initial_input={
                "source_owner": "sqlite",
                "targets": [{"table": "orders", "column": "status"}],
            }
        )
        result = await DbAgentLoop(runtime, planner).run(
            operation,
            safety_frame={"max_access": "read"},
            max_turns=6,
        )
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert result.status == "finished"
    capability_order = [task.capability_id for task in tasks]
    assert capability_order.index("db.planning.context.build") < (
        capability_order.index("db.query.plan")
    )
    assert {
        "db.schema.inspect",
        "catalog.value_grounding.plan",
        "catalog.column_value_hints.resolve",
        "db.planning.context.build",
        "db.query.plan",
        "db.query.plan.validate",
        "db.sql.validate",
        "db.sql.execute_read",
    } <= set(capability_order)
    query_plan_task = next(
        task for task in tasks if task.capability_id == "db.query.plan"
    )
    context_task = next(
        task for task in tasks if task.capability_id == "db.planning.context.build"
    )
    assert "query_plan_ref" not in query_plan_task.input
    assert "plan_evidence_id" not in query_plan_task.input
    assert "planning_context_evidence_id" not in query_plan_task.input
    assert query_plan_task.dependencies[0].evidence_kind == "planning.context"
    assert query_plan_task.dependencies[0].producer_task_id == context_task.id
    contexts = [item for item in evidence if item.kind == "planning.context"]
    assert contexts
    proposal = next(item for item in evidence if item.kind == "query.plan.proposal")
    assert proposal.accepted is True
    assert proposal.payload["planning_context_evidence_id"] == contexts[-1].id
    assert proposal.payload["sql"].endswith("orders.status = 'complete'")
    query_result = next(item for item in evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"order_count": 1}]
    hint = next(item for item in evidence if item.kind == "schema.column_value_hint")
    assert hint.payload["hints"][0]["table"] == "orders"
    assert hint.payload["hints"][0]["column"] == "status"
    assert hint.payload["hints"][0]["observed_values"][0]["value"] == "complete"
    assert any(task.capability_id == "db.column_values.profile" for task in tasks)


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


async def test_sqlite_schema_inspect_distinguishes_rowid_reuse_from_autoincrement():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup()
    await sqlite.execute_script("""
        CREATE TABLE ordinary_ids (
            id INTEGER PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE monotonic_ids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value TEXT NOT NULL
        );
        """)

    try:
        evidence = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
    finally:
        await runtime.teardown()

    tables = {table["name"]: table for table in evidence[0].payload["tables"]}
    ordinary = tables["ordinary_ids"]["columns"][0]
    monotonic = tables["monotonic_ids"]["columns"][0]
    assert ordinary["is_identity"] is True
    assert ordinary["is_generated"] is True
    assert ordinary["is_autoincrement"] is False
    assert ordinary["is_monotonic"] is False
    assert ordinary["identity_proof"]["rowid_reuse_possible"] is True
    assert monotonic["is_identity"] is True
    assert monotonic["is_generated"] is True
    assert monotonic["is_autoincrement"] is True
    assert monotonic["is_monotonic"] is True
    assert monotonic["identity_proof"]["rowid_reuse_possible"] is False


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


async def test_sqlite_runtime_supports_two_safe_reads_through_one_plugin():
    sqlite = SQLitePlugin(path=":memory:", query_default_limit=10)
    runtime = DbRuntime(plugins=(sqlite,))
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


async def test_planning_context_value_hints_profile_catalog_target_before_context():
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
            status TEXT NOT NULL,
            total REAL NOT NULL
        );
        INSERT INTO orders (id, status, total)
        VALUES (1, 'complete', 120.0), (2, 'pending', 80.0);
        """)
    executed_capabilities = []
    original_execute_task = runtime.kernel.execute_task

    async def execute_task_spy(task_id, **kwargs):
        task = await runtime.store.load_task(task_id)
        executed_capabilities.append(task.capability_id)
        return await original_execute_task(task_id, **kwargs)

    runtime.kernel.execute_task = execute_task_spy

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "What are completed order totals?",
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
                    input={
                        "source_owner": "sqlite",
                        "targets": [{"table": "orders", "column": "status"}],
                    },
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
        for task in plan.tasks[:4]:
            await runtime.execute_task(task, operation)
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.schema.inspect",
        "catalog.schema.search",
        "catalog.value_grounding.plan",
        "catalog.column_value_hints.resolve",
        "db.planning.context.build",
    ]
    assert "catalog.value_grounding.plan" in executed_capabilities
    assert executed_capabilities.index("catalog.value_grounding.plan") < (
        executed_capabilities.index("db.column_values.profile")
    )
    grounding_plan = next(
        item for item in evidence if item.kind == "catalog.value_grounding.plan"
    )
    assert grounding_plan.payload["targets"] == [
        {
            "table": "orders",
            "column": "status",
            "reason": "explicit_target",
            "confidence": 1.0,
            "requires_profile_read": True,
            "source": {"kind": "explicit_target"},
        }
    ]
    hint_task = next(
        task
        for task in tasks
        if task.capability_id == "catalog.column_value_hints.resolve"
    )
    assert hint_task.input["profile_pairs"] == [{"table": "orders", "column": "status"}]
    prerequisite_tasks = {
        task.capability_id: task
        for task in tasks
        if task.metadata.get("prerequisite_for") == "catalog.column_value_hints.resolve"
    }
    assert {
        "db.column_values.profile",
        "catalog.column_values.register",
    } <= set(prerequisite_tasks)
    assert (
        next(item for item in evidence if item.kind == "column_values.profile").payload[
            "top_values"
        ][0]["value"]
        == "complete"
    )
    hint = next(item for item in evidence if item.kind == "schema.column_value_hint")
    assert hint.payload["hints"][0]["table"] == "orders"
    assert hint.payload["hints"][0]["column"] == "status"
    assert hint.payload["hints"][0]["observed_values"][0]["value"] == "complete"


async def test_planning_context_value_grounding_does_not_require_known_prompt_keywords():
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
        CREATE TABLE account_revenue (
            id INTEGER PRIMARY KEY,
            loyalty_band TEXT NOT NULL,
            amount REAL NOT NULL
        );
        INSERT INTO account_revenue (id, loyalty_band, amount)
        VALUES (1, 'platinum', 1200.0), (2, 'gold', 800.0);
        """)

    try:
        schema = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="source.profile",
        )
        await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": schema[0].payload,
                "store_type": "sqlite",
                "store_id": "store:sqlite",
                "persist": False,
            },
        )
        await catalog.register_column_value_profiles(
            "store:sqlite",
            [
                {
                    "table": "account_revenue",
                    "column": "loyalty_band",
                    "distinct_count": 2,
                    "top_values": [
                        {"value": "platinum", "count": 1},
                        {"value": "gold", "count": 1},
                    ],
                }
            ],
        )
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Summarize revenue for platinum accounts.",
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
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    grounding_plan = next(
        item for item in evidence if item.kind == "catalog.value_grounding.plan"
    )
    assert [
        (target["table"], target["column"], target["reason"])
        for target in grounding_plan.payload["targets"]
    ] == [("account_revenue", "loyalty_band", "catalog_profile")]
    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)
    contexts = [item for item in evidence if item.kind == "planning.context"]
    assert contexts
    hints = contexts[-1].payload["column_value_hints"]
    assert ("account_revenue", "loyalty_band", "platinum") in {
        (hint["table"], hint["column"], observed["value"])
        for hint in hints
        for observed in hint.get("observed_values", [])
    }


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


async def test_validation_fact_value_grounding_profiles_only_target_column():
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
            status TEXT NOT NULL,
            channel TEXT NOT NULL
        );
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status, channel)
        VALUES (1, 'complete', 'web'), (2, 'pending', 'store');
        INSERT INTO customers (id, status)
        VALUES (1, 'active'), (2, 'inactive');
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Repair the draft SQL using validation facts.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        await runtime.store.save_evidence(
            Evidence(
                id="validation-unobserved-status",
                kind="sql.validation",
                owner="sqlite",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "valid": False,
                    "sql": (
                        "SELECT COUNT(*) FROM orders "
                        "WHERE orders.status = 'completed'"
                    ),
                    "operation": "query",
                    "warnings": [
                        {
                            "kind": "unobserved_filter_literal",
                            "table": "orders",
                            "column": "status",
                            "literal": "completed",
                            "candidates": ["complete", "pending"],
                        }
                    ],
                    "validation_facts": [
                        {
                            "kind": "unobserved_filter_literal",
                            "table": "orders",
                            "column": "status",
                            "literal": "completed",
                            "candidates": ["complete", "pending"],
                        }
                    ],
                },
            )
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
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert compilation.rejected_action_summaries == ()
    grounding_plan = next(
        item for item in evidence if item.kind == "catalog.value_grounding.plan"
    )
    assert [
        (target["table"], target["column"], target["requires_profile_read"])
        for target in grounding_plan.payload["targets"]
    ] == [("orders", "status", True)]
    profile_tasks = [
        task for task in tasks if task.capability_id == "db.column_values.profile"
    ]
    assert [(task.input["table"], task.input["column"]) for task in profile_tasks] == [
        ("orders", "status")
    ]
    register_tasks = [
        task for task in tasks if task.capability_id == "catalog.column_values.register"
    ]
    assert len(register_tasks) == 1
    hint = next(item for item in evidence if item.kind == "schema.column_value_hint")
    assert [(item["table"], item["column"]) for item in hint.payload["hints"]] == [
        ("orders", "status")
    ]
    assert hint.payload["hints"][0]["observed_values"][0]["value"] == "complete"
    assert hint.payload["hints"][0]["candidate_mapping"]["prompt_term"] == "completed"
    assert hint.payload["hints"][0]["candidate_mapping"]["closest_value"] == "complete"
    contexts = [item for item in evidence if item.kind == "planning.context"]
    assert contexts
    latest_context = contexts[-1]
    assert (
        latest_context.payload["diagnostics"]["validation_grounding_repair_attempted"]
        is True
    )
    assert latest_context.payload["diagnostics"]["validation_grounding_repair"][
        "target_refs"
    ] == ["orders.status"]
    context_hint = latest_context.payload["column_value_hints"][0]
    assert context_hint["table"] == "orders"
    assert context_hint["column"] == "status"
    assert context_hint["candidate_mapping"]["closest_value"] == "complete"


async def test_validation_grounding_runtime_context_precedes_planner_search_choice():
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
            status TEXT NOT NULL,
            channel TEXT NOT NULL
        );
        INSERT INTO orders (id, status, channel)
        VALUES (1, 'complete', 'web'), (2, 'pending', 'store');
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Show completed order count.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        await runtime.store.save_evidence(
            Evidence(
                id="validation-requires-status-grounding",
                kind="query.plan.validation",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload={
                    "valid": False,
                    "errors": [
                        "filter_literal_requires_grounding:" "orders.status=completed"
                    ],
                    "validation_facts": [
                        {
                            "kind": "filter_literal_requires_grounding",
                            "table": "orders",
                            "column": "status",
                            "operator": "=",
                            "literal": "completed",
                            "source": "query.plan.validation",
                            "reason": (
                                "proposed_sql_filter_literal_without_accepted_"
                                "value_evidence"
                            ),
                        }
                    ],
                },
            )
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
                    action_id="planner_search_values",
                    kind=DbPlannerActionKind.SEARCH_COLUMN_VALUES,
                    input={"owner": "catalog"},
                    metadata={
                        "source_owner": "sqlite",
                        "target": "orders.status",
                        "query": "completed",
                    },
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
        evidence = await runtime.store.list_evidence(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert compilation.rejected_action_summaries == ()
    assert compilation.accepted_action_summaries[0]["kind"] == "build_planning_context"
    assert compilation.accepted_action_summaries[1]["kind"] == "search_column_values"
    capability_order = [spec.capability_id for spec in compilation.task_specs]
    assert capability_order.index("db.planning.context.build") < capability_order.index(
        "catalog.column_values.search"
    )
    hint = next(item for item in evidence if item.kind == "schema.column_value_hint")
    assert [(item["table"], item["column"]) for item in hint.payload["hints"]] == [
        ("orders", "status")
    ]
    latest_context = [item for item in evidence if item.kind == "planning.context"][-1]
    assert latest_context.payload["column_value_hints"][0]["table"] == "orders"
    assert latest_context.payload["column_value_hints"][0]["column"] == "status"


async def test_validation_grounding_runtime_context_exhaustion_blocks_loop():
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

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Show completed order count.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        await runtime.store.save_evidence(
            Evidence(
                id="validation-requires-status-grounding",
                kind="query.plan.validation",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=False,
                payload={
                    "valid": False,
                    "errors": [
                        "filter_literal_requires_grounding:" "orders.status=completed"
                    ],
                    "validation_facts": [
                        {
                            "kind": "filter_literal_requires_grounding",
                            "table": "orders",
                            "column": "status",
                            "operator": "=",
                            "literal": "completed",
                            "source": "query.plan.validation",
                            "reason": (
                                "proposed_sql_filter_literal_without_accepted_"
                                "value_evidence"
                            ),
                        }
                    ],
                },
            )
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
                    action_id="planner_search_values",
                    kind=DbPlannerActionKind.SEARCH_COLUMN_VALUES,
                    input={"owner": "catalog"},
                    metadata={"target": "orders.status", "query": "completed"},
                ),
            ),
        )
        first_compilation = loop.compile_actions(decision, state)
        fingerprint = next(
            spec.metadata["validation_grounding_fingerprint"]
            for spec in first_compilation.task_specs
            if spec.capability_id == "db.planning.context.build"
        )
        await runtime.store.save_evidence(
            Evidence(
                id="context-with-exhausted-grounding",
                kind="planning.context",
                owner="db_runtime",
                operation_id=operation.id,
                accepted=True,
                payload={
                    "diagnostics": {
                        "validation_grounding_repair": {
                            "fingerprint": fingerprint,
                            "target_refs": ["orders.status"],
                            "targets": [
                                {
                                    "kind": "filter_literal_requires_grounding",
                                    "table": "orders",
                                    "column": "status",
                                    "literal": "completed",
                                }
                            ],
                        }
                    },
                    "column_value_hints": [],
                },
            )
        )
        state = await loop.build_loop_state(
            operation,
            safety_frame={"max_access": "read"},
            turn=2,
            remaining_turns=1,
        )

        exhausted = loop.compile_actions(decision, state)
    finally:
        await runtime.teardown()

    assert exhausted.task_specs == ()
    assert exhausted.rejected_action_summaries[0]["error"] == (
        "validation_grounding_context_refresh_exhausted"
    )
    assert exhausted.rejected_action_summaries[0]["continuation"][
        "missing_target_refs"
    ] == ["orders.status"]


async def test_validation_grounding_repair_replans_with_refreshed_context():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    llm_service = _SequentialLLMService(
        {
            "operation": "read",
            "selected_sql": (
                "SELECT COUNT(*) AS order_count FROM orders "
                "WHERE orders.status = 'completed'"
            ),
            "selected_tables": ["orders"],
            "filters": [
                {
                    "column": "orders.status",
                    "operator": "=",
                    "value": "completed",
                }
            ],
            "confidence": 0.91,
        },
        {
            "operation": "read",
            "selected_sql": (
                "SELECT COUNT(*) AS order_count FROM orders "
                "WHERE orders.status = 'complete'"
            ),
            "selected_tables": ["orders"],
            "filters": [
                {
                    "column": "orders.status",
                    "operator": "=",
                    "value": "complete",
                }
            ],
            "confidence": 0.93,
        },
    )
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
        db_llm_service=llm_service,
    )
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            channel TEXT NOT NULL
        );
        INSERT INTO orders (id, status, channel)
        VALUES (1, 'complete', 'web'), (2, 'pending', 'store');
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Show the requested order count.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        planner = _ValidationRepairPlanner()
        result = await DbAgentLoop(runtime, planner).run(
            operation,
            safety_frame={"max_access": "read"},
            max_turns=8,
        )
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert result.status == "finished"
    validations = [item for item in evidence if item.kind == "query.plan.validation"]
    assert [item.accepted for item in validations] == [False, True]
    assert validations[0].payload["validation_facts"] == [
        {
            "kind": "filter_literal_requires_grounding",
            "table": "orders",
            "column": "status",
            "operator": "=",
            "literal": "completed",
            "source": "query.plan.validation",
            "reason": ("proposed_sql_filter_literal_without_accepted_value_evidence"),
        }
    ]
    repaired_plan = [item for item in evidence if item.kind == "query.plan.proposal"][
        -1
    ]
    assert "orders.status = 'complete'" in repaired_plan.payload["sql"]
    query_result = next(item for item in evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"order_count": 1}]
    contexts = [item for item in evidence if item.kind == "planning.context"]
    assert validations[0].payload["planning_context_evidence_id"] == contexts[0].id
    assert validations[1].payload["planning_context_evidence_id"] == contexts[-1].id
    assert validations[1].payload["planning_context_evidence_id"] != (
        validations[0].payload["planning_context_evidence_id"]
    )
    repair_task = next(
        task for task in tasks if task.capability_id == "db.query.repair"
    )
    plans = [item for item in evidence if item.kind == "query.plan.proposal"]
    assert repair_task.input["planning_context_evidence_id"] == contexts[-1].id
    assert repair_task.input["prior_plan_evidence_id"] == plans[0].id
    assert repair_task.input["failure_evidence_id"] == validations[0].id
    assert "query_plan_ref" not in repair_task.input
    repair_dependencies = {
        dependency.evidence_kind: dependency for dependency in repair_task.dependencies
    }
    assert repair_dependencies["planning.context"].evidence_id == contexts[-1].id
    assert repair_dependencies["query.plan.proposal"].evidence_id == plans[0].id
    assert repair_dependencies["query.plan.validation"].evidence_id == validations[0].id
    assert repair_dependencies["query.plan.validation"].evidence_accepted is False
    assert not any(
        "missing_dependency:a3" in json.dumps(item.payload, sort_keys=True)
        for item in evidence
        if item.kind in {"planner.compilation", "planner.observation"}
    )
    assert (
        contexts[-1].payload["column_value_hints"][0]["candidate_mapping"][
            "closest_value"
        ]
        == "complete"
    )
    profile_tasks = [
        task for task in tasks if task.capability_id == "db.column_values.profile"
    ]
    assert [(task.input["table"], task.input["column"]) for task in profile_tasks] == [
        ("orders", "status")
    ]
    assert not any(
        task.capability_id == "db.column_values.profile"
        and task.input["column"] == "channel"
        for task in tasks
    )
    register_tasks = [
        task for task in tasks if task.capability_id == "catalog.column_values.register"
    ]
    assert len(register_tasks) == 1
    assert (
        len(
            [
                item
                for item in evidence
                if item.kind == "schema.column_value_hint" and item.payload.get("hints")
            ]
        )
        == 1
    )


async def test_runtime_validation_repair_executes_after_plan_is_durable():
    stale_sql = (
        "SELECT COUNT(*) AS order_count FROM orders "
        "WHERE orders.status = 'completed'"
    )
    repaired_sql = (
        "SELECT COUNT(*) AS order_count FROM orders " "WHERE orders.status = 'complete'"
    )
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    llm_service = _SequentialLLMService(
        {
            "operation": "read",
            "selected_sql": stale_sql,
            "selected_tables": ["orders"],
            "filters": [
                {
                    "column": "orders.status",
                    "operator": "=",
                    "value": "completed",
                }
            ],
            "confidence": 0.91,
        },
        {
            "operation": "read",
            "selected_sql": repaired_sql,
            "selected_tables": ["orders"],
            "filters": [
                {
                    "column": "orders.status",
                    "operator": "=",
                    "value": "complete",
                }
            ],
            "confidence": 0.93,
        },
    )
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            metadata={
                "from_db_options": {
                    "catalog_store_id": "store:sqlite",
                }
            },
        ),
        plugins=(catalog, sqlite),
        db_llm_service=llm_service,
    )
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            channel TEXT NOT NULL
        );
        INSERT INTO orders (id, status, channel)
        VALUES (1, 'complete', 'web'), (2, 'pending', 'store');
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Show the requested order count.",
                "source_scope": ["sqlite"],
            },
            metadata={"safety_frame": {"max_access": "read"}},
            evaluate_governance=False,
        )
        planner = _ValidationRepairSameTurnExecutePlanner()
        result = await DbAgentLoop(runtime, planner).run(
            operation,
            safety_frame={"max_access": "read"},
            max_turns=8,
        )
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    assert result.status == "finished"
    repair_compilation = next(
        item
        for item in evidence
        if item.kind == "planner.compilation"
        and any(
            spec["capability_id"] == "db.query.repair"
            for spec in item.payload["compilation"]["task_specs"]
        )
    )
    repair_payload = repair_compilation.payload["compilation"]
    assert repair_compilation.accepted is True
    assert [spec["capability_id"] for spec in repair_payload["task_specs"]] == [
        "db.query.repair"
    ]
    assert repair_payload["rejected_action_summaries"] == []
    assert stale_sql not in json.dumps(repair_payload["task_specs"], sort_keys=True)
    plans = [item for item in evidence if item.kind == "query.plan.proposal"]
    repaired_plan = plans[-1]
    assert repaired_plan.accepted is True
    assert repaired_plan.payload["sql"] == repaired_sql
    repaired_validation_task = next(
        task
        for task in tasks
        if task.capability_id == "db.query.plan.validate"
        and task.input.get("plan_evidence_id") == repaired_plan.id
    )
    assert repaired_validation_task.input["planning_context_evidence_id"]
    query_result = next(item for item in evidence if item.kind == "query.result")
    assert query_result.payload["rows"] == [{"order_count": 1}]


async def test_session_scope_value_grounding_profiles_non_keyword_value():
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
            fulfillment_status TEXT NOT NULL,
            channel TEXT NOT NULL
        );
        INSERT INTO orders (id, fulfillment_status, channel)
        VALUES (1, 'fulfilled', 'web'), (2, 'queued', 'store');
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Summarize the same orders.",
                "source_scope": ["sqlite"],
                "session_context": {
                    "query_scopes": [
                        {
                            "operation_id": "op-prior",
                            "tables": ["orders"],
                            "filters": [
                                {
                                    "column": "fulfillment_status",
                                    "operator": "=",
                                    "values": ["fulfilled"],
                                }
                            ],
                        }
                    ]
                },
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
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    grounding_plan = next(
        item for item in evidence if item.kind == "catalog.value_grounding.plan"
    )
    assert [
        (target["table"], target["column"], target["reason"])
        for target in grounding_plan.payload["targets"]
    ] == [("orders", "fulfillment_status", "session_query_scope_filter")]
    profile_tasks = [
        task for task in tasks if task.capability_id == "db.column_values.profile"
    ]
    assert [(task.input["table"], task.input["column"]) for task in profile_tasks] == [
        ("orders", "fulfillment_status")
    ]
    hint = next(item for item in evidence if item.kind == "schema.column_value_hint")
    assert hint.payload["hints"][0]["observed_values"][0]["value"] == "fulfilled"


async def test_existing_catalog_profile_grounding_does_not_profile_again():
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
        schema = await runtime.execute_capability(
            "db.schema.inspect",
            owner="sqlite",
            operation_type="schema.query",
        )
        await runtime.execute_capability(
            "catalog.source.register",
            owner="catalog",
            operation_type="schema.register",
            input={
                "schema": schema[0].payload,
                "store_type": "sqlite",
                "store_id": "store:sqlite",
                "persist": False,
            },
        )
        await catalog.register_column_value_profiles(
            "store:sqlite",
            [
                {
                    "table": "orders",
                    "column": "status",
                    "distinct_count": 2,
                    "top_values": [
                        {"value": "complete", "count": 1},
                        {"value": "pending", "count": 1},
                    ],
                }
            ],
        )
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "What are completed order totals?",
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
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    grounding_plan = next(
        item for item in evidence if item.kind == "catalog.value_grounding.plan"
    )
    assert [
        (target["table"], target["column"], target["requires_profile_read"])
        for target in grounding_plan.payload["targets"]
    ] == [("orders", "status", False)]
    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)
    hint = next(item for item in evidence if item.kind == "schema.column_value_hint")
    assert hint.payload["hints"][0]["observed_values"][0]["value"] == "complete"


async def test_value_grounding_skipped_budget_target_does_not_profile():
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
                "prompt": "Resolve status values.",
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
                    input={
                        "source_owner": "sqlite",
                        "targets": [{"table": "orders", "column": "status"}],
                        "profile_budget": 0,
                    },
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
        evidence = await runtime.store.list_evidence(operation.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert_no_invalid_accepted_query_plans(evidence)
    grounding_plan = next(
        item for item in evidence if item.kind == "catalog.value_grounding.plan"
    )
    assert grounding_plan.payload["targets"] == []
    assert grounding_plan.payload["skipped"] == [
        {
            "table": "orders",
            "column": "status",
            "reason": "profile_budget_exhausted",
        }
    ]
    assert not any(task.capability_id == "db.column_values.profile" for task in tasks)


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


async def _sqlite_column_value_registration_runtime(*, register_store=True):
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(catalog, sqlite))
    await runtime.setup(agent_id="column-value-registration-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (status) VALUES ('complete'), ('pending');
        """)
    if register_store:
        await catalog.register_schema(
            {
                "database_type": "sqlite",
                "database_name": "registration-tests",
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
            store_type="sqlite",
            store_id="store:sqlite-registration",
            persist=False,
        )
    return runtime, catalog, sqlite


async def _plan_evidence_derived_column_value_registration(
    runtime,
    *,
    operation_id="evidence-derived-column-value-registration",
    store_id="store:sqlite-registration",
    register_input=None,
    include_dependency=True,
    duplicate_dependency=False,
    dependency_overrides=None,
):
    operation = await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="source.profile",
        request={"prompt": "register the profiled orders.status values"},
        evaluate_governance=False,
    )
    profile_input = {
        "table": "orders",
        "column": "status",
        "profile_kind": "categorical_values",
        "max_values": 25,
    }
    profile_task_id = f"{operation_id}-profile"
    profile_capability = runtime.registry.get_capability(
        "db.column_values.profile",
        owner="sqlite",
    )
    dependency_values = {
        "kind": "evidence",
        "evidence_kind": "column_values.profile",
        "evidence_owner": profile_capability.owner,
        "producer_task_id": profile_task_id,
        "producer_capability_id": profile_capability.id,
        "producer_executor_id": profile_capability.executor,
        "evidence_accepted": True,
        "input_hash": persisted_fingerprint(profile_input),
        "operation_id": operation.id,
    }
    dependency_values.update(dependency_overrides or {})
    dependency = TaskDependency(**dependency_values)
    dependencies = (dependency,) if include_dependency else ()
    if duplicate_dependency:
        dependencies = (
            dependency,
            TaskDependency(
                **{
                    **dependency_values,
                    "metadata": {"duplicate_dependency": True},
                }
            ),
        )
    registration_input = register_input or {
        "store_id": store_id,
        "table": "orders",
        "column": "status",
        "profile_kind": "categorical_values",
    }
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="db.column_values.profile",
                task_id=profile_task_id,
                owner="sqlite",
                input=profile_input,
                reason="test_profile_dependency",
                sequence=1,
            ),
            DbTaskSpec(
                capability_id="catalog.column_values.register",
                task_id=f"{operation_id}-register",
                owner="catalog",
                input=registration_input,
                reason="test_evidence_derived_registration",
                sequence=2,
                dependencies=dependencies,
            ),
        ),
    )
    return operation, plan.tasks[0], plan.tasks[1]


async def test_sqlite_column_value_profile_registers_with_catalog():
    catalog = CatalogPlugin(auto_persist=False)
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(catalog, sqlite))
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


async def test_sqlite_explicit_profile_registration_skips_schema_inspection():
    runtime, _, sqlite = await _sqlite_column_value_registration_runtime()

    async def fail_schema_inspection():
        raise AssertionError("explicit registration must not inspect schema")

    sqlite.tables = fail_schema_inspection
    try:
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
            operation_id="sqlite-explicit-profile-registration",
            input={
                "store_id": "store:sqlite-registration",
                "profiles": [raw_profile[0].payload],
                "source_evidence_id": raw_profile[0].id,
            },
        )
        tasks = await runtime.store.list_tasks(
            "sqlite-explicit-profile-registration"
        )
    finally:
        await runtime.teardown()

    assert registered[0].accepted is True
    assert [task.capability_id for task in tasks] == [
        "catalog.column_values.register"
    ]


async def test_explicit_empty_profiles_remain_explicit_without_bootstrap():
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        registered = await runtime.execute_capability(
            "catalog.column_values.register",
            owner="catalog",
            operation_type="source.profile",
            operation_id="explicit-empty-profile-registration",
            input={
                "store_id": "store:sqlite-registration",
                "profiles": [],
            },
        )
        tasks = await runtime.store.list_tasks("explicit-empty-profile-registration")
    finally:
        await runtime.teardown()

    assert registered[0].accepted is True
    assert registered[0].payload["profile_count"] == 0
    assert len(tasks) == 1
    assert tasks[0].capability_id == "catalog.column_values.register"
    assert tasks[0].dependencies == ()


async def test_evidence_derived_registration_consumes_exact_profile_dependency():
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        operation, profile_task, registration_task = (
            await _plan_evidence_derived_column_value_registration(runtime)
        )
        profile_evidence = await runtime.execute_task(profile_task, operation)
        registered = await runtime.execute_task(registration_task, operation)
        hydrated = await runtime.store.load_task(registration_task.id)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert registered[0].accepted is True
    assert hydrated is not None
    assert hydrated.input["profiles"] == [profile_evidence[0].payload]
    assert hydrated.input["source_evidence_id"] == profile_evidence[0].id
    assert hydrated.input["persist"] is False
    assert len(registration_task.dependencies) == 1
    dependency = registration_task.dependencies[0]
    assert dependency.producer_task_id == profile_task.id
    assert dependency.producer_capability_id == "db.column_values.profile"
    assert dependency.producer_executor_id == "sqlite.column_values.profile"
    assert {task.capability_id for task in tasks} == {
        "db.column_values.profile",
        "catalog.column_values.register",
    }


@pytest.mark.parametrize(
    "register_input",
    (
        {
            "store_id": "store:sqlite-registration",
            "table": "orders",
            "column": "status",
        },
        {
            "store_id": "store:sqlite-registration",
            "table": "orders",
            "column": "status",
            "profile_kind": "",
        },
        {
            "store_id": "store:sqlite-registration",
            "table": "shipments",
            "column": "status",
            "profile_kind": "categorical_values",
        },
        {
            "store_id": "store:sqlite-registration",
            "table": "orders",
            "column": "state",
            "profile_kind": "categorical_values",
        },
        {
            "store_id": "store:sqlite-registration",
            "table": "orders",
            "column": "status",
            "profile_kind": "logical_type_validation",
        },
    ),
    ids=(
        "missing-kind",
        "empty-kind",
        "wrong-table",
        "wrong-column",
        "wrong-kind",
    ),
)
async def test_evidence_derived_registration_rejects_incomplete_or_wrong_selection(
    register_input,
):
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        operation, profile_task, registration_task = (
            await _plan_evidence_derived_column_value_registration(
                runtime,
                register_input=register_input,
            )
        )
        await runtime.execute_task(profile_task, operation)
        with pytest.raises(
            RuntimeError,
            match="catalog.column_value_profile_selection_mismatch",
        ):
            await runtime.execute_task(registration_task, operation)
        tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    assert not any(task.capability_id == "db.schema.inspect" for task in tasks)
    assert not any(task.capability_id == "catalog.source.register" for task in tasks)


@pytest.mark.parametrize(
    ("include_dependency", "duplicate_dependency", "execute_profile"),
    (
        (False, False, False),
        (True, True, True),
    ),
    ids=("missing", "ambiguous"),
)
async def test_evidence_derived_registration_requires_exactly_one_dependency(
    include_dependency,
    duplicate_dependency,
    execute_profile,
):
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        operation, profile_task, registration_task = (
            await _plan_evidence_derived_column_value_registration(
                runtime,
                include_dependency=include_dependency,
                duplicate_dependency=duplicate_dependency,
            )
        )
        if execute_profile:
            await runtime.execute_task(profile_task, operation)
        with pytest.raises(
            RuntimeError,
            match="catalog.column_value_profile_dependency_required",
        ):
            await runtime.execute_task(registration_task, operation)
    finally:
        await runtime.teardown()


@pytest.mark.parametrize(
    ("dependency_overrides", "expected_error"),
    (
        (
            {"evidence_owner": "catalog"},
            "catalog.column_value_profile_dependency_identity_mismatch",
        ),
        (
            {"producer_capability_id": "db.schema.inspect"},
            "catalog.column_value_profile_dependency_identity_mismatch",
        ),
        (
            {"producer_executor_id": "sqlite.schema.inspect"},
            "catalog.column_value_profile_dependency_identity_mismatch",
        ),
        (
            {"operation_id": "other-operation"},
            "catalog.column_value_profile_dependency_identity_mismatch",
        ),
        (
            {"input_hash": "wrong-input-hash"},
            "catalog.column_value_profile_dependency_identity_mismatch",
        ),
        (
            {"producer_task_id": "missing-profile-task"},
            "catalog.column_value_profile_dependency_task_missing",
        ),
    ),
    ids=(
        "wrong-owner",
        "wrong-capability",
        "wrong-executor",
        "wrong-operation",
        "wrong-input",
        "missing-producer-task",
    ),
)
async def test_evidence_derived_registration_rejects_wrong_dependency_identity(
    dependency_overrides,
    expected_error,
):
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        operation, profile_task, registration_task = (
            await _plan_evidence_derived_column_value_registration(
                runtime,
                dependency_overrides=dependency_overrides,
            )
        )
        await runtime.execute_task(profile_task, operation)
        with pytest.raises(RuntimeError, match=expected_error):
            await runtime.execute_task(registration_task, operation)
    finally:
        await runtime.teardown()


async def test_evidence_derived_registration_rejects_missing_profile_evidence():
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        operation, _, registration_task = (
            await _plan_evidence_derived_column_value_registration(runtime)
        )
        with pytest.raises(
            RuntimeError,
            match="catalog.column_value_profile_evidence_missing",
        ):
            await runtime.execute_task(registration_task, operation)
    finally:
        await runtime.teardown()


async def test_evidence_derived_registration_rejects_rejected_profile_evidence():
    runtime, _, _ = await _sqlite_column_value_registration_runtime()
    try:
        operation, profile_task, registration_task = (
            await _plan_evidence_derived_column_value_registration(
                runtime,
                dependency_overrides={"evidence_accepted": False},
            )
        )
        await runtime.store.save_evidence(
            Evidence(
                id="rejected-column-value-profile",
                kind="column_values.profile",
                owner="sqlite",
                operation_id=operation.id,
                task_id=profile_task.id,
                accepted=False,
                payload={
                    "table": "orders",
                    "column": "status",
                    "profile_kind": "categorical_values",
                },
                metadata={"task_input_hash": profile_task.metadata["input_hash"]},
            )
        )
        with pytest.raises(
            RuntimeError,
            match="catalog.column_value_profile_dependency_identity_mismatch",
        ):
            await runtime.execute_task(registration_task, operation)
    finally:
        await runtime.teardown()


async def test_missing_catalog_store_response_is_bounded_for_both_registration_modes():
    runtime, _, _ = await _sqlite_column_value_registration_runtime(
        register_store=False
    )
    try:
        explicit = await runtime.execute_capability(
            "catalog.column_values.register",
            owner="catalog",
            operation_type="source.profile",
            operation_id="missing-store-explicit-registration",
            input={
                "store_id": "store:missing",
                "profiles": [
                    {
                        "table": "orders",
                        "column": "status",
                        "profile_kind": "categorical_values",
                    }
                ],
            },
        )
        operation, profile_task, registration_task = (
            await _plan_evidence_derived_column_value_registration(
                runtime,
                operation_id="missing-store-evidence-registration",
                store_id="store:missing",
            )
        )
        await runtime.execute_task(profile_task, operation)
        evidence_derived = await runtime.execute_task(registration_task, operation)
        explicit_tasks = await runtime.store.list_tasks(
            "missing-store-explicit-registration"
        )
        evidence_tasks = await runtime.store.list_tasks(operation.id)
    finally:
        await runtime.teardown()

    for evidence in (explicit[0], evidence_derived[0]):
        assert evidence.accepted is False
        assert evidence.payload == {
            "success": False,
            "store_id": "store:missing",
            "error": (
                "No profiled schema for store 'store:missing'. "
                "Profile or register the schema first."
            ),
        }
    all_tasks = (*explicit_tasks, *evidence_tasks)
    assert not any(task.capability_id == "db.schema.inspect" for task in all_tasks)
    assert not any(
        task.capability_id == "catalog.source.register" for task in all_tasks
    )


async def test_sqlite_logical_type_profile_is_bounded_and_exposes_no_values():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
    await runtime.setup(agent_id="db-runtime-test")
    await sqlite.execute_script("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            timestamp_named_only TEXT NOT NULL
        );
        INSERT INTO events (id, created_at, timestamp_named_only) VALUES
            (1, '2026-01-02T10:00:00Z', 'not a timestamp'),
            (2, '2026-01-03T11:00:00Z', 'still arbitrary');
        """)

    try:
        proven = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "events",
                "column": "created_at",
                "profile_kind": "logical_type_validation",
                "max_values": 64,
            },
        )
        arbitrary = await runtime.execute_capability(
            "db.column_values.profile",
            owner="sqlite",
            operation_type="source.profile",
            input={
                "table": "events",
                "column": "timestamp_named_only",
                "profile_kind": "logical_type_validation",
                "max_values": 64,
            },
        )
    finally:
        await runtime.teardown()

    payload = proven[0].payload
    assert payload["logical_type"] == "timestamp"
    assert payload["logical_type_proof"]["method"] == "bounded_value_profile"
    assert payload["logical_type_proof"]["representation"] == ("iso8601_utc_second")
    assert payload["logical_type_proof"]["all_values_matched"] is True
    assert payload["logical_type_proof"]["values_exposed"] is False
    assert payload["policy"]["sample_limit"] == 64
    assert payload["policy"]["raw_values_persisted"] is False
    assert payload["top_values"] == []
    assert "2026-01-02T10:00:00Z" not in str(payload)
    assert "logical_type" not in arbitrary[0].payload
    assert arbitrary[0].payload["top_values"] == []
    assert "not a timestamp" not in str(arbitrary[0].payload)


async def test_sqlite_column_value_profile_fingerprint_only_uses_live_revision():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(plugins=(sqlite,))
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


async def test_sqlite_column_value_profile_honors_source_value_policy():
    sqlite = SQLitePlugin(path=":memory:", include_sample_values=False)
    runtime = DbRuntime(plugins=(sqlite,))
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
    runtime = DbRuntime(plugins=(sqlite,))
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
    runtime = DbRuntime(plugins=(sqlite,))
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
