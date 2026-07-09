import json

from daita.agents.agent import Agent
from daita.agents.conversation import ConversationHistory
from daita.db import DbAgent, DbRequest, DbRuntime, DbRuntimeConfig
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbOperationContract,
    DbOperationResult,
)
from daita.db.planning_context import DbPlanningContextBuilder
from daita.db.query_plan import DbQueryPlan
from daita.db.runtime.tasks.runtime import DbTaskSpec
from daita.db.session_context import (
    DbSessionContextBuilder,
    db_session_context_from_request,
    session_query_scope_evidence_for,
    session_scope_binding_evidence_for,
)
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import AccessMode, Evidence, Operation, OperationStatus


class SpyRuntime(DbRuntime):
    def __init__(self) -> None:
        super().__init__(runtime_id="db-session-spy")
        self.run_requests: list[DbRequest] = []

    async def run(self, request):
        db_request = request if isinstance(request, DbRequest) else DbRequest(request)
        self.run_requests.append(db_request)
        return DbOperationResult(
            operation_id=f"spy-{len(self.run_requests)}",
            request=db_request,
            intent=DbIntent(
                kind=DbIntentKind.CONVERSATIONAL,
                confidence=1.0,
                access=AccessMode.NONE,
            ),
            contract=DbOperationContract(operation_type="conversational"),
            status=OperationStatus.SUCCEEDED,
            answer="spy answer",
        )


class _BlockedColumnSource:
    read_only = True
    allowed_tables = set()
    blocked_tables = set()
    blocked_columns = {"customers.loyalty_band"}


async def _seed_agent_schema(path):
    plugin = SQLitePlugin(path=str(path))
    await plugin.execute_script("""
        CREATE TABLE agent_profiles (
            id INTEGER PRIMARY KEY,
            agent_name TEXT NOT NULL
        );
        CREATE TABLE agent_runs (
            id INTEGER PRIMARY KEY,
            profile_id INTEGER REFERENCES agent_profiles(id),
            status TEXT NOT NULL
        );
        CREATE TABLE billing_events (
            id INTEGER PRIMARY KEY,
            amount_cents INTEGER NOT NULL
        );
        INSERT INTO agent_profiles (agent_name) VALUES ('alpha');
        INSERT INTO agent_runs (profile_id, status) VALUES (1, 'ok');
        INSERT INTO billing_events (amount_cents) VALUES (100);
        """)
    await plugin.disconnect()


async def test_session_context_builder_collects_runtime_referents_and_diagnostics():
    runtime = DbRuntime(runtime_id="session-context-builder")
    operation = Operation(
        id="op-1",
        operation_type="schema.query",
        status=OperationStatus.SUCCEEDED,
        request={"prompt": "What columns are in customers?", "session_id": "s1"},
        metadata={"session_id": "s1"},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_evidence(
        Evidence(
            kind="schema.asset_profile",
            owner="sqlite",
            operation_id="op-1",
            payload={
                "tables": [
                    {
                        "name": "customers",
                        "columns": [{"name": "id"}, {"name": "name"}],
                    }
                ],
                "metadata": {"scope": "asset"},
            },
        )
    )
    await runtime.store.save_evidence(
        Evidence(
            kind="query.plan.proposal",
            owner="db_runtime",
            operation_id="op-1",
            payload={
                "sql": (
                    "SELECT c.customer_id, c.name FROM customers c "
                    "JOIN regions r ON r.region_id = c.region_id "
                    "WHERE c.tier = 'enterprise' "
                    "AND r.region_code = 'NA' "
                    "AND c.email = 'ada@example.com'"
                ),
                "structured_plan": {
                    "selected_tables": ["customers", "regions"],
                    "filters": [
                        {
                            "column": "customers.tier",
                            "operator": "=",
                            "value": "enterprise",
                        },
                        {
                            "column": "regions.region_code",
                            "operator": "=",
                            "value": "NA",
                        },
                        {
                            "column": "customers.email",
                            "operator": "=",
                            "value": "ada@example.com",
                        },
                    ],
                },
            },
        )
    )
    await runtime.store.save_evidence(
        Evidence(
            kind="query.result",
            owner="sqlite",
            operation_id="op-1",
            payload={"rows": [{"customer_id": 1}, {"customer_id": 3}]},
        )
    )

    context = await DbSessionContextBuilder(runtime).build(
        DbRequest("What are their completed order totals?", session_id="s1"),
        conversation_messages=[
            {"role": "user", "content": "What columns are in customers?"},
            {"role": "assistant", "content": "customers: id, name"},
        ],
    )

    assert context.session_id == "s1"
    assert "customers" in context.referents.tables
    assert {"id", "name"} <= set(context.referents.columns)
    assert context.referents.operations == ("op-1",)
    assert len(context.query_scopes) == 1
    scope = context.query_scopes[0].to_dict()
    assert {"customers", "regions"} <= set(scope["tables"])
    assert {
        (item["column"], item["operator"], tuple(item["values"]))
        for item in scope["filters"]
    } >= {
        ("customers.tier", "=", ("enterprise",)),
        ("regions.region_code", "=", ("NA",)),
    }
    assert "ada@example.com" not in json.dumps(scope)
    assert scope["result_row_count"] == 2
    assert context.diagnostics["query_scope_count"] == 1
    assert (
        "runtime.evidence" in context.diagnostics["referent_sources"]["tables"].values()
    )
    assert "conversation_history" in context.diagnostics["sources"]


def test_session_query_scope_evidence_captures_safe_runtime_facts():
    operation = Operation(
        id="op-scope",
        operation_type="data.query",
        request={"prompt": "Show enterprise customers", "session_id": "s1"},
        metadata={"session_id": "s1"},
    )
    evidence = (
        Evidence(
            id="plan-scope",
            kind="query.plan.proposal",
            owner="db_runtime",
            operation_id="op-scope",
            accepted=True,
            payload={
                "sql": (
                    "SELECT c.id, r.name AS region_name FROM customers c "
                    "JOIN regions r ON c.region_id = r.id "
                    "WHERE c.tier = 'enterprise'"
                ),
                "structured_plan": {
                    "selected_tables": ["customers", "regions"],
                    "selected_columns": ["customers.id", "regions.name"],
                    "joins": [
                        {
                            "left_table": "customers",
                            "left_column": "region_id",
                            "right_table": "regions",
                            "right_column": "id",
                        }
                    ],
                    "filters": [
                        {
                            "column": "customers.tier",
                            "operator": "=",
                            "value": "enterprise",
                        }
                    ],
                },
            },
        ),
        Evidence(
            id="sql-scope",
            kind="sql.validation",
            owner="sqlite",
            operation_id="op-scope",
            accepted=True,
            payload={
                "valid": True,
                "sql": (
                    "SELECT c.id, r.name AS region_name FROM customers c "
                    "JOIN regions r ON c.region_id = r.id "
                    "WHERE c.tier = 'enterprise'"
                ),
                "tables": ["customers", "regions"],
            },
        ),
        Evidence(
            id="result-scope",
            kind="query.result",
            owner="sqlite",
            operation_id="op-scope",
            accepted=True,
            payload={"rows": [{"id": 1, "region_name": "NA"}]},
        ),
    )

    scope = session_query_scope_evidence_for(operation, evidence, task_id="task-read")

    assert scope is not None
    assert scope.kind == "session.query_scope"
    assert scope.owner == "db_runtime"
    assert scope.task_id == "task-read"
    assert scope.payload["source_operation_id"] == "op-scope"
    assert {"customers", "regions"} <= set(scope.payload["tables"])
    assert {
        (
            item["left_table"],
            item["left_column"],
            item["right_table"],
            item["right_column"],
        )
        for item in scope.payload["joins"]
    } >= {("customers", "region_id", "regions", "id")}
    assert {
        (item["column"], item["operator"], tuple(item["values"]))
        for item in scope.payload["filters"]
    } >= {("customers.tier", "=", ("enterprise",))}
    assert "customers.id" in scope.payload["selected_columns"]
    assert scope.payload["result_row_count"] == 1
    assert scope.payload["scope_id"].startswith("session-scope-")


async def test_stateful_read_execution_emits_session_query_scope():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(config=DbRuntimeConfig(plugins=(sqlite,)))
    await runtime.setup(agent_id="db-session-scope-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status) VALUES (1, 'complete'), (2, 'pending');
        """)

    try:
        operation = await runtime.kernel.create_operation(
            operation_type="data.query",
            request={
                "prompt": "Show complete orders",
                "session_id": "stateful-scope",
            },
            metadata={"session_id": "stateful-scope"},
            evaluate_governance=False,
        )
        plan = await runtime.plan_task_specs(
            operation,
            (
                DbTaskSpec(
                    capability_id="db.sql.validate",
                    owner="sqlite",
                    input={
                        "sql": "SELECT id FROM orders WHERE status = 'complete'",
                        "operation": "query",
                    },
                    sequence=1,
                ),
                DbTaskSpec(
                    capability_id="db.sql.execute_read",
                    owner="sqlite",
                    input={"sql_ref": "sql.validation", "params": []},
                    sequence=2,
                ),
            ),
        )
        for task in plan.tasks:
            await runtime.execute_task(task, operation)
        evidence = await runtime.store.list_evidence(operation.id)
    finally:
        await runtime.teardown()

    scope = next(item for item in evidence if item.kind == "session.query_scope")
    assert scope.payload["source_operation_id"] == operation.id
    assert scope.payload["tables"] == ["orders"]
    assert scope.payload["filters"] == [
        {"column": "status", "operator": "=", "values": ["complete"]}
    ]
    assert scope.payload["result_row_count"] == 1


def test_follow_up_with_prior_scope_emits_scope_binding():
    operation = Operation(
        id="op-follow-up",
        operation_type="data.query",
        request={"prompt": "Now total them", "session_id": "s1"},
        metadata={"session_id": "s1"},
    )
    plan = DbQueryPlan(
        operation="read",
        selected_sql=(
            "SELECT SUM(total) AS total FROM orders WHERE status = 'complete'"
        ),
        selected_tables=("orders",),
        confidence=0.9,
    )
    binding = session_scope_binding_evidence_for(
        operation,
        plan,
        {
            "session_context": {
                "query_scopes": [
                    {
                        "scope_id": "scope-prior",
                        "operation_id": "op-prior",
                        "tables": ["orders"],
                        "filters": [
                            {
                                "column": "orders.status",
                                "operator": "=",
                                "values": ["complete"],
                            }
                        ],
                    }
                ]
            }
        },
        task_id="task-validate",
    )

    assert binding is not None
    assert binding.kind == "session.scope_binding"
    assert binding.task_id == "task-validate"
    assert binding.payload["source_scope_id"] == "scope-prior"
    assert binding.payload["source_operation_id"] == "op-prior"
    assert binding.payload["binding_status"] == "bound"
    assert binding.payload["required_filters"] == [
        {"column": "orders.status", "operator": "=", "values": ["complete"]}
    ]


def test_follow_up_without_required_scope_facts_does_not_emit_empty_binding():
    operation = Operation(
        id="op-follow-up",
        operation_type="data.query",
        request={"prompt": "Show another summary", "session_id": "s1"},
        metadata={"session_id": "s1"},
    )
    plan = DbQueryPlan(
        operation="read",
        selected_sql="SELECT COUNT(*) AS count FROM orders",
        selected_tables=("orders",),
        confidence=0.9,
    )

    binding = session_scope_binding_evidence_for(
        operation,
        plan,
        {
            "session_context": {
                "query_scopes": [
                    {
                        "scope_id": "scope-prior",
                        "operation_id": "op-prior",
                        "tables": ["orders"],
                    }
                ]
            }
        },
        task_id="task-validate",
    )

    assert binding is None


def test_planner_visible_session_context_redacts_blocked_columns():
    request = DbRequest(
        "Continue summarizing customer revenue.",
        session_id="s-blocked",
        session_context={
            "session_id": "s-blocked",
            "referents": {
                "tables": ["customers"],
                "columns": ["customers.loyalty_band"],
                "schemas": [],
                "metrics": [],
                "monitors": [],
                "approvals": [],
                "operations": ["op-prior"],
            },
            "query_scopes": [
                {
                    "tables": ["customers"],
                    "filters": [
                        {
                            "column": "customers.loyalty_band",
                            "operator": "=",
                            "values": ["platinum"],
                        }
                    ],
                    "result_row_count": 3,
                }
            ],
        },
    )
    context = DbPlanningContextBuilder(DbRuntimeConfig()).build(
        request=request,
        intent=DbIntent(
            kind=DbIntentKind.DATA_QUERY,
            confidence=1.0,
            access=AccessMode.READ,
        ),
        operation=Operation(id="op-current", operation_type="data.query"),
        schema_evidence=Evidence(
            id="schema-blocked-context",
            kind="schema.asset_profile",
            owner="sqlite",
            accepted=True,
            payload={
                "database_type": "sqlite",
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "data_type": "INTEGER"},
                            {"name": "loyalty_band", "data_type": "TEXT"},
                            {"name": "revenue", "data_type": "REAL"},
                        ],
                    }
                ],
            },
        ),
        source=_BlockedColumnSource(),
    )

    planner_visible = json.dumps(
        {
            "session_context": context.session_context,
            "rendered_context": context.rendered_context,
        },
        sort_keys=True,
    )
    assert "customers.loyalty_band" not in planner_visible
    assert "platinum" not in planner_visible
    assert "blocked_columns" in context.policy_summary
    assert context.policy_summary["blocked_columns"] == ["<redacted>"]
    assert "customers.loyalty_band" not in json.dumps(context.policy_summary)


def test_session_context_from_dict_filters_invalid_conversation_messages():
    context = db_session_context_from_request(
        DbRequest(
            "continue",
            session_context={
                "session_id": "s1",
                "user_id": "u1",
                "current_prompt": "continue",
                "conversation_messages": [
                    {"role": "user", "content": "show customers"},
                    None,
                    {"role": "invalid", "content": "ignored"},
                    {"role": "assistant", "content": ""},
                    "ignored",
                    {"role": "assistant", "content": "customers: id, name"},
                ],
            },
        )
    )

    assert context is not None
    assert context.conversation_messages == (
        {"role": "user", "content": "show customers"},
        {"role": "assistant", "content": "customers: id, name"},
    )


async def test_db_agent_history_session_id_flows_and_turn_is_appended():
    history = ConversationHistory(session_id="history-session", workspace="db-test")
    runtime = SpyRuntime()
    agent = DbAgent(runtime=runtime, name="db-history-test")

    result = await agent.run_detailed("Hello there", history=history)

    assert result.request.session_id == "history-session"
    assert runtime.run_requests[0].session_id == "history-session"
    assert history.turn_count == 1
    assert history.messages[-2:] == [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "spy answer"},
    ]


async def test_db_agent_explicit_session_id_overrides_history_for_routing():
    history = ConversationHistory(session_id="history-session", workspace="db-test")
    runtime = SpyRuntime()
    agent = DbAgent(runtime=runtime, name="db-history-test")

    result = await agent.run_detailed(
        "Hello there",
        history=history,
        session_id="explicit-session",
    )

    assert result.request.session_id == "explicit-session"
    assert runtime.run_requests[0].session_id == "explicit-session"
    assert history.turn_count == 1


async def test_db_agent_session_context_payload_compacts_history_and_operations():
    history = ConversationHistory(session_id="compact-session", workspace="db-test")
    await history.add_turn(
        "prior user transcript text that should not be transported",
        "The useful table is `agent_profiles`.",
    )
    runtime = SpyRuntime()
    await runtime.store.save_operation(
        Operation(
            id="op-compact",
            operation_type="schema.query",
            status=OperationStatus.SUCCEEDED,
            request={
                "prompt": "prior operation prompt that should not be transported",
                "session_id": "compact-session",
            },
            metadata={"session_id": "compact-session"},
        )
    )
    agent = DbAgent(runtime=runtime, name="db-history-test")

    result = await agent.run_detailed(
        "What columns do those tables have?", history=history
    )

    payload = result.request.session_context
    assert payload is not None
    dumped = json.dumps(payload)
    assert "conversation_messages" not in payload
    assert "current_prompt" not in payload
    assert "prior user transcript text" not in dumped
    assert "prior operation prompt" not in dumped
    assert payload["recent_operations"][0]["operation_id"] == "op-compact"
    assert "prompt" not in payload["recent_operations"][0]
    assert "prompt_fingerprint" in payload["recent_operations"][0]


async def test_from_db_stateful_true_creates_default_history(tmp_path):
    db_path = tmp_path / "stateful.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path), stateful=True, name="stateful-db-test")

    try:
        result = await agent.run_detailed("How many agent profiles are there?")
    finally:
        await agent.stop()

    assert result.request.session_id is not None
    assert agent._default_history is not None
    assert agent._default_history.turn_count == 1


async def test_stateless_default_does_not_create_session_history(tmp_path):
    db_path = tmp_path / "stateless.sqlite"
    await _seed_agent_schema(db_path)
    agent = await Agent.from_db(str(db_path))

    try:
        result = await agent.run_detailed("How many agent profiles are there?")
    finally:
        await agent.stop()

    assert result.request.session_id is None
    assert agent._default_history is None
