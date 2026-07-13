import json

import pytest

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
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.db.query_plan import DbQueryPlan
from daita.db.session_context import (
    DbSessionContextBuilder,
    db_session_context_from_request,
    persist_session_query_scopes,
    session_query_scope_evidence_for,
    session_scope_binding_evidence_for,
)
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import (
    AccessMode,
    Evidence,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskStatus,
)


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


class _SessionReadPlanner:
    def __init__(self, *sql_statements: str) -> None:
        self.sql_statements = sql_statements

    async def plan(self, state):
        return DbPlannerDecision(
            status=DbPlannerDecisionStatus.CONTINUE,
            intent={"operation_type": "data.query"},
            actions=tuple(
                DbPlannerAction(
                    action_id=f"session_read_{index}",
                    kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                    input={"owner": "sqlite", "sql": sql},
                )
                for index, sql in enumerate(self.sql_statements, start=1)
            ),
        )


def _scope_read_facts(
    operation_id: str,
    suffix: str,
    sql: str,
    *,
    status: TaskStatus = TaskStatus.SUCCEEDED,
    result_payload: dict | None = None,
    result_accepted: bool = True,
) -> tuple[Task, tuple[Evidence, ...]]:
    plan_id = f"plan-{suffix}"
    validation_id = f"validation-{suffix}"
    task = Task(
        id=f"read-{suffix}",
        operation_id=operation_id,
        capability_id="db.sql.execute_read",
        executor_id="sqlite.sql.execute_read",
        status=status,
        dependencies=(
            TaskDependency(
                kind="evidence",
                evidence_kind="query.plan.proposal",
                evidence_id=plan_id,
                operation_id=operation_id,
            ),
            TaskDependency(
                kind="evidence",
                evidence_kind="sql.validation",
                evidence_id=validation_id,
                operation_id=operation_id,
            ),
        ),
        metadata={"owner": "sqlite"},
    )
    evidence = [
        Evidence(
            id=plan_id,
            kind="query.plan.proposal",
            owner="db_runtime",
            operation_id=operation_id,
            accepted=True,
            payload={"sql": sql},
        ),
        Evidence(
            id=validation_id,
            kind="sql.validation",
            owner="sqlite",
            operation_id=operation_id,
            accepted=True,
            payload={"valid": True, "sql": sql},
        ),
    ]
    if result_payload is not None:
        evidence.append(
            Evidence(
                id=f"result-{suffix}",
                kind="query.result",
                owner="sqlite",
                operation_id=operation_id,
                task_id=task.id,
                accepted=result_accepted,
                payload=result_payload,
            )
        )
    return task, tuple(evidence)


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


async def test_successful_db_run_persists_one_scope_and_follow_up_retrieves_it():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(sqlite,)),
        host_services={
            "db_agent_planner": _SessionReadPlanner(
                "SELECT id FROM orders WHERE status = 'complete'"
            )
        },
    )
    verification_evidence_kinds = []
    original_verify = runtime.verifier.verify

    def capture_verification_evidence(contract, intent, evidence, tasks):
        verification_evidence_kinds.append(tuple(item.kind for item in evidence))
        return original_verify(contract, intent, evidence, tasks)

    runtime.verifier.verify = capture_verification_evidence
    await runtime.setup(agent_id="db-session-scope-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL
        );
        INSERT INTO orders (id, status) VALUES (1, 'complete'), (2, 'pending');
        """)

    try:
        result = await runtime.run(
            DbRequest("Show complete orders", session_id="stateful-scope")
        )
        operation = await runtime.store.load_operation(result.operation_id)
        tasks = tuple(await runtime.store.list_tasks(result.operation_id))
        evidence = tuple(await runtime.store.list_evidence(result.operation_id))
        follow_up = await DbSessionContextBuilder(runtime).build(
            DbRequest("Now total them", session_id="stateful-scope")
        )
        assert operation is not None
        repeated = await persist_session_query_scopes(
            runtime.store,
            operation,
            tasks,
            evidence,
        )
        evidence_after_repeat = tuple(
            await runtime.store.list_evidence(result.operation_id)
        )
    finally:
        await runtime.teardown()

    scopes = [item for item in evidence if item.kind == "session.query_scope"]
    scopes_after_repeat = [
        item for item in evidence_after_repeat if item.kind == "session.query_scope"
    ]
    assert result.status is OperationStatus.SUCCEEDED
    assert verification_evidence_kinds
    assert "session.query_scope" in verification_evidence_kinds[-1]
    assert len(scopes) == 1
    assert repeated == ()
    assert [item.id for item in scopes_after_repeat] == [item.id for item in scopes]
    scope = scopes[0]
    assert scope.payload["source_operation_id"] == result.operation_id
    assert scope.payload["tables"] == ["orders"]
    assert scope.payload["filters"] == [
        {"column": "status", "operator": "=", "values": ["complete"]}
    ]
    assert scope.payload["result_row_count"] == 1
    assert scope.task_id == next(
        task.id for task in tasks if task.capability_id == "db.sql.execute_read"
    )
    assert [item.scope_id for item in follow_up.query_scopes] == [
        scope.payload["scope_id"]
    ]


async def test_multiple_successful_reads_persist_deterministic_scopes_once():
    runtime = DbRuntime()
    operation = Operation(
        id="op-multiple-scopes",
        operation_type="db.run",
        request={
            "prompt": "Show complete and pending orders",
            "session_id": "multiple-scopes",
            "source_scope": ["orders"],
        },
        metadata={"session_id": "multiple-scopes"},
    )
    first_task, first_evidence = _scope_read_facts(
        operation.id,
        "complete",
        "SELECT id FROM orders WHERE status = 'complete'",
        result_payload={"rows": [{"id": 1}]},
    )
    second_task, second_evidence = _scope_read_facts(
        operation.id,
        "pending",
        "SELECT id FROM orders WHERE status = 'pending'",
        result_payload={"rows": [{"id": 2}, {"id": 3}]},
    )
    tasks = (first_task, second_task)
    evidence = (*first_evidence, *second_evidence)
    await runtime.store.save_operation(operation)
    for task in tasks:
        await runtime.store.save_task(task)
    for item in evidence:
        await runtime.store.save_evidence(item)

    first_pass = await persist_session_query_scopes(
        runtime.store,
        operation,
        tasks,
        evidence,
    )
    second_pass = await persist_session_query_scopes(
        runtime.store,
        operation,
        tasks,
        evidence,
    )
    scopes = [
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "session.query_scope"
    ]

    assert len(first_pass) == 2
    assert second_pass == ()
    assert len(scopes) == 2
    assert len({item.payload["scope_id"] for item in scopes}) == 2
    assert {item.task_id for item in scopes} == {first_task.id, second_task.id}
    assert {
        tuple(filter_item["values"])
        for item in scopes
        for filter_item in item.payload["filters"]
    } == {("complete",), ("pending",)}
    assert {item.payload["result_row_count"] for item in scopes} == {1, 2}
    assert all(item.payload["source_scope"] == ["orders"] for item in scopes)
    assert all(
        {ref["kind"] for ref in item.payload["source_evidence_refs"]}
        == {"query.plan.proposal", "sql.validation", "query.result"}
        for item in scopes
    )


@pytest.mark.parametrize(
    ("status", "result_payload", "result_accepted"),
    (
        (TaskStatus.FAILED, {"rows": [{"id": 1}]}, True),
        (TaskStatus.BLOCKED, {"rows": [{"id": 1}]}, True),
        (TaskStatus.PENDING, {"rows": [{"id": 1}]}, True),
        (TaskStatus.RUNNING, {"rows": [{"id": 1}]}, True),
        (TaskStatus.CANCELLED, {"rows": [{"id": 1}]}, True),
        (TaskStatus.SKIPPED, {"rows": [{"id": 1}]}, True),
        (TaskStatus.SUCCEEDED, None, True),
        (TaskStatus.SUCCEEDED, {"success": False, "rows": []}, True),
        (TaskStatus.SUCCEEDED, {"rows": [{"id": 1}]}, False),
    ),
)
async def test_unsuccessful_or_missing_result_reads_do_not_persist_scopes(
    status,
    result_payload,
    result_accepted,
):
    runtime = DbRuntime()
    operation = Operation(
        id=f"op-scope-{status.value}-{result_payload is None}-{result_accepted}",
        operation_type="db.run",
        request={"prompt": "Show orders", "session_id": "scope-status"},
    )
    task, evidence = _scope_read_facts(
        operation.id,
        "status",
        "SELECT id FROM orders",
        status=status,
        result_payload=result_payload,
        result_accepted=result_accepted,
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(task)
    for item in evidence:
        await runtime.store.save_evidence(item)

    persisted = await persist_session_query_scopes(
        runtime.store,
        operation,
        (task,),
        evidence,
    )

    assert persisted == ()
    assert not [
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "session.query_scope"
    ]


@pytest.mark.parametrize("operation_type", ("monitor.source", "scheduled.operation"))
async def test_monitor_and_scheduler_operations_do_not_persist_session_scopes(
    operation_type,
):
    runtime = DbRuntime()
    operation = Operation(
        id=f"op-{operation_type}",
        operation_type=operation_type,
        request={"prompt": "Show orders", "session_id": "non-run-session"},
    )
    task, evidence = _scope_read_facts(
        operation.id,
        "non-run",
        "SELECT id FROM orders",
        result_payload={"rows": [{"id": 1}]},
    )
    await runtime.store.save_operation(operation)
    await runtime.store.save_task(task)
    for item in evidence:
        await runtime.store.save_evidence(item)

    persisted = await persist_session_query_scopes(
        runtime.store,
        operation,
        (task,),
        evidence,
    )

    assert persisted == ()


async def test_direct_capability_execution_does_not_persist_session_scope():
    sqlite = SQLitePlugin(path=":memory:")
    runtime = DbRuntime(config=DbRuntimeConfig(plugins=(sqlite,)))
    await runtime.setup(agent_id="db-direct-scope-test")
    await sqlite.execute_script("""
        CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT NOT NULL);
        INSERT INTO orders (id, status) VALUES (1, 'complete');
        """)

    try:
        result_evidence = await runtime.execute_capability(
            "db.sql.execute_read",
            owner="sqlite",
            operation_type="data.query",
            input={
                "prompt": "Show complete orders",
                "session_id": "direct-session",
                "sql": "SELECT id FROM orders WHERE status = 'complete'",
            },
        )
        operation = (await runtime.store.list_operations())[-1]
        stored_evidence = await runtime.store.list_evidence(operation.id)
    finally:
        await runtime.teardown()

    assert any(item.kind == "query.result" for item in result_evidence)
    assert not [item for item in stored_evidence if item.kind == "session.query_scope"]


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
    assert context.session_context["session_id"] == "s-blocked"
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
