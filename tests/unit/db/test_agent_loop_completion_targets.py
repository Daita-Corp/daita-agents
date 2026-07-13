from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.loop import DbAgentLoop
from daita.db.models import DbIntent, DbIntentKind, DbOperationContract, DbRequest
from daita.db.planner_protocol import (
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import (
    AccessMode,
    Capability,
    Evidence,
    EvidenceSchema,
    Operation,
    OperationStatus,
    RiskLevel,
    Task,
    TaskStatus,
)
from daita.db.verification import DbVerifier, db_run_finalization_check

OWNER = "completion_target"


class CompletionTargetExecutor:
    def __init__(self, executor_id, capability_ids):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)
        self.calls = []

    async def execute(self, task, operation, context):
        self.calls.append(
            {
                "task_id": task.id,
                "capability_id": task.capability_id,
                "status": task.status.value,
                "input": dict(task.input),
            }
        )
        if task.capability_id == "db.schema.inspect":
            return [
                Evidence(
                    kind="database.schema",
                    owner=OWNER,
                    accepted=True,
                    payload={
                        "tables": [
                            {
                                "name": "orders",
                                "columns": ["id", "status", "total"],
                            }
                        ],
                        "summary": "orders(id, status, total)",
                    },
                )
            ]
        if task.capability_id == "catalog.schema.search":
            return [
                Evidence(
                    kind="schema.search_result",
                    owner=OWNER,
                    accepted=True,
                    payload={
                        "query": task.input.get("query"),
                        "matches": [
                            {
                                "asset": "orders",
                                "columns": ["id", "status", "total"],
                            }
                        ],
                    },
                )
            ]
        if task.capability_id == "db.sql.validate":
            sql = task.input["sql"]
            if "missing_table" in sql:
                raise ValueError("validation_failed: missing_table")
            return [
                Evidence(
                    kind="sql.validation",
                    owner=OWNER,
                    accepted=True,
                    payload={"valid": True, "sql": sql, "operation": "query"},
                )
            ]
        if task.capability_id == "db.sql.execute_read":
            return [
                Evidence(
                    kind="query.result",
                    owner=OWNER,
                    accepted=True,
                    payload={
                        "rows": [{"status": "paid", "count": 2}],
                        "total_rows": 1,
                        "sql": task.input["sql"],
                        "validated_evidence_id": task.input["validated_evidence_id"],
                    },
                )
            ]
        raise AssertionError(f"unexpected capability: {task.capability_id}")


class CompletionTargetPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id=OWNER,
        display_name="Completion Target",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.schema = CompletionTargetExecutor(
            f"{OWNER}.schema.inspect",
            {"db.schema.inspect"},
        )
        self.catalog = CompletionTargetExecutor(
            f"{OWNER}.schema.search",
            {"catalog.schema.search"},
        )
        self.source_register = CompletionTargetExecutor(
            f"{OWNER}.source.register",
            {"catalog.source.register"},
        )
        self.column_search = CompletionTargetExecutor(
            f"{OWNER}.column_values.search",
            {"catalog.column_values.search"},
        )
        self.column_profile = CompletionTargetExecutor(
            f"{OWNER}.column_values.profile",
            {"db.column_values.profile"},
        )
        self.column_register = CompletionTargetExecutor(
            f"{OWNER}.column_values.register",
            {"catalog.column_values.register"},
        )
        self.validation = CompletionTargetExecutor(
            f"{OWNER}.sql.validate",
            {"db.sql.validate"},
        )
        self.read = CompletionTargetExecutor(
            f"{OWNER}.sql.execute_read",
            {"db.sql.execute_read"},
        )

    def declare_capabilities(self):
        common = {
            "domains": frozenset({"db"}),
            "operation_types": frozenset(
                {"db.run", "schema.query", "data.query", "data.query.catalog_assisted"}
            ),
            "risk": RiskLevel.LOW,
            "input_schema": {"type": "object"},
            "runtime_only": True,
            "side_effecting": False,
            "replay_safe": True,
            "idempotent": True,
        }
        return [
            Capability(
                id="db.schema.inspect",
                owner=OWNER,
                description="Inspect schema.",
                access=AccessMode.METADATA_READ,
                output_evidence=frozenset({"database.schema"}),
                executor=self.schema.id,
                **common,
            ),
            Capability(
                id="catalog.schema.search",
                owner=OWNER,
                description="Search schema catalog.",
                access=AccessMode.METADATA_READ,
                output_evidence=frozenset({"schema.search_result"}),
                executor=self.catalog.id,
                **common,
            ),
            Capability(
                id="catalog.source.register",
                owner=OWNER,
                description="Register catalog source.",
                access=AccessMode.METADATA_READ,
                output_evidence=frozenset({"catalog.source_registered"}),
                executor=self.source_register.id,
                **common,
            ),
            Capability(
                id="catalog.column_values.search",
                owner=OWNER,
                description="Search catalog column values.",
                access=AccessMode.METADATA_READ,
                output_evidence=frozenset({"schema.column_value_search_result"}),
                executor=self.column_search.id,
                **common,
            ),
            Capability(
                id="db.column_values.profile",
                owner=OWNER,
                description="Profile source column values.",
                access=AccessMode.READ,
                risk=RiskLevel.MEDIUM,
                output_evidence=frozenset({"column_values.profile"}),
                executor=self.column_profile.id,
                **{key: value for key, value in common.items() if key != "risk"},
            ),
            Capability(
                id="catalog.column_values.register",
                owner=OWNER,
                description="Register catalog column values.",
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.MEDIUM,
                output_evidence=frozenset({"schema.column_value_profile"}),
                executor=self.column_register.id,
                **{key: value for key, value in common.items() if key != "risk"},
            ),
            Capability(
                id="db.sql.validate",
                owner=OWNER,
                description="Validate SQL.",
                access=AccessMode.METADATA_READ,
                output_evidence=frozenset({"sql.validation"}),
                executor=self.validation.id,
                **common,
            ),
            Capability(
                id="db.sql.execute_read",
                owner=OWNER,
                description="Execute validated read.",
                access=AccessMode.READ,
                output_evidence=frozenset({"query.result"}),
                executor=self.read.id,
                **common,
            ),
        ]

    def get_executors(self):
        return [
            self.schema,
            self.catalog,
            self.source_register,
            self.column_search,
            self.column_profile,
            self.column_register,
            self.validation,
            self.read,
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="database.schema",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="schema.search_result",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="catalog.source_registered",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="schema.column_value_search_result",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="column_values.profile",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="schema.column_value_profile",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="sql.validation",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="query.result",
                owner=OWNER,
                json_schema={"type": "object"},
            ),
        ]


def test_finalization_policy_blocks_data_query_with_schema_evidence_only():
    check = _finalization_check(
        intent_kind=DbIntentKind.DATA_QUERY,
        required_evidence=("database.schema",),
        evidence=(
            Evidence(
                kind="database.schema",
                owner=OWNER,
                accepted=True,
                payload={"tables": [{"name": "orders"}]},
            ),
        ),
    )

    assert check.finalizable is False
    assert check.query_result_required is True
    assert check.query_result_present is False
    assert "query_result_missing" in check.verification.warnings


def test_finalization_policy_blocks_data_query_with_query_plan_only():
    check = _finalization_check(
        intent_kind=DbIntentKind.DATA_QUERY,
        required_evidence=("query.plan.proposal",),
        evidence=(
            Evidence(
                kind="query.plan.proposal",
                owner=OWNER,
                accepted=True,
                payload={"sql": "select count(*) as count from orders"},
            ),
        ),
    )

    assert check.finalizable is False
    assert check.query_result_required is True
    assert check.query_result_present is False
    assert "query_result_missing" in check.verification.warnings


def test_finalization_policy_accepts_data_query_with_query_result():
    sql = "select count(*) as count from orders"
    check = _finalization_check(
        intent_kind=DbIntentKind.DATA_QUERY,
        required_evidence=("query.result",),
        evidence=(
            Evidence(
                kind="sql.validation",
                owner=OWNER,
                task_id="validate",
                accepted=True,
                payload={"valid": True, "sql": sql, "operation": "query"},
            ),
            Evidence(
                kind="query.result",
                owner=OWNER,
                task_id="read",
                accepted=True,
                payload={"rows": [{"count": 3}], "total_rows": 1, "sql": sql},
            ),
        ),
        tasks=_validated_read_tasks(),
    )

    assert check.finalizable is True
    assert check.query_result_required is True
    assert check.query_result_present is True


def test_finalization_policy_ignores_result_from_older_query_plan():
    old_sql = "select count(*) as count from orders"
    new_sql = "select count(*) as count from customers"
    tasks = (
        Task(
            id="validate-old",
            operation_id="finalization-policy-target",
            capability_id="db.sql.validate",
            executor_id=f"{OWNER}.sql.validate",
            status=TaskStatus.SUCCEEDED,
        ),
        Task(
            id="read-old",
            operation_id="finalization-policy-target",
            capability_id="db.sql.execute_read",
            executor_id=f"{OWNER}.sql.execute_read",
            input={"plan_evidence_id": "plan-old"},
            status=TaskStatus.SUCCEEDED,
        ),
    )
    check = _finalization_check(
        intent_kind=DbIntentKind.DATA_QUERY,
        required_evidence=("query.result",),
        evidence=(
            Evidence(
                id="plan-old",
                kind="query.plan.proposal",
                accepted=True,
                payload={"valid": True, "sql": old_sql},
            ),
            Evidence(
                kind="sql.validation",
                task_id="validate-old",
                accepted=True,
                payload={"valid": True, "sql": old_sql, "operation": "query"},
            ),
            Evidence(
                kind="query.result",
                task_id="read-old",
                accepted=True,
                payload={"rows": [{"count": 3}], "sql": old_sql},
            ),
            Evidence(
                id="plan-new",
                kind="query.plan.proposal",
                accepted=True,
                payload={"valid": True, "sql": new_sql},
            ),
        ),
        tasks=tasks,
    )

    assert check.finalizable is False
    assert check.query_result_present is False
    assert "query_result_missing" in check.verification.warnings


def test_finalization_policy_accepts_schema_query_without_query_result():
    check = _finalization_check(
        intent_kind=DbIntentKind.SCHEMA_QUERY,
        operation_type="schema.query",
        required_evidence=("database.schema",),
        evidence=(
            Evidence(
                kind="database.schema",
                owner=OWNER,
                accepted=True,
                payload={"tables": [{"name": "orders"}]},
            ),
        ),
    )

    assert check.finalizable is True
    assert check.query_result_required is False
    assert check.query_result_present is False


def test_finalization_policy_accepts_write_execution_without_query_result():
    tasks = (
        Task(
            id="validate",
            operation_id="finalization-policy-target",
            capability_id="db.sql.validate",
            executor_id=f"{OWNER}.sql.validate",
            status=TaskStatus.SUCCEEDED,
        ),
        Task(
            id="write",
            operation_id="finalization-policy-target",
            capability_id="db.sql.execute_write",
            executor_id=f"{OWNER}.sql.execute_write",
            status=TaskStatus.SUCCEEDED,
        ),
    )
    check = _finalization_check(
        intent_kind=DbIntentKind.WRITE_EXECUTE,
        operation_type="write.execute",
        required_evidence=("sql.validation", "write.execution"),
        tasks=tasks,
        evidence=(
            Evidence(
                kind="sql.validation",
                owner=OWNER,
                task_id="validate",
                accepted=True,
                payload={
                    "valid": True,
                    "sql": "update orders set status = 'approved'",
                    "operation": "write",
                },
            ),
            Evidence(
                kind="write.execution",
                owner=OWNER,
                task_id="write",
                accepted=True,
                payload={
                    "affected_rows": 1,
                    "sql": "update orders set status = 'approved'",
                },
            ),
        ),
    )

    assert check.finalizable is True
    assert check.query_result_required is False
    assert check.query_result_present is False


def test_finalization_policy_blocks_write_execute_without_write_execution():
    check = _finalization_check(
        intent_kind=DbIntentKind.WRITE_EXECUTE,
        operation_type="write.execute",
        required_evidence=("sql.validation",),
        evidence=(
            Evidence(
                kind="sql.validation",
                owner=OWNER,
                task_id="validate",
                accepted=True,
                payload={
                    "valid": True,
                    "sql": "update orders set status = 'approved'",
                    "operation": "write",
                },
            ),
        ),
    )

    assert check.finalizable is False
    assert check.query_result_required is False
    assert "write_execution_missing" in check.verification.warnings


def test_finalization_policy_rejects_unaccepted_write_validation():
    check = _finalization_check(
        intent_kind=DbIntentKind.WRITE_EXECUTE,
        operation_type="write.execute",
        required_evidence=("sql.validation", "write.execution"),
        evidence=(
            Evidence(
                kind="sql.validation",
                owner=OWNER,
                task_id="validate",
                accepted=False,
                payload={
                    "valid": True,
                    "sql": "update orders set status = 'approved'",
                    "operation": "write",
                },
            ),
            Evidence(
                kind="write.execution",
                owner=OWNER,
                task_id="write",
                accepted=True,
                payload={
                    "affected_rows": 1,
                    "sql": "update orders set status = 'approved'",
                },
            ),
        ),
    )

    assert check.finalizable is False
    assert "sql.validation" in check.verification.missing_evidence
    assert "sql_validation_missing_for_write_proposal" in check.verification.warnings


def test_finalization_policy_ignores_answer_synthesis_as_supporting_evidence():
    check = _finalization_check(
        intent_kind=DbIntentKind.SCHEMA_QUERY,
        operation_type="schema.query",
        evidence=(
            Evidence(
                kind="answer.synthesis",
                owner=OWNER,
                accepted=True,
                payload={"answer": "There is an orders table."},
            ),
        ),
    )

    assert check.finalizable is False
    assert check.verification.passed is True
    assert check.supporting_evidence == ()


class ScriptedPlanner:
    def __init__(self, *decisions, repeat_last=False):
        self.decisions = list(decisions)
        self.repeat_last = repeat_last
        self.states = []
        self._last = decisions[-1] if decisions else None

    async def plan(self, state):
        self.states.append(state)
        if self.decisions:
            self._last = self.decisions.pop(0)
            return self._last
        if self.repeat_last and self._last is not None:
            return self._last
        raise AssertionError("planner was called after scripted decisions ended")


async def test_multi_turn_schema_to_sql_loop_persists_observation_before_second_turn():
    planner = ScriptedPlanner(
        _inspect_schema_decision(),
        _read_decision(sql="select status, count(*) as count from orders"),
    )
    runtime, plugin = await _runtime_with_planner(planner)
    executed_capabilities = []
    original_execute_task = runtime.execute_task

    async def execute_task_spy(task, operation, context=None):
        executed_capabilities.append(task.capability_id)
        return await original_execute_task(task, operation, context)

    runtime.execute_task = execute_task_spy

    result = await runtime.run("How many paid orders are there?")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert len(planner.states) == 2
    assert result.status is OperationStatus.SUCCEEDED
    assert executed_capabilities[:3] == [
        "db.schema.inspect",
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    second_state = planner.states[1]
    assert _state_has_evidence(second_state, "database.schema")
    assert _state_has_observation_evidence(second_state, "database.schema")
    assert [item["capability_id"] for item in plugin.schema.calls] == [
        "db.schema.inspect"
    ]
    assert _evidence_kinds(snapshot) >= {
        "database.schema",
        "sql.validation",
        "query.result",
        "planner.observation",
        "answer.synthesis",
    }


async def test_multi_turn_catalog_search_to_sql_loop_uses_persisted_search_evidence():
    planner = ScriptedPlanner(
        _catalog_search_decision(),
        _read_decision(
            sql="select status, count(*) as count from orders group by status",
            operation_type="data.query.catalog_assisted",
        ),
    )
    runtime, _ = await _runtime_with_planner(planner)

    result = await runtime.run("Use the catalog to count orders by status.")

    assert len(planner.states) == 2
    assert result.status is OperationStatus.SUCCEEDED
    second_state = planner.states[1]
    assert _state_has_evidence(second_state, "schema.search_result")
    assert _state_has_observation_evidence(second_state, "schema.search_result")
    snapshot = await runtime.inspect_operation(result.operation_id)
    assert _evidence_kinds(snapshot) >= {
        "schema.search_result",
        "sql.validation",
        "query.result",
        "answer.synthesis",
    }


async def test_explicit_schema_mode_keeps_loop_contract_schema_only():
    planner = ScriptedPlanner(
        _inspect_schema_decision(),
        _read_decision(sql="select status, count(*) as count from orders"),
    )
    runtime, _ = await _runtime_with_planner(planner)

    result = await runtime.run(
        DbRequest("Tell me about the orders table.", mode="schema.query")
    )
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.SUCCEEDED
    assert result.intent.kind is DbIntentKind.SCHEMA_QUERY
    assert result.contract.operation_type == "schema.query"
    assert len(planner.states) == 1
    assert "query.result" not in _evidence_kinds(snapshot)
    assert not any(
        task.capability_id == "db.sql.execute_read" for task in snapshot.tasks
    )


async def test_explicit_schema_mode_rejects_sql_read_actions():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    loop = DbAgentLoop(runtime, ScriptedPlanner())
    state = DbLoopState(
        operation_id="schema-mode-compile-target",
        normalized_user_request={"prompt": "Tell me about the orders table."},
        explicit_mode="schema.query",
        safety_frame={"max_access": AccessMode.METADATA_READ.value},
    )

    compilation = loop.compile_actions(
        _read_decision(sql="select * from orders"),
        state,
    )

    assert compilation.task_specs == ()
    assert {item["error"] for item in compilation.rejected_action_summaries} == {
        "action_outside_explicit_mode:execute_validated_read:schema.query"
    }


async def test_explicit_relationship_mode_rejects_sql_read_actions():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    loop = DbAgentLoop(runtime, ScriptedPlanner())
    state = DbLoopState(
        operation_id="relationship-mode-compile-target",
        normalized_user_request={"prompt": "Tell me how orders relate to customers."},
        explicit_mode="schema.relationship_query",
        safety_frame={"max_access": AccessMode.METADATA_READ.value},
    )

    compilation = loop.compile_actions(
        _read_decision(sql="select * from orders"),
        state,
    )

    assert compilation.task_specs == ()
    assert {item["error"] for item in compilation.rejected_action_summaries} == {
        (
            "action_outside_explicit_mode:"
            "execute_validated_read:schema.relationship_query"
        )
    }


async def test_column_value_search_contract_declares_profile_prerequisites():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(
        runtime,
        "column-value-contract-target",
    )
    loop = DbAgentLoop(runtime, ScriptedPlanner())
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
                input={"owner": OWNER},
                metadata={"target": "orders.status", "query": "completed orders"},
            ),
        ),
    )

    compilation = loop.compile_actions(decision, state)
    contract = compilation.compiled_contract_snapshot

    assert compilation.rejected_action_summaries == ()
    assert {
        "catalog.column_values.search",
        "db.schema.inspect",
        "catalog.source.register",
        "db.column_values.profile",
        "catalog.column_values.register",
    } <= set(contract["required_capabilities"])
    assert "column_values.profile" in contract["required_evidence"]
    assert contract["access"] == "read"
    prerequisites = contract["metadata"]["runtime_prerequisites"]
    prerequisite_ids = {item["capability_id"] for item in prerequisites}
    assert {
        "db.schema.inspect",
        "catalog.source.register",
        "db.column_values.profile",
        "catalog.column_values.register",
    } <= prerequisite_ids
    assert all(item["for_action_id"] == "search_values" for item in prerequisites)
    assert all(
        item["for_capability_id"] == "catalog.column_values.search"
        for item in prerequisites
    )
    assert {
        item["capability_id"]
        for item in prerequisites
        if item["reason"] == "catalog_column_value_grounding"
    } == prerequisite_ids


async def test_column_value_search_contract_infers_scope_from_simple_sql_input():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(
        runtime,
        "column-value-sql-scope-target",
    )
    loop = DbAgentLoop(runtime, ScriptedPlanner())
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
                input={
                    "owner": OWNER,
                    "sql": (
                        "SELECT DISTINCT status FROM support_tickets "
                        "WHERE status IS NOT NULL"
                    ),
                },
            ),
        ),
    )

    compilation = loop.compile_actions(decision, state)
    prerequisites = compilation.compiled_contract_snapshot["metadata"][
        "runtime_prerequisites"
    ]

    assert compilation.rejected_action_summaries == ()
    profile = next(
        item
        for item in prerequisites
        if item["capability_id"] == "db.column_values.profile"
    )
    assert profile["tables"] == ["support_tickets"]
    assert profile["columns"] == ["status"]


async def test_resume_reenters_agent_loop_after_first_turn_evidence():
    first_turn = ScriptedPlanner(_inspect_schema_decision())
    runtime, _ = await _runtime_with_planner(first_turn)
    operation = await _bootstrap_run_operation(runtime, "resume-loop-target")
    await DbAgentLoop(runtime, first_turn).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    before = await runtime.inspect_operation(operation.id)
    resume_planner = ScriptedPlanner(
        _read_decision(sql="select status, count(*) as count from orders")
    )
    select_calls = 0

    def select_planner():
        nonlocal select_calls
        select_calls += 1
        return resume_planner

    runtime._select_db_agent_planner = select_planner

    resumed = await runtime.resume_operation(operation.id)

    assert select_calls == 1
    assert len(resume_planner.states) == 1
    assert _state_has_evidence(resume_planner.states[0], "database.schema")
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert [
        task.id for task in resumed.tasks if task.capability_id == "db.schema.inspect"
    ] == [task.id for task in before.tasks if task.capability_id == "db.schema.inspect"]
    assert any(task.capability_id == "db.sql.execute_read" for task in resumed.tasks)
    assert "answer.synthesis" in _evidence_kinds(resumed)


async def test_resume_finalization_uses_latest_compiled_contract_intent():
    first_turn = ScriptedPlanner(_inspect_schema_decision())
    runtime, _ = await _runtime_with_planner(first_turn)
    operation = await _bootstrap_run_operation(runtime, "resume-intent-target")
    await DbAgentLoop(runtime, first_turn).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    runtime.host_services["db_agent_planner"] = ScriptedPlanner(
        _read_decision(sql="select status, count(*) as count from orders")
    )

    await runtime.resume_operation(operation.id)

    assert runtime.operation_results[-1].intent.kind is DbIntentKind.DATA_QUERY


async def test_resume_finalizes_sufficient_evidence_without_planner_or_llm():
    first_turn = ScriptedPlanner(
        _read_decision(sql="select status, count(*) as count from orders")
    )
    runtime, _ = await _runtime_with_planner(first_turn)
    operation = await _bootstrap_run_operation(runtime, "resume-finalizable-target")
    loop_result = await DbAgentLoop(runtime, first_turn).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    before = await runtime.inspect_operation(operation.id)
    runtime.host_services.clear()
    runtime._select_db_agent_planner = _fail_planner_selection

    resumed = await runtime.resume_operation(operation.id)

    assert loop_result.status == "finished"
    assert before.operation.status is not OperationStatus.SUCCEEDED
    assert "query.result" in _evidence_kinds(before)
    assert "answer.synthesis" not in _evidence_kinds(before)
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert {"verification.result", "answer.synthesis"} <= _evidence_kinds(resumed)
    synthesis = next(
        evidence for evidence in resumed.evidence if evidence.kind == "answer.synthesis"
    )
    assert runtime.operation_results[-1].answer == synthesis.payload["answer"]


async def test_resume_finalization_does_not_finalize_schema_only_data_query():
    first_turn = ScriptedPlanner(_inspect_schema_decision())
    runtime, _ = await _runtime_with_planner(first_turn)
    operation = await _bootstrap_run_operation(runtime, "resume-schema-only-target")
    loop_result = await DbAgentLoop(runtime, first_turn).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    snapshot = await runtime.inspect_operation(operation.id)

    finalized = await runtime._try_finalize_run_operation_from_snapshot(
        snapshot,
        request=DbRequest("completion target"),
        fallback_intent=DbIntent(
            kind=DbIntentKind.DATA_QUERY,
            access=AccessMode.READ,
            evidence_mode="test",
        ),
        fallback_contract=DbOperationContract(
            operation_type="data.query",
            required_evidence=("database.schema",),
            access=AccessMode.READ,
        ),
    )

    assert loop_result.status == "budget_exhausted"
    assert finalized is None
    assert "database.schema" in _evidence_kinds(snapshot)
    assert "query.result" not in _evidence_kinds(snapshot)
    assert "answer.synthesis" not in _evidence_kinds(snapshot)


async def test_runtime_finalization_state_uses_shared_policy_for_query_plan_only_data_query():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "facade-query-plan-only-target")
    await runtime.store.save_evidence(
        Evidence(
            kind="query.plan.proposal",
            owner=OWNER,
            operation_id=operation.id,
            accepted=True,
            payload={"sql": "select count(*) as count from orders"},
        )
    )

    finalizable, diagnostics = await runtime._run_operation_finalization_state(
        operation.id,
        fallback_intent=DbIntent(
            kind=DbIntentKind.DATA_QUERY,
            access=AccessMode.READ,
            evidence_mode="test",
        ),
        fallback_contract=DbOperationContract(
            operation_type="data.query",
            required_evidence=("query.plan.proposal",),
            access=AccessMode.READ,
        ),
    )

    assert finalizable is False
    assert diagnostics["query_result_required"] is True
    assert diagnostics["query_result_present"] is False
    assert "query_result_missing" in diagnostics["verification"]["warnings"]


async def test_resume_executes_persisted_runnable_tasks_then_finalizes(monkeypatch):
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "resume-pending-tasks-target")
    operation, tasks = await _plan_read_tasks_without_execution(
        runtime,
        operation,
        sql="select status, count(*) as count from orders",
    )
    executed = []
    original_execute_task = runtime.kernel.execute_task

    async def execute_task_spy(task_id, **kwargs):
        task = await runtime.store.load_task(task_id)
        executed.append(task.capability_id)
        return await original_execute_task(task_id, **kwargs)

    monkeypatch.setattr(runtime.kernel, "execute_task", execute_task_spy)
    runtime._select_db_agent_planner = _fail_planner_selection

    resumed = await runtime.resume_operation(operation.id)

    assert [task.status for task in tasks] == [TaskStatus.PENDING, TaskStatus.PENDING]
    assert executed == [
        "db.sql.validate",
        "db.sql.execute_read",
        "db.answer.synthesize",
    ]
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert {"sql.validation", "query.result", "answer.synthesis"} <= _evidence_kinds(
        resumed
    )


async def test_resume_skips_completed_tasks_and_does_not_replay_them():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "resume-skip-completed-target")
    operation, tasks = await _plan_read_tasks_without_execution(
        runtime,
        operation,
        sql="select status, count(*) as count from orders",
    )
    validation_task, read_task = tasks
    await runtime.execute_task(validation_task, operation)
    executed = []
    original_execute_task = runtime.execute_task

    async def execute_task_spy(task, operation, context=None):
        executed.append(task.id)
        return await original_execute_task(task, operation, context)

    runtime.execute_task = execute_task_spy
    runtime._select_db_agent_planner = _fail_planner_selection

    resumed = await runtime.resume_operation(operation.id)

    assert validation_task.id not in executed
    assert read_task.id in executed
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert [
        task.status
        for task in resumed.tasks
        if task.id in {validation_task.id, read_task.id}
    ] == [TaskStatus.SUCCEEDED, TaskStatus.SUCCEEDED]


async def test_agent_loop_finishes_without_planner_when_operation_is_finalizable():
    first_turn = ScriptedPlanner(
        _read_decision(sql="select status, count(*) as count from orders")
    )
    runtime, _ = await _runtime_with_planner(first_turn)
    operation = await _bootstrap_run_operation(runtime, "loop-finalizable-target")
    await DbAgentLoop(runtime, first_turn).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    no_planner = ScriptedPlanner()

    result = await DbAgentLoop(runtime, no_planner).run(
        operation,
        safety_frame={"max_access": "read"},
    )

    assert result.status == "finished"
    assert result.diagnostics["pre_planner_finalization"] is True
    assert no_planner.states == []


async def test_agent_loop_rejects_finish_decision_when_actions_present():
    finish_with_read = DbPlannerDecision(
        status=DbPlannerDecisionStatus.FINISH,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="read_orders",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": OWNER,
                    "sql": "select status, count(*) as count from orders",
                },
            ),
        ),
        stop_conditions=("query.result evidence exists",),
    )
    planner = ScriptedPlanner(_inspect_schema_decision(), finish_with_read)
    runtime, plugin = await _runtime_with_planner(planner)
    operation = await _bootstrap_run_operation(runtime, "finish-with-actions-target")

    result = await DbAgentLoop(runtime, planner).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=2,
    )
    tasks = await runtime.store.list_tasks(operation.id)
    evidence = await runtime.store.list_evidence(operation.id)

    assert result.status == "failed"
    assert len(planner.states) == 2
    assert plugin.read.calls == []
    assert "db.sql.validate" not in {task.capability_id for task in tasks}
    assert "query.result" not in {item.kind for item in evidence}
    compilation = next(
        item for item in reversed(evidence) if item.kind == "planner.compilation"
    )
    rejected = compilation.payload["compilation"]["rejected_action_summaries"]
    assert rejected[0]["error"] == "terminal_status_must_not_include_actions"


async def test_planner_dag_dependencies_become_durable_task_dependencies():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "dag-valid-target")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "schema.query"},
        actions=(
            DbPlannerAction(
                action_id="schema",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": OWNER},
            ),
            DbPlannerAction(
                action_id="catalog",
                kind=DbPlannerActionKind.SEARCH_SCHEMA,
                input={"owner": OWNER, "query": "orders"},
                depends_on=("schema",),
            ),
        ),
    )

    result = await DbAgentLoop(runtime, ScriptedPlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    tasks = await runtime.store.list_tasks(operation.id)
    schema_task = next(
        task for task in tasks if task.capability_id == "db.schema.inspect"
    )
    catalog_task = next(
        task for task in tasks if task.capability_id == "catalog.schema.search"
    )

    assert result.status == "finished"
    assert any(
        dependency.producer_task_id == schema_task.id
        or dependency.producer_capability_id == "db.schema.inspect"
        for dependency in catalog_task.dependencies
    )


async def test_planner_dag_missing_dependency_is_rejected_clearly():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "dag-missing-target")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="catalog",
                kind=DbPlannerActionKind.SEARCH_SCHEMA,
                input={"owner": OWNER, "query": "orders"},
                depends_on=("schema",),
            ),
        ),
    )

    result = await DbAgentLoop(runtime, ScriptedPlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    compilation = _latest_compilation(await runtime.store.list_evidence(operation.id))

    assert result.status in {"blocked", "failed"}
    assert compilation["rejected_action_summaries"]
    assert "missing_dependency:schema" in {
        item["error"] for item in compilation["rejected_action_summaries"]
    }
    assert not await runtime.store.list_tasks(operation.id)


async def test_prior_turn_action_dependency_uses_durable_task_summary():
    first_turn = ScriptedPlanner(_inspect_schema_decision())
    runtime, _ = await _runtime_with_planner(first_turn)
    operation = await _bootstrap_run_operation(runtime, "dag-prior-action-target")
    loop = DbAgentLoop(runtime, first_turn)
    await loop.run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "read"},
        turn=2,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "schema.query"},
        actions=(
            DbPlannerAction(
                action_id="catalog",
                kind=DbPlannerActionKind.SEARCH_SCHEMA,
                input={"owner": OWNER, "query": "orders"},
                depends_on=("schema",),
            ),
        ),
    )

    compilation = loop.compile_actions(decision, state)

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "catalog.schema.search"
    ]
    dependency = compilation.task_specs[0].dependencies[0]
    assert dependency.metadata["durable_prior_action"] is True
    assert dependency.metadata["producer_action_id"] == "schema"
    assert dependency.producer_task_id


async def test_planner_dag_cycle_is_rejected_clearly():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "dag-cycle-target")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="a",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": OWNER},
                depends_on=("b",),
            ),
            DbPlannerAction(
                action_id="b",
                kind=DbPlannerActionKind.SEARCH_SCHEMA,
                input={"owner": OWNER, "query": "orders"},
                depends_on=("a",),
            ),
        ),
    )

    result = await DbAgentLoop(runtime, ScriptedPlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    compilation = _latest_compilation(await runtime.store.list_evidence(operation.id))

    assert result.status in {"blocked", "failed"}
    assert compilation["rejected_action_summaries"]
    assert any(
        item["error"].startswith("dependency_cycle")
        for item in compilation["rejected_action_summaries"]
    )
    assert not await runtime.store.list_tasks(operation.id)


async def test_repeated_failing_action_stops_with_no_progress_observation():
    repeated = _read_decision(sql="select * from missing_table")
    planner = ScriptedPlanner(repeated, repeat_last=True)
    runtime, _ = await _runtime_with_planner(planner)
    operation = await _bootstrap_run_operation(runtime, "no-progress-target")

    result = await DbAgentLoop(runtime, planner).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=6,
    )
    observations = [
        evidence.payload["observation"]
        for evidence in await runtime.store.list_evidence(operation.id)
        if evidence.kind == "planner.observation"
    ]

    assert result.status in {"blocked", "failed"}
    assert len(planner.states) < 6
    assert {
        "db_agent_loop_no_progress",
        "db_agent_loop_repeated_action",
    } & set(result.warnings)
    assert any(
        observation["diagnostics"].get("status")
        in {"db_agent_loop_no_progress", "db_agent_loop_repeated_action"}
        or observation["no_progress_facts"]
        for observation in observations
    )


async def test_reused_terminal_failed_task_is_not_counted_as_progress():
    failing = _read_decision(sql="select * from missing_table")
    runtime, _ = await _runtime_with_planner(ScriptedPlanner(failing))
    operation = await _bootstrap_run_operation(runtime, "terminal-failed-target")
    await DbAgentLoop(runtime, ScriptedPlanner(failing)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )
    before = await runtime.store.list_tasks(operation.id)
    failed_validation = next(
        task
        for task in before
        if task.capability_id == "db.sql.validate" and task.status is TaskStatus.FAILED
    )
    planner = ScriptedPlanner(failing, repeat_last=True)

    result = await DbAgentLoop(runtime, planner).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=5,
    )
    after = await runtime.store.list_tasks(operation.id)

    assert result.status in {"blocked", "failed"}
    assert len(planner.states) < 5
    assert [task.id for task in after if task.capability_id == "db.sql.validate"].count(
        failed_validation.id
    ) == 1
    assert {
        "db_agent_loop_no_progress",
        "db_agent_loop_repeated_action",
    } & set(result.warnings)


async def test_repeated_sql_failure_stops_with_specific_warning():
    sql = "select * from missing_table"
    planner = ScriptedPlanner(
        _read_decision(sql=sql, action_id="read_orders_once"),
        _read_decision(sql=sql, action_id="read_orders_twice"),
        _read_decision(sql=sql, action_id="read_orders_thrice"),
        repeat_last=True,
    )
    runtime, _ = await _runtime_with_planner(planner)
    operation = await _bootstrap_run_operation(runtime, "repeated-sql-target")

    result = await DbAgentLoop(runtime, planner).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=6,
    )
    observations = [
        evidence.payload["observation"]
        for evidence in await runtime.store.list_evidence(operation.id)
        if evidence.kind == "planner.observation"
    ]

    assert result.status == "failed"
    assert len(planner.states) < 6
    assert "db_agent_loop_repeated_sql_failure" in result.warnings
    assert any(
        fact.get("warning") == "db_agent_loop_repeated_sql_failure"
        for observation in observations
        for fact in observation["retry_facts"]
    )
    assert any(
        observation["diagnostics"].get("status") == "db_agent_loop_repeated_sql_failure"
        for observation in observations
    )


async def _runtime_with_planner(planner):
    plugin = CompletionTargetPlugin()
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(plugin,)),
        host_services={"db_agent_planner": planner},
    )
    await runtime.setup()
    return runtime, plugin


async def _bootstrap_run_operation(runtime, operation_id):
    return await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="db.run",
        request={
            "prompt": "completion target",
            "source_scope": ["orders"],
            "requested_capabilities": [],
        },
        required_evidence=frozenset(),
        metadata={
            "safety_frame": {"max_access": "read"},
            "resume_context": {
                "request": {
                    "prompt": "completion target",
                    "source_scope": ["orders"],
                    "requested_capabilities": [],
                    "constraints": {},
                    "metadata": {},
                },
                "intent": {
                    "kind": "conversational",
                    "confidence": 1.0,
                    "access": "none",
                    "evidence_mode": "bootstrap",
                    "requested_outputs": [],
                    "constraints": {},
                    "diagnostics": {"source": "bootstrap"},
                },
                "contract": {
                    "operation_type": "db.run",
                    "required_capabilities": [],
                    "required_evidence": [],
                    "access": "none",
                    "limits": runtime.config.limits.to_dict(),
                    "policy_ids": [],
                    "metadata": {},
                },
            },
        },
        evaluate_governance=False,
    )


async def _plan_read_tasks_without_execution(runtime, operation, *, sql):
    loop = DbAgentLoop(runtime, ScriptedPlanner())
    decision = _read_decision(sql=sql)
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": "read"},
        turn=1,
        remaining_turns=1,
    )
    compilation = loop.compile_actions(decision, state)
    await loop._persist_compilation(operation, compilation, decision, turn=1)
    operation = await loop._persist_compiled_contract(operation, compilation)
    task_plan = await runtime.plan_task_specs(
        operation,
        compilation.task_specs,
        contract=compilation.compiled_contract_snapshot,
    )
    return operation, task_plan.tasks


def _fail_planner_selection():
    raise AssertionError("planner selection should not be required")


def _finalization_check(
    *,
    intent_kind,
    evidence,
    tasks=(),
    operation_type="db.run",
    required_evidence=(),
):
    operation = Operation(
        id="finalization-policy-target",
        operation_type=operation_type,
        request={"prompt": "completion target"},
    )
    intent = DbIntent(
        kind=intent_kind,
        access=(
            AccessMode.METADATA_READ
            if intent_kind
            in {DbIntentKind.SCHEMA_QUERY, DbIntentKind.SCHEMA_RELATIONSHIP_QUERY}
            else AccessMode.READ
        ),
        evidence_mode="test",
    )
    contract = DbOperationContract(
        operation_type=operation_type,
        required_evidence=required_evidence,
        access=intent.access,
    )
    return db_run_finalization_check(
        operation=operation,
        verifier=DbVerifier(),
        contract=contract,
        intent=intent,
        evidence=tuple(evidence),
        tasks=tuple(tasks),
    )


def _validated_read_tasks():
    return (
        Task(
            id="validate",
            operation_id="finalization-policy-target",
            capability_id="db.sql.validate",
            executor_id=f"{OWNER}.sql.validate",
            status=TaskStatus.SUCCEEDED,
        ),
        Task(
            id="read",
            operation_id="finalization-policy-target",
            capability_id="db.sql.execute_read",
            executor_id=f"{OWNER}.sql.execute_read",
            status=TaskStatus.SUCCEEDED,
        ),
    )


def _inspect_schema_decision():
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="schema",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": OWNER},
            ),
        ),
    )


def _catalog_search_decision():
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query.catalog_assisted"},
        actions=(
            DbPlannerAction(
                action_id="catalog_search",
                kind=DbPlannerActionKind.SEARCH_SCHEMA,
                input={"owner": OWNER, "query": "orders status"},
            ),
        ),
    )


def _read_decision(sql, operation_type="data.query", action_id="read_orders"):
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": operation_type},
        actions=(
            DbPlannerAction(
                action_id=action_id,
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={"owner": OWNER, "sql": sql},
            ),
        ),
    )


def _state_has_evidence(state: DbLoopState, kind: str) -> bool:
    return any(item["kind"] == kind for item in state.accepted_evidence_summaries)


def _state_has_observation_evidence(state: DbLoopState, kind: str) -> bool:
    return any(
        any(item["kind"] == kind for item in observation.accepted_evidence_summaries)
        for observation in state.planner_observations
    )


def _evidence_kinds(snapshot) -> set[str]:
    return {item.kind for item in snapshot.evidence}


def _latest_compilation(evidence):
    return [
        item.payload["compilation"]
        for item in evidence
        if item.kind == "planner.compilation"
    ][-1]
