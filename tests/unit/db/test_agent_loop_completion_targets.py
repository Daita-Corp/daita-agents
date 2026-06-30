from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.agent_loop import DbAgentLoop
from daita.db.models import DbIntentKind
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
    OperationStatus,
    RiskLevel,
    TaskStatus,
)


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
                {"db.run", "data.query", "data.query.catalog_assisted"}
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
        return [self.schema, self.catalog, self.validation, self.read]

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
    runtime.host_services["db_agent_planner"] = resume_planner

    resumed = await runtime.resume_operation(operation.id)

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


async def test_planner_dag_dependencies_become_durable_task_dependencies():
    runtime, _ = await _runtime_with_planner(ScriptedPlanner())
    operation = await _bootstrap_run_operation(runtime, "dag-valid-target")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
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


def _read_decision(sql, operation_type="data.query"):
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": operation_type},
        actions=(
            DbPlannerAction(
                action_id="read_orders",
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
