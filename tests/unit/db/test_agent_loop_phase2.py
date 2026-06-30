import json

from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.agent_loop import DbAgentLoop
from daita.db.llm_agent_planner import DbLLMAgentPlanner
from daita.db.llm_service import DbLLMResponse, DbLLMService
from daita.db.planner_protocol import (
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, Evidence, EvidenceSchema, RiskLevel


class PhaseTwoExecutor:
    def __init__(self, executor_id, capability_ids):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)

    async def execute(self, task, operation, context):
        assert operation.metadata["latest_compiled_contract_snapshot"]
        if task.capability_id == "db.schema.inspect":
            return [
                Evidence(
                    kind="database.schema",
                    owner="phase_two",
                    payload={"tables": [{"name": "orders"}]},
                )
            ]
        if task.capability_id == "db.sql.validate":
            sql = task.input["sql"]
            return [
                Evidence(
                    kind="sql.validation",
                    owner="phase_two",
                    accepted=True,
                    payload={"valid": True, "sql": sql, "operation": "query"},
                )
            ]
        if task.capability_id == "db.sql.execute_read":
            return [
                Evidence(
                    kind="query.result",
                    owner="phase_two",
                    payload={
                        "rows": [{"answer": 1}],
                        "sql": task.input.get("sql"),
                        "validated_evidence_id": task.input.get(
                            "validated_evidence_id"
                        ),
                    },
                )
            ]
        if task.capability_id == "db.sql.execute_write":
            return [
                Evidence(
                    kind="write.execution",
                    owner="phase_two",
                    payload={"status": "executed"},
                )
            ]
        raise AssertionError(f"unexpected capability: {task.capability_id}")


class PhaseTwoPlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="phase_two",
        display_name="Phase Two",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="db.schema.inspect",
                owner="phase_two",
                description="Inspect schema.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"database.schema"}),
                executor="phase_two.schema.inspect",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.validate",
                owner="phase_two",
                description="Validate SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"sql.validation"}),
                executor="phase_two.sql.validate",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.execute_read",
                owner="phase_two",
                description="Execute read.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="phase_two.sql.execute_read",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.execute_write",
                owner="phase_two",
                description="Execute write.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                input_schema={"type": "object"},
                output_evidence=frozenset({"write.execution"}),
                executor="phase_two.sql.execute_write",
                runtime_only=True,
                side_effecting=True,
                replay_safe=False,
                idempotent=False,
            ),
        ]

    def get_executors(self):
        return [
            PhaseTwoExecutor("phase_two.schema.inspect", {"db.schema.inspect"}),
            PhaseTwoExecutor("phase_two.sql.validate", {"db.sql.validate"}),
            PhaseTwoExecutor("phase_two.sql.execute_read", {"db.sql.execute_read"}),
            PhaseTwoExecutor("phase_two.sql.execute_write", {"db.sql.execute_write"}),
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="database.schema",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="sql.validation",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="query.result",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="write.execution",
                owner="phase_two",
                json_schema={"type": "object"},
            ),
        ]


class FakePlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.decisions.pop(0)


class FakeLLMService:
    available = True

    def __init__(self, content):
        self.content = content
        self.messages = None

    async def generate_json(self, messages):
        self.messages = messages
        return DbLLMResponse(
            content=self.content,
            diagnostics={"provider": "fake", "model": "phase-two"},
        )


async def test_agent_loop_runs_schema_and_read_flow_through_task_specs():
    runtime, operation = await _runtime_and_operation("phase-two-read")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "db.run"},
        actions=(
            DbPlannerAction(
                action_id="schema",
                kind=DbPlannerActionKind.INSPECT_SCHEMA,
                input={"owner": "phase_two"},
            ),
            DbPlannerAction(
                action_id="read",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={"owner": "phase_two", "sql": "select 1 as answer"},
            ),
        ),
    )

    result = await DbAgentLoop(runtime, FakePlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )

    assert result.status == "finished"
    tasks = await runtime.store.list_tasks(operation.id)
    assert [task.capability_id for task in tasks] == [
        "db.schema.inspect",
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert [task.status.value for task in tasks] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    read_task = tasks[-1]
    assert read_task.dependencies[0].producer_task_id == tasks[1].id
    evidence = await runtime.store.list_evidence(operation.id)
    kinds = [item.kind for item in evidence]
    assert kinds.index("planner.decision") < kinds.index("database.schema")
    assert {"planner.decision", "planner.compilation", "planner.observation"} <= set(
        kinds
    )
    query_result = next(item for item in evidence if item.kind == "query.result")
    assert query_result.payload["sql"] == "select 1 as answer"


async def test_agent_loop_rejects_action_outside_contract_before_task_creation():
    runtime, operation = await _runtime_and_operation("phase-two-reject")
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "db.run"},
        actions=(
            DbPlannerAction(
                action_id="write",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
                input={
                    "owner": "phase_two",
                    "sql": "update orders set status = 'paid'",
                },
            ),
        ),
    )

    result = await DbAgentLoop(runtime, FakePlanner(decision)).run(
        operation,
        safety_frame={"max_access": "read"},
        max_turns=1,
    )

    assert result.status == "budget_exhausted"
    assert await runtime.store.list_tasks(operation.id) == []
    compilation = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.compilation"
    )
    rejected = compilation.payload["compilation"]["rejected_action_summaries"]
    assert rejected
    assert rejected[0]["error"].startswith("access_outside_contract")
    observation = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.observation"
    )
    assert observation.payload["observation"]["diagnostics"]["status"] == (
        "compilation_rejected"
    )


async def test_llm_agent_planner_emits_typed_decision_from_mocked_response():
    content = json.dumps(
        {
            "status": "continue",
            "intent": {"operation_type": "db.run"},
            "actions": [
                {
                    "action_id": "read",
                    "kind": "execute_validated_read",
                    "input": {"owner": "phase_two", "sql": "select 1"},
                    "depends_on": [],
                    "rationale": "Need one row.",
                    "metadata": {"source": "test"},
                }
            ],
            "stop_conditions": ["verified"],
            "clarification_question": None,
            "rationale": "Read from the database.",
            "metadata": {"planner": "fake"},
        }
    )
    service = FakeLLMService(content)
    state = _loop_state()

    decision = await DbLLMAgentPlanner(service).plan(state)

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    assert decision.actions[0].kind is DbPlannerActionKind.EXECUTE_VALIDATED_READ
    assert decision.metadata["planner"] == "fake"
    assert decision.metadata["llm"]["model"] == "phase-two"
    assert service.messages is not None
    request_payload = json.loads(service.messages[-1]["content"])
    assert request_payload["state"]["operation_id"] == "op-loop"


async def test_llm_agent_planner_rejects_invalid_action_kind_without_tasks():
    content = json.dumps(
        {
            "status": "continue",
            "intent": {"operation_type": "db.run"},
            "actions": [
                {
                    "action_id": "bad",
                    "kind": "made_up_action",
                    "input": {"owner": "phase_two"},
                    "depends_on": [],
                    "rationale": "Invalid action should not compile.",
                    "metadata": {},
                }
            ],
            "stop_conditions": [],
            "clarification_question": None,
            "rationale": "Malformed planner action.",
            "metadata": {},
        }
    )
    planner = DbLLMAgentPlanner(FakeLLMService(content))

    decision = await planner.plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.FAILED
    assert decision.actions == ()
    assert decision.metadata["failure"] == "planner_decision_invalid"

    runtime, operation = await _runtime_and_operation("phase-two-invalid-action")
    result = await DbAgentLoop(runtime, planner).run(operation, max_turns=1)

    assert result.status == "failed"
    assert await runtime.store.list_tasks(operation.id) == []
    decision_evidence = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.decision"
    )
    persisted = decision_evidence.payload["decision"]
    assert persisted["actions"] == []
    assert persisted["metadata"]["failure"] == "planner_decision_invalid"


async def test_missing_db_llm_configuration_returns_no_planner_actions():
    decision = await DbLLMAgentPlanner(DbLLMService(None)).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.FAILED
    assert decision.actions == ()
    assert decision.metadata["configuration_required"] is True

    runtime, operation = await _runtime_and_operation("phase-two-missing-llm")
    result = await DbAgentLoop(runtime, DbLLMAgentPlanner(DbLLMService(None))).run(
        operation,
        max_turns=1,
    )

    assert result.status == "configuration_required"
    assert await runtime.store.list_tasks(operation.id) == []
    decision_evidence = next(
        item
        for item in await runtime.store.list_evidence(operation.id)
        if item.kind == "planner.decision"
    )
    persisted = decision_evidence.payload["decision"]
    assert persisted["actions"] == []
    assert persisted["metadata"]["configuration_required"] is True


async def _runtime_and_operation(operation_id):
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(PhaseTwoPlugin(),)),
        runtime_id="phase-two-runtime",
    )
    await runtime.setup(agent_id="agent-phase-two")
    operation = await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="db.run",
        request={"prompt": "phase two", "source_scope": ["orders"]},
        required_evidence=frozenset(),
        metadata={},
        evaluate_governance=False,
    )
    return runtime, operation


def _loop_state():
    return DbLoopState(
        operation_id="op-loop",
        normalized_user_request={"prompt": "show one row"},
        safety_frame={"max_access": "read"},
        available_action_kinds=tuple(DbPlannerActionKind),
        capability_summaries=(
            {
                "id": "db.sql.execute_read",
                "owner": "phase_two",
                "access": "read",
            },
        ),
        runtime_limits={"max_tasks": 3},
        remaining_budget={"planner_turns": 1},
    )
