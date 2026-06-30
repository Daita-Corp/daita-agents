from pathlib import Path

from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, Evidence, EvidenceSchema
from daita.runtime import OperationStatus, RiskLevel, TaskStatus


class PhaseFiveExecutor:
    def __init__(self, executor_id, capability_ids):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)
        self.calls = []

    async def execute(self, task, operation, context):
        self.calls.append(
            {
                "capability_id": task.capability_id,
                "operation_type": operation.operation_type,
                "input": dict(task.input),
            }
        )
        if task.capability_id == "db.sql.validate":
            return [
                Evidence(
                    kind="sql.validation",
                    owner="phase_five",
                    accepted=True,
                    payload={
                        "valid": True,
                        "sql": task.input["sql"],
                        "operation": "query",
                    },
                )
            ]
        if task.capability_id == "db.sql.execute_read":
            return [
                Evidence(
                    kind="query.result",
                    owner="phase_five",
                    payload={
                        "rows": [{"count": 2}],
                        "total_rows": 1,
                        "sql": task.input["sql"],
                        "validated_evidence_id": task.input["validated_evidence_id"],
                    },
                )
            ]
        raise AssertionError(f"unexpected capability: {task.capability_id}")


class PhaseFivePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="phase_five",
        display_name="Phase Five",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def __init__(self):
        self.validation = PhaseFiveExecutor(
            "phase_five.sql.validate",
            {"db.sql.validate"},
        )
        self.read = PhaseFiveExecutor(
            "phase_five.sql.execute_read",
            {"db.sql.execute_read"},
        )

    def declare_capabilities(self):
        common = {
            "domains": frozenset({"db"}),
            "operation_types": frozenset({"db.run", "data.query"}),
            "risk": RiskLevel.LOW,
            "input_schema": {"type": "object"},
            "runtime_only": True,
            "side_effecting": False,
            "replay_safe": True,
            "idempotent": True,
        }
        return [
            Capability(
                id="db.sql.validate",
                owner="phase_five",
                description="Validate SQL.",
                access=AccessMode.METADATA_READ,
                output_evidence=frozenset({"sql.validation"}),
                executor=self.validation.id,
                **common,
            ),
            Capability(
                id="db.sql.execute_read",
                owner="phase_five",
                description="Execute a read.",
                access=AccessMode.READ,
                output_evidence=frozenset({"query.result"}),
                executor=self.read.id,
                **common,
            ),
        ]

    def get_executors(self):
        return [self.validation, self.read]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="sql.validation",
                owner="phase_five",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="query.result",
                owner="phase_five",
                json_schema={"type": "object"},
            ),
        ]


class FakeLoopPlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.decisions.pop(0)


def _read_decision():
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="count_orders",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "phase_five",
                    "sql": "select count(*) as count from orders",
                },
            ),
        ),
    )


async def _runtime_with_planner(planner, plugin=None):
    plugin = plugin or PhaseFivePlugin()
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(plugin,)),
        host_services={"db_agent_planner": planner},
    )
    await runtime.setup()
    return runtime, plugin


async def test_normal_run_enters_agent_loop_and_creates_neutral_operation():
    planner = FakeLoopPlanner(_read_decision())
    runtime, _ = await _runtime_with_planner(planner)

    result = await runtime.run("How many orders are there?")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.SUCCEEDED
    assert len(planner.states) == 1
    assert snapshot.operation.operation_type == "db.run"
    assert planner.states[0].normalized_user_request["operation_type"] == "db.run"
    assert snapshot.operation.metadata["resume_context"]["request"]["prompt"] == (
        "How many orders are there?"
    )


async def test_missing_db_llm_configuration_fails_before_planner_actions():
    runtime = DbRuntime()

    result = await runtime.run("How many orders are there?")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert "DB LLM service is required" in result.answer
    assert "db_runtime_llm_configuration_required" in result.warnings
    assert snapshot.tasks == ()
    assert not [item for item in snapshot.evidence if item.kind == "planner.decision"]


async def test_loop_decision_executes_persisted_task_specs_through_execute_task():
    planner = FakeLoopPlanner(_read_decision())
    runtime, _ = await _runtime_with_planner(planner)
    original = runtime.execute_task
    executed = []

    async def spy(task, operation, context=None):
        executed.append(
            {
                "capability_id": task.capability_id,
                "persisted_before_execute": await runtime.store.load_task(task.id)
                is not None,
            }
        )
        return await original(task, operation, context)

    runtime.execute_task = spy

    result = await runtime.run("How many orders are there?")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.SUCCEEDED
    planner_task_calls = [
        item
        for item in executed
        if item["capability_id"] in {"db.sql.validate", "db.sql.execute_read"}
    ]
    assert planner_task_calls == [
        {"capability_id": "db.sql.validate", "persisted_before_execute": True},
        {"capability_id": "db.sql.execute_read", "persisted_before_execute": True},
    ]
    assert [task.capability_id for task in snapshot.tasks[:2]] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert all(task.status is TaskStatus.SUCCEEDED for task in snapshot.tasks[:2])


async def test_final_answer_is_synthesized_from_accepted_evidence():
    planner = FakeLoopPlanner(_read_decision())
    runtime, _ = await _runtime_with_planner(planner)

    result = await runtime.run("How many orders are there?")
    snapshot = await runtime.inspect_operation(result.operation_id)

    synthesis = next(
        item for item in snapshot.evidence if item.kind == "answer.synthesis"
    )
    verification = next(
        item for item in snapshot.evidence if item.kind == "verification.result"
    )
    synthesis_task = next(
        task for task in snapshot.tasks if task.capability_id == "db.answer.synthesize"
    )
    assert result.answer == synthesis.payload["answer"]
    assert synthesis.accepted
    assert verification.accepted
    assert synthesis_task.status is TaskStatus.SUCCEEDED
    assert synthesis_task.id == synthesis.task_id
    assert "loop" not in result.diagnostics
    assert result.diagnostics["planner"]["status"] == "ran_tasks"
    execution = result.diagnostics["execution"]
    assert execution["task_count"] == len(snapshot.tasks)
    assert [item["id"] for item in execution["tasks"]] == [
        task.id for task in snapshot.tasks
    ]
    assert execution["planned_sql"] == "select count(*) as count from orders"
    assert [item["id"] for item in execution["evidence_refs"]] == [
        item.id for item in result.evidence
    ]


async def test_resume_uses_persisted_tasks_without_rebuilding_completed_tasks():
    planner = FakeLoopPlanner(_read_decision())
    runtime, _ = await _runtime_with_planner(planner)

    result = await runtime.run("How many orders are there?")
    before = await runtime.inspect_operation(result.operation_id)
    resumed = await runtime.resume_operation(result.operation_id)

    assert result.status is OperationStatus.SUCCEEDED
    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert [task.id for task in resumed.tasks] == [task.id for task in before.tasks]
    assert len(planner.states) == 1


def test_phase5_source_scan_normal_run_has_no_executor_or_intent_classifier():
    facade = Path("daita/db/runtime/facade.py").read_text()
    loop = Path("daita/db/agent_loop.py").read_text()
    planner = Path("daita/db/llm_agent_planner.py").read_text()

    assert "DbOperationExecutor" not in facade
    assert "DbIntentClassifier" not in facade
    assert "DbOperationExecutor" not in loop
    assert "DbIntentClassifier" not in loop
    assert "DbOperationExecutor" not in planner
    assert "DbIntentClassifier" not in planner
    assert "_planner_route" not in facade
    assert "DbQueryPlanner" not in facade
    assert "db.query.prepare_read" not in facade
