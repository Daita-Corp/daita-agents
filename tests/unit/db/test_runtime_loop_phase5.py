from dataclasses import asdict
import json
from pathlib import Path

import pytest

from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.models import (
    DbIntent,
    DbIntentKind,
    DbLimits,
    DbOperationContract,
    DbOperationResult,
    DbRequest,
)
from daita.db.planner_protocol import (
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
)
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, Evidence, EvidenceSchema
from daita.runtime import OperationStatus, RiskLevel, Task, TaskStatus


class PhaseFiveExecutor:
    def __init__(self, executor_id, capability_ids, *, error=None):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)
        self.error = error
        self.calls = []

    async def execute(self, task, operation, context):
        self.calls.append(
            {
                "capability_id": task.capability_id,
                "operation_type": operation.operation_type,
                "input": dict(task.input),
            }
        )
        if self.error is not None:
            raise RuntimeError(self.error)
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

    def __init__(self, *, read_error=None):
        self.validation = PhaseFiveExecutor(
            "phase_five.sql.validate",
            {"db.sql.validate"},
        )
        self.read = PhaseFiveExecutor(
            "phase_five.sql.execute_read",
            {"db.sql.execute_read"},
            error=read_error,
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


def _read_decision(*, sql="select count(*) as count from orders"):
    return DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(
            DbPlannerAction(
                action_id="count_orders",
                kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
                input={
                    "owner": "phase_five",
                    "sql": sql,
                },
            ),
        ),
    )


async def _runtime_with_planner(planner, plugin=None, *, limits=None):
    plugin = plugin or PhaseFivePlugin()
    runtime = DbRuntime(
        config=DbRuntimeConfig(
            plugins=(plugin,),
            limits=limits or DbLimits(),
        ),
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
    assert synthesis.payload["answer_facts"]["primary_scalar"]["value"] == 2
    assert (
        synthesis.payload["answer_facts"]["primary_scalar"]["aggregate_kind"] == "count"
    )
    assert synthesis.accepted
    assert verification.accepted
    assert synthesis_task.status is TaskStatus.SUCCEEDED
    assert synthesis_task.id == synthesis.task_id
    assert "loop" not in result.diagnostics
    assert result.diagnostics["planner"]["status"] == "finished"
    execution = result.diagnostics["execution"]
    assert execution["task_count"] == len(snapshot.tasks)
    assert [item["id"] for item in execution["task_refs"]] == [
        task.id for task in snapshot.tasks
    ]
    assert all(
        set(item) == {"id", "capability_id", "status"}
        for item in execution["task_refs"]
    )
    assert "planned_sql" not in execution
    assert [item["id"] for item in execution["evidence_refs"]] == [
        item.id for item in result.evidence
    ]


@pytest.mark.parametrize("terminal_status", ("budget_exhausted", "blocked"))
async def test_real_non_final_loop_result_projects_persisted_execution_refs(
    terminal_status,
):
    canary = "NON_FINAL_CANARY::task-input-and-error"
    decisions = [_read_decision(sql=f"select '{canary}'")]
    max_turns = 1
    if terminal_status == "blocked":
        decisions.append(DbPlannerDecision(status=DbPlannerDecisionStatus.BLOCKED))
        max_turns = 2
    planner = FakeLoopPlanner(*decisions)
    runtime, _ = await _runtime_with_planner(
        planner,
        PhaseFivePlugin(read_error=f"read failed for {canary}"),
        limits=DbLimits(max_tasks=max_turns),
    )

    result = await runtime.run("Run a safe non-final query.")
    snapshot = await runtime.inspect_operation(result.operation_id)

    assert result.status is OperationStatus.BLOCKED
    assert result.diagnostics["planner"]["status"] == terminal_status
    execution = result.diagnostics["execution"]
    assert execution["operation_id"] == result.operation_id
    assert execution["task_count"] == len(snapshot.tasks)
    assert execution["evidence_count"] == len(result.evidence)
    assert [item["id"] for item in execution["task_refs"]] == [
        task.id for task in snapshot.tasks
    ]
    assert all(
        set(item) == {"id", "capability_id", "status"}
        for item in execution["task_refs"]
    )
    assert canary not in json.dumps(asdict(result), default=str, sort_keys=True)
    assert any(canary in str(task.input.get("sql")) for task in snapshot.tasks)


_PUBLIC_RESULT_CANARY = "PACKAGE_A_CANARY::9f4d7e2b@secret"


@pytest.mark.parametrize(
    ("status", "planner_status"),
    (
        (OperationStatus.SUCCEEDED, "finished"),
        (OperationStatus.FAILED, "failed"),
        (OperationStatus.BLOCKED, "blocked"),
        (OperationStatus.BLOCKED, "clarification_required"),
        (OperationStatus.BLOCKED, "budget_exhausted"),
    ),
    ids=(
        "succeeded",
        "failed",
        "blocked",
        "clarification-required",
        "budget-exhausted",
    ),
)
async def test_every_terminal_result_uses_whole_result_public_projection(
    status,
    planner_status,
):
    runtime = DbRuntime()
    operation = await runtime.kernel.create_operation(
        operation_type="db.run",
        request={"prompt": "Safe public prompt"},
        metadata={
            "safety_frame": {},
            "trace": {
                "trace_id": "trace-safe",
                "root_span_id": "span-safe",
                "span_name": _PUBLIC_RESULT_CANARY,
            },
        },
        evaluate_governance=False,
    )
    task = Task(
        id=f"{operation.id}.query",
        operation_id=operation.id,
        capability_id="db.sql.execute_read",
        executor_id="phase_five.sql.execute_read",
        input={
            "sql": f"select '{_PUBLIC_RESULT_CANARY}'",
            "parameters": {"account": _PUBLIC_RESULT_CANARY},
            "task_input": _PUBLIC_RESULT_CANARY,
        },
        status=TaskStatus.SUCCEEDED,
        metadata={"task_metadata": _PUBLIC_RESULT_CANARY},
    )
    evidence = (
        Evidence(
            id=f"{operation.id}.query-result",
            kind="query.result",
            owner="phase_five",
            operation_id=operation.id,
            task_id=task.id,
            payload={
                "rows": [{"secret_value": _PUBLIC_RESULT_CANARY}],
                "sql": f"select '{_PUBLIC_RESULT_CANARY}'",
                "parameters": {"account": _PUBLIC_RESULT_CANARY},
                "total_rows": 1,
                "success": True,
            },
        ),
        Evidence(
            id=f"{operation.id}.validation",
            kind="sql.validation",
            owner="phase_five",
            operation_id=operation.id,
            task_id=task.id,
            payload={
                "valid": False,
                "operation": "query",
                "validation_errors": [f"validation rejected {_PUBLIC_RESULT_CANARY}"],
            },
        ),
        Evidence(
            id=f"{operation.id}.planner-decision",
            kind="planner.decision",
            owner="db_runtime",
            operation_id=operation.id,
            payload={
                "status": planner_status,
                "actions": [{"input": {"planner_action_input": _PUBLIC_RESULT_CANARY}}],
                "task_plan": {
                    "tasks": [
                        {
                            "input": {"sql": _PUBLIC_RESULT_CANARY},
                            "metadata": {"secret": _PUBLIC_RESULT_CANARY},
                        }
                    ]
                },
                "warnings": [f"warning text {_PUBLIC_RESULT_CANARY}"],
            },
        ),
        Evidence(
            id=f"{operation.id}.planning-context",
            kind="planning.context",
            owner="db_runtime",
            operation_id=operation.id,
            payload={"memory_text": _PUBLIC_RESULT_CANARY},
        ),
        Evidence(
            id=f"{operation.id}.synthesis",
            kind="answer.synthesis",
            owner="db_runtime",
            operation_id=operation.id,
            payload={
                "answer": "Safe synthesized answer.",
                "diagnostics": {
                    "mode": "llm",
                    "provider": "safe-provider",
                    "model": "safe-model",
                    "input_tokens": 8,
                    "output_tokens": 5,
                    "total_tokens": 13,
                    "estimated_cost_usd": 0.02,
                    "latency_ms": 7.5,
                    "private_diagnostic": _PUBLIC_RESULT_CANARY,
                },
            },
        ),
    )
    await runtime.store.save_task(task)
    for item in evidence:
        await runtime.store.save_evidence(item)

    raw_result = DbOperationResult(
        operation_id=operation.id,
        request=DbRequest("Safe public prompt"),
        intent=DbIntent(kind=DbIntentKind.DATA_QUERY, access=AccessMode.READ),
        contract=DbOperationContract(
            operation_type="db.run",
            access=AccessMode.READ,
        ),
        status=status,
        answer="Safe synthesized answer.",
        evidence=evidence,
        warnings=(
            "db_runtime_safe_code",
            f"warning text {_PUBLIC_RESULT_CANARY}",
        ),
        diagnostics={
            "planner": {
                "status": planner_status,
                "warnings": [f"planner warning {_PUBLIC_RESULT_CANARY}"],
                "diagnostics": {
                    "turn": 3,
                    "actions": [{"input": {"secret": _PUBLIC_RESULT_CANARY}}],
                    "task_plan": {"tasks": [{"input": {"sql": _PUBLIC_RESULT_CANARY}}]},
                },
            },
            "execution": {
                "operation_id": _PUBLIC_RESULT_CANARY,
                "task_count": 1,
                "evidence_count": len(evidence),
                "tasks": [task.to_dict()],
                "evidence_refs": [
                    {
                        "id": item.id,
                        "kind": item.kind,
                        "task_id": item.task_id,
                        "accepted": item.accepted,
                    }
                    for item in evidence
                ],
                "planned_sql": f"select '{_PUBLIC_RESULT_CANARY}'",
            },
            "verification": {
                "passed": status is OperationStatus.SUCCEEDED,
                "missing_evidence": ["verification.result"],
                "warnings": [f"validation warning {_PUBLIC_RESULT_CANARY}"],
                "diagnostics": {
                    "required_evidence": ["query.result", "verification.result"],
                    "validation_errors": [_PUBLIC_RESULT_CANARY],
                },
            },
            "synthesis": {
                "diagnostics": {
                    "provider": "safe-provider",
                    "model": "safe-model",
                    "private_diagnostic": _PUBLIC_RESULT_CANARY,
                }
            },
            "trace": {
                "trace_id": "trace-safe",
                "root_span_id": "span-safe",
                "span_name": _PUBLIC_RESULT_CANARY,
            },
            "error": {
                "type": "ValidationError",
                "code": "db_runtime_validation_failed",
                "message": f"validation failed for {_PUBLIC_RESULT_CANARY}",
            },
            "memory_text": _PUBLIC_RESULT_CANARY,
        },
    )

    result = await runtime._record_operation_result(raw_result, operation=operation)
    snapshot = await runtime.inspect_operation(operation.id)

    serialized_result = json.dumps(asdict(result), default=str, sort_keys=True)
    assert _PUBLIC_RESULT_CANARY not in serialized_result
    assert result is runtime.operation_results[-1]
    assert _PUBLIC_RESULT_CANARY not in json.dumps(runtime.audit_log, sort_keys=True)
    assert all(
        item.metadata["projection_mode"] == "public_result"
        and item.payload["projection_mode"] == "public_result"
        for item in result.evidence
    )
    task_refs = result.diagnostics["execution"]["task_refs"]
    assert task_refs == [
        {
            "id": task.id,
            "capability_id": task.capability_id,
            "status": task.status.value,
        }
    ]
    assert all("input" not in item and "metadata" not in item for item in task_refs)
    assert set(result.diagnostics["planner"]) == {
        "status",
        "turn_count",
        "warning_codes",
        "terminal_reason_code",
    }
    assert result.diagnostics["planner"]["turn_count"] == 3
    assert result.diagnostics["error"] == {
        "type": "ValidationError",
        "code": "db_runtime_validation_failed",
    }
    assert result.telemetry == {
        "provider": "safe-provider",
        "model": "safe-model",
        "input_tokens": 8,
        "output_tokens": 5,
        "total_tokens": 13,
        "llm_calls": 1,
        "estimated_cost_usd": 0.02,
        "latency_ms": 7.5,
        "mode": "llm",
    }
    assert result.diagnostics["synthesis"] == result.telemetry

    raw_snapshot = json.dumps(
        {
            "operation": snapshot.operation.to_dict(),
            "tasks": [item.to_dict() for item in snapshot.tasks],
            "evidence": [item.to_dict() for item in snapshot.evidence],
            "events": [item.to_dict() for item in snapshot.events],
        },
        default=str,
        sort_keys=True,
    )
    assert _PUBLIC_RESULT_CANARY in raw_snapshot
    assert snapshot.events[-1].payload["warnings"][-1].endswith(_PUBLIC_RESULT_CANARY)
    assert snapshot.tasks[0].input["task_input"] == _PUBLIC_RESULT_CANARY
    assert snapshot.tasks[0].metadata["task_metadata"] == _PUBLIC_RESULT_CANARY
    raw_by_kind = {item.kind: item for item in snapshot.evidence}
    assert raw_by_kind["query.result"].payload["rows"][0]["secret_value"] == (
        _PUBLIC_RESULT_CANARY
    )
    assert raw_by_kind["query.result"].payload["parameters"]["account"] == (
        _PUBLIC_RESULT_CANARY
    )
    assert _PUBLIC_RESULT_CANARY in raw_by_kind["query.result"].payload["sql"]
    assert (
        raw_by_kind["planner.decision"].payload["actions"][0]["input"][
            "planner_action_input"
        ]
        == _PUBLIC_RESULT_CANARY
    )
    assert (
        raw_by_kind["planner.decision"].payload["task_plan"]["tasks"][0]["input"]["sql"]
        == _PUBLIC_RESULT_CANARY
    )
    assert (
        raw_by_kind["planner.decision"]
        .payload["warnings"][0]
        .endswith(_PUBLIC_RESULT_CANARY)
    )
    assert (
        raw_by_kind["sql.validation"]
        .payload["validation_errors"][0]
        .endswith(_PUBLIC_RESULT_CANARY)
    )
    assert raw_by_kind["planning.context"].payload["memory_text"] == (
        _PUBLIC_RESULT_CANARY
    )
    assert (
        raw_by_kind["answer.synthesis"].payload["diagnostics"]["private_diagnostic"]
        == _PUBLIC_RESULT_CANARY
    )


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
    loop = Path("daita/db/loop/runner.py").read_text()
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
