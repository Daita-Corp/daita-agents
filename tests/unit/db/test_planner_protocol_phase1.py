import json
from pathlib import Path

import pytest

from daita.db import DbRuntime, DbRuntimeConfig
from daita.db.planner_protocol import (
    DbAgentPlanner,
    DbLoopState,
    DbPlannerAction,
    DbPlannerActionKind,
    DbPlannerDecision,
    DbPlannerDecisionStatus,
    DbPlannerObservation,
)
from daita.db.runtime.extensions.plugin import DbRuntimePlanningPlugin
from daita.db.runtime.tasks import DbTaskSpec
from daita.plugins import PluginKind, PluginManifest, RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, EvidenceSchema, RiskLevel, Task


class PhaseOneExecutor:
    def __init__(self, executor_id, capability_ids):
        self.id = executor_id
        self.capability_ids = frozenset(capability_ids)

    async def execute(self, task, operation, context):
        raise AssertionError("phase 1 task materialization tests must not execute")


class PhaseOnePlugin(RuntimeExtensionPlugin):
    manifest = PluginManifest(
        id="phase_one",
        display_name="Phase One",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"db"}),
    )

    def declare_capabilities(self):
        return [
            Capability(
                id="phase.one.generic",
                owner="phase_one",
                description="Generic phase one task.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"phase.one.result"}),
                executor="phase_one.generic",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.validate",
                owner="phase_one",
                description="Validate SQL.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.METADATA_READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"sql.validation"}),
                executor="phase_one.sql.validate",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.execute_read",
                owner="phase_one",
                description="Execute validated read.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.READ,
                risk=RiskLevel.LOW,
                input_schema={"type": "object"},
                output_evidence=frozenset({"query.result"}),
                executor="phase_one.sql.execute_read",
                runtime_only=True,
                side_effecting=False,
                replay_safe=True,
                idempotent=True,
            ),
            Capability(
                id="db.sql.execute_write",
                owner="phase_one",
                description="Execute validated write.",
                domains=frozenset({"db"}),
                operation_types=frozenset({"db.run"}),
                access=AccessMode.WRITE,
                risk=RiskLevel.HIGH,
                input_schema={"type": "object"},
                output_evidence=frozenset({"write.execution"}),
                executor="phase_one.sql.execute_write",
                runtime_only=True,
                side_effecting=True,
                replay_safe=False,
                idempotent=False,
            ),
        ]

    def get_executors(self):
        return [
            PhaseOneExecutor("phase_one.generic", {"phase.one.generic"}),
            PhaseOneExecutor("phase_one.sql.validate", {"db.sql.validate"}),
            PhaseOneExecutor("phase_one.sql.execute_read", {"db.sql.execute_read"}),
            PhaseOneExecutor("phase_one.sql.execute_write", {"db.sql.execute_write"}),
        ]

    def declare_evidence_schemas(self):
        return [
            EvidenceSchema(
                kind="phase.one.result",
                owner="phase_one",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="sql.validation",
                owner="phase_one",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="query.result",
                owner="phase_one",
                json_schema={"type": "object"},
            ),
            EvidenceSchema(
                kind="write.execution",
                owner="phase_one",
                json_schema={"type": "object"},
            ),
        ]


def test_planner_protocol_records_serialize_cleanly():
    action = DbPlannerAction(
        action_id="a1",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"sql": "select 1"},
        depends_on=("a0",),
        rationale="Need rows.",
        metadata={"source": "test"},
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
        stop_conditions=("verified",),
        metadata={"decision_id": "d1"},
    )
    observation = DbPlannerObservation(
        accepted_evidence_summaries=({"kind": "query.result"},),
        task_statuses=({"task_id": "t1", "status": "succeeded"},),
    )
    state = DbLoopState(
        operation_id="op-1",
        normalized_user_request={"prompt": "show orders"},
        source_scope=("orders",),
        explicit_requested_capabilities=("db.sql.execute_read",),
        safety_frame={"max_access": "read"},
        available_action_kinds=(DbPlannerActionKind.EXECUTE_VALIDATED_READ,),
        capability_summaries=({"id": "db.sql.execute_read"},),
        planner_observations=(observation,),
        runtime_limits={"max_tasks": 5},
        remaining_budget={"planner_turns": 2},
    )

    for payload in (
        action.to_dict(),
        decision.to_dict(),
        observation.to_dict(),
        state.to_dict(),
    ):
        json.dumps(payload)

    assert DbPlannerDecision.from_dict(decision.to_dict()) == decision
    assert DbLoopState.from_dict(state.to_dict()) == state


def test_db_agent_planner_requires_plan_implementation():
    class MissingPlan(DbAgentPlanner):
        pass

    with pytest.raises(TypeError, match="abstract class"):
        MissingPlan()


def test_planner_decision_status_and_action_kind_values_are_stable():
    assert [status.value for status in DbPlannerDecisionStatus] == [
        "continue",
        "finish",
        "clarify",
        "blocked",
        "failed",
    ]
    assert [kind.value for kind in DbPlannerActionKind] == [
        "inspect_schema",
        "register_catalog_source",
        "search_schema",
        "inspect_asset",
        "find_relationship_paths",
        "search_column_values",
        "build_planning_context",
        "propose_sql_read",
        "repair_query_plan",
        "execute_validated_read",
        "propose_sql_write",
        "execute_validated_write",
        "recall_memory",
        "plan_memory_update",
        "commit_memory_update",
        "plan_analysis",
        "execute_analysis_step",
        "summarize_analysis",
        "plan_monitor_create",
        "commit_monitor_create",
        "plan_monitor_lifecycle",
        "commit_monitor_lifecycle",
        "read_monitor_state",
        "resolve_monitor_approval",
        "synthesize",
        "clarify",
        "finish",
    ]


async def test_task_specs_materialize_generic_tasks():
    runtime, operation = await _runtime_and_operation("op-generic")
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="phase.one.generic",
                owner="phase_one",
                input={"value": 1},
                reason="phase_1_test",
                sequence=3,
                metadata={"planner_action_id": "a1"},
            ),
        ),
    )

    task = plan.tasks[0]
    assert task.id.startswith("db-task-")
    assert task.capability_id == "phase.one.generic"
    assert task.executor_id == "phase_one.generic"
    assert task.input["input_hash"] == task.metadata["input_hash"]
    assert task.metadata["owner"] == "phase_one"
    assert task.metadata["planner_action_id"] == "a1"
    assert task.metadata["idempotent"] is True
    assert task.metadata["replay_safe"] is True
    assert task.metadata["side_effecting"] is False
    assert await runtime.store.load_task(task.id) == task


async def test_task_specs_create_validation_before_execute_read():
    runtime, operation = await _runtime_and_operation("op-read")
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="db.sql.validate",
                owner="phase_one",
                input={"sql": "select * from orders", "operation": "query"},
                reason="phase_1_read_validation",
                sequence=1,
            ),
            DbTaskSpec(
                capability_id="db.sql.execute_read",
                owner="phase_one",
                input={"sql_ref": "sql.validation", "params": [1]},
                reason="phase_1_read",
                sequence=2,
            ),
        ),
    )

    validation_task, read_task = plan.tasks
    assert validation_task.capability_id == "db.sql.validate"
    assert read_task.capability_id == "db.sql.execute_read"
    dependency = read_task.dependencies[0]
    assert dependency.evidence_kind == "sql.validation"
    assert dependency.producer_task_id == validation_task.id
    assert dependency.input_hash == validation_task.metadata["input_hash"]
    assert read_task.input["sql_ref"] == "sql.validation"


async def test_task_specs_create_validation_and_approval_dependencies_for_write():
    runtime, operation = await _runtime_and_operation("op-write")
    plan = await runtime.plan_task_specs(
        operation,
        (
            DbTaskSpec(
                capability_id="db.sql.validate",
                owner="phase_one",
                input={
                    "sql": "update orders set status = 'paid'",
                    "operation": operation.operation_type,
                },
                reason="phase_1_write_validation",
                sequence=1,
            ),
            DbTaskSpec(
                capability_id="db.sql.execute_write",
                owner="phase_one",
                input={"sql_ref": "sql.validation", "params": []},
                reason="phase_1_write",
                sequence=2,
            ),
        ),
    )

    validation_task, write_task = plan.tasks
    assert validation_task.capability_id == "db.sql.validate"
    assert write_task.capability_id == "db.sql.execute_write"
    assert {dependency.kind.value for dependency in write_task.dependencies} == {
        "evidence",
        "approval",
    }
    approval = next(
        dependency
        for dependency in write_task.dependencies
        if dependency.kind.value == "approval"
    )
    assert approval.approval_policy_id == "approval_required_for_writes"
    assert approval.approval_name == "human"
    assert write_task.metadata["side_effecting"] is True
    assert write_task.metadata["idempotent"] is False


async def test_task_spec_materialization_is_deterministic_for_same_operation_input():
    runtime, operation = await _runtime_and_operation("op-idempotent")
    spec = DbTaskSpec(
        capability_id="phase.one.generic",
        owner="phase_one",
        input={"value": 7},
        reason="same_input",
        sequence=1,
    )

    first = await runtime.plan_task_specs(operation, (spec,))
    second = await runtime.plan_task_specs(operation, (spec,))

    assert second.tasks[0].id == first.tasks[0].id
    assert second.tasks[0].metadata["idempotency_key"] == (
        first.tasks[0].metadata["idempotency_key"]
    )
    assert second.diagnostics["reused_task_count"] == 1
    assert len(await runtime.store.list_tasks(operation.id)) == 1


def test_planner_evidence_schemas_are_declared_by_runtime_extension_plugin():
    kinds = {
        schema.kind for schema in DbRuntimePlanningPlugin().declare_evidence_schemas()
    }

    assert {
        "planner.decision",
        "planner.observation",
        "planner.compilation",
    }.issubset(kinds)


def test_new_protocol_and_loop_code_do_not_construct_raw_tasks_outside_task_owner():
    repo = Path(__file__).parents[3]
    checked = [
        repo / "daita/db/planner_protocol.py",
        repo / "daita/db/agent_loop.py",
        repo / "daita/db/llm_agent_planner.py",
    ]

    for path in checked:
        if path.exists():
            assert "Task(" not in path.read_text()


async def _runtime_and_operation(operation_id):
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(PhaseOnePlugin(),)),
        runtime_id="phase-one-runtime",
    )
    await runtime.setup(agent_id="agent-phase-one")
    operation = await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="db.run",
        request={"prompt": "phase one"},
        required_evidence=frozenset(),
        metadata={},
        evaluate_governance=False,
    )
    return runtime, operation
