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
from daita.runtime import (
    AccessMode,
    ApprovalRequest,
    ApprovalStatus,
    Capability,
    Evidence,
    EvidenceSchema,
    RiskLevel,
)
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin

import pytest


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


async def test_direct_sql_compiles_validation_and_read_task_specs():
    sql = "select 1 as answer"
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "sql": sql},
    )

    compilation, _, _ = await _compile_single_action("phase-four-direct", action)

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    validation, read = compilation.task_specs
    assert validation.input == {"sql": sql, "operation": "query"}
    assert validation.dependencies == ()
    assert validation.metadata["sql_provenance"]["provenance"] == "direct"
    assert read.input["sql_ref"] == "sql.validation"


async def test_explicit_plan_evidence_id_attaches_dependency_to_validation():
    sql = "select count(*) as count from orders"
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-explicit"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-explicit-plan",
        action,
        plan_evidence=(
            {"evidence_id": "plan-explicit", "sql": sql, "task_id": "plan-task"},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation, read = compilation.task_specs
    assert validation.input == {"sql": sql, "operation": "query"}
    assert read.input["sql_ref"] == "sql.validation"
    assert len(validation.dependencies) == 1
    dependency = validation.dependencies[0]
    assert dependency.evidence_kind == "query.plan.proposal"
    assert dependency.evidence_id == "plan-explicit"
    assert dependency.evidence_owner == "phase_two"
    assert dependency.producer_task_id == "plan-task"
    assert dependency.payload_fingerprint == "fp-plan-explicit"
    provenance = validation.metadata["sql_provenance"]
    assert provenance["provenance"] == "plan_evidence_id"
    assert provenance["source_evidence_id"] == "plan-explicit"
    assert provenance["source_evidence_kind"] == "query.plan.proposal"
    assert provenance["source_evidence_owner"] == "phase_two"
    assert provenance["source_task_id"] == "plan-task"
    assert provenance["source_payload_fingerprint"] == "fp-plan-explicit"
    assert provenance["sql_fingerprint"]


async def test_latest_accepted_query_plan_ref_selects_latest_accepted_plan():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-latest-plan",
        action,
        plan_evidence=(
            {"evidence_id": "plan-old", "sql": "select 1 as answer"},
            {"evidence_id": "plan-new", "sql": "select 2 as answer"},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 2 as answer"
    assert validation.dependencies[0].evidence_id == "plan-new"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )


async def test_prior_turn_query_plan_dependency_recovers_to_latest_plan():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two"},
        depends_on=("plan_previous_turn",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-prior-turn-plan-dependency",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )


async def test_prior_turn_latest_plan_ref_dependency_recovers_to_latest_plan():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
        depends_on=("plan_previous_turn",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-prior-turn-plan-ref-dependency",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )


async def test_direct_sql_matching_latest_plan_ref_uses_plan_provenance():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "sql": "select 1 as answer;",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-direct-sql-matching-latest-plan",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"
    assert (
        validation.metadata["sql_provenance"]["provenance"]
        == "latest_accepted_query_plan"
    )


async def test_direct_sql_mismatching_latest_plan_ref_is_ambiguous():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "sql": "select 2 as answer",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-direct-sql-mismatching-latest-plan",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == "ambiguous_sql_input"


async def test_write_execute_with_direct_sql_and_plan_ref_uses_matching_validation():
    sql = "UPDATE orders SET status = 'approved' WHERE order_id = 101"
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={
            "owner": "phase_two",
            "sql": f"{sql};",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-from-validation",
        action,
        sql_validation_evidence=(
            {
                "evidence_id": "validation-write",
                "sql": sql,
                "operation": "write",
                "task_id": "validation-task",
            },
        ),
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.sql.validate",
        "db.sql.execute_write",
    ]
    validation, write = compilation.task_specs
    assert validation.input == {"sql": f"{sql};", "operation": "write"}
    assert validation.dependencies == ()
    assert write.input["sql_ref"] == "sql.validation"
    provenance = validation.metadata["sql_provenance"]
    assert provenance["provenance"] == "latest_accepted_sql_validation"
    assert provenance["source_evidence_id"] == "validation-write"
    assert provenance["source_evidence_kind"] == "sql.validation"
    assert provenance["source_task_id"] == "validation-task"


async def test_write_execute_plan_evidence_id_can_reference_sql_validation():
    sql = "UPDATE orders SET status = 'approved' WHERE order_id = 101"
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={
            "owner": "phase_two",
            "sql": sql,
            "plan_evidence_id": "validation-write",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-validation-id",
        action,
        sql_validation_evidence=(
            {
                "evidence_id": "validation-write",
                "sql": sql,
                "operation": "write",
                "task_id": "validation-task",
            },
        ),
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    validation, write = compilation.task_specs
    assert validation.input == {"sql": sql, "operation": "write"}
    assert write.input["sql_ref"] == "sql.validation"
    provenance = validation.metadata["sql_provenance"]
    assert provenance["provenance"] == "validation_evidence_id"
    assert provenance["source_evidence_id"] == "validation-write"
    assert provenance["source_evidence_kind"] == "sql.validation"


async def test_write_execute_ignores_invalid_matching_validation():
    sql = "UPDATE orders SET status = 'approved' WHERE order_id = 101"
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={
            "owner": "phase_two",
            "sql": sql,
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-invalid-validation",
        action,
        sql_validation_evidence=(
            {
                "evidence_id": "validation-invalid",
                "sql": sql,
                "operation": "write",
                "valid": False,
            },
        ),
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.task_specs == ()
    assert (
        compilation.rejected_action_summaries[0]["error"]
        == "missing_valid_sql_validation"
    )


async def test_plan_evidence_with_context_validates_plan_before_sql():
    sql = "select count(*) as count from orders"
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-explicit"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-plan-validation",
        action,
        plan_evidence=(
            {"evidence_id": "plan-explicit", "sql": sql, "task_id": "plan-task"},
        ),
        planning_context=True,
    )

    assert compilation.rejected_action_summaries == ()
    assert [spec.capability_id for spec in compilation.task_specs] == [
        "db.query.plan.validate",
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    plan_validation, validation, read = compilation.task_specs
    assert plan_validation.input == {
        "plan_evidence_id": "plan-explicit",
        "planning_context_evidence_id": "planning-context",
    }
    assert validation.input == {"sql": sql, "operation": "query"}
    assert validation.dependencies[0].evidence_kind == "query.plan.validation"
    assert validation.dependencies[0].producer_task_id == plan_validation.task_id
    assert read.input["sql_ref"] == "sql.validation"


async def test_explicit_write_execute_mode_overrides_planner_data_query_intent():
    action = DbPlannerAction(
        action_id="write",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_WRITE,
        input={"owner": "phase_two", "sql": "update orders set status = 'approved'"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-write-mode-contract",
        action,
        explicit_mode="write.execute",
        max_access="write",
    )

    assert compilation.rejected_action_summaries == ()
    assert compilation.compiled_contract_snapshot["operation_type"] == "write.execute"
    assert compilation.compiled_contract_snapshot["access"] == "write"
    assert (
        compilation.compiled_contract_snapshot["metadata"]["planner_intent"][
            "operation_type"
        ]
        == "write.execute"
    )
    assert (
        compilation.compiled_contract_snapshot["metadata"]["planner_raw_intent"][
            "operation_type"
        ]
        == "data.query"
    )


async def test_approval_state_uses_requested_policy_id():
    runtime, operation = await _runtime_and_operation("phase-two-approval-state")
    approval = ApprovalRequest(
        approval_id="approval-1",
        operation_id=operation.id,
        reason="Approve write execution.",
        risk=RiskLevel.HIGH,
        requested_by_policy_id="approval_required_for_writes",
        proposed_action={"approval": "human"},
        status=ApprovalStatus.PENDING,
    )
    await runtime.store.save_approval_request(approval)

    state = await DbAgentLoop(runtime, FakePlanner())._approval_state(operation.id)

    assert state["requests"] == [
        {
            "approval_id": "approval-1",
            "status": "pending",
            "policy_id": "approval_required_for_writes",
            "task_id": None,
        }
    ]


async def test_latest_accepted_query_plan_ref_ignores_rejected_plan_evidence():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-rejected-ignored",
        action,
        plan_evidence=(
            {"evidence_id": "plan-accepted", "sql": "select 1 as answer"},
            {
                "evidence_id": "plan-rejected",
                "sql": "select 999 as answer",
                "accepted": False,
            },
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-accepted"


async def test_latest_accepted_query_plan_ref_skips_accepted_plans_without_sql():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_accepted_query_plan",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-latest-plan-skips-nosql",
        action,
        plan_evidence=(
            {"evidence_id": "plan-with-sql", "sql": "select 1 as answer"},
            {"evidence_id": "plan-without-sql", "sql": None},
        ),
    )

    assert compilation.rejected_action_summaries == ()
    validation = compilation.task_specs[0]
    assert validation.input["sql"] == "select 1 as answer"
    assert validation.dependencies[0].evidence_id == "plan-with-sql"


async def test_rejected_plan_evidence_id_is_rejected_clearly():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-rejected"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-rejected-explicit",
        action,
        plan_evidence=(
            {
                "evidence_id": "plan-rejected",
                "sql": "select 999 as answer",
                "accepted": False,
            },
        ),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "rejected_plan_evidence:plan-rejected"
    )


async def test_ambiguous_sql_inputs_are_rejected_clearly():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "sql": "select 1 as answer",
            "plan_evidence_id": "plan-explicit",
        },
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-ambiguous-sql-input",
        action,
        plan_evidence=({"evidence_id": "plan-explicit", "sql": "select 2 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == ("ambiguous_sql_input")


async def test_ambiguous_plan_evidence_id_is_rejected_clearly():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two", "plan_evidence_id": "plan-ambiguous"},
    )
    runtime, operation = await _runtime_and_operation("phase-four-ambiguous-plan")
    state = DbLoopState(
        operation_id=operation.id,
        normalized_user_request={"prompt": "phase four"},
        safety_frame={"max_access": "read"},
        available_action_kinds=tuple(DbPlannerActionKind),
        accepted_evidence_summaries=(
            {
                "id": "plan-ambiguous",
                "kind": "query.plan.proposal",
                "owner": "phase_two",
                "accepted": True,
                "sql": "select 1 as answer",
            },
            {
                "id": "plan-ambiguous",
                "kind": "query.plan.proposal",
                "owner": "phase_two",
                "accepted": True,
                "sql": "select 2 as answer",
            },
        ),
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
    )

    compilation = DbAgentLoop(runtime, FakePlanner()).compile_actions(
        decision,
        state,
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "ambiguous_plan_evidence:plan-ambiguous"
    )


async def test_invalid_sql_reference_inputs_are_rejected_clearly():
    cases = (
        (
            "missing-plan-id",
            {"owner": "phase_two", "plan_evidence_id": ""},
            (),
            "missing_plan_evidence_id",
        ),
        (
            "unsupported-ref",
            {"owner": "phase_two", "query_plan_ref": "latest_sql"},
            (),
            "unsupported_query_plan_ref:latest_sql",
        ),
        (
            "plan-without-sql",
            {"owner": "phase_two", "plan_evidence_id": "plan-nosql"},
            ({"evidence_id": "plan-nosql", "sql": None},),
            "plan_evidence_without_sql:plan-nosql",
        ),
    )
    for suffix, action_input, plan_evidence, expected_error in cases:
        action = DbPlannerAction(
            action_id="read",
            kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
            input=action_input,
        )

        compilation, _, _ = await _compile_single_action(
            f"phase-four-{suffix}",
            action,
            plan_evidence=plan_evidence,
        )

        assert compilation.task_specs == ()
        assert compilation.rejected_action_summaries[0]["error"] == expected_error


async def test_prior_turn_depends_on_with_unsupported_ref_remains_missing_dependency_error():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={
            "owner": "phase_two",
            "query_plan_ref": "latest_sql",
        },
        depends_on=("prior_turn_plan",),
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-prior-depends-on",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == (
        "missing_dependency:prior_turn_plan"
    )


async def test_missing_sql_no_longer_falls_back_to_latest_plan_silently():
    action = DbPlannerAction(
        action_id="read",
        kind=DbPlannerActionKind.EXECUTE_VALIDATED_READ,
        input={"owner": "phase_two"},
    )

    compilation, _, _ = await _compile_single_action(
        "phase-four-missing-sql",
        action,
        plan_evidence=({"evidence_id": "plan-accepted", "sql": "select 1 as answer"},),
    )

    assert compilation.task_specs == ()
    assert compilation.rejected_action_summaries[0]["error"] == "missing_sql"


async def test_validation_grounding_repair_targets_only_validation_column():
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
    await runtime.setup(agent_id="agent-phase-two")
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
                id="schema-targeted-repair",
                kind="schema.asset_profile",
                owner="sqlite",
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
                                {"name": "channel", "data_type": "TEXT"},
                            ],
                        },
                        {
                            "name": "customers",
                            "columns": [
                                {"name": "id", "data_type": "INTEGER"},
                                {"name": "status", "data_type": "TEXT"},
                            ],
                        },
                    ],
                },
            )
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
    finally:
        await runtime.teardown()

    grounding_specs = [
        spec
        for spec in compilation.task_specs
        if spec.capability_id
        in {
            "catalog.value_grounding.plan",
            "catalog.column_value_hints.resolve",
        }
    ]
    assert grounding_specs
    targets = {
        (target["table"], target["column"])
        for spec in grounding_specs
        for target in _phase0_value_grounding_targets(spec.input)
    }
    assert ("orders", "status") in targets
    assert ("orders", "channel") not in targets
    assert ("customers", "status") not in targets


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
    content = json.dumps(_llm_planner_payload())
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


async def test_llm_agent_planner_parses_fenced_json_at_boundary():
    content = f"```json\n{json.dumps(_llm_planner_payload())}\n```"

    decision = await DbLLMAgentPlanner(FakeLLMService(content)).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    diagnostics = decision.metadata["planner_json_normalization"]
    assert any(
        step["step"] == "json_fence_stripped"
        for step in diagnostics["normalization_steps"]
    )


async def test_llm_agent_planner_unwraps_common_decision_envelopes():
    for envelope_key in ("decision", "planner_decision", "DbPlannerDecision"):
        content = json.dumps({envelope_key: _llm_planner_payload()})

        decision = await DbLLMAgentPlanner(FakeLLMService(content)).plan(_loop_state())

        assert decision.status is DbPlannerDecisionStatus.CONTINUE
        diagnostics = decision.metadata["planner_json_normalization"]
        assert diagnostics["unwrapped_envelope"] == envelope_key


async def test_llm_agent_planner_normalizes_unknown_keys_and_tuple_fields():
    payload = _llm_planner_payload(
        stop_conditions={"name": "verified"},
        unexpected_decision_key="drop me",
    )
    payload["actions"][0]["depends_on"] = [{"action_id": "schema"}]
    payload["actions"][0]["unexpected_action_key"] = "drop me too"
    content = json.dumps({"planner_decision": payload, "wrapper_note": "ignored"})

    decision = await DbLLMAgentPlanner(FakeLLMService(content)).plan(_loop_state())

    assert decision.status is DbPlannerDecisionStatus.CONTINUE
    assert decision.actions[0].depends_on == ("schema",)
    assert decision.stop_conditions == ("verified",)
    assert decision.metadata["planner"] == "fake"

    diagnostics = decision.metadata["planner_json_normalization"]
    assert diagnostics["dropped_envelope_keys"] == ["wrapper_note"]
    assert "unexpected_decision_key" in diagnostics["dropped_decision_keys"]
    assert diagnostics["dropped_action_keys"] == [
        {
            "index": 0,
            "action_id": "read",
            "keys": ["unexpected_action_key"],
        }
    ]
    assert {item["path"] for item in diagnostics["coerced_fields"]} == {
        "actions[0].depends_on",
        "stop_conditions",
    }


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


async def _runtime_and_operation(operation_id, *, mode=None):
    runtime = DbRuntime(
        config=DbRuntimeConfig(plugins=(PhaseTwoPlugin(),)),
        runtime_id="phase-two-runtime",
    )
    await runtime.setup(agent_id="agent-phase-two")
    operation = await runtime.kernel.create_operation(
        operation_id=operation_id,
        operation_type="db.run",
        request={
            "prompt": "phase two",
            "source_scope": ["orders"],
            **({"mode": mode} if mode else {}),
        },
        required_evidence=frozenset(),
        metadata=({"mode": mode} if mode else {}),
        evaluate_governance=False,
    )
    return runtime, operation


async def _compile_single_action(
    operation_id,
    action,
    *,
    plan_evidence=(),
    sql_validation_evidence=(),
    planning_context=False,
    explicit_mode=None,
    max_access="read",
):
    runtime, operation = await _runtime_and_operation(
        operation_id,
        mode=explicit_mode,
    )
    for item in plan_evidence:
        await runtime.store.save_evidence(
            _query_plan_evidence(operation.id, **dict(item))
        )
    for item in sql_validation_evidence:
        await runtime.store.save_evidence(
            _sql_validation_evidence(operation.id, **dict(item))
        )
    if planning_context:
        await runtime.store.save_evidence(_planning_context_evidence(operation.id))
    loop = DbAgentLoop(runtime, FakePlanner())
    state = await loop.build_loop_state(
        operation,
        safety_frame={"max_access": max_access},
        turn=1,
        remaining_turns=1,
    )
    decision = DbPlannerDecision(
        status=DbPlannerDecisionStatus.CONTINUE,
        intent={"operation_type": "data.query"},
        actions=(action,),
    )
    return loop.compile_actions(decision, state), runtime, operation


def _query_plan_evidence(
    operation_id,
    *,
    evidence_id,
    sql,
    accepted=True,
    task_id=None,
):
    payload = {"valid": bool(sql)}
    if sql is not None:
        payload["sql"] = sql
    return Evidence(
        id=evidence_id,
        kind="query.plan.proposal",
        owner="phase_two",
        operation_id=operation_id,
        task_id=task_id or f"task-{evidence_id}",
        accepted=accepted,
        payload=payload,
        metadata={"payload_fingerprint": f"fp-{evidence_id}"},
    )


def _sql_validation_evidence(
    operation_id,
    *,
    evidence_id,
    sql,
    operation="query",
    valid=True,
    accepted=True,
    task_id=None,
):
    return Evidence(
        id=evidence_id,
        kind="sql.validation",
        owner="phase_two",
        operation_id=operation_id,
        task_id=task_id or f"task-{evidence_id}",
        accepted=accepted,
        payload={"valid": valid, "sql": sql, "operation": operation},
        metadata={"payload_fingerprint": f"fp-{evidence_id}"},
    )


def _planning_context_evidence(operation_id):
    return Evidence(
        id="planning-context",
        kind="planning.context",
        owner="db_runtime",
        operation_id=operation_id,
        accepted=True,
        payload={
            "schema": {
                "database_type": "sqlite",
                "tables": [{"name": "orders", "columns": [{"name": "status"}]}],
            },
            "column_value_hints": [],
        },
        metadata={"payload_fingerprint": "fp-planning-context"},
    )


def _phase0_value_grounding_targets(task_input):
    targets = []
    for key in ("targets", "profile_pairs"):
        for item in task_input.get(key) or []:
            if isinstance(item, dict) and item.get("table") and item.get("column"):
                targets.append({"table": item["table"], "column": item["column"]})
    for key in ("validation_facts", "warnings", "validation_warnings"):
        for item in task_input.get(key) or []:
            if isinstance(item, dict) and item.get("table") and item.get("column"):
                targets.append({"table": item["table"], "column": item["column"]})
    return targets


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


def _llm_planner_payload(**overrides):
    payload = {
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
    payload.update(overrides)
    return payload
