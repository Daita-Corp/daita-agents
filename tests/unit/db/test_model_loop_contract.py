from dataclasses import replace

import pytest

from daita.db.agent_loop import DbAgentLoop, DbAgentLoopBlocked
from daita.db.contracts import DbContractBuilder
from daita.db.fallback_planner import ContractFallbackDbAgentPlanner
from daita.db.models import DbRequest, DbRuntimeConfig
from daita.db.planner_protocol import DbPlannerAction, DbPlannerDecision
from daita.db.runtime import resume as resume_module
from daita.db.runtime import DbRuntime
from daita.db.runtime.tasks import DbRuntimeTasksMixin
from daita.db.safety import DbSafetyVerifier
from daita.plugins.catalog import CatalogPlugin
from daita.plugins.sqlite import SQLitePlugin
from daita.runtime import (
    Evidence,
    InMemoryRuntimeStore,
    Operation,
    OperationStatus,
    Task,
)


class MockPlanner:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, observations):
        self.contexts.append((context, observations))
        if self.decisions:
            return self.decisions.pop(0)
        return DbPlannerDecision(
            actions=(DbPlannerAction(kind="finish", payload={"message": "done"}),)
        )


class RecordingRuntime(DbRuntimeTasksMixin):
    def __init__(self, registry):
        self.registry = registry
        self.store = InMemoryRuntimeStore()
        self.executed_tasks = []
        self.materialized_specs = []

    def materialize_task_specs(self, operation, specs):
        self.materialized_specs.append(specs)
        return super().materialize_task_specs(operation, specs)

    async def execute_task(self, task: Task, operation: Operation, context=None):
        await self.store.save_task(task)
        self.executed_tasks.append((task, context or {}))
        return ()


def _contract(prompt, *, requested_capabilities=(), plugins=()):
    request = DbRequest(prompt, requested_capabilities=tuple(requested_capabilities))
    frame = DbSafetyVerifier().verify(request)
    runtime = DbRuntime(plugins=(SQLitePlugin(path=":memory:"), *plugins))
    contract = DbContractBuilder(runtime.registry, DbRuntimeConfig()).build(
        request,
        frame,
    )
    operation = Operation(
        id="op-model-loop",
        operation_type=contract.operation_type,
        request={"prompt": request.prompt},
        required_evidence=frozenset(contract.required_evidence),
        metadata={"contract": contract.metadata},
    )
    return request, runtime.registry, contract, operation


async def _prepared_loop(prompt, *decisions, requested_capabilities=(), plugins=()):
    request, registry, contract, operation = _contract(
        prompt,
        requested_capabilities=requested_capabilities,
        plugins=plugins,
    )
    runtime = RecordingRuntime(registry)
    await runtime.store.save_operation(operation)
    loop = DbAgentLoop(
        runtime,
        MockPlanner(*decisions),
        max_steps=2,
    )
    return runtime, loop, request, operation, contract


async def _run(prompt, *decisions, requested_capabilities=(), plugins=()):
    runtime, loop, request, operation, contract = await _prepared_loop(
        prompt,
        *decisions,
        requested_capabilities=requested_capabilities,
        plugins=plugins,
    )
    result = await loop.run(
        request=request,
        operation=operation,
        contract=contract,
    )
    return runtime, result


def _fallback_decision(prompt, lanes, *, evidence=(), available=()):
    planner = ContractFallbackDbAgentPlanner(
        {
            "from_db_options": {
                "catalog_store_id": "runtime_source",
                "memory": {"limit": 2, "score_threshold": 0.5},
            }
        }
    )
    return planner.decide(
        {
            "request": {"prompt": prompt},
            "contract": {"granted_lanes": list(lanes)},
            "evidence_observations": list(evidence),
            "available_capabilities": [{"id": capability} for capability in available],
        },
        (),
    )


def _action_kind(decision):
    return decision.actions[0].kind


def test_fallback_planner_emits_expected_schema_read_memory_and_write_actions():
    schema = {
        "kind": "schema.asset_profile",
        "id": "schema-1",
        "accepted": True,
        "payload": {"tables": [{"name": "orders"}]},
    }
    validation = {
        "kind": "query.plan.validation",
        "accepted": True,
        "payload": {"valid": True, "sql": "select count(*) from orders"},
    }

    assert (
        _action_kind(
            _fallback_decision(
                "what columns are in orders",
                ("schema",),
                available=("db.schema.inspect",),
            )
        )
        == "inspect_schema"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "register the schema",
                ("schema",),
                evidence=(schema,),
                available=("catalog.source.register",),
            )
        )
        == "register_catalog_source"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "how many orders are there",
                ("read",),
                evidence=(schema,),
                available=("db.query.prepare_read",),
            )
        )
        == "propose_sql_read"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "how many orders are there",
                ("read",),
                evidence=(schema, validation),
                available=("db.sql.execute_read",),
            )
        )
        == "execute_validated_read"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "what did we learn about orders",
                ("memory_answer",),
            )
        )
        == "recall_memory"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "remember that orders joins customers by customer_id",
                ("memory_write",),
            )
        )
        == "write_memory"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "update orders set status = 'closed'",
                ("write_propose",),
            )
        )
        == "propose_sql_write"
    )
    assert (
        _action_kind(
            _fallback_decision(
                "update orders set status = 'closed'",
                ("write_execute",),
            )
        )
        == "execute_validated_write"
    )


async def test_fallback_planner_uses_same_loop_task_spec_path():
    request, registry, contract, operation = _contract("how many orders are there")
    runtime = RecordingRuntime(registry)
    await runtime.store.save_operation(operation)
    await runtime.store.save_evidence(
        Evidence(
            kind="schema.asset_profile",
            id="schema-1",
            operation_id=operation.id,
            payload={"tables": [{"name": "orders"}]},
        )
    )
    loop = DbAgentLoop(
        runtime,
        ContractFallbackDbAgentPlanner({}),
        max_steps=1,
    )

    result = await loop.run(
        request=request,
        operation=operation,
        contract=contract,
    )

    assert result.status == "blocked"
    assert [task.capability_id for task, _ in runtime.executed_tasks] == [
        "db.query.prepare_read"
    ]
    assert [
        [spec.capability_id for spec in specs] for specs in runtime.materialized_specs
    ] == [["db.query.prepare_read"]]


async def test_catalog_and_value_parity_actions_materialize_task_specs():
    runtime, result = await _run(
        "how many orders for acme by customer",
        DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind="find_relationship_paths",
                    payload={
                        "store_id": "runtime_source",
                        "source_assets": ["orders"],
                        "target_assets": ["customers"],
                    },
                ),
                DbPlannerAction(
                    kind="search_column_values",
                    payload={"store_id": "runtime_source", "query": "acme"},
                ),
                DbPlannerAction(
                    kind="resolve_column_value_hints",
                    payload={"store_id": "runtime_source", "prompt": "acme orders"},
                ),
            )
        ),
        _finish(),
        plugins=(CatalogPlugin(auto_persist=False),),
    )

    assert result.status == "finish"
    assert [task.capability_id for task, _ in runtime.executed_tasks] == [
        "catalog.relationship_paths.find",
        "catalog.column_values.search",
        "catalog.column_value_hints.resolve",
    ]


async def test_model_read_action_inside_schema_lane_is_blocked():
    decision = DbPlannerDecision(
        actions=(
            DbPlannerAction(
                kind="execute_validated_read",
                payload={"sql": "select * from orders"},
            ),
        )
    )

    runtime, loop, request, operation, contract = await _prepared_loop(
        "what columns are in orders",
        decision,
    )

    with pytest.raises(DbAgentLoopBlocked):
        await loop.run(request=request, operation=operation, contract=contract)

    assert await runtime.store.list_tasks("op-model-loop") == []
    assert runtime.executed_tasks == []


async def test_model_write_execution_inside_write_propose_lane_is_blocked():
    decision = DbPlannerDecision(
        actions=(
            DbPlannerAction(
                kind="execute_validated_write",
                payload={"sql": "update orders set status = 'closed'"},
            ),
        )
    )

    runtime, loop, request, operation, contract = await _prepared_loop(
        "update orders set status = 'closed'",
        decision,
    )

    with pytest.raises(DbAgentLoopBlocked):
        await loop.run(request=request, operation=operation, contract=contract)

    assert await runtime.store.list_tasks("op-model-loop") == []
    assert runtime.executed_tasks == []


async def test_model_valid_read_inside_read_lane_persists_tasks():
    read_decision = DbPlannerDecision(
        actions=(
            DbPlannerAction(
                kind="execute_validated_read",
                payload={"sql": "select count(*) from orders"},
            ),
        )
    )
    finish_decision = DbPlannerDecision(
        actions=(DbPlannerAction(kind="finish", payload={"message": "done"}),)
    )

    runtime, result = await _run(
        "how many orders are there",
        read_decision,
        finish_decision,
    )

    persisted = await runtime.store.list_tasks("op-model-loop")
    assert result.status == "finish"
    assert [task.capability_id for task in persisted] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert [task.capability_id for task, _ in runtime.executed_tasks] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert [
        [spec.capability_id for spec in specs] for specs in runtime.materialized_specs
    ] == [["db.sql.validate", "db.sql.execute_read"]]
    assert runtime.materialized_specs[0][1].depends_on_validation is True
    assert persisted[1].dependencies[0].producer_task_id == persisted[0].id
    assert persisted[1].metadata["required_lane"] == "read"


async def test_model_clarification_response_returns_without_task_execution():
    decision = DbPlannerDecision(
        actions=(
            DbPlannerAction(
                kind="clarify",
                payload={"message": "Which table should I use?"},
            ),
        )
    )

    runtime, result = await _run("how many orders are there", decision)

    assert result.status == "clarify"
    assert result.message == "Which table should I use?"
    assert await runtime.store.list_tasks("op-model-loop") == []
    assert runtime.executed_tasks == []


async def test_model_finish_response_returns_without_task_execution():
    decision = DbPlannerDecision(
        actions=(DbPlannerAction(kind="finish", payload={"message": "done"}),)
    )

    runtime, result = await _run("how many orders are there", decision)

    assert result.status == "finish"
    assert result.message == "done"
    assert await runtime.store.list_tasks("op-model-loop") == []
    assert runtime.executed_tasks == []


async def test_forbidden_capabilities_win_over_requested_actions():
    decision = DbPlannerDecision(
        actions=(
            DbPlannerAction(
                kind="execute_validated_read",
                payload={"sql": "select * from orders"},
            ),
        )
    )

    runtime, loop, request, operation, contract = await _prepared_loop(
        "schema only; do not query rows",
        decision,
        requested_capabilities=("db.sql.execute_read",),
    )

    with pytest.raises(DbAgentLoopBlocked):
        await loop.run(request=request, operation=operation, contract=contract)

    assert await runtime.store.list_tasks("op-model-loop") == []
    assert runtime.executed_tasks == []


def _runtime_for_run(*decisions):
    planner = MockPlanner(*decisions)
    runtime = DbRuntime(
        plugins=(
            CatalogPlugin(auto_persist=False),
            SQLitePlugin(path=":memory:"),
        ),
        host_services={"db_agent_planner": planner},
    )
    return runtime, planner


def _finish(message="done"):
    return DbPlannerDecision(
        actions=(DbPlannerAction(kind="finish", payload={"message": message}),)
    )


async def _spy_execute_task(runtime):
    executed = []

    async def execute_task(task: Task, operation: Operation, context=None):
        await runtime.store.save_task(task)
        executed.append((task, context or {}))
        return ()

    runtime.execute_task = execute_task
    return executed


async def test_db_runtime_run_persists_safety_contract_metadata_without_intent_metadata():
    runtime, _ = _runtime_for_run(_finish())
    executed = await _spy_execute_task(runtime)

    result = await runtime.run("how many orders are there")

    operation = await runtime.store.load_operation(result.operation_id)
    assert operation is not None
    assert result.status is OperationStatus.SUCCEEDED
    assert operation.metadata["safety_frame"]["granted_lanes"] == ["read"]
    assert operation.metadata["granted_lanes"] == ["read"]
    assert operation.metadata["forbidden_capabilities"] == ["db.sql.execute_write"]
    assert operation.metadata["contract_metadata"]["granted_lanes"] == ["read"]
    legacy_intent_key = "_".join(("intent", "kind"))
    legacy_frame_key = "_".join(("intent", "frame"))
    assert legacy_intent_key not in operation.metadata
    assert "intent" not in operation.metadata
    assert legacy_frame_key not in operation.metadata
    assert legacy_intent_key not in operation.metadata["resume_context"]
    assert "intent" not in operation.metadata["resume_context"]
    assert executed == []


async def test_db_runtime_schema_only_prompt_does_not_create_sql_execution_tasks():
    runtime, _ = _runtime_for_run(
        DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind="inspect_schema",
                    payload={"focus": "orders"},
                ),
            )
        ),
        _finish(),
    )
    executed = await _spy_execute_task(runtime)

    result = await runtime.run("schema only; what columns are in orders")

    tasks = await runtime.store.list_tasks(result.operation_id)
    assert result.status is OperationStatus.SUCCEEDED
    assert [task.capability_id for task in tasks] == ["db.schema.inspect"]
    assert all("execute" not in task.capability_id for task in tasks)
    assert [task.capability_id for task, _ in executed] == ["db.schema.inspect"]


async def test_db_runtime_read_prompt_creates_validate_and_read_tasks_through_loop():
    runtime, _ = _runtime_for_run(
        DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind="execute_validated_read",
                    payload={"sql": "select count(*) from orders"},
                ),
            )
        ),
        _finish(),
    )
    executed = await _spy_execute_task(runtime)

    result = await runtime.run("how many orders are there")

    tasks = await runtime.store.list_tasks(result.operation_id)
    assert result.status is OperationStatus.SUCCEEDED
    assert [task.capability_id for task in tasks] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert [task.capability_id for task, _ in executed] == [
        "db.sql.validate",
        "db.sql.execute_read",
    ]
    assert tasks[1].dependencies[0].producer_task_id == tasks[0].id
    assert result.diagnostics["planner"]["observations"]


async def test_db_runtime_invalid_planner_action_blocks_before_executor_tasks():
    runtime, _ = _runtime_for_run({"actions": [{"kind": "unsupported_action"}]})
    executed = await _spy_execute_task(runtime)

    result = await runtime.run("how many orders are there")

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_planner_action_blocked",)
    assert await runtime.store.list_tasks(result.operation_id) == []
    assert executed == []


async def test_db_runtime_contract_block_blocks_before_executor_tasks():
    runtime, _ = _runtime_for_run(
        DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind="execute_validated_read",
                    payload={"sql": "select * from orders"},
                ),
            )
        )
    )
    executed = await _spy_execute_task(runtime)

    result = await runtime.run("schema only; do not query rows")

    assert result.status is OperationStatus.BLOCKED
    assert result.warnings == ("db_planner_action_blocked",)
    assert await runtime.store.list_tasks(result.operation_id) == []
    assert executed == []


async def test_db_runtime_clarify_and_finish_return_without_executor_execution():
    clarify_runtime, _ = _runtime_for_run(
        DbPlannerDecision(
            actions=(
                DbPlannerAction(
                    kind="clarify",
                    payload={"message": "Which table should I use?"},
                ),
            )
        )
    )
    clarify_executed = await _spy_execute_task(clarify_runtime)

    clarify = await clarify_runtime.run("how many orders are there")

    assert clarify.status is OperationStatus.BLOCKED
    assert clarify.answer == "Which table should I use?"
    assert await clarify_runtime.store.list_tasks(clarify.operation_id) == []
    assert clarify_executed == []

    finish_runtime, _ = _runtime_for_run(_finish("Already answered."))
    finish_executed = await _spy_execute_task(finish_runtime)

    finish = await finish_runtime.run("how many orders are there")

    assert finish.status is OperationStatus.SUCCEEDED
    assert finish.answer == "Already answered."
    assert await finish_runtime.store.list_tasks(finish.operation_id) == []
    assert finish_executed == []


async def test_db_runtime_resume_new_path_uses_lane_contract_context():
    runtime, _ = _runtime_for_run(_finish("Already answered."))

    result = await runtime.run("how many orders are there")
    operation = await runtime.store.load_operation(result.operation_id)
    assert operation is not None
    await runtime.store.save_operation(
        replace(operation, status=OperationStatus.BLOCKED)
    )

    assert not hasattr(resume_module, "_db_intent_from_context")

    resumed = await runtime.resume_operation(result.operation_id)

    assert resumed.operation.status is OperationStatus.SUCCEEDED
    assert runtime.operation_results[-1].answer == "Already answered."
    assert runtime.operation_results[-1].diagnostics["resume"] == {
        "source": "lane_contract_context",
        "granted_lanes": ["read"],
        "forbidden_capabilities": ["db.sql.execute_write"],
    }
