from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.catalog import ResourceKind, catalog_resource_id
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRouteAttempt,
    ModelRouteAttemptOutcome,
    ModelRequest,
    ModelResponse,
    ModelRoutingTrace,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopExit, LoopExitKind, LoopPhase, Readiness
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    ActionValidationFacts,
    AgentTrigger,
    Observation,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

from .assertions import (
    assert_catalog_graph_use,
    assert_cited_evidence_supports,
    assert_count_and_money,
    assert_discrepancies_explained,
    assert_inspectable_runtime_state,
    assert_labeled_money,
    assert_session_cited_evidence_supports,
)
from .fixture_oracles import Discrepancy
from .harness import (
    MVP_EVALUATOR_VERSION,
    LiveMvpConfiguration,
    LiveRowRecorder,
    summarize_live_run,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


class _ReadExecutor:
    executor_id = "evaluator.fake.read.executor"

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        return EvidenceCandidate(
            kind="evaluator.fake.read_result",
            schema_version=1,
            payload={"key": request.arguments["key"], "value": "alpha"},
        )


class _StaticExecutor:
    def __init__(self, executor_id: str, kind: str, payload: dict[str, object]) -> None:
        self.executor_id = executor_id
        self._kind = kind
        self._payload = payload

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        del request
        return EvidenceCandidate(
            kind=self._kind,
            schema_version=1,
            payload=self._payload,
        )


def _registry() -> CapabilityRegistry:
    executor = _ReadExecutor()
    capability = Capability(
        id="evaluator.fake.read",
        owner="evaluator-contract-test",
        description="Read one deterministic test value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="evaluator.fake.read_result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        executor_id=executor.executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="evaluator_fake_read",
                capability_id=capability.id,
                description=capability.description,
            ),
        ),
    )


_GRAPH_RESOURCE_NAMES = (
    "refunds",
    "payments",
    "orders",
    "customers",
    "regions",
)
_GRAPH_RESOURCES = tuple(
    catalog_resource_id("source-graph", ResourceKind.TABLE, f"main.{name}")
    for name in _GRAPH_RESOURCE_NAMES
)
_GRAPH_FIELDS = (
    ("payment_id", "id"),
    ("order_id", "id"),
    ("customer_id", "id"),
    ("region_id", "id"),
)


def _graph_registry() -> CapabilityRegistry:
    steps = [
        {
            "relationship_id": f"relationship-{index}",
            "from_resource_id": _GRAPH_RESOURCES[index],
            "to_resource_id": _GRAPH_RESOURCES[index + 1],
            "field_pairs": [
                {
                    "source_field": source_field,
                    "target_field": target_field,
                    "ordinal": 0,
                }
            ],
        }
        for index, (source_field, target_field) in enumerate(_GRAPH_FIELDS)
    ]
    traversal_executor = _StaticExecutor(
        "catalog.traverse.executor",
        "catalog.traversal_result",
        {
            "reachable": True,
            "truncated": False,
            "request": {
                "from_resource_ids": [_GRAPH_RESOURCES[0]],
                "to_resource_ids": [_GRAPH_RESOURCES[-1]],
            },
            "paths": [
                {
                    "resource_ids": list(_GRAPH_RESOURCES),
                    "steps": steps,
                }
            ],
        },
    )
    sql_executor = _StaticExecutor(
        "data.sqlite.query.executor",
        "data.sqlite.query_result",
        {"rows": [{"region": "North America", "net_cents": 27000}]},
    )
    capabilities = (
        Capability(
            id="catalog.traverse",
            owner="catalog",
            description="Traverse the deterministic graph.",
            input_schema={
                "type": "object",
                "properties": {
                    "from_resource_ids": {"type": "array"},
                    "to_resource_ids": {"type": "array"},
                },
                "required": ["from_resource_ids", "to_resource_ids"],
                "additionalProperties": False,
            },
            output_evidence_kind="catalog.traversal_result",
            output_schema_version=1,
            output_schema={"type": "object", "additionalProperties": True},
            executor_id=traversal_executor.executor_id,
            access_mode=AccessMode.READ,
            risk=RiskLevel.LOW,
            side_effecting=False,
            idempotent=True,
            replay_safe=True,
        ),
        Capability(
            id="data.sqlite.query",
            owner="data",
            description="Run the deterministic graph query.",
            input_schema={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
            output_evidence_kind="data.sqlite.query_result",
            output_schema_version=1,
            output_schema={"type": "object", "additionalProperties": True},
            executor_id=sql_executor.executor_id,
            access_mode=AccessMode.READ,
            risk=RiskLevel.LOW,
            side_effecting=False,
            idempotent=True,
            replay_safe=True,
        ),
    )
    return CapabilityRegistry(
        capabilities=capabilities,
        executors=(traversal_executor, sql_executor),
        tool_views=tuple(
            ToolView(
                name=capability.id.replace(".", "_"),
                capability_id=capability.id,
                description=capability.description,
            )
            for capability in capabilities
        ),
    )


async def _commit_tool_call(
    runtime: OperationRuntime,
    operation_id: str,
    registry: CapabilityRegistry,
    call: ToolCall,
) -> str:
    turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-evaluator",
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Read the test value."),),
            ),
        ),
        tools=registry.tool_definitions(),
        allow_parallel_tool_calls=False,
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:evaluator-contract",
        request,
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(call,),
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return turn.id


async def _completed_repaired_snapshot():
    registry = _registry()
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-evaluator",
            agent_id="agent-evaluator",
            kind=TriggerKind.USER,
            source_id="user:evaluator",
            payload={"message": "Read the test value."},
            created_at=NOW,
        )
    )
    operation_id = started.operation.id

    rejected_call = ToolCall(
        id="call-rejected",
        name="evaluator_fake_read",
        arguments={"key": "invalid"},
    )
    rejected_turn_id = await _commit_tool_call(
        runtime,
        operation_id,
        registry,
        rejected_call,
    )
    await runtime.record_action_rejection(
        operation_id,
        rejected_turn_id,
        rejected_call,
        ActionRejection(
            code="evaluator.invalid_key",
            message="Use the supported key.",
            details={"allowed_keys": ["alpha"]},
        ),
    )

    accepted_call = ToolCall(
        id="call-accepted",
        name="evaluator_fake_read",
        arguments={"key": "alpha"},
    )
    accepted_turn_id = await _commit_tool_call(
        runtime,
        operation_id,
        registry,
        accepted_call,
    )
    evidence = await runtime.submit(
        ActionProposal(
            operation_id=operation_id,
            turn_id=accepted_turn_id,
            call_id=accepted_call.id,
            capability_id="evaluator.fake.read",
            arguments=accepted_call.arguments,
            proposed_at=NOW,
        )
    )
    assert evidence is not None
    await runtime.append_observation(
        Observation(
            operation_id=operation_id,
            turn_id=accepted_turn_id,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            code="evaluator.fake.read_succeeded",
            message="The deterministic read succeeded.",
            payload=evidence.payload,
            success=True,
            created_at=NOW,
        )
    )

    final_text = f"The value is alpha. [evidence:{evidence.id}]"
    final_turn = await runtime.begin_turn(operation_id)
    final_request = ModelRequest(
        operation_id=operation_id,
        turn_id=final_turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-evaluator",
                operation_id=operation_id,
                turn_id=final_turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Answer from accepted evidence."),),
            ),
        ),
        allow_parallel_tool_calls=False,
    )
    final_call = await runtime.begin_model_call(
        operation_id,
        final_turn.id,
        "mock:evaluator-contract",
        final_request,
    )
    await runtime.record_model_response(
        operation_id,
        final_call.id,
        ModelResponse(text=final_text, finish_reason=FinishReason.STOP),
        next_phase=LoopPhase.SYNTHESIZING,
    )
    await runtime.record_readiness(
        operation_id,
        final_text,
        Readiness(
            allowed=True,
            code="data.response_contract_satisfied",
            message="Evidence-linking and disclosure requirements passed.",
            evaluated_at=NOW,
        ),
    )
    await runtime.complete(operation_id, final_text)
    return await runtime.inspect(operation_id)


async def _graph_snapshot():
    registry = _graph_registry()
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-graph",
            agent_id="agent-evaluator",
            kind=TriggerKind.USER,
            source_id="user:evaluator",
            payload={"message": "Use the graph, then query."},
            created_at=NOW,
        )
    )
    operation_id = started.operation.id
    traversal_call = ToolCall(
        id="call-traversal",
        name="catalog_traverse",
        arguments={
            "from_resource_ids": [_GRAPH_RESOURCES[0]],
            "to_resource_ids": [_GRAPH_RESOURCES[-1]],
        },
    )
    traversal_turn = await _commit_tool_call(
        runtime,
        operation_id,
        registry,
        traversal_call,
    )
    traversal = await runtime.submit(
        ActionProposal(
            operation_id=operation_id,
            turn_id=traversal_turn,
            call_id=traversal_call.id,
            capability_id="catalog.traverse",
            arguments=traversal_call.arguments,
            proposed_at=NOW,
        )
    )
    assert traversal is not None
    await runtime.append_observation(
        Observation(
            operation_id=operation_id,
            turn_id=traversal_turn,
            task_id=traversal.task_id,
            evidence_id=traversal.id,
            code="catalog.traversal_succeeded",
            message="The graph path was accepted.",
            payload=traversal.payload,
            success=True,
            created_at=NOW,
        )
    )

    sql = (
        "SELECT regions.name FROM refunds "
        "JOIN payments ON payments.id = refunds.payment_id "
        "JOIN orders ON orders.id = payments.order_id "
        "JOIN customers ON customers.id = orders.customer_id "
        "JOIN regions ON regions.id = customers.region_id"
    )
    query_call = ToolCall(
        id="call-query",
        name="data_sqlite_query",
        arguments={"sql": sql},
    )
    query_turn = await _commit_tool_call(
        runtime,
        operation_id,
        registry,
        query_call,
    )
    revision = "sha256:" + ("2" * 64)
    query = await runtime.submit(
        ActionProposal(
            operation_id=operation_id,
            turn_id=query_turn,
            call_id=query_call.id,
            capability_id="data.sqlite.query",
            arguments=query_call.arguments,
            proposed_at=NOW,
            validation_facts=ActionValidationFacts(
                schema_version=1,
                source_id="source-graph",
                source_ids=("source-graph",),
                source_revision="revision-graph",
                resource_ids=_GRAPH_RESOURCES,
                resource_revisions=tuple(
                    (resource_id, revision) for resource_id in _GRAPH_RESOURCES
                ),
                freshness_state="current",
            ),
        )
    )
    assert query is not None
    await runtime.append_observation(
        Observation(
            operation_id=operation_id,
            turn_id=query_turn,
            task_id=query.task_id,
            evidence_id=query.id,
            code="data.sqlite.query_succeeded",
            message="The graph-grounded query was accepted.",
            payload=query.payload,
            success=True,
            created_at=NOW,
        )
    )
    return await runtime.inspect(operation_id)


async def test_completed_repaired_operation_is_inspectable() -> None:
    snapshot = await _completed_repaired_snapshot()

    assert snapshot.loop_state.repair_count == 1
    assert any(not observation.success for observation in snapshot.observations)
    assert any(observation.success for observation in snapshot.observations)
    assert_inspectable_runtime_state(snapshot)


async def test_inspectability_rejects_duplicate_observation_correlation() -> None:
    snapshot = await _completed_repaired_snapshot()
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "observations",
        (*snapshot.observations, snapshot.observations[-1]),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


async def test_inspectability_rejects_missing_observation_event() -> None:
    snapshot = await _completed_repaired_snapshot()
    successful = next(item for item in snapshot.observations if item.success)
    task = next(item for item in snapshot.tasks if item.id == successful.task_id)
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "events",
        tuple(
            event
            for event in snapshot.events
            if not (
                event.type == "observation.recorded"
                and event.task_id == task.id
                and event.call_id == task.call_id
            )
        ),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


async def test_inspectability_rejects_cross_operation_observation() -> None:
    snapshot = await _completed_repaired_snapshot()
    cross_operation = copy.copy(snapshot.observations[-1])
    object.__setattr__(cross_operation, "operation_id", "operation-other")
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "observations",
        (*snapshot.observations[:-1], cross_operation),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


async def test_inspectability_does_not_depend_on_evidence_order() -> None:
    snapshot = await _completed_repaired_snapshot()
    reordered = copy.copy(snapshot)
    object.__setattr__(reordered, "evidence", tuple(reversed(snapshot.evidence)))

    assert_inspectable_runtime_state(reordered)


async def test_inspectability_rejects_dangling_task_turn_scope() -> None:
    snapshot = await _completed_repaired_snapshot()
    task = snapshot.tasks[0]
    evidence = next(item for item in snapshot.evidence if item.task_id == task.id)
    observation = next(item for item in snapshot.observations if item.success)
    dangling_turn_id = "turn-does-not-exist"
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "tasks",
        tuple(
            replace(item, turn_id=dangling_turn_id) if item.id == task.id else item
            for item in snapshot.tasks
        ),
    )
    object.__setattr__(
        malformed,
        "evidence",
        tuple(
            replace(item, turn_id=dangling_turn_id) if item.id == evidence.id else item
            for item in snapshot.evidence
        ),
    )
    object.__setattr__(
        malformed,
        "observations",
        tuple(
            replace(item, turn_id=dangling_turn_id) if item is observation else item
            for item in snapshot.observations
        ),
    )
    object.__setattr__(
        malformed,
        "events",
        tuple(
            (
                replace(item, turn_id=dangling_turn_id)
                if item.task_id == task.id or item.evidence_id == evidence.id
                else item
            )
            for item in snapshot.events
        ),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


async def test_inspectability_rejects_dangling_tool_call_scope() -> None:
    snapshot = await _completed_repaired_snapshot()
    task = snapshot.tasks[0]
    evidence = next(item for item in snapshot.evidence if item.task_id == task.id)
    dangling_call_id = "call-does-not-exist"
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "tasks",
        tuple(
            replace(item, call_id=dangling_call_id) if item.id == task.id else item
            for item in snapshot.tasks
        ),
    )
    object.__setattr__(
        malformed,
        "events",
        tuple(
            (
                replace(item, call_id=dangling_call_id)
                if item.task_id == task.id or item.evidence_id == evidence.id
                else item
            )
            for item in snapshot.events
        ),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


async def test_inspectability_rejects_wrong_observation_event_model_call() -> None:
    snapshot = await _completed_repaired_snapshot()
    evidence = next(
        item
        for item in snapshot.evidence
        if any(
            observation.success and observation.evidence_id == item.id
            for observation in snapshot.observations
        )
    )
    wrong_model_call_id = snapshot.model_calls[-1].id
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "events",
        tuple(
            (
                replace(item, model_call_id=wrong_model_call_id)
                if item.type == "observation.recorded"
                and item.evidence_id == evidence.id
                else item
            )
            for item in snapshot.events
        ),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


@pytest.mark.parametrize("event_type", ("task.created", "evidence.accepted"))
async def test_inspectability_rejects_wrong_task_or_evidence_event_model_call(
    event_type: str,
) -> None:
    snapshot = await _completed_repaired_snapshot()
    task = snapshot.tasks[0]
    wrong_model_call_id = snapshot.model_calls[-1].id
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "events",
        tuple(
            (
                replace(item, model_call_id=wrong_model_call_id)
                if item.type == event_type and item.task_id == task.id
                else item
            )
            for item in snapshot.events
        ),
    )

    with pytest.raises(AssertionError):
        assert_inspectable_runtime_state(malformed)


async def test_out_of_band_graph_lookup_cannot_substitute_for_operation_evidence() -> (
    None
):
    snapshot = await _completed_repaired_snapshot()
    out_of_band_lookup = {
        "reachable": True,
        "resource_ids": ("resource-a", "resource-b"),
    }
    assert out_of_band_lookup["reachable"] is True

    with pytest.raises(AssertionError, match="catalog traversal"):
        assert_catalog_graph_use(
            snapshot,
            from_resource_id="resource-a",
            to_resource_id="resource-b",
        )


def test_semantic_oracles_ignore_evidence_identifier_tokens() -> None:
    with pytest.raises(AssertionError):
        assert_count_and_money(
            "No result was computed. [evidence:2-27000.00]",
            2,
            2_700_000,
        )


def test_count_and_money_oracle_binds_roles_instead_of_swapped_numbers() -> None:
    assert_count_and_money(
        "2 active customers; net revenue was $162.00.",
        2,
        16_200,
    )
    assert_labeled_money("Europe led with $125.00.", "Europe", 12_500)

    with pytest.raises(AssertionError):
        assert_count_and_money(
            "162 active customers; net revenue was $2.00.",
            2,
            16_200,
        )
    with pytest.raises(AssertionError):
        assert_labeled_money(
            "Europe led with $75.00. North America had $125.00.",
            "Europe",
            12_500,
        )


async def test_cited_query_rows_collectively_support_exact_and_money_facts() -> None:
    snapshot = await _graph_snapshot()
    query_evidence = next(
        item for item in snapshot.evidence if item.capability_id == "data.sqlite.query"
    )
    count_evidence = replace(
        query_evidence,
        id="evidence-count",
        payload=FrozenJsonObject.from_mapping({"rows": [{"customer_count": 2}]}),
    )
    money_evidence = replace(
        query_evidence,
        id="evidence-money",
        payload=FrozenJsonObject.from_mapping({"rows": [{"net_revenue": 162}]}),
    )
    persisted = copy.copy(snapshot)
    object.__setattr__(
        persisted,
        "evidence",
        (*snapshot.evidence, count_evidence, money_evidence),
    )
    candidates = (count_evidence, money_evidence)

    assert_cited_evidence_supports(
        "2 customers and $162.00. [evidence:evidence-count] "
        "[evidence:evidence-money]",
        persisted,
        candidates,
        exact_values=(2,),
        money_cents=(16_200,),
    )
    assert_session_cited_evidence_supports(
        "2 customers and $162.00. [evidence:evidence-count] "
        "[evidence:evidence-money]",
        (persisted, snapshot),
        candidates,
        exact_values=(2,),
        money_cents=(16_200,),
    )
    with pytest.raises(AssertionError):
        assert_cited_evidence_supports(
            "2 customers and $162.00. [evidence:evidence-count]",
            persisted,
            candidates,
            exact_values=(2,),
            money_cents=(16_200,),
        )


def test_discrepancy_explanation_preserves_values_nulls_and_direction() -> None:
    discrepancies = (
        Discrepancy(
            kind="value_mismatch",
            customer_id="2",
            column="plan",
            file_value="enterprise",
            database_value="growth",
        ),
        Discrepancy(
            kind="type_mismatch",
            customer_id="5",
            column="email",
            file_value="chloe@example.test",
            database_value=None,
        ),
        Discrepancy(kind="right_only", customer_id="7"),
        Discrepancy(kind="left_only", customer_id="8"),
    )
    answer = (
        "Customer 2 has plan enterprise in the export and growth in the database.\n"
        "Customer 5 has email chloe@example.test in the export, while the database "
        "value is null (a type mismatch).\n"
        "Customer 7 appears only in the current database and is missing from the "
        "export.\n"
        "Customer 8 appears only in the export and is absent from the current "
        "database. [evidence:customer-2-5-7-8]"
    )
    assert_discrepancies_explained(answer, discrepancies)

    reversed_direction = answer.replace(
        "Customer 8 appears only in the export and is absent from the current database",
        "Customer 8 appears only in the current database and is absent from the export",
    )
    with pytest.raises(AssertionError):
        assert_discrepancies_explained(reversed_direction, discrepancies)

    swapped_values = answer.replace(
        "plan enterprise in the export and growth in the database",
        "plan growth in the export and enterprise in the database",
    )
    with pytest.raises(AssertionError):
        assert_discrepancies_explained(swapped_values, discrepancies)


def test_discrepancy_explanation_accepts_multiline_identity_blocks() -> None:
    discrepancies = (
        Discrepancy(kind="right_only", customer_id="7"),
        Discrepancy(kind="left_only", customer_id="8"),
        Discrepancy(
            kind="value_mismatch",
            customer_id="2",
            column="plan",
            file_value="enterprise",
            database_value="growth",
        ),
    )
    answer = (
        "Customer 7:\n"
        "- Present only in the current database.\n"
        "- Missing from the export.\n"
        "Customer 8:\n"
        "- Present only in the export.\n"
        "- Missing from the current database.\n"
        "Customer 2:\n"
        "- plan in the export: enterprise\n"
        "- plan in the database: growth"
    )

    assert_discrepancies_explained(answer, discrepancies)


def test_discrepancy_explanation_covers_invalid_and_duplicate_keys() -> None:
    discrepancies = (
        Discrepancy(
            kind="invalid_key",
            customer_id=None,
            source="file",
            missing_columns=("id",),
        ),
        Discrepancy(
            kind="invalid_key",
            customer_id=None,
            source="database",
            null_columns=("id",),
        ),
        Discrepancy(
            kind="duplicate_key",
            customer_id="6",
            source="file",
            duplicate_count=2,
        ),
        Discrepancy(
            kind="duplicate_key",
            customer_id="9",
            source="database",
            duplicate_count=3,
        ),
    )
    answer = (
        "The export has a row with a missing id, so its comparison key is invalid.\n"
        "The database has a row whose id is null, making that key unusable.\n"
        "Customer 6 is duplicated in the export with 2 rows.\n"
        "Customer 9 is repeated in the current database with 3 rows."
    )

    assert_discrepancies_explained(answer, discrepancies)

    for invalid_answer in (
        answer.replace(
            "missing id, so its comparison key is invalid",
            "present id, so its comparison key is valid",
        ),
        answer.replace(
            "database has a row whose id is null",
            "export has a row whose id is null",
        ),
        answer.replace("export with 2 rows", "database with 2 rows"),
        answer.replace("database with 3 rows", "database with 4 rows"),
    ):
        with pytest.raises(AssertionError):
            assert_discrepancies_explained(invalid_answer, discrepancies)


async def test_graph_use_correlates_accepted_path_to_later_qualified_sql() -> None:
    snapshot = await _graph_snapshot()

    assert_catalog_graph_use(
        snapshot,
        from_resource_id=_GRAPH_RESOURCES[0],
        to_resource_id=_GRAPH_RESOURCES[-1],
    )


async def test_graph_use_rejects_same_turn_or_wrong_endpoint_join() -> None:
    snapshot = await _graph_snapshot()
    traversal_task = next(
        task for task in snapshot.tasks if task.capability_id == "catalog.traverse"
    )
    query_task = next(
        task for task in snapshot.tasks if task.capability_id == "data.sqlite.query"
    )
    same_turn = copy.copy(snapshot)
    object.__setattr__(
        same_turn,
        "tasks",
        tuple(
            (
                replace(task, turn_id=query_task.turn_id)
                if task.id == traversal_task.id
                else task
            )
            for task in snapshot.tasks
        ),
    )
    with pytest.raises(AssertionError, match="did not inform"):
        assert_catalog_graph_use(
            same_turn,
            from_resource_id=_GRAPH_RESOURCES[0],
            to_resource_id=_GRAPH_RESOURCES[-1],
        )

    wrong_endpoint = copy.copy(snapshot)
    wrong_sql = str(query_task.arguments["sql"]).replace(
        "refunds.payment_id",
        "orders.payment_id",
    )
    malformed_query_task = copy.copy(query_task)
    object.__setattr__(
        malformed_query_task,
        "arguments",
        FrozenJsonObject.from_mapping({"sql": wrong_sql}),
    )
    object.__setattr__(
        wrong_endpoint,
        "tasks",
        tuple(
            malformed_query_task if task.id == query_task.id else task
            for task in snapshot.tasks
        ),
    )
    with pytest.raises(AssertionError, match="did not inform"):
        assert_catalog_graph_use(
            wrong_endpoint,
            from_resource_id=_GRAPH_RESOURCES[0],
            to_resource_id=_GRAPH_RESOURCES[-1],
        )


async def test_failed_fake_live_row_emits_complete_redacted_metrics(
    tmp_path: Path,
) -> None:
    snapshot = await _completed_repaired_snapshot()
    successful_observation = next(
        item for item in snapshot.observations if item.success
    )
    assert successful_observation.evidence_id is not None
    truncated_snapshot = copy.copy(snapshot)
    object.__setattr__(
        truncated_snapshot,
        "evidence",
        tuple(
            (
                replace(
                    item,
                    payload={**dict(item.payload), "truncated": True},
                )
                if item.id == successful_observation.evidence_id
                else item
            )
            for item in snapshot.evidence
        ),
    )
    object.__setattr__(
        truncated_snapshot,
        "observations",
        tuple(
            (
                replace(
                    item,
                    payload={
                        "body": {"value": "alpha"},
                        "projection_truncated": True,
                        "repair_details": {},
                        "schema_version": 2,
                        "source_truncated": True,
                    },
                    truncated=True,
                )
                if item is successful_observation
                else item
            )
            for item in snapshot.observations
        ),
    )
    snapshot = truncated_snapshot
    result = LoopExit(
        operation_id=snapshot.operation.id,
        kind=LoopExitKind.COMPLETED,
        reason=snapshot.operation.terminal_reason or "completed",
        final_text=snapshot.operation.final_text,
        created_at=NOW,
    )
    properties: dict[str, object] = {}

    def record_property(name: str, value: object) -> None:
        properties[name] = value

    sidecar = tmp_path / "reports" / "wave1.json"
    retained_home = tmp_path / "state" / "fake-agent"
    retained_home.mkdir(parents=True)
    (retained_home / "agent.toml").write_text("schema_version = 1\n")
    credential = "test-credential-must-not-leak"
    sentinel = "SESSION_SENTINEL_MUST_NOT_LEAK"
    recorder = LiveRowRecorder(
        row_id="tests/live/mvp/test.py::test_fake[failed]",
        scenario_id="LIVE-MVP-03",
        variant_id="conversational",
        configuration=LiveMvpConfiguration(provider="openai", model="fake-model"),
        fixture_version="fixture-v1",
        fixture_digest="sha256:" + ("1" * 64),
        prompt_version="prompts-v1",
        sidecar_path=sidecar,
        record_property=record_property,
    )
    recorder.register_home(retained_home, (credential,))
    recorder.register_report_prohibited(credential, sentinel)
    recorder.capture(result, snapshot, wall_time_seconds=0.25)
    with recorder.hard_check("outcome", "answer_exact"):
        assert True
    with recorder.hard_check("safety", "current_authority"):
        raise AssertionError("the hard-check detail must not be persisted")
    with recorder.hard_check("evidence", "citation_invalid"):
        raise AssertionError("a known failure must dominate an incomplete check")
    recorder.record_not_evaluated(
        layer="evidence",
        code="citation_support_unavailable",
    )
    with recorder.diagnostic("catalog_inspection_missing"):
        raise AssertionError("the diagnostic detail must not be persisted")
    with pytest.raises(AssertionError, match="current_authority"):
        recorder.assert_mvp_passed()

    row = recorder.finalize(outcome="failed")

    assert row["outcome"] == "failed"
    assert row["failure_category"] == "safety"
    assert row["benchmark_order"] == 7
    assert row["evaluator_version"] == MVP_EVALUATOR_VERSION
    assert row["mvp_status"] == "fail"
    assert row["outcome_status"] == "pass"
    assert row["safety_status"] == "fail"
    assert row["evidence_status"] == "fail"
    assert row["hard_failure_codes"] == [
        "current_authority",
        "citation_invalid",
        "citation_support_unavailable",
    ]
    assert row["diagnostic_codes"] == ["catalog_inspection_missing"]
    assert row["hard_checks"] == [
        {"layer": "outcome", "code": "answer_exact", "status": "pass"},
        {"layer": "safety", "code": "current_authority", "status": "fail"},
        {"layer": "evidence", "code": "citation_invalid", "status": "fail"},
        {
            "layer": "evidence",
            "code": "citation_support_unavailable",
            "status": "not_evaluated",
        },
    ]
    assert row["operation_count"] == 1
    assert row["operation_ids"] == [snapshot.operation.id]
    required_metrics = {
        "actions",
        "evidence_count",
        "fallbacks",
        "input_tokens",
        "model_calls",
        "observation_characters",
        "omitted_context_tokens",
        "output_tokens",
        "provider_latency_ms",
        "rejected_actions",
        "repairs",
        "retries",
        "selected_context_tokens",
        "source_truncated_observations",
        "task_count",
        "tool_calls",
        "truncated_evidence",
        "truncated_observations",
        "truncation_history",
        "projection_truncated_observations",
        "wall_time_seconds",
    }
    assert required_metrics <= set(row)
    assert required_metrics <= set(properties)
    assert row["truncated_evidence"] == 1
    assert row["source_truncated_observations"] == 1
    assert row["projection_truncated_observations"] == 1
    assert row["truncated_observations"] == 1
    truncation_history = row["truncation_history"]
    assert isinstance(truncation_history, list)
    assert len(truncation_history) == 1
    assert truncation_history[0] == {
        "call_id": snapshot.tasks[0].call_id,
        "code": successful_observation.code,
        "evidence_id": successful_observation.evidence_id,
        "evidence_truncated": True,
        "operation_id": snapshot.operation.id,
        "projection_truncated": True,
        "source_truncated": True,
        "task_id": successful_observation.task_id,
        "turn_id": successful_observation.turn_id,
    }
    assert json.loads(str(properties["truncation_history"])) == truncation_history
    decoded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == 1
    assert decoded["rows"] == {
        recorder.row_id: json.loads(json.dumps(row, sort_keys=True))
    }
    serialized = sidecar.read_text(encoding="utf-8")
    assert credential not in serialized
    assert sentinel not in serialized


async def test_summary_rejects_malformed_persisted_truncation_facts() -> None:
    snapshot = await _completed_repaired_snapshot()
    successful_observation = next(
        item for item in snapshot.observations if item.success
    )
    malformed = copy.copy(snapshot)
    object.__setattr__(
        malformed,
        "observations",
        tuple(
            (
                replace(
                    item,
                    payload={
                        "projection_truncated": "yes",
                        "schema_version": 2,
                        "source_truncated": False,
                    },
                )
                if item is successful_observation
                else item
            )
            for item in snapshot.observations
        ),
    )
    result = LoopExit(
        operation_id=malformed.operation.id,
        kind=LoopExitKind.COMPLETED,
        reason=malformed.operation.terminal_reason or "completed",
        final_text=malformed.operation.final_text,
        created_at=NOW,
    )

    with pytest.raises(AssertionError, match="truncation facts are malformed"):
        summarize_live_run(result, malformed, wall_time_seconds=0.5)


async def test_summary_uses_only_persisted_route_and_call_metrics() -> None:
    snapshot = await _completed_repaired_snapshot()
    final_call = snapshot.model_calls[-1]
    assert final_call.response is not None
    provider_id = final_call.provider_id
    routing = ModelRoutingTrace(
        route_id="route-persisted",
        primary_provider_id=provider_id,
        selected_provider_id=provider_id,
        attempts=(
            ModelRouteAttempt(
                provider_id=provider_id,
                attempt=1,
                outcome=ModelRouteAttemptOutcome.SUCCEEDED,
                latency_ms=321,
            ),
        ),
    )
    routed_call = replace(
        final_call,
        response=replace(
            final_call.response,
            provider_id=provider_id,
            routing=routing,
        ),
    )
    persisted = copy.copy(snapshot)
    object.__setattr__(
        persisted,
        "model_calls",
        (*snapshot.model_calls[:-1], routed_call),
    )
    result = LoopExit(
        operation_id=persisted.operation.id,
        kind=LoopExitKind.COMPLETED,
        reason=persisted.operation.terminal_reason or "completed",
        final_text=persisted.operation.final_text,
        created_at=NOW,
    )

    summary = summarize_live_run(
        result,
        persisted,
        wall_time_seconds=0.5,
    )

    assert summary.provider_ids == (provider_id,)
    assert summary.route_ids == ("route-persisted",)
    assert summary.provider_latency_ms == 321
    assert summary.retries == 0
    assert summary.fallbacks == 0
    assert summary.model_call_ids == tuple(call.id for call in persisted.model_calls)
    assert summary.finish_reasons[-1] == FinishReason.STOP.value


def test_actual_failed_pytest_row_writes_junit_and_json_metrics(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_failed_live_row.py"
    sidecar = tmp_path / "failed-row.json"
    junit = tmp_path / "failed-row.xml"
    log = tmp_path / "failed-row.log"
    credential = "subprocess-credential-must-not-leak"
    sentinel = "SUBPROCESS_SENTINEL_MUST_NOT_LEAK"
    test_file.write_text(
        textwrap.dedent(f"""
            import asyncio
            import os
            from pathlib import Path
            import pytest

            from daita.loop.models import LoopExit, LoopExitKind
            from mvp.fixture_oracles import CommerceFixture
            from mvp.harness import LiveMvpConfiguration
            from mvp.test_evaluator_contracts import (
                NOW,
                _completed_repaired_snapshot,
            )

            pytest_plugins = ("mvp.conftest",)

            @pytest.fixture
            def commerce_fixture(tmp_path: Path) -> CommerceFixture:
                return CommerceFixture(
                    root=tmp_path,
                    database_path=tmp_path / "fixture.db",
                    files_root=tmp_path / "files",
                    fixture_version="fixture-v1",
                    manifest_digest="sha256:" + ("1" * 64),
                    reporting_start="2026-01-01",
                    reporting_end_exclusive="2026-04-01",
                )

            @pytest.fixture
            def live_mvp_configuration() -> LiveMvpConfiguration:
                return LiveMvpConfiguration(
                    provider="openai",
                    model="fake-model",
                    credential_environment="FAKE_LIVE_KEY",
                )

            def test_deliberate_failure(live_row_recorder) -> None:
                live_row_recorder.register_report_prohibited(
                    os.environ["FAKE_SENTINEL"]
                )
                snapshot = asyncio.run(_completed_repaired_snapshot())
                result = LoopExit(
                    operation_id=snapshot.operation.id,
                    kind=LoopExitKind.COMPLETED,
                    reason=snapshot.operation.terminal_reason or "completed",
                    final_text=snapshot.operation.final_text,
                    created_at=NOW,
                )
                live_row_recorder.capture(
                    result,
                    snapshot,
                    wall_time_seconds=0.25,
                )
                assert False
            """),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment.update(
        {
            "DAITA_LIVE_MVP_JSON_SIDECAR": str(sidecar),
            "FAKE_LIVE_KEY": credential,
            "FAKE_SENTINEL": sentinel,
            "PYTHONPATH": os.pathsep.join(
                (
                    str(repository / "src"),
                    str(repository / "tests" / "live"),
                    str(repository),
                    environment.get("PYTHONPATH", ""),
                )
            ),
        }
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test_file),
            f"--junitxml={junit}",
            "-o",
            "junit_family=xunit1",
            "-o",
            f"log_file={log}",
        ),
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    row = next(iter(document["rows"].values()))
    assert row["outcome"] == "failed"
    assert row["operation_count"] == 1
    assert row["truncated_evidence"] == 0
    assert row["source_truncated_observations"] == 0
    assert row["projection_truncated_observations"] == 0
    assert row["truncated_observations"] == 0
    assert row["truncation_history"] == []
    property_nodes = tuple(ET.parse(junit).getroot().iter("property"))
    property_names = tuple(item.attrib["name"] for item in property_nodes)
    assert len(property_names) == len(set(property_names))
    properties = {
        item.attrib["name"]: item.attrib.get("value", "") for item in property_nodes
    }
    assert properties["outcome"] == "failed"
    assert properties["operation_count"] == "1"
    assert properties["truncated_evidence"] == "0"
    assert properties["source_truncated_observations"] == "0"
    assert properties["projection_truncated_observations"] == "0"
    assert properties["truncated_observations"] == "0"
    assert json.loads(properties["truncation_history"]) == []
    retained = sidecar.read_text(encoding="utf-8") + junit.read_text(encoding="utf-8")
    if log.exists():
        retained += log.read_text(encoding="utf-8")
    assert credential not in retained
    assert sentinel not in retained


def test_exact_live_mvp_collection_is_variant_first_and_twelve_rows() -> None:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
            "-m",
            "live_mvp",
            "tests/live/mvp/test_data_journeys_live.py",
            "tests/live/mvp/test_sessions_live.py",
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    collected = tuple(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("tests/live/mvp/")
    )
    assert collected == (
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_01_grounded_multi_table_analyst_query[direct]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_02_ambiguous_catalog_and_graph_resolution[direct]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_03_newest_cross_source_comparison[direct]",
        "tests/live/mvp/test_sessions_live.py::"
        "test_live_mvp_04_session_continuity_and_cold_reopen[direct]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_01_grounded_multi_table_analyst_query[conversational]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_02_ambiguous_catalog_and_graph_resolution[conversational]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_03_newest_cross_source_comparison[conversational]",
        "tests/live/mvp/test_sessions_live.py::"
        "test_live_mvp_04_session_continuity_and_cold_reopen[conversational]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_01_grounded_multi_table_analyst_query[answerable-ambiguous]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_02_ambiguous_catalog_and_graph_resolution[answerable-ambiguous]",
        "tests/live/mvp/test_data_journeys_live.py::"
        "test_live_mvp_03_newest_cross_source_comparison[answerable-ambiguous]",
        "tests/live/mvp/test_sessions_live.py::"
        "test_live_mvp_04_session_continuity_and_cold_reopen[answerable-ambiguous]",
    )


@pytest.mark.parametrize(
    ("marker", "expected_count"),
    (
        ("live_mvp and live_mvp_smoke", 4),
        ("live_mvp and live_mvp_reliability", 8),
        ("live_precutover", 3),
    ),
)
def test_live_mvp_collection_tiers_are_explicit(
    marker: str,
    expected_count: int,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
            "-m",
            marker,
            "tests/live/mvp/test_data_journeys_live.py",
            "tests/live/mvp/test_sessions_live.py",
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    collected = tuple(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("tests/live/mvp/")
    )
    assert len(collected) == expected_count
