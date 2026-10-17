"""Planner-created graph work is admitted exactly like owner-admitted work."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import CapabilityInputError, CapabilityRegistry
from daita.jobs.graph.admission import RegistryInitialTaskProposalResolver
from daita.jobs.graph.calls import admit_graph_call
from daita.jobs.graph.models import (
    GraphMutationRequest,
    GraphState,
    GraphTask,
    TaskRole,
    TaskState,
)
from daita.jobs.graph.validation import GraphValidationError, validate_mutation
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from tests.support.graph import GRAPH_NOW, graph_admission
from tests.support.model_graph_integration import (
    CONVERSATION_ID,
    MODEL_ROUTE_ID,
    READ_CAPABILITY_ID,
    READ_TOOL_NAME,
    ModelGraphIntegration,
    graph_read_declarations,
)
from tests.support.toolbox_model import RefCitingMockModelProvider, result_ref

pytestmark = pytest.mark.integration

_USAGE = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))
_Call = tuple[str, str, Mapping[str, object]]
_FIX = "correct_arguments"
_EXTRA = "unexpected_arguments"
_OUTSIDE = "source_scope_violation"
_UNFIT = "execution_not_supported_for_requested_shape"


def _response(*calls: _Call) -> ModelResponse:
    tool_calls = tuple(ToolCall(id=i, name=n, arguments=dict(a)) for i, n, a in calls)
    return ModelResponse(FinishReason.TOOL_CALLS, tool_calls=tool_calls, usage=_USAGE)


def _complete(call_id: str, result_kind: str, cited: str) -> ModelResponse:
    payload: dict[str, object] = {
        "result_kind": result_kind,
        "summary": "Completed.",
        "payload": {"done": True},
        "evidence_refs": (result_ref(cited),),
        "artifact_ids": (),
        "residual_risk": None,
        "downstream_constraints": {},
    }
    return _response((call_id, "task_complete", payload))


def _script(ids: tuple[str, ...], *turns: tuple[_Call, ...], cited: str):
    """Block for replan, run the planner turns, then execute the replacement."""
    message = "The admitted work needs one replacement."
    details: dict[str, object] = {"evidence_references": ()}
    block: dict[str, object] = {
        "kind": "needs_replan",
        "message": message,
        "details": details,
    }
    return (
        _response(("discover", "task_block", block)),
        *(_response(*turn) for turn in turns),
        _complete("complete-plan", "graph.plan", cited),
        _response(("replacement-read", READ_TOOL_NAME, {"resource_ids": ids})),
        _complete("complete-replacement", "test.graph.read", "replacement-read"),
    )


def _supersede(call_id: str, worker: GraphTask, **extra: object) -> _Call:
    common: dict[str, object] = {"expected_revision": 1, "idempotency_key": call_id}
    arguments = {**common, "task_id": worker.task_id, **extra}
    return (call_id, "graph_supersede_unstarted", arguments)


def _children(call_id: str, label: str, arguments: Mapping[str, object]) -> _Call:
    child: dict[str, object] = {
        "client_key": label,
        "title": "Read one bounded source",
        "description": "Read inside the admitted scope.",
        "result_kind": "test.graph.read",
        "capability_id": READ_CAPABILITY_ID,
        "arguments": arguments,
        "parent_task_ids": (),
        "input_result_ids": (),
        "reduction": False,
    }
    common = {"expected_revision": 1, "idempotency_key": call_id}
    return (call_id, "graph_create_children", {**common, "children": (child,)})


def _worker(tasks: tuple[GraphTask, ...]) -> GraphTask:
    return next(item for item in tasks if item.role is TaskRole.WORKER)


@pytest.fixture
async def integration(tmp_path: Path) -> AsyncIterator[ModelGraphIntegration]:
    opened = await ModelGraphIntegration.open(tmp_path)
    try:
        yield opened
    finally:
        await opened.close()


async def _run_to_replacement(integration: ModelGraphIntegration, admission, script):
    """Run one admission to success; return its tool results, terminal, replacement."""
    provider = RefCitingMockModelProvider(
        script, provider_id=MODEL_ROUTE_ID, complete_pricing=True
    )
    assert integration.supervisor._graph_model_loop is not None
    integration.supervisor._graph_model_loop._model = provider
    integration.provider = provider
    await integration.admit_and_start(admission)
    terminal = await integration.wait_terminal(admission.job.job_id)
    assert terminal.job.state is GraphState.SUCCEEDED
    provider.assert_consumed()
    results = {
        block.call_id: block
        for request in provider.requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }
    by_id = {item.task_id: item for item in terminal.tasks}
    replaced = by_id[_worker(admission.tasks).task_id]
    assert replaced.state is TaskState.SUPERSEDED
    assert replaced.superseded_by_task_id is not None
    return results, terminal, by_id[replaced.superseded_by_task_id]


async def test_invalid_planner_calls_are_rejected_and_supersede_copies_by_reference(
    integration: ModelGraphIntegration,
) -> None:
    admission = integration.build()
    worker = _worker(admission.tasks)
    ids = worker.specification.authority.resource_ids
    nested = {
        "title": "Read",
        "capability_id": READ_CAPABILITY_ID,
        "arguments": {"resource_ids": ids},
        "result_kind": "test.graph.read",
    }
    invalid: dict[str, tuple[Mapping[str, object], str, str | None]] = (
        {  # label: (arguments, error code, recovery kind when asserted)
            "wrong-type": ({"resource_ids": ids[0]}, "invalid_argument_type", None),
            "empty": ({}, "missing_arguments", _FIX),
            "out-of-scope": ({"resource_ids": ("resource-99",)}, _OUTSIDE, None),
            "extra-key": ({"resource_ids": ids, "sql": "select 1"}, _EXTRA, _FIX),
            "nested-spec": (nested, "missing_arguments", _FIX),
        }
    )
    bad_calls: list[_Call] = []
    expected: dict[str, tuple[str, str | None]] = {}
    for label, (arguments, code, recovery) in invalid.items():
        bad_calls.append(_supersede(f"supersede-{label}", worker, arguments=arguments))
        bad_calls.append(_children(f"children-{label}", label, arguments))
        expected |= {
            f"{kind}-{label}": (code, recovery) for kind in ("supersede", "children")
        }
    foreign = {"capability_id": "jobs.graph.task_complete", "arguments": {}}
    bad_calls.append(_supersede("supersede-foreign-capability", worker, **foreign))
    expected["supersede-foreign-capability"] = ("planner_capability_invalid", None)
    by_reference = (_supersede("replace-by-reference", worker),)
    script = _script(ids, tuple(bad_calls), by_reference, cited="replace-by-reference")

    results, terminal, replacement = await _run_to_replacement(
        integration, admission, script
    )

    for call_id, (code, recovery) in expected.items():
        assert results[call_id].is_error, call_id
        error = results[call_id].output["error"]
        assert isinstance(error, Mapping) and error["code"] == code, call_id
        if recovery is not None:
            assert isinstance(details := error["details"], Mapping)
            assert details["recovery"]["kind"] == recovery, call_id
    assert not results["replace-by-reference"].is_error
    # Only the policy planner and the one by-reference replacement were added.
    original = {item.task_id for item in admission.tasks}
    created = [item for item in terminal.tasks if item.task_id not in original]
    assert sorted(item.role.value for item in created) == ["planner", "worker"]
    assert replacement.state is TaskState.SUCCEEDED
    copied = ("expected_result_contract", "authority", "budgets", "title", "max_steps")
    for field in (*copied, "max_wall_time_seconds"):
        assert getattr(replacement.specification, field) == getattr(
            worker.specification, field
        ), field
    initial_call = replacement.specification.expected_result_contract["initial_call"]
    assert isinstance(initial_call, Mapping)
    assert initial_call["callable_tool_names"] == (READ_TOOL_NAME,)
    assert len(integration.reader.calls) == 1


async def test_owner_and_planner_admission_freeze_the_same_call(
    integration: ModelGraphIntegration,
) -> None:
    resource_ids = ("resource-1", "resource-0")
    scope = tuple(sorted(resource_ids))
    raw_arguments = {"resource_ids": list(resource_ids)}
    resolver = RegistryInitialTaskProposalResolver(
        agent_id=integration.owner.agent_id,
        registry=integration.registry,
        contract_reader=integration.contracts,
        model_route_id=MODEL_ROUTE_ID,
        max_steps=6,
        per_run_max_tokens=10_000,
    )
    references = {"source_ids": (), "resource_ids": scope, "connector_binding_ids": ()}
    proposal = await resolver(
        proposal={
            "capability_id": READ_CAPABILITY_ID,
            "arguments": raw_arguments,
            "expected_result_contract": {"result_kind": "test.graph.read"},
            "retained_references": references,
        },
        source_scope=None,
        sensitivity=ModelSensitivity.INTERNAL,
    )
    admission = integration.builder.build(
        run_id="run-" + "b" * 32,
        call_id="start-parity-graph",
        conversation_id=CONVERSATION_ID,
        objective="Read the exact admitted resources.",
        outcome_contract={"required_result_kind": "test.graph.read"},
        deadline_seconds=300,
        proposal=proposal,
    )
    worker = _worker(admission.tasks)
    owner_call = worker.specification.expected_result_contract["initial_call"]
    note = "Same call, admitted again by the planner."
    override = {"capability_id": READ_CAPABILITY_ID, "arguments": raw_arguments}
    turn = (_supersede("replace-with-override", worker, **override, note=note),)
    script = _script(resource_ids, turn, cited="replace-with-override")

    _, _, replacement = await _run_to_replacement(integration, admission, script)

    spec = replacement.specification
    assert spec.expected_result_contract["initial_call"] == owner_call
    assert isinstance(owner_call, Mapping)
    assert owner_call["callable_tool_names"] == (READ_TOOL_NAME,)
    assert spec.description.endswith(f"Planner replacement note: {note}")
    direct = await admit_graph_call(
        integration.registry, READ_CAPABILITY_ID, raw_arguments, resource_ids=scope
    )
    assert FrozenJsonObject.from_mapping(direct.material()) == owner_call


def _optional_registry() -> CapabilityRegistry:
    """The graph read with one optional bound and a graph limit of three targets."""
    declarations, executor = graph_read_declarations()
    read = declarations.capabilities[0]
    assert read.execution_admission_policy is not None
    properties = read.input_schema["properties"]
    assert isinstance(properties, Mapping)
    properties = {**properties, "limit": {"type": "integer", "minimum": 1}}
    capability = replace(
        read,
        input_schema={**read.input_schema, "properties": properties},
        execution_admission_policy=replace(
            read.execution_admission_policy, inline_max_targets=2, graph_max_targets=3
        ),
    )
    declarations = replace(declarations, capabilities=(capability,))
    return CapabilityRegistry(declarations=(declarations,), executors=(executor,))


async def test_shared_call_admission_normalises_unset_nulls() -> None:
    admitted = await admit_graph_call(
        _optional_registry(),
        READ_CAPABILITY_ID,
        {"resource_ids": ["a"], "limit": None},
        resource_ids=("a", "b"),
    )
    assert dict(admitted.arguments) == {"resource_ids": ("a",)}
    assert admitted.callable_tool_names == (READ_TOOL_NAME,)


@pytest.mark.parametrize(
    ("capability_id", "arguments", "scope", "code"),
    (
        (READ_CAPABILITY_ID, {"resource_ids": list("abcd")}, tuple("abcd"), _UNFIT),
        (READ_CAPABILITY_ID, {"resource_ids": ["a", "z"]}, ("a",), _OUTSIDE),
        ("test.graph.unknown", {}, (), _UNFIT),
    ),
    ids=("over-graph-bound", "outside-scope", "unregistered"),
)
async def test_shared_call_admission_checks_bounds(
    capability_id: str, arguments: dict[str, object], scope: tuple[str, ...], code: str
) -> None:
    with pytest.raises(CapabilityInputError) as error:
        await admit_graph_call(
            _optional_registry(), capability_id, arguments, resource_ids=scope
        )
    assert error.value.code == code


_CALL = {"capability_id": "capability.read", "callable_tool_names": ("read",)}


@pytest.mark.parametrize(
    "initial_call",
    (
        None,
        {"capability_id": "capability.read", "arguments": {}},
        {**_CALL, "callable_tool_names": (), "arguments": {}},
        {**_CALL, "arguments": "not-an-object"},
        {**_CALL, "capability_id": "capability.other", "arguments": {}},
    ),
    ids=("missing", "no-tool-names", "empty-tool-names", "bad-arguments", "outside"),
)
def test_validate_mutation_rejects_unadmitted_model_created_calls(
    initial_call: object,
) -> None:
    admission = graph_admission()
    worker = _worker(admission.tasks)
    contract = dict(worker.specification.expected_result_contract)
    contract.pop("initial_call", None)
    if initial_call is not None:
        contract["initial_call"] = initial_call
    spec = replace(
        worker.specification,
        expected_result_contract=contract,
        created_by="planner-task",
    )
    child = replace(
        worker,
        task_id="model-created",
        state=TaskState.READY,
        specification=spec,
        task_spec_digest=spec.digest,
    )
    job, graph = admission.job, admission.graph
    limits = job.specification.limits
    bounds = "max_tasks max_edges max_depth max_direct_parents max_fan_out".split()
    request = GraphMutationRequest(
        agent_id=job.agent_id,
        job_id=job.job_id,
        mutation_id="mutation-model-created",
        actor_kind="planner_attempt",
        actor_key="planner-task",
        idempotency_key="model-created",
        expected_revision=graph.revision,
        created_at=GRAPH_NOW,
        tasks=(child,),
        creator_task_id="planner-task",
        creator_attempt_id="attempt-planner",
        claim_token="claim-planner",
        fencing_epoch=1,
    )
    with pytest.raises(GraphValidationError) as error:
        validate_mutation(
            job_authority=job.specification.authority,
            graph=graph,
            existing_tasks=admission.tasks,
            existing_dependencies=admission.dependencies,
            request=request,
            **{name: getattr(limits, name) for name in bounds},
        )
    assert error.value.code == "model_task_call_invalid"
