"""Planner-created graph work is admitted exactly like owner-admitted work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    ExecutionAdmissionPolicy,
    ToolboxId,
    ToolLoadMode,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from daita.jobs.graph.admission import RegistryInitialTaskProposalResolver
from daita.jobs.graph.calls import admit_graph_call
from daita.jobs.graph.models import (
    GraphMutationRequest,
    GraphState,
    GraphTask,
    TaskRole,
    TaskState,
)
from daita.jobs.graph.validation import GraphValidationError
from daita.llm.models import (
    FinishReason,
    ModelRequest,
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
)
from tests.support.toolbox_model import RefCitingMockModelProvider, result_ref

pytestmark = pytest.mark.integration

_USAGE = ModelUsage(cost_estimate=CostEstimate.complete(Decimal(0)))


def _response(*calls: tuple[str, str, Mapping[str, object]]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(
            ToolCall(id=call_id, name=name, arguments=dict(arguments))
            for call_id, name, arguments in calls
        ),
        usage=_USAGE,
    )


def _complete(call_id: str, result_kind: str, cited: str) -> ModelResponse:
    return _response(
        (
            call_id,
            "task_complete",
            {
                "result_kind": result_kind,
                "summary": "Completed.",
                "payload": {"done": True},
                "evidence_refs": (result_ref(cited),),
                "artifact_ids": (),
                "residual_risk": None,
                "downstream_constraints": {},
            },
        )
    )


def _block_for_replan() -> ModelResponse:
    return _response(
        (
            "discover",
            "task_block",
            {
                "kind": "needs_replan",
                "message": "The admitted work needs one replacement.",
                "details": {"evidence_references": ()},
            },
        )
    )


def _results(requests: tuple[ModelRequest, ...]) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for request in requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _child(client_key: str, arguments: Mapping[str, object]) -> dict[str, object]:
    return {
        "client_key": client_key,
        "title": "Read one bounded source",
        "description": "Read inside the admitted scope.",
        "result_kind": "test.graph.read",
        "capability_id": READ_CAPABILITY_ID,
        "arguments": arguments,
        "parent_task_ids": (),
        "input_result_ids": (),
        "reduction": False,
    }


async def _run(
    integration: ModelGraphIntegration,
    admission,
    script: tuple[ModelResponse, ...],
):
    provider = RefCitingMockModelProvider(
        script, provider_id=MODEL_ROUTE_ID, complete_pricing=True
    )
    assert integration.supervisor._graph_model_loop is not None
    integration.supervisor._graph_model_loop._model = provider
    integration.provider = provider
    await integration.admit_and_start(admission)
    terminal = await integration.wait_terminal(admission.job.job_id)
    return provider, terminal


def _replacement(terminal, original: GraphTask) -> GraphTask:
    current = next(item for item in terminal.tasks if item.task_id == original.task_id)
    assert current.state is TaskState.SUPERSEDED
    return next(
        item for item in terminal.tasks if item.task_id == current.superseded_by_task_id
    )


async def test_invalid_planner_calls_are_rejected_and_supersede_copies_by_reference(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    admission = integration.build()
    worker = next(item for item in admission.tasks if item.role is TaskRole.WORKER)
    resource_ids = worker.specification.authority.resource_ids
    nested_spec = {
        "title": "Read",
        "capability_id": READ_CAPABILITY_ID,
        "arguments": {"resource_ids": resource_ids},
        "result_kind": "test.graph.read",
    }
    invalid_arguments: dict[str, Mapping[str, object]] = {
        "wrong-type": {"resource_ids": resource_ids[0]},
        "empty": {},
        "out-of-scope": {"resource_ids": ("resource-99",)},
        "extra-key": {"resource_ids": resource_ids, "sql": "select 1"},
        "nested-spec": nested_spec,
    }
    supersede_common = {"expected_revision": 1, "task_id": worker.task_id}
    bad_calls: list[tuple[str, str, Mapping[str, object]]] = []
    for label, arguments in invalid_arguments.items():
        bad_calls.append(
            (
                f"supersede-{label}",
                "graph_supersede_unstarted",
                {
                    **supersede_common,
                    "idempotency_key": f"supersede-{label}",
                    "arguments": arguments,
                },
            )
        )
        bad_calls.append(
            (
                f"children-{label}",
                "graph_create_children",
                {
                    "expected_revision": 1,
                    "idempotency_key": f"children-{label}",
                    "children": (_child(label, arguments),),
                },
            )
        )
    bad_calls.append(
        (
            "supersede-foreign-capability",
            "graph_supersede_unstarted",
            {
                **supersede_common,
                "idempotency_key": "supersede-foreign-capability",
                "capability_id": "jobs.graph.task_complete",
                "arguments": {},
            },
        )
    )
    script = (
        _block_for_replan(),
        _response(*bad_calls),
        _response(
            (
                "replace-by-reference",
                "graph_supersede_unstarted",
                {**supersede_common, "idempotency_key": "replace-by-reference"},
            )
        ),
        _complete("complete-plan", "graph.plan", "replace-by-reference"),
        _response(("replacement-read", READ_TOOL_NAME, {"resource_ids": resource_ids})),
        _complete("complete-replacement", "test.graph.read", "replacement-read"),
    )
    try:
        provider, terminal = await _run(integration, admission, script)

        assert terminal.job.state is GraphState.SUCCEEDED
        provider.assert_consumed()
        results = _results(provider.requests)
        codes = {}
        for call_id, _name, _arguments in bad_calls:
            block = results[call_id]
            assert block.is_error, call_id
            error = block.output["error"]
            assert isinstance(error, Mapping)
            codes[call_id] = error["code"]
        assert codes["supersede-out-of-scope"] == "source_scope_violation"
        assert codes["children-out-of-scope"] == "source_scope_violation"
        assert codes["supersede-wrong-type"] == "invalid_argument_type"
        assert codes["children-wrong-type"] == "invalid_argument_type"
        assert codes["supersede-foreign-capability"] == "planner_capability_invalid"
        expected_codes = {
            "empty": "missing_arguments",
            "extra-key": "unexpected_arguments",
            "nested-spec": "missing_arguments",
        }
        for label, code in expected_codes.items():
            for kind in ("supersede", "children"):
                error = results[f"{kind}-{label}"].output["error"]
                assert isinstance(error, Mapping)
                assert error["code"] == code, (kind, label)
                details = error["details"]
                assert isinstance(details, Mapping)
                assert details["recovery"]["kind"] == "correct_arguments"
        assert not results["replace-by-reference"].is_error

        # Only the policy planner and the one by-reference replacement were added.
        created = {item.task_id for item in terminal.tasks} - {
            item.task_id for item in admission.tasks
        }
        roles = sorted(
            item.role.value for item in terminal.tasks if item.task_id in created
        )
        assert roles == ["planner", "worker"]

        replacement = _replacement(terminal, worker)
        assert replacement.state is TaskState.SUCCEEDED
        original_spec = worker.specification
        replacement_spec = replacement.specification
        assert (
            replacement_spec.expected_result_contract
            == original_spec.expected_result_contract
        )
        initial_call = replacement_spec.expected_result_contract["initial_call"]
        assert isinstance(initial_call, Mapping)
        assert initial_call["callable_tool_names"] == (READ_TOOL_NAME,)
        assert replacement_spec.authority == original_spec.authority
        assert replacement_spec.budgets == original_spec.budgets
        assert replacement_spec.title == original_spec.title
        assert replacement_spec.max_steps == original_spec.max_steps
        assert (
            replacement_spec.max_wall_time_seconds
            == original_spec.max_wall_time_seconds
        )
        assert len(integration.reader.calls) == 1
    finally:
        await integration.close()


async def test_owner_and_planner_admission_freeze_the_same_call(
    tmp_path: Path,
) -> None:
    integration = await ModelGraphIntegration.open(tmp_path)
    resource_ids = ("resource-1", "resource-0")
    raw_arguments = {"resource_ids": list(resource_ids)}
    resolver = RegistryInitialTaskProposalResolver(
        agent_id=integration.owner.agent_id,
        registry=integration.registry,
        contract_reader=integration.contracts,
        model_route_id=MODEL_ROUTE_ID,
        max_steps=6,
        per_run_max_tokens=10_000,
    )
    proposal = await resolver(
        proposal={
            "capability_id": READ_CAPABILITY_ID,
            "arguments": raw_arguments,
            "expected_result_contract": {"result_kind": "test.graph.read"},
            "retained_references": {
                "source_ids": (),
                "resource_ids": tuple(sorted(resource_ids)),
                "connector_binding_ids": (),
            },
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
    worker = next(item for item in admission.tasks if item.role is TaskRole.WORKER)
    owner_call = worker.specification.expected_result_contract["initial_call"]
    script = (
        _block_for_replan(),
        _response(
            (
                "replace-with-override",
                "graph_supersede_unstarted",
                {
                    "expected_revision": 1,
                    "idempotency_key": "replace-with-override",
                    "task_id": worker.task_id,
                    "capability_id": READ_CAPABILITY_ID,
                    "arguments": raw_arguments,
                    "note": "Same call, admitted again by the planner.",
                },
            )
        ),
        _complete("complete-plan", "graph.plan", "replace-with-override"),
        _response(("replacement-read", READ_TOOL_NAME, {"resource_ids": resource_ids})),
        _complete("complete-replacement", "test.graph.read", "replacement-read"),
    )
    try:
        provider, terminal = await _run(integration, admission, script)

        assert terminal.job.state is GraphState.SUCCEEDED
        provider.assert_consumed()
        replacement = _replacement(terminal, worker)
        planner_call = replacement.specification.expected_result_contract[
            "initial_call"
        ]
        assert planner_call == owner_call
        assert isinstance(owner_call, Mapping)
        assert owner_call["callable_tool_names"] == (READ_TOOL_NAME,)
        assert replacement.specification.description.endswith(
            "Planner replacement note: Same call, admitted again by the planner."
        )
        direct = await admit_graph_call(
            integration.registry,
            READ_CAPABILITY_ID,
            raw_arguments,
            resource_ids=tuple(sorted(resource_ids)),
        )
        assert FrozenJsonObject.from_mapping(direct.material()) == owner_call
    finally:
        await integration.close()


def _optional_registry() -> CapabilityRegistry:
    capability = Capability(
        id="test.graph.optional_read",
        description="Read with one optional bound.",
        input_schema={
            "type": "object",
            "properties": {
                "resource_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 16,
                },
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["resource_ids"],
            "additionalProperties": False,
        },
        output_kind="test.graph.optional_read",
        output_schema={"type": "object", "additionalProperties": True},
        executor_id="test.graph.optional_read.executor",
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
        execution_admission_policy=ExecutionAdmissionPolicy(
            shape="relational_read",
            inline_eligible=True,
            graph_v1_eligible=True,
            target_count_argument="resource_ids",
            inline_max_targets=2,
            graph_max_targets=3,
        ),
    )
    view = ToolView(
        name="optional_read",
        capability_id=capability.id,
        description=capability.description,
        presentation=ToolPresentation(
            toolbox_id=ToolboxId.SOURCES,
            load_mode=ToolLoadMode.PINNED,
            text_trust=ToolTextTrust.CODE,
            summary=capability.description,
            when_to_use="Use for the optional graph read test.",
            keywords=("graph", "read"),
        ),
    )

    class _Executor:
        executor_id = capability.executor_id

        async def execute(self, request):  # pragma: no cover - never dispatched
            raise AssertionError(request)

    return CapabilityRegistry(
        declarations=(
            CapabilityDeclarations(
                domain_owner_id="test.graph.optional",
                capabilities=(capability,),
                executor_ids=(capability.executor_id,),
                tool_views=(view,),
            ),
        ),
        executors=(_Executor(),),
    )


async def test_shared_call_admission_normalises_unset_nulls_and_checks_bounds() -> None:
    registry = _optional_registry()
    admitted = await admit_graph_call(
        registry,
        "test.graph.optional_read",
        {"resource_ids": ["a"], "limit": None},
        resource_ids=("a", "b"),
    )
    assert dict(admitted.arguments) == {"resource_ids": ("a",)}
    assert admitted.callable_tool_names == ("optional_read",)

    with pytest.raises(CapabilityInputError) as error:
        await admit_graph_call(
            registry,
            "test.graph.optional_read",
            {"resource_ids": ["a", "b", "c", "d"]},
            resource_ids=("a", "b", "c", "d"),
        )
    assert error.value.code == "execution_not_supported_for_requested_shape"
    with pytest.raises(CapabilityInputError) as error:
        await admit_graph_call(
            registry,
            "test.graph.optional_read",
            {"resource_ids": ["a", "z"]},
            resource_ids=("a",),
        )
    assert error.value.code == "source_scope_violation"
    with pytest.raises(CapabilityInputError) as error:
        await admit_graph_call(registry, "test.graph.unknown", {})
    assert error.value.code == "execution_not_supported_for_requested_shape"


@pytest.mark.parametrize(
    "initial_call",
    (
        None,
        {"capability_id": "capability.read", "arguments": {}},
        {
            "capability_id": "capability.read",
            "callable_tool_names": (),
            "arguments": {},
        },
        {
            "capability_id": "capability.read",
            "callable_tool_names": ("read",),
            "arguments": "not-an-object",
        },
        {
            "capability_id": "capability.other",
            "callable_tool_names": ("read",),
            "arguments": {},
        },
    ),
)
def test_validate_mutation_rejects_unadmitted_model_created_calls(
    initial_call: object,
) -> None:
    from daita.jobs.graph.models import GraphJob, JobGraph
    from daita.jobs.graph.validation import validate_mutation

    admission = graph_admission()
    worker = next(item for item in admission.tasks if item.role is TaskRole.WORKER)
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
    job: GraphJob = admission.job
    graph: JobGraph = admission.graph
    limits = job.specification.limits
    with pytest.raises(GraphValidationError) as error:
        validate_mutation(
            job_authority=job.specification.authority,
            graph=graph,
            existing_tasks=admission.tasks,
            existing_dependencies=admission.dependencies,
            request=GraphMutationRequest(
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
            ),
            max_tasks=limits.max_tasks,
            max_edges=limits.max_edges,
            max_depth=limits.max_depth,
            max_direct_parents=limits.max_direct_parents,
            max_fan_out=limits.max_fan_out,
        )
    assert error.value.code == "model_task_call_invalid"
