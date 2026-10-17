"""Run-scoped source/resource handles and their exact full-ID resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from daita import Agent
from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalRequest,
    AutomationEligibility,
    CallTargetLane,
    Capability,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    EffectReceiptPolicy,
    OperationalEffect,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolView,
    resource_id_schema,
    source_id_schema,
)
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
)
from daita.llm.providers.anthropic.messages import _message_input
from daita.llm.providers.openai.messages import _response_input
from daita.loop.models import RunInput, Transcript
from daita.scope import EffectiveSourceScope, scope_refs
from tests.capabilities._effect_receipt_support import STARTED_AT, _store
from tests.support.capability_runtime import (
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
    static_registry,
)
from tests.support.workspace import workspace_for

SOURCE_A = "source:sha256:" + "a" * 64
SOURCE_B = "source:sha256:" + "b" * 64
RESOURCE_A = "catalog-resource:sha256:" + "1" * 64
RESOURCE_B = "catalog-resource:sha256:" + "2" * 64
RESOURCE_C = "catalog-resource:sha256:" + "3" * 64
SCOPE = EffectiveSourceScope(
    frozenset({SOURCE_B, SOURCE_A}),
    frozenset({RESOURCE_C, RESOURCE_A, RESOURCE_B}),
)


def test_handles_number_the_frozen_scope_and_differ_when_the_scope_differs():
    refs = scope_refs(SCOPE)
    assert dict(refs.handles) == {
        SOURCE_A: "s1",
        SOURCE_B: "s2",
        RESOURCE_A: "t1",
        RESOURCE_B: "t2",
        RESOURCE_C: "t3",
    }
    assert {handle: refs.ids[handle] for handle in refs.ids} == {
        handle: identity for identity, handle in refs.handles.items()
    }
    # Every step of one run sees the same frozen scope, hence the same handles.
    again = EffectiveSourceScope(
        frozenset({SOURCE_A, SOURCE_B}),
        frozenset({RESOURCE_A, RESOURCE_B, RESOURCE_C}),
    )
    assert dict(scope_refs(again).handles) == dict(refs.handles)
    # Another run's scope numbers its own identities: a handle names nothing
    # outside the run that issued it.
    other = scope_refs(EffectiveSourceScope(frozenset({SOURCE_B}), frozenset()))
    assert dict(other.handles) == {SOURCE_B: "s1"}
    assert other.resolve("s", "s2") is None
    assert other.resolve("t", "s1") is None
    assert not scope_refs(None).handles
    assert not scope_refs(EffectiveSourceScope(frozenset(), frozenset())).handles


class _Recorder:
    def __init__(self, executor_id: str) -> None:
        self.executor_id = executor_id
        self.requests: list[ToolExecution] = []

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.requests.append(request)
        return ToolOutput(kind="test.scope.output", data={"ok": True})


class _Writer:
    executor_id = "test.scope.write.executor"

    def __init__(self) -> None:
        self.requests: list[ToolExecution] = []

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping({"current": True})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.requests.append(request)
        payload = FrozenJsonObject.from_mapping(
            {
                "source_id": request.arguments["source_id"],
                "resource_ids": request.arguments["resource_ids"],
            }
        )
        return ToolOutput(
            kind="test.scope.write",
            data=payload,
            effect_observation=EffectObservation(
                EffectOutcome.SUCCEEDED, EffectEvidenceBasis.SERVER_REPORTED, payload
            ),
        )


_IDENTITY_INPUT = {
    "type": "object",
    "properties": {
        "source_id": source_id_schema(),
        "resource_ids": {
            "type": "array",
            "items": resource_id_schema(),
            "minItems": 1,
            "maxItems": 4,
        },
    },
    "required": ["source_id", "resource_ids"],
    "additionalProperties": False,
}
_OK_OUTPUT = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _fixture(store) -> tuple[CapabilityRuntime, _Recorder, _Recorder, _Writer, list]:
    read = Capability(
        id="test.scope.read",
        description="Read one exact source.",
        input_schema=_IDENTITY_INPUT,
        output_kind="test.scope.output",
        output_schema=_OK_OUTPUT,
        executor_id="test.scope.read.executor",
        access_mode=AccessMode.READ,
    )
    propose = Capability(
        id="test.scope.propose",
        description="Retain another capability's exact arguments.",
        input_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "object",
                    "properties": {
                        "capability_id": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["capability_id", "arguments"],
                    "additionalProperties": False,
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        output_kind="test.scope.output",
        output_schema=_OK_OUTPUT,
        executor_id="test.scope.propose.executor",
        delegated_arguments=(("task", "arguments", "input"),),
    )
    write = Capability(
        id="test.scope.write",
        description="Act on one exact source.",
        input_schema=_IDENTITY_INPUT,
        output_kind="test.scope.write",
        output_schema=_IDENTITY_INPUT,
        executor_id=_Writer.executor_id,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.EXTERNAL_ACTION,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        effect_receipt_policy=EffectReceiptPolicy(
            "test.scope.write", _IDENTITY_INPUT, EffectEvidenceBasis.SERVER_REPORTED
        ),
    )
    views = tuple(
        ToolView(
            name=capability.id.rsplit(".", 1)[1],
            capability_id=capability.id,
            description=capability.description,
            presentation=presentation_metadata(
                load_mode=(
                    ToolLoadMode.ON_DEMAND
                    if capability.operational_effect is not OperationalEffect.NONE
                    else ToolLoadMode.PINNED
                )
            ),
        )
        for capability in (read, propose, write)
    )
    domain = StaticTestDomain((read, propose, write), views)
    reader = _Recorder(read.executor_id)
    proposer = _Recorder(propose.executor_id)
    writer = _Writer()
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    async def resolve_scope(run: RunInput) -> EffectiveSourceScope:
        return SCOPE

    runtime = CapabilityRuntime(
        static_registry(domain, (reader, proposer, writer)),
        (domain,),
        effect_receipts=store,
        approval_handler=approve,
        clock=lambda: STARTED_AT,
        source_scope_resolver=resolve_scope,
    )
    return runtime, reader, proposer, writer, approvals


_RUN = RunInput(
    id="run-effect",
    agent_id="agent-effect",
    message="exact handles",
    created_at=STARTED_AT,
    conversation_id="conversation-effect",
)


def _frozen(value: Mapping[str, object]) -> FrozenJsonObject:
    return FrozenJsonObject.from_mapping(value)


def _error(result: ToolResultBlock) -> Mapping[str, object]:
    assert result.is_error
    error = result.output["error"]
    assert isinstance(error, Mapping)
    details = error["details"]
    assert isinstance(details, Mapping)
    assert details["execution_disposition"] == "not_dispatched"
    return error


async def test_handles_reach_execution_approval_and_receipt_as_full_ids(tmp_path):
    store = await _store(tmp_path / "state.db")
    runtime, reader, proposer, writer, approvals = _fixture(store)
    sent = (
        ToolCall("read", "read", {"source_id": "s2", "resource_ids": ("t1", "t3")}),
        ToolCall(
            "propose",
            "propose",
            {
                "task": {
                    "capability_id": "test.scope.read",
                    "arguments": {"source_id": "s1", "resource_ids": ["t2"]},
                }
            },
        ),
        ToolCall("write", "write", {"source_id": "s1", "resource_ids": ("t2",)}),
    )
    try:
        outcome = await execute_projected(runtime, _RUN, sent)
        assert not any(result.is_error for result in outcome.ordered_results)
        assert reader.requests[0].arguments == _frozen(
            {"source_id": SOURCE_B, "resource_ids": (RESOURCE_A, RESOURCE_C)}
        )
        task = proposer.requests[0].arguments["task"]
        assert isinstance(task, Mapping)
        full = _frozen({"source_id": SOURCE_A, "resource_ids": (RESOURCE_B,)})
        assert task["arguments"] == full
        assert writer.requests[0].arguments == full
        assert [request.arguments for request in approvals] == [full]
        (receipt,) = await store.list_effect_receipts("agent-effect")
        assert receipt.payload == full
        # The model's calls keep exactly what it sent.
        assert sent[2].arguments["source_id"] == "s1"

        # The same operation named by full IDs is the same reserved effect.
        again = await execute_projected(
            runtime, _RUN, (ToolCall("write-again", "write", full),)
        )
        reference = outcome.ordered_results[2].output["effect_receipt"]
        assert again.ordered_results[0].output["effect_receipt"] == reference
        assert len(writer.requests) == 1
    finally:
        await store.close()


async def test_unissued_handles_and_garbled_ids_fail_before_dispatch(tmp_path):
    store = await _store(tmp_path / "state.db")
    runtime, reader, proposer, writer, approvals = _fixture(store)
    garbled = SOURCE_A[:-1]
    calls = (
        ToolCall("unknown", "read", {"source_id": "s3", "resource_ids": ("t1",)}),
        ToolCall("family", "read", {"source_id": "t1", "resource_ids": ("t1",)}),
        ToolCall("resource", "read", {"source_id": "s1", "resource_ids": ("t9",)}),
        ToolCall(
            "nested",
            "propose",
            {
                "task": {
                    "capability_id": "test.scope.read",
                    "arguments": {"source_id": "s7", "resource_ids": ["t1"]},
                }
            },
        ),
        ToolCall("garbled", "read", {"source_id": garbled, "resource_ids": ("t1",)}),
        ToolCall("effect", "write", {"source_id": "s5", "resource_ids": ("t1",)}),
    )
    try:
        results = (await execute_projected(runtime, _RUN, calls)).ordered_results
        errors = [_error(result) for result in results]
        assert [error["code"] for error in errors] == [
            "source_ref_invalid",
            "source_ref_invalid",
            "resource_ref_invalid",
            "source_ref_invalid",
            "invalid_argument_value",
            "source_ref_invalid",
        ]
        recovery = errors[0]["details"]
        assert isinstance(recovery, Mapping)
        assert recovery["name"] == "source_id"
        assert recovery["recovery"] == _frozen(
            {
                "kind": "correct_arguments",
                "constraint": "Use a source_ref (s1-s2) shown beside its source_id "
                "in this run, or the exact full source_id.",
            }
        )
        nested = errors[3]["details"]
        assert isinstance(nested, Mapping)
        assert nested["name"] == "task.arguments.source_id"
        garbled_details = errors[4]["details"]
        assert isinstance(garbled_details, Mapping)
        assert garbled_details["name"] == "source_id"
        assert garbled_details["constraint"] == "pattern"
        assert not reader.requests and not proposer.requests
        assert not writer.requests and not approvals
        assert not await store.list_effect_receipts("agent-effect")
    finally:
        await store.close()


def test_rendered_results_show_each_issued_handle_beside_its_full_id():
    output = {
        "kind": "test.scope.output",
        "data": {
            "hits": [
                {"source_id": SOURCE_A, "resource_id": RESOURCE_C, "name": "orders"},
                {"source_id": "source:sha256:" + "f" * 64, "name": "outside"},
            ],
            "resource_ids": [RESOURCE_A, "catalog-resource:sha256:" + "9" * 64],
            "evidence_call_ids": ["call-1"],
        },
    }
    block = ToolResultBlock(call_id="call-1", output=output)
    handles = scope_refs(SCOPE).handles
    rendered = model_tool_result(block, "r1", handles)
    data = rendered["output"]["data"]  # type: ignore[index]
    assert data["hits"][0]["source_ref"] == "s1"
    assert data["hits"][0]["resource_ref"] == "t3"
    assert "source_ref" not in data["hits"][1]
    assert data["resource_refs"] == ["t1", None]
    assert "evidence_call_refs" not in data
    # Stored output is unchanged; only the rendering carries handles.
    assert "source_ref" not in canonical_json(block.output)

    request = ModelRequest(
        messages=(
            CanonicalMessage(
                MessageRole.ASSISTANT,
                tool_calls=(ToolCall("call-1", "read", {"source_id": "s1"}),),
            ),
            CanonicalMessage(MessageRole.TOOL, content=(block,)),
        ),
        tool_result_refs={"call-1": "r1"},
        scope_refs=handles,
    )
    _, anthropic = _message_input(
        request.messages, "anthropic", request.tool_result_refs, request.scope_refs
    )
    openai = _response_input(
        request.messages, "openai", request.tool_result_refs, request.scope_refs
    )
    for text in (
        str(anthropic[-1]["content"][0]["content"]),  # type: ignore[index]
        str(next(item["output"] for item in openai if "output" in item)),
    ):
        assert '"resource_ref":"t3"' in text and '"source_ref":"s1"' in text


async def test_every_model_facing_identity_argument_admits_handles(
    tmp_path: Path,
) -> None:
    agent = await Agent.create(
        "scope-ref-inventory", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    names = ("source_id", "source_ids", "resource_id", "resource_ids")
    patterns = {
        "source": "^(source:sha256:[0-9a-f]{64}|s[1-9][0-9]{0,4})$",
        "resource": "^(catalog-resource:sha256:[0-9a-f]{64}|t[1-9][0-9]{0,4})$",
    }
    found: dict[str, str | None] = {}

    def visit(tool: str, rule: object, path: str) -> None:
        if not isinstance(rule, Mapping):
            return
        properties = rule.get("properties")
        if isinstance(properties, Mapping):
            for key, child in properties.items():
                if isinstance(child, Mapping) and str(key).endswith(names):
                    leaf = child.get("items", child)
                    assert isinstance(leaf, Mapping)
                    if leaf.get("type") == "string":
                        found[f"{tool}:{path}.{key}"] = leaf.get("pattern")
                visit(tool, child, f"{path}.{key}")
        for key in ("items", "additionalProperties"):
            visit(tool, rule.get(key), f"{path}[{key}]")

    try:
        registry = agent._embedded._capabilities
        for name in sorted(registry.tool_names):
            definition = registry.tool_definition(name, lane=CallTargetLane.FOREGROUND)
            visit(name, definition.input_schema, "input")
    finally:
        await agent.close()

    assert {"data_query:input.source_id", "catalog_describe:input.resource_ids"} <= (
        set(found)
    )
    wrong = {
        path: pattern
        for path, pattern in found.items()
        if pattern != patterns["resource" if "resource" in path else "source"]
    }
    assert wrong == {}


async def test_routine_save_persists_full_ids_named_by_handles(tmp_path: Path) -> None:
    import sqlite3
    from datetime import UTC, datetime, timedelta

    from daita import SQLiteSource
    from daita.distribution import outcome_contract_projection
    from daita.llm.models import FinishReason, ModelProfile, ModelResponse
    from daita.llm.providers.mock import MockModelProvider
    from tests.support.distribution import no_artifact_outcome_contract

    database = tmp_path / "current.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
    connection.commit()
    connection.close()
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="Authorize it."),)
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "scope-ref-routine",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=64_000,
            max_output_tokens=1_000,
            supports_tools=True,
            supports_parallel_tools=True,
        ),
        approval_handler=approve,
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Current"))
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        origin = await agent.run("Create a scheduled report.")
        assert origin.conversation_id is not None
        destinations = await agent.distribution_destinations(origin.conversation_id)
        run = (await agent._embedded._store.load(origin.run_id)).run
        now = datetime.now(UTC)
        arguments = {
            "title": "Report",
            "authorized_instruction": "Describe the current_value resource.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
            "misfire_policy": "latest_only",
            "reporting_mode": "always",
            "allowed_source_ids": ("s1",),
            "allowed_resource_ids": ("t1",),
            "allowed_connector_binding_ids": (),
            "allowed_capability_ids": ("catalog.describe",),
            "sensitivity_ceiling": "internal",
            "outcome_contract": outcome_contract_projection(
                no_artifact_outcome_contract()
            ),
            "distribution_destination_id": destinations[0].destination_id,
            "eligible_model_routes": (provider.provider_id,),
            "per_run_max_tokens": 1000,
            "per_run_max_cost_usd": "0.01",
            "cumulative_max_tokens": 10000,
            "cumulative_max_cost_usd": "0.10",
            "cumulative_max_attempts": 10,
            "cumulative_max_occurrences": 10,
            "maximum_consecutive_failures": 3,
            "expires_at": (now + timedelta(days=30)).isoformat(),
            "skill_names": (),
            "run_immediately": False,
        }
        created = await execute_projected(
            agent._embedded._capability_runtime,
            run,
            (ToolCall("handle-routine", "routine_save", arguments),),
        )
        result = created.ordered_results[0]
        assert not result.is_error, result.output
        (summary,) = await agent.list_routines()
        inspection = await agent.inspect_routine(summary.routine_id)
        assert inspection is not None
        routine = inspection.routine
        assert routine.allowed_source_ids == (source.id,)
        assert routine.allowed_resource_ids == (resource.id,)
        assert approvals
        approved = canonical_json(approvals[-1].arguments)
        assert source.id in approved and resource.id in approved
        assert '"s1"' not in approved and '"t1"' not in approved
    finally:
        await agent.close()


class _Citer:
    executor_id = "test.scope.cite.executor"

    def __init__(self) -> None:
        self.requests: list[ToolExecution] = []

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping({"current": True})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.requests.append(request)
        payload = FrozenJsonObject.from_mapping(
            {
                "source_id": request.arguments["source_id"],
                "resource_ids": request.arguments["resource_ids"],
            }
        )
        return ToolOutput(
            kind="test.scope.write",
            data=payload,
            effect_observation=EffectObservation(
                EffectOutcome.SUCCEEDED, EffectEvidenceBasis.SERVER_REPORTED, payload
            ),
        )


async def test_executed_record_is_exactly_what_execution_approval_and_receipt_saw(
    tmp_path,
):
    """One resolver derives the executed record every reader shares.

    A call naming a result ref and run-scoped handles is persisted as the model
    sent it; the registry's executed record of that persisted call equals the
    executor's request, the approval request and the receipt payload.
    """

    store = await _store(tmp_path / "state.db")
    read = Capability(
        id="test.scope.read",
        description="Read one exact source.",
        input_schema=_IDENTITY_INPUT,
        output_kind="test.scope.output",
        output_schema=_OK_OUTPUT,
        executor_id="test.scope.read.executor",
        access_mode=AccessMode.READ,
    )
    cite_input = {
        "type": "object",
        "properties": {
            "source_id": source_id_schema(),
            "resource_ids": {
                "type": "array",
                "items": resource_id_schema(),
                "minItems": 1,
                "maxItems": 4,
            },
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
        },
        "required": ["source_id", "resource_ids", "evidence_refs"],
        "additionalProperties": False,
    }
    cite = Capability(
        id="test.scope.cite",
        description="Act on one exact source citing earlier evidence.",
        input_schema=cite_input,
        output_kind="test.scope.write",
        output_schema=_IDENTITY_INPUT,
        executor_id=_Citer.executor_id,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.EXTERNAL_ACTION,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        effect_receipt_policy=EffectReceiptPolicy(
            "test.scope.cite", _IDENTITY_INPUT, EffectEvidenceBasis.SERVER_REPORTED
        ),
        result_ref_arguments=("evidence_refs[]",),
    )
    views = tuple(
        ToolView(
            name=capability.id.rsplit(".", 1)[1],
            capability_id=capability.id,
            description=capability.description,
            presentation=presentation_metadata(
                load_mode=(
                    ToolLoadMode.ON_DEMAND
                    if capability.operational_effect is not OperationalEffect.NONE
                    else ToolLoadMode.PINNED
                )
            ),
        )
        for capability in (read, cite)
    )
    domain = StaticTestDomain((read, cite), views)
    reader = _Recorder(read.executor_id)
    citer = _Citer()
    registry = static_registry(domain, (reader, citer))
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    async def resolve_scope(run: RunInput) -> EffectiveSourceScope:
        return SCOPE

    runtime = CapabilityRuntime(
        registry,
        (domain,),
        effect_receipts=store,
        approval_handler=approve,
        clock=lambda: STARTED_AT,
        source_scope_resolver=resolve_scope,
    )
    run = await runtime.freeze_source_scope(_RUN)
    assert run.resolved_source_scope == SCOPE
    load = ToolCall("load", "toolbox_load", {"tool_names": ("cite",)})
    sent_read = ToolCall("read", "read", {"source_id": "s2", "resource_ids": ("t3",)})
    # The control result is numbered r1 but is not citable; the read is r2.
    sent_cite = ToolCall(
        "cite",
        "cite",
        {
            "source_id": "s1",
            "resource_ids": ("t2", RESOURCE_A),
            "evidence_refs": ("r2",),
        },
    )
    try:
        catalog = await runtime.prepare_run(run)
        messages: tuple[CanonicalMessage, ...] = (run.start_message(),)
        for call in (load, sent_read, sent_cite):
            outcome = await runtime.execute_all(
                run,
                (call,),
                projection=runtime.project(catalog, messages),
                messages=messages,
                sensitivity=ModelSensitivity.INTERNAL,
            )
            messages = (
                *messages,
                CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
                CanonicalMessage(MessageRole.TOOL, content=outcome.ordered_results),
            )
        transcript = Transcript(run, messages)
        results = [result for _, result in transcript.tool_pairs]
        assert all(result is not None and not result.is_error for result in results)

        load_pair, read_pair, cite_pair = registry.executed_tool_pairs(transcript)
        assert [load_pair[1], read_pair[1], cite_pair[1]] == results
        assert load_pair[0] == load
        assert read_pair[0].arguments == reader.requests[0].arguments
        assert read_pair[0].arguments == _frozen(
            {"source_id": SOURCE_B, "resource_ids": (RESOURCE_C,)}
        )
        executed = cite_pair[0].arguments
        assert executed == _frozen(
            {
                "source_id": SOURCE_A,
                "resource_ids": (RESOURCE_B, RESOURCE_A),
                "evidence_refs": ("read",),
            }
        )
        assert executed == citer.requests[0].arguments
        assert [request.arguments for request in approvals] == [executed]
        (receipt,) = await store.list_effect_receipts("agent-effect")
        assert (receipt.run_id, receipt.call_id) == (run.id, cite_pair[0].id)
        assert receipt.payload == _frozen(
            {"source_id": SOURCE_A, "resource_ids": (RESOURCE_B, RESOURCE_A)}
        )
        # The transcript keeps the model's exact text for provider replay.
        assert [call for call, _ in transcript.tool_pairs] == [
            load,
            sent_read,
            sent_cite,
        ]

        # A call that never resolved failed before execution and is unchanged.
        unissued = ToolCall(
            "unissued", "read", {"source_id": "s9", "resource_ids": ("t1",)}
        )
        assert (
            registry.executed_call(unissued, messages=messages, source_scope=SCOPE)
            == unissued
        )
        unknown = ToolCall("unknown", "not_a_tool", {"source_id": "s1"})
        assert (
            registry.executed_call(unknown, messages=(), source_scope=SCOPE) == unknown
        )
    finally:
        await store.close()
