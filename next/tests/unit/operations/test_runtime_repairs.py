from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import LoopPhase, Readiness
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Observation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationStateError

NOW = datetime(2026, 7, 17, 11, 0, tzinfo=timezone.utc)

READ_FAKE_TOOL = ToolDefinition(
    name="read_fake",
    description="Read one fake value.",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
)


class SuccessfulExecutor:
    executor_id = "fake.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={
                "key": request.arguments["key"],
                "value": "ALPHA",
            },
        )


def _registry(executor: SuccessfulExecutor) -> CapabilityRegistry:
    return CapabilityRegistry(
        capabilities=(
            Capability(
                id="fake.read",
                owner="loop-lab",
                description="Read one fake value.",
                input_schema=READ_FAKE_TOOL.input_schema,
                output_evidence_kind="fake.read.result",
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
            ),
        ),
        executors=(executor,),
        tool_views=(
            ToolView(
                name=READ_FAKE_TOOL.name,
                capability_id="fake.read",
                description=READ_FAKE_TOOL.description,
            ),
        ),
    )


async def _begin_runtime(
    *,
    capabilities: CapabilityRegistry | None = None,
    id_factory: Callable[[str], str] | None = None,
) -> tuple[OperationRuntime, str]:
    if id_factory is None:
        runtime = OperationRuntime(
            clock=lambda: NOW,
            capabilities=capabilities,
        )
    else:
        runtime = OperationRuntime(
            clock=lambda: NOW,
            capabilities=capabilities,
            id_factory=id_factory,
        )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-1",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "Read alpha."},
            created_at=NOW,
        )
    )
    return runtime, started.operation.id


async def _commit_tool_call(
    runtime: OperationRuntime,
    operation_id: str,
    call: ToolCall,
) -> str:
    return await _commit_tool_calls(runtime, operation_id, (call,))


async def _commit_tool_calls(
    runtime: OperationRuntime,
    operation_id: str,
    calls: tuple[ToolCall, ...],
) -> str:
    turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Read alpha."),),
            ),
        ),
        tools=(READ_FAKE_TOOL,),
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=calls,
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return turn.id


async def _commit_text_response(
    runtime: OperationRuntime,
    operation_id: str,
    text: str,
) -> str:
    turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Answer now."),),
            ),
        ),
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(text=text, finish_reason=FinishReason.STOP),
        next_phase=LoopPhase.SYNTHESIZING,
    )
    return turn.id


def _rejection() -> ActionRejection:
    return ActionRejection(
        code="action.invalid_arguments",
        message="The requested key is not allowed.",
        details={"field": "key", "expected": "an allowed key"},
    )


def _fingerprint(call: ToolCall) -> str:
    normalized = canonical_json(
        {
            "arguments": call.arguments,
            "tool_name": call.name,
        }
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_action_rejection_is_bounded_and_mutation_isolated() -> None:
    details: dict[str, object] = {"field": "key"}
    rejection = ActionRejection(
        code="action.invalid_arguments",
        message="The requested key is invalid.",
        details=details,
    )
    details["field"] = "mutated"

    assert isinstance(rejection.details, FrozenJsonObject)
    assert rejection.details.to_dict() == {"field": "key"}

    with pytest.raises(ValueError, match="bounded"):
        ActionRejection(
            code="action.invalid_arguments",
            message="x" * 513,
        )


@pytest.mark.parametrize(
    "forged_call",
    [
        ToolCall(id="forged-call", name="read_fake", arguments={"key": "alpha"}),
        ToolCall(id="call-1", name="forged_tool", arguments={"key": "alpha"}),
        ToolCall(id="call-1", name="read_fake", arguments={"key": "beta"}),
    ],
)
async def test_action_rejection_must_bind_to_the_exact_committed_tool_call(
    forged_call: ToolCall,
) -> None:
    runtime, operation_id = await _begin_runtime()
    turn_id = await _commit_tool_call(
        runtime,
        operation_id,
        ToolCall(id="call-1", name="read_fake", arguments={"key": "alpha"}),
    )
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="committed|tool call"):
        await runtime.record_action_rejection(
            operation_id,
            turn_id,
            forged_call,
            _rejection(),
        )

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.observations == ()
    assert after.tasks == ()


async def test_action_rejection_commits_an_unsuccessful_no_task_observation() -> None:
    runtime, operation_id = await _begin_runtime()
    call = ToolCall(
        id="call-1",
        name="read_fake",
        arguments={"key": "alpha"},
    )
    turn_id = await _commit_tool_call(runtime, operation_id, call)

    observation = await runtime.record_action_rejection(
        operation_id,
        turn_id,
        call,
        _rejection(),
    )

    snapshot = await runtime.inspect(operation_id)
    fingerprint = _fingerprint(call)
    assert observation.operation_id == operation_id
    assert observation.turn_id == turn_id
    assert observation.call_id == "call-1"
    assert observation.code == "action.invalid_arguments"
    assert observation.message == "The requested key is not allowed."
    assert observation.success is False
    assert observation.task_id is None
    assert observation.evidence_id is None
    assert isinstance(observation.payload, FrozenJsonObject)
    assert observation.payload.to_dict() == {
        "field": "key",
        "expected": "an allowed key",
    }
    assert snapshot.observations == (observation,)
    assert snapshot.tasks == ()
    assert snapshot.evidence == ()
    assert snapshot.loop_state.phase is LoopPhase.OBSERVING
    assert snapshot.loop_state.repair_count == 1
    assert snapshot.loop_state.identical_failure_count == 1
    assert snapshot.loop_state.no_progress_fingerprints == (fingerprint,)
    assert [event.type for event in snapshot.events[-2:]] == [
        "action.rejected",
        "observation.recorded",
    ]
    assert all(event.call_id == "call-1" for event in snapshot.events[-2:])
    action_event = snapshot.events[-2]
    assert action_event.payload["tool_name"] == "read_fake"
    assert action_event.payload["fingerprint"] == fingerprint
    assert "failure_fingerprint" not in action_event.payload


async def test_rejections_track_consecutive_duplicates_and_full_history() -> None:
    runtime, operation_id = await _begin_runtime()

    calls: list[ToolCall] = []
    for index in range(1, 4):
        call = ToolCall(
            id=f"call-{index}",
            name="read_fake",
            arguments={"key": "alpha" if index < 3 else "beta"},
        )
        calls.append(call)
        turn_id = await _commit_tool_call(runtime, operation_id, call)
        await runtime.record_action_rejection(
            operation_id,
            turn_id,
            call,
            _rejection(),
        )

        snapshot = await runtime.inspect(operation_id)
        assert snapshot.loop_state.repair_count == index
        assert snapshot.loop_state.identical_failure_count == (
            index if index < 3 else 1
        )

    assert snapshot.loop_state.no_progress_fingerprints == (
        _fingerprint(calls[0]),
        _fingerprint(calls[2]),
    )


async def test_rejection_fingerprint_ignores_call_id_and_rejection_code() -> None:
    runtime, operation_id = await _begin_runtime()
    rejection_codes = ("action.invalid_type", "action.disallowed_value")

    for index, rejection_code in enumerate(rejection_codes, start=1):
        call = ToolCall(
            id=f"provider-call-{index}",
            name="read_fake",
            arguments={"key": "alpha"},
        )
        turn_id = await _commit_tool_call(runtime, operation_id, call)
        await runtime.record_action_rejection(
            operation_id,
            turn_id,
            call,
            ActionRejection(
                code=rejection_code,
                message="The same normalized action remains invalid.",
            ),
        )

    snapshot = await runtime.inspect(operation_id)
    assert snapshot.loop_state.identical_failure_count == 2
    assert snapshot.loop_state.no_progress_fingerprints == (_fingerprint(call),)
    assert [
        event.payload["fingerprint"]
        for event in snapshot.events
        if event.type == "action.rejected"
    ] == [_fingerprint(call), _fingerprint(call)]


async def test_rejection_atomically_skips_every_later_committed_call() -> None:
    runtime, operation_id = await _begin_runtime()
    calls = (
        ToolCall(id="call-rejected", name="read_fake", arguments={"key": 7}),
        ToolCall(id="call-skipped-a", name="read_fake", arguments={"key": "a"}),
        ToolCall(id="call-skipped-b", name="read_fake", arguments={"key": "b"}),
    )
    turn_id = await _commit_tool_calls(runtime, operation_id, calls)

    rejected = await runtime.record_action_rejection(
        operation_id,
        turn_id,
        calls[0],
        _rejection(),
    )

    snapshot = await runtime.inspect(operation_id)
    assert snapshot.observations[0] == rejected
    assert [observation.call_id for observation in snapshot.observations] == [
        call.id for call in calls
    ]
    assert [observation.code for observation in snapshot.observations] == [
        "action.invalid_arguments",
        "action.skipped_after_rejection",
        "action.skipped_after_rejection",
    ]
    for skipped in snapshot.observations[1:]:
        assert not skipped.success
        assert skipped.task_id is None
        assert skipped.evidence_id is None
        assert isinstance(skipped.payload, FrozenJsonObject)
        assert skipped.payload.to_dict() == {
            "blocked_by_call_id": "call-rejected",
            "blocked_by_code": "action.invalid_arguments",
        }
    assert snapshot.tasks == ()
    assert snapshot.evidence == ()
    assert snapshot.loop_state.repair_count == 1
    assert snapshot.loop_state.identical_failure_count == 1
    assert snapshot.loop_state.no_progress_fingerprints == (_fingerprint(calls[0]),)
    assert [event.type for event in snapshot.events[-6:]] == [
        "action.rejected",
        "observation.recorded",
        "action.skipped",
        "observation.recorded",
        "action.skipped",
        "observation.recorded",
    ]
    assert [event.call_id for event in snapshot.events[-6:]] == [
        "call-rejected",
        "call-rejected",
        "call-skipped-a",
        "call-skipped-a",
        "call-skipped-b",
        "call-skipped-b",
    ]


async def test_accepted_evidence_resets_the_consecutive_failure_count() -> None:
    executor = SuccessfulExecutor()
    runtime, operation_id = await _begin_runtime(capabilities=_registry(executor))
    rejected_call = ToolCall(
        id="call-1",
        name="read_fake",
        arguments={"key": "invalid"},
    )
    rejected_turn_id = await _commit_tool_call(
        runtime,
        operation_id,
        rejected_call,
    )
    await runtime.record_action_rejection(
        operation_id,
        rejected_turn_id,
        rejected_call,
        _rejection(),
    )

    successful_call = ToolCall(
        id="call-2",
        name="read_fake",
        arguments={"key": "alpha"},
    )
    successful_turn_id = await _commit_tool_call(
        runtime,
        operation_id,
        successful_call,
    )
    evidence = await runtime.submit(
        ActionProposal(
            operation_id=operation_id,
            turn_id=successful_turn_id,
            call_id=successful_call.id,
            capability_id="fake.read",
            arguments=successful_call.arguments,
            proposed_at=NOW,
        )
    )
    await runtime.append_observation(
        Observation(
            operation_id=operation_id,
            turn_id=successful_turn_id,
            call_id=successful_call.id,
            code="fake.read.succeeded",
            message="Fake read completed.",
            payload=evidence.payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=NOW,
        )
    )

    snapshot = await runtime.inspect(operation_id)
    assert snapshot.loop_state.repair_count == 1
    assert snapshot.loop_state.identical_failure_count == 0
    assert snapshot.loop_state.no_progress_fingerprints == ()
    assert len(snapshot.evidence) == 1
    assert snapshot.evidence[0].accepted is True
    assert len(snapshot.observations) == 2
    assert len(executor.requests) == 1


async def test_success_observation_call_id_must_match_its_task() -> None:
    executor = SuccessfulExecutor()
    runtime, operation_id = await _begin_runtime(capabilities=_registry(executor))
    call = ToolCall(id="call-1", name="read_fake", arguments={"key": "alpha"})
    turn_id = await _commit_tool_call(runtime, operation_id, call)
    evidence = await runtime.submit(
        ActionProposal(
            operation_id=operation_id,
            turn_id=turn_id,
            call_id=call.id,
            capability_id="fake.read",
            arguments=call.arguments,
            proposed_at=NOW,
        )
    )
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="call_id.*task"):
        await runtime.append_observation(
            Observation(
                operation_id=operation_id,
                turn_id=turn_id,
                call_id="forged-call",
                code="fake.read.succeeded",
                message="Fake read completed.",
                payload=evidence.payload,
                success=True,
                task_id=evidence.task_id,
                evidence_id=evidence.id,
                created_at=NOW,
            )
        )

    assert await runtime.inspect(operation_id) == before


async def test_action_rejection_and_observation_events_commit_atomically() -> None:
    issued_id_count = 0
    reject_event_count = 0
    inject_failure = False

    def fail_second_rejection_event(prefix: str) -> str:
        nonlocal issued_id_count, reject_event_count
        issued_id_count += 1
        if inject_failure and prefix == "event":
            reject_event_count += 1
            if reject_event_count == 2:
                raise RuntimeError("injected rejection observation failure")
        return f"{prefix}-{issued_id_count}"

    runtime, operation_id = await _begin_runtime(id_factory=fail_second_rejection_event)
    call = ToolCall(
        id="call-1",
        name="read_fake",
        arguments={"key": "alpha"},
    )
    turn_id = await _commit_tool_call(runtime, operation_id, call)
    before = await runtime.inspect(operation_id)
    inject_failure = True

    with pytest.raises(RuntimeError, match="injected rejection observation failure"):
        await runtime.record_action_rejection(
            operation_id,
            turn_id,
            call,
            _rejection(),
        )

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.observations == ()
    assert after.loop_state.repair_count == 0
    assert after.loop_state.identical_failure_count == 0
    assert "action.rejected" not in [event.type for event in after.events]


async def test_rejection_skip_batch_commits_atomically() -> None:
    issued_id_count = 0
    rejection_event_count = 0
    inject_failure = False

    def fail_skip_observation_event(prefix: str) -> str:
        nonlocal issued_id_count, rejection_event_count
        issued_id_count += 1
        if inject_failure and prefix == "event":
            rejection_event_count += 1
            if rejection_event_count == 4:
                raise RuntimeError("injected skip observation failure")
        return f"{prefix}-{issued_id_count}"

    runtime, operation_id = await _begin_runtime(id_factory=fail_skip_observation_event)
    calls = (
        ToolCall(id="call-rejected", name="read_fake", arguments={"key": 7}),
        ToolCall(id="call-skipped", name="read_fake", arguments={"key": "a"}),
    )
    turn_id = await _commit_tool_calls(runtime, operation_id, calls)
    before = await runtime.inspect(operation_id)
    inject_failure = True

    with pytest.raises(RuntimeError, match="injected skip observation failure"):
        await runtime.record_action_rejection(
            operation_id,
            turn_id,
            calls[0],
            _rejection(),
        )

    assert await runtime.inspect(operation_id) == before


async def test_no_progress_requires_a_rejection_for_the_current_call() -> None:
    runtime, operation_id = await _begin_runtime()
    first_call = ToolCall(
        id="provider-call-a",
        name="read_fake",
        arguments={"key": "alpha"},
    )
    first_turn_id = await _commit_tool_call(runtime, operation_id, first_call)
    await runtime.record_action_rejection(
        operation_id,
        first_turn_id,
        first_call,
        _rejection(),
    )
    current_call = ToolCall(
        id="provider-call-b",
        name="read_fake",
        arguments={"key": "alpha"},
    )
    await _commit_tool_call(runtime, operation_id, current_call)
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="current rejection"):
        await runtime.fail_no_progress(operation_id, current_call.id)

    assert await runtime.inspect(operation_id) == before


async def test_no_progress_terminal_events_commit_atomically() -> None:
    issued_id_count = 0
    terminal_event_count = 0
    inject_failure = False

    def fail_operation_failed_event(prefix: str) -> str:
        nonlocal issued_id_count, terminal_event_count
        issued_id_count += 1
        if inject_failure and prefix == "event":
            terminal_event_count += 1
            if terminal_event_count == 2:
                raise RuntimeError("injected operation failure event")
        return f"{prefix}-{issued_id_count}"

    runtime, operation_id = await _begin_runtime(id_factory=fail_operation_failed_event)
    call = ToolCall(
        id="call-rejected",
        name="read_fake",
        arguments={"key": "alpha"},
    )
    turn_id = await _commit_tool_call(runtime, operation_id, call)
    await runtime.record_action_rejection(
        operation_id,
        turn_id,
        call,
        _rejection(),
    )
    before = await runtime.inspect(operation_id)
    inject_failure = True

    with pytest.raises(RuntimeError, match="injected operation failure event"):
        await runtime.fail_no_progress(operation_id, call.id)

    assert await runtime.inspect(operation_id) == before


async def test_denied_readiness_commits_a_structured_correction() -> None:
    runtime, operation_id = await _begin_runtime()
    turn_id = await _commit_text_response(runtime, operation_id, "Answer too early.")
    readiness = Readiness(
        allowed=False,
        code="readiness.missing_evidence",
        message="Accepted evidence is required before answering.",
        missing_facts=("accepted_evidence", "observed_result"),
        evaluated_at=NOW,
    )

    correction = await runtime.record_readiness(
        operation_id,
        "Answer too early.",
        readiness,
    )

    snapshot = await runtime.inspect(operation_id)
    assert correction is not None
    assert correction.operation_id == operation_id
    assert correction.turn_id == turn_id
    assert correction.call_id is None
    assert correction.code == readiness.code
    assert correction.message == readiness.message
    assert correction.success is False
    assert correction.task_id is None
    assert correction.evidence_id is None
    assert isinstance(correction.payload, FrozenJsonObject)
    assert correction.payload.to_dict() == {
        "missing_facts": ["accepted_evidence", "observed_result"]
    }
    assert snapshot.readiness == (readiness,)
    assert snapshot.observations == (correction,)
    assert snapshot.operation.status is OperationStatus.RUNNING
    assert snapshot.operation.final_text is None
    assert snapshot.loop_state.final_answer_candidate is None
    assert snapshot.loop_state.phase is LoopPhase.OBSERVING
    assert [event.type for event in snapshot.events[-2:]] == [
        "readiness.recorded",
        "observation.recorded",
    ]


async def test_allowed_readiness_returns_no_correction_and_stores_candidate() -> None:
    runtime, operation_id = await _begin_runtime()
    await _commit_text_response(runtime, operation_id, "Supported answer.")
    readiness = Readiness(
        allowed=True,
        code="readiness.ready",
        message="The answer is supported.",
        evaluated_at=NOW,
    )

    correction = await runtime.record_readiness(
        operation_id,
        "Supported answer.",
        readiness,
    )

    snapshot = await runtime.inspect(operation_id)
    assert correction is None
    assert snapshot.readiness == (readiness,)
    assert snapshot.observations == ()
    assert snapshot.loop_state.final_answer_candidate == "Supported answer."
    assert snapshot.loop_state.phase is LoopPhase.SYNTHESIZING
    assert snapshot.events[-1].type == "readiness.recorded"


async def test_one_model_response_accepts_only_one_readiness_decision() -> None:
    runtime, operation_id = await _begin_runtime()
    await _commit_text_response(runtime, operation_id, "Answer too early.")
    readiness = Readiness(
        allowed=False,
        code="readiness.missing_evidence",
        message="Accepted evidence is required before answering.",
        missing_facts=("accepted_evidence",),
        evaluated_at=NOW,
    )
    await runtime.record_readiness(operation_id, "Answer too early.", readiness)
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="already.*readiness"):
        await runtime.record_readiness(operation_id, "Answer too early.", readiness)

    assert await runtime.inspect(operation_id) == before


async def test_denied_readiness_correction_commits_atomically() -> None:
    issued_id_count = 0
    readiness_event_count = 0
    inject_failure = False

    def fail_second_readiness_event(prefix: str) -> str:
        nonlocal issued_id_count, readiness_event_count
        issued_id_count += 1
        if inject_failure and prefix == "event":
            readiness_event_count += 1
            if readiness_event_count == 2:
                raise RuntimeError("injected readiness observation failure")
        return f"{prefix}-{issued_id_count}"

    runtime, operation_id = await _begin_runtime(id_factory=fail_second_readiness_event)
    await _commit_text_response(runtime, operation_id, "Answer too early.")
    before = await runtime.inspect(operation_id)
    inject_failure = True

    with pytest.raises(RuntimeError, match="injected readiness observation failure"):
        await runtime.record_readiness(
            operation_id,
            "Answer too early.",
            Readiness(
                allowed=False,
                code="readiness.missing_evidence",
                message="Accepted evidence is required before answering.",
                missing_facts=("accepted_evidence",),
                evaluated_at=NOW,
            ),
        )

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.readiness == ()
    assert after.observations == ()
    assert after.loop_state.final_answer_candidate is None
    assert "readiness.recorded" not in [event.type for event in after.events]
