from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityInputError,
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
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopBudgets, LoopExitKind, Readiness, Turn
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationSnapshot

NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
REJECTION_CODE = "invalid_arguments"
REJECTION_MESSAGE = "The key argument must be a string."
REJECTION_DETAILS = {"field": "key", "expected": "string"}
PREMATURE_ANSWER = "The unverified balance is 42 dollars."
HONEST_CORRECTION = "I cannot verify the balance without current evidence."
MISSING_FACT = "accepted current evidence for the requested balance"


def _capability() -> Capability:
    return Capability(
        id="fake.read",
        owner="loop-lab",
        description="Read one deterministic fake value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
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
        executor_id="fake.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class CountingReadExecutor:
    executor_id = "fake.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": key, "value": key.upper()},
        )


class RepairingFakeReadDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry
        self.validated_call_ids: list[str] = []

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        assert operation.operation.status is OperationStatus.RUNNING
        return self.registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal | ActionRejection:
        self.validated_call_ids.append(call.id)
        view, capability = self.registry.resolve_tool(call.name)
        try:
            arguments = self.registry.validate_arguments(
                capability.id,
                call.arguments,
            )
        except CapabilityInputError:
            # Provider and validator exception text are deliberately not exposed.
            return ActionRejection(
                code=REJECTION_CODE,
                message=REJECTION_MESSAGE,
                details=REJECTION_DETAILS,
            )
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=arguments,
            proposed_at=NOW,
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            code="fake.read.succeeded",
            message="Fake read completed.",
            payload=evidence.payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=NOW,
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert text == "The repaired fake read succeeded."
        assert len(operation.evidence) == 1
        assert operation.evidence[0].accepted
        return Readiness(
            allowed=True,
            code="ready.fake_read",
            message="The repaired read has accepted evidence.",
            evaluated_at=NOW,
        )


class RepairTranscriptContextBuilder:
    """Project committed responses and observations into canonical messages."""

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        messages: list[CanonicalMessage] = [
            CanonicalMessage(
                agent_id=operation.operation.agent_id,
                operation_id=operation.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Read the fake value for the requested key."),),
            )
        ]
        for model_call in operation.model_calls:
            response = model_call.response
            assert response is not None
            if response.tool_calls:
                messages.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=model_call.turn_id,
                        role=MessageRole.ASSISTANT,
                        tool_calls=response.tool_calls,
                    )
                )
            else:
                assert response.text is not None
                messages.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=model_call.turn_id,
                        role=MessageRole.ASSISTANT,
                        content=(TextBlock(response.text),),
                    )
                )

            for observation in operation.observations:
                if observation.turn_id != model_call.turn_id:
                    continue
                call_id = observation.call_id
                if call_id is None:
                    assert observation.task_id is not None
                    call_id = next(
                        task.call_id
                        for task in operation.tasks
                        if task.id == observation.task_id
                    )
                messages.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=model_call.turn_id,
                        role=MessageRole.TOOL,
                        content=(
                            ToolResultBlock(
                                call_id=call_id,
                                output={
                                    "code": observation.code,
                                    "message": observation.message,
                                    "details": observation.payload,
                                },
                                is_error=not observation.success,
                            ),
                        ),
                    )
                )

        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=tuple(messages),
            tools=tools,
        )


class CorrectingReadinessDomain:
    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        assert operation.operation.status is OperationStatus.RUNNING
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal | ActionRejection:
        raise AssertionError("readiness correction must not validate an action")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("readiness correction must not project evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert operation.tasks == ()
        assert operation.evidence == ()
        if text == HONEST_CORRECTION:
            return Readiness(
                allowed=True,
                code="ready.honest_limitation",
                message="The answer honestly reports its missing evidence.",
                evaluated_at=NOW,
            )
        return Readiness(
            allowed=False,
            code="readiness.missing_evidence",
            message="Do not claim a balance without accepted current evidence.",
            missing_facts=(MISSING_FACT,),
            evaluated_at=NOW,
        )


class ReadinessCorrectionContextBuilder:
    """Replay readiness feedback as canonical user-side correction context."""

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        messages: list[CanonicalMessage] = [
            CanonicalMessage(
                agent_id=operation.operation.agent_id,
                operation_id=operation.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Report the requested balance."),),
            )
        ]
        for model_call in operation.model_calls:
            response = model_call.response
            assert response is not None
            assert response.text is not None
            assert response.tool_calls == ()
            messages.append(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=model_call.turn_id,
                    role=MessageRole.ASSISTANT,
                    content=(TextBlock(response.text),),
                )
            )
            for observation in operation.observations:
                if observation.turn_id != model_call.turn_id:
                    continue
                assert not observation.success
                assert observation.call_id is None
                assert observation.task_id is None
                assert observation.evidence_id is None
                messages.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=model_call.turn_id,
                        role=MessageRole.USER,
                        content=(
                            TextBlock(
                                canonical_json(
                                    {
                                        "type": "readiness_correction",
                                        "code": observation.code,
                                        "message": observation.message,
                                        "missing_facts": observation.payload[
                                            "missing_facts"
                                        ],
                                    }
                                )
                            ),
                        ),
                    )
                )

        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=tuple(messages),
            tools=(),
        )


def _registry(
    executor: CountingReadExecutor,
) -> CapabilityRegistry:
    capability = _capability()
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id=capability.id,
                description="Read one fake value by key.",
            ),
        ),
    )


def _trigger(identifier: str) -> AgentTrigger:
    return AgentTrigger(
        id=identifier,
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"key": "alpha"},
        created_at=NOW,
    )


def _tool_response(call_id: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id=call_id,
                name="read_fake_value",
                arguments=arguments,
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=ModelUsage(input_tokens=8, output_tokens=3),
    )


def _important_events(snapshot: OperationSnapshot) -> list[str]:
    important = {
        "action.rejected",
        "action.skipped",
        "evidence.accepted",
        "executor.completed",
        "executor.started",
        "model_response.recorded",
        "no_progress.detected",
        "observation.recorded",
        "operation.failed",
        "operation.succeeded",
        "readiness.recorded",
        "task.claimed",
        "task.created",
        "task.ready",
        "task.succeeded",
    }
    return [event.type for event in snapshot.events if event.type in important]


async def test_invalid_action_is_observed_then_changed_action_repairs() -> None:
    executor = CountingReadExecutor()
    registry = _registry(executor)
    provider = MockModelProvider(
        (
            _tool_response("invalid-call", {"key": 7}),
            _tool_response("repaired-call", {"key": "alpha"}),
            ModelResponse(
                text="The repaired fake read succeeded.",
                finish_reason=FinishReason.STOP,
                usage=ModelUsage(input_tokens=18, output_tokens=6),
            ),
        )
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=RepairTranscriptContextBuilder(),
        domain=RepairingFakeReadDomain(registry),
        budgets=LoopBudgets(max_repairs=2, max_identical_failures=2),
    )

    result = await loop.run(_trigger("trigger-repaired-action"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert len(final.observations) == 2
    rejection = final.observations[0]
    assert not rejection.success
    assert rejection.call_id == "invalid-call"
    assert rejection.task_id is None
    assert rejection.evidence_id is None
    assert rejection.code == REJECTION_CODE
    assert rejection.message == REJECTION_MESSAGE
    assert isinstance(rejection.payload, FrozenJsonObject)
    assert rejection.payload.to_dict() == REJECTION_DETAILS
    assert len(canonical_json(rejection.payload)) < 2048
    assert len(final.tasks) == 1
    assert len(final.evidence) == 1
    assert [request.arguments["key"] for request in executor.requests] == ["alpha"]
    assert final.loop_state.repair_count == 1
    assert final.loop_state.identical_failure_count == 0
    assert final.loop_state.no_progress_fingerprints == ()

    repair_request = provider.requests[1]
    assert [message.role for message in repair_request.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    repair_block = repair_request.messages[-1].content[0]
    assert isinstance(repair_block, ToolResultBlock)
    assert repair_block.call_id == "invalid-call"
    assert repair_block.is_error
    assert isinstance(repair_block.output, FrozenJsonObject)
    assert repair_block.output.to_dict() == {
        "code": REJECTION_CODE,
        "details": REJECTION_DETAILS,
        "message": REJECTION_MESSAGE,
    }
    assert [event.type for event in final.events] == [
        "trigger.received",
        "operation.created",
        "turn.created",
        "context.built",
        "model_call.started",
        "model_response.recorded",
        "action.rejected",
        "observation.recorded",
        "turn.created",
        "context.built",
        "model_call.started",
        "model_response.recorded",
        "task.created",
        "task.ready",
        "task.claimed",
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
        "observation.recorded",
        "turn.created",
        "context.built",
        "model_call.started",
        "model_response.recorded",
        "readiness.recorded",
        "operation.succeeded",
    ]

    task = final.tasks[0]
    evidence = final.evidence[0]
    successful_observation = final.observations[1]
    assert task.call_id == "repaired-call"
    assert task.turn_id == final.turns[1].id
    assert task.evidence_ids == (evidence.id,)
    assert evidence.operation_id == final.operation.id
    assert evidence.task_id == task.id
    assert evidence.turn_id == task.turn_id
    assert evidence.capability_id == task.capability_id
    assert evidence.executor_id == task.executor_id
    assert successful_observation.operation_id == final.operation.id
    assert successful_observation.turn_id == task.turn_id
    assert successful_observation.task_id == task.id
    assert successful_observation.evidence_id == evidence.id

    rejected_event = next(
        event for event in final.events if event.type == "action.rejected"
    )
    rejected_observation_event = next(
        event
        for event in final.events
        if event.type == "observation.recorded" and event.call_id == "invalid-call"
    )
    assert rejected_event.turn_id == final.turns[0].id
    assert rejected_event.call_id == rejection.call_id
    assert rejected_event.task_id is None
    assert rejected_event.evidence_id is None
    assert rejected_observation_event.turn_id == final.turns[0].id
    assert rejected_observation_event.task_id is None
    assert rejected_observation_event.evidence_id is None

    task_events = [
        event
        for event in final.events
        if event.type
        in {
            "task.claimed",
            "task.created",
            "task.ready",
            "executor.started",
            "executor.completed",
            "evidence.accepted",
            "task.succeeded",
        }
    ]
    assert len(task_events) == 7
    for event in task_events:
        assert event.turn_id == task.turn_id
        assert event.call_id == task.call_id
        assert event.task_id == task.id
        assert event.capability_id == task.capability_id
        assert event.executor_id == task.executor_id
    evidence_events = [
        event
        for event in task_events
        if event.type in {"evidence.accepted", "task.succeeded"}
    ]
    assert all(event.evidence_id == evidence.id for event in evidence_events)
    successful_observation_event = next(
        event
        for event in final.events
        if event.type == "observation.recorded" and event.evidence_id == evidence.id
    )
    assert successful_observation_event.turn_id == task.turn_id
    assert successful_observation_event.call_id == task.call_id
    assert successful_observation_event.task_id == task.id
    assert successful_observation_event.capability_id == task.capability_id
    assert successful_observation_event.executor_id == task.executor_id
    failure_facts = canonical_json(
        {
            "code": rejection.code,
            "message": rejection.message,
            "details": rejection.payload,
        }
    )
    assert "CapabilityInputError" not in failure_facts
    assert "Traceback" not in failure_facts
    assert "mock:scripted" not in failure_facts
    provider.assert_consumed()


async def test_rejection_first_skips_later_call_and_repair_transcript_is_complete() -> (
    None
):
    executor = CountingReadExecutor()
    registry = _registry(executor)
    domain = RepairingFakeReadDomain(registry)
    provider = MockModelProvider(
        (
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="rejected-first",
                        name="read_fake_value",
                        arguments={"key": 7},
                    ),
                    ToolCall(
                        id="skipped-second",
                        name="read_fake_value",
                        arguments={"key": "must-not-run"},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            _tool_response("repaired-call", {"key": "alpha"}),
            ModelResponse(
                text="The repaired fake read succeeded.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=RepairTranscriptContextBuilder(),
        domain=domain,
        budgets=LoopBudgets(max_repairs=2, max_identical_failures=2),
    )

    result = await loop.run(_trigger("trigger-rejection-first-batch"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert domain.validated_call_ids == ["rejected-first", "repaired-call"]
    assert [request.arguments["key"] for request in executor.requests] == ["alpha"]
    assert [observation.call_id for observation in final.observations[:2]] == [
        "rejected-first",
        "skipped-second",
    ]
    assert final.observations[1].code == "action.skipped_after_rejection"
    assert isinstance(final.observations[1].payload, FrozenJsonObject)
    assert final.observations[1].payload.to_dict() == {
        "blocked_by_call_id": "rejected-first",
        "blocked_by_code": REJECTION_CODE,
    }
    repair_request = provider.requests[1]
    assert [message.role for message in repair_request.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
    ]
    repair_blocks: list[ToolResultBlock] = []
    for message in repair_request.messages:
        if message.role is not MessageRole.TOOL:
            continue
        block = message.content[0]
        assert isinstance(block, ToolResultBlock)
        repair_blocks.append(block)
    assert [block.call_id for block in repair_blocks] == [
        "rejected-first",
        "skipped-second",
    ]
    assert all(block.is_error for block in repair_blocks)
    assert final.loop_state.repair_count == 1
    assert final.loop_state.identical_failure_count == 0
    assert final.loop_state.no_progress_fingerprints == ()
    provider.assert_consumed()


async def test_success_then_rejection_skips_rest_in_stable_transcript_order() -> None:
    executor = CountingReadExecutor()
    registry = _registry(executor)
    domain = RepairingFakeReadDomain(registry)
    provider = MockModelProvider(
        (
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="successful-first",
                        name="read_fake_value",
                        arguments={"key": "alpha"},
                    ),
                    ToolCall(
                        id="rejected-second",
                        name="read_fake_value",
                        arguments={"key": 7},
                    ),
                    ToolCall(
                        id="skipped-third",
                        name="read_fake_value",
                        arguments={"key": "must-not-run"},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            ModelResponse(
                text="The repaired fake read succeeded.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=RepairTranscriptContextBuilder(),
        domain=domain,
        budgets=LoopBudgets(max_repairs=2, max_identical_failures=2),
    )

    result = await loop.run(_trigger("trigger-success-rejection-batch"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert domain.validated_call_ids == ["successful-first", "rejected-second"]
    assert [request.arguments["key"] for request in executor.requests] == ["alpha"]
    assert len(final.tasks) == 1
    assert len(final.evidence) == 1
    assert [observation.code for observation in final.observations] == [
        "fake.read.succeeded",
        REJECTION_CODE,
        "action.skipped_after_rejection",
    ]
    final_request = provider.requests[1]
    assert [message.role for message in final_request.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.TOOL,
    ]
    result_blocks: list[ToolResultBlock] = []
    for message in final_request.messages:
        if message.role is not MessageRole.TOOL:
            continue
        block = message.content[0]
        assert isinstance(block, ToolResultBlock)
        result_blocks.append(block)
    assert [block.call_id for block in result_blocks] == [
        "successful-first",
        "rejected-second",
        "skipped-third",
    ]
    assert [block.is_error for block in result_blocks] == [False, True, True]
    assert final.loop_state.repair_count == 1
    assert final.loop_state.identical_failure_count == 1
    assert len(final.loop_state.no_progress_fingerprints) == 1
    assert _important_events(final) == [
        "model_response.recorded",
        "task.created",
        "task.ready",
        "task.claimed",
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
        "observation.recorded",
        "action.rejected",
        "observation.recorded",
        "action.skipped",
        "observation.recorded",
        "model_response.recorded",
        "readiness.recorded",
        "operation.succeeded",
    ]
    model_call_by_turn = {
        model_call.turn_id: model_call.id for model_call in final.model_calls
    }
    for event in final.events:
        if event.type not in {
            "action.rejected",
            "action.skipped",
            "observation.recorded",
        }:
            continue
        assert event.turn_id is not None
        assert event.model_call_id == model_call_by_turn[event.turn_id]
    provider.assert_consumed()


async def test_identical_failure_limit_one_stops_on_first_failed_attempt() -> None:
    executor = CountingReadExecutor()
    registry = _registry(executor)
    provider = MockModelProvider(
        (
            _tool_response("provider-call-a", {"key": 7}),
            ModelResponse(
                text="This scripted answer must remain unconsumed.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=RepairTranscriptContextBuilder(),
        domain=RepairingFakeReadDomain(registry),
        budgets=LoopBudgets(max_repairs=5, max_identical_failures=1),
    )

    result = await loop.run(_trigger("trigger-first-action-stop"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "no_progress_action_failure_limit"
    assert len(provider.requests) == 1
    assert executor.requests == []
    assert final.tasks == ()
    assert final.evidence == ()
    assert len(final.observations) == 1
    assert final.loop_state.repair_count == 1
    assert final.loop_state.identical_failure_count == 1
    assert len(final.loop_state.no_progress_fingerprints) == 1
    with pytest.raises(AssertionError, match="1 unconsumed"):
        provider.assert_consumed()


async def test_repeated_normalized_failure_stops_before_more_model_or_io() -> None:
    executor = CountingReadExecutor()
    registry = _registry(executor)
    provider = MockModelProvider(
        (
            _tool_response("provider-call-a", {"key": 7}),
            _tool_response("provider-call-b", {"key": 7}),
            ModelResponse(
                text="This scripted answer must remain unconsumed.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=RepairTranscriptContextBuilder(),
        domain=RepairingFakeReadDomain(registry),
        # Both limits cross on the second failure. The more specific
        # no-progress reason takes precedence over aggregate repair exhaustion.
        budgets=LoopBudgets(max_repairs=1, max_identical_failures=2),
    )

    result = await loop.run(_trigger("trigger-repeated-action"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "no_progress_action_failure_limit"
    assert final.operation.status is OperationStatus.FAILED
    assert final.operation.terminal_reason == "no_progress_action_failure_limit"
    assert len(provider.requests) == 2
    with pytest.raises(AssertionError, match="1 unconsumed"):
        provider.assert_consumed()
    assert executor.requests == []
    assert final.tasks == ()
    assert final.evidence == ()
    assert len(final.observations) == 2
    assert [observation.call_id for observation in final.observations] == [
        "provider-call-a",
        "provider-call-b",
    ]
    assert all(not observation.success for observation in final.observations)
    assert all(observation.task_id is None for observation in final.observations)
    assert all(observation.evidence_id is None for observation in final.observations)
    assert final.loop_state.repair_count == 2
    assert final.loop_state.identical_failure_count == 2
    assert len(final.loop_state.no_progress_fingerprints) == 1
    fingerprint = final.loop_state.no_progress_fingerprints[0]
    assert fingerprint.startswith("sha256:")
    assert "provider-call-a" not in fingerprint
    assert "provider-call-b" not in fingerprint

    rejection_events = [
        event for event in final.events if event.type == "action.rejected"
    ]
    assert [event.call_id for event in rejection_events] == [
        "provider-call-a",
        "provider-call-b",
    ]
    assert [event.payload["fingerprint"] for event in rejection_events] == [
        fingerprint,
        fingerprint,
    ]
    no_progress = next(
        event for event in final.events if event.type == "no_progress.detected"
    )
    assert no_progress.call_id == "provider-call-b"
    assert no_progress.turn_id is not None
    model_call_by_turn = {
        model_call.turn_id: model_call.id for model_call in final.model_calls
    }
    assert no_progress.model_call_id == model_call_by_turn[no_progress.turn_id]
    assert isinstance(no_progress.payload, FrozenJsonObject)
    assert no_progress.payload.to_dict() == {
        "count": 2,
        "fingerprint": fingerprint,
        "reason": "no_progress_action_failure_limit",
    }
    assert _important_events(final) == [
        "model_response.recorded",
        "action.rejected",
        "observation.recorded",
        "model_response.recorded",
        "action.rejected",
        "observation.recorded",
        "no_progress.detected",
        "operation.failed",
    ]
    serialized_failures = canonical_json(
        {
            "observations": [
                {
                    "call_id": observation.call_id,
                    "code": observation.code,
                    "message": observation.message,
                    "details": observation.payload,
                }
                for observation in final.observations
            ],
            "event_payloads": [
                event.payload
                for event in final.events
                if event.type
                in {"action.rejected", "no_progress.detected", "operation.failed"}
            ],
        }
    )
    assert "CapabilityInputError" not in serialized_failures
    assert "OperationStateError" not in serialized_failures
    assert "Traceback" not in serialized_failures
    assert "mock:scripted" not in serialized_failures


async def test_denied_readiness_is_observed_then_corrected_by_the_model() -> None:
    provider = MockModelProvider(
        (
            ModelResponse(
                text=PREMATURE_ANSWER,
                finish_reason=FinishReason.STOP,
                usage=ModelUsage(input_tokens=7, output_tokens=4),
            ),
            ModelResponse(
                text=HONEST_CORRECTION,
                finish_reason=FinishReason.STOP,
                usage=ModelUsage(input_tokens=16, output_tokens=8),
            ),
        )
    )
    runtime = OperationRuntime(clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=ReadinessCorrectionContextBuilder(),
        domain=CorrectingReadinessDomain(),
        budgets=LoopBudgets(max_repairs=1, max_identical_failures=2),
    )

    result = await loop.run(_trigger("trigger-readiness-correction"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == HONEST_CORRECTION
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert final.operation.final_text == HONEST_CORRECTION
    assert final.tasks == ()
    assert final.evidence == ()
    assert len(final.readiness) == 2
    assert not final.readiness[0].allowed
    assert final.readiness[0].missing_facts == (MISSING_FACT,)
    assert final.readiness[1].allowed
    assert len(final.observations) == 1
    correction = final.observations[0]
    assert not correction.success
    assert correction.call_id is None
    assert correction.task_id is None
    assert correction.evidence_id is None
    assert correction.code == "readiness.missing_evidence"
    assert isinstance(correction.payload, FrozenJsonObject)
    assert correction.payload.to_dict() == {"missing_facts": [MISSING_FACT]}
    assert final.loop_state.repair_count == 1
    assert final.loop_state.identical_failure_count == 0
    assert final.loop_state.no_progress_fingerprints == ()

    assert len(provider.requests) == 2
    corrected_request = provider.requests[1]
    assert [message.role for message in corrected_request.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    correction_block = corrected_request.messages[-1].content[0]
    assert isinstance(correction_block, TextBlock)
    assert correction_block.text == canonical_json(
        {
            "type": "readiness_correction",
            "code": correction.code,
            "message": correction.message,
            "missing_facts": [MISSING_FACT],
        }
    )
    assert _important_events(final) == [
        "model_response.recorded",
        "readiness.recorded",
        "observation.recorded",
        "model_response.recorded",
        "readiness.recorded",
        "operation.succeeded",
    ]
    provider.assert_consumed()


async def test_repeated_denied_readiness_exhausts_repairs_before_extra_call() -> None:
    provider = MockModelProvider(
        (
            ModelResponse(
                text=PREMATURE_ANSWER,
                finish_reason=FinishReason.STOP,
            ),
            ModelResponse(
                text=PREMATURE_ANSWER,
                finish_reason=FinishReason.STOP,
            ),
            ModelResponse(
                text="This scripted response must remain unconsumed.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    runtime = OperationRuntime(clock=lambda: NOW)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=ReadinessCorrectionContextBuilder(),
        domain=CorrectingReadinessDomain(),
        budgets=LoopBudgets(max_repairs=1, max_identical_failures=2),
    )

    result = await loop.run(_trigger("trigger-readiness-budget"))
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "repair_budget_exhausted"
    assert final.operation.status is OperationStatus.FAILED
    assert final.operation.terminal_reason == "repair_budget_exhausted"
    assert len(provider.requests) == 2
    with pytest.raises(AssertionError, match="1 unconsumed"):
        provider.assert_consumed()
    assert final.tasks == ()
    assert final.evidence == ()
    assert len(final.readiness) == 2
    assert all(not readiness.allowed for readiness in final.readiness)
    assert all(
        readiness.missing_facts == (MISSING_FACT,) for readiness in final.readiness
    )
    assert len(final.observations) == 2
    assert all(not observation.success for observation in final.observations)
    assert all(observation.call_id is None for observation in final.observations)
    assert all(observation.task_id is None for observation in final.observations)
    assert all(observation.evidence_id is None for observation in final.observations)
    assert final.loop_state.repair_count == 2
    assert final.loop_state.identical_failure_count == 0
    assert final.loop_state.no_progress_fingerprints == ()
    assert _important_events(final) == [
        "model_response.recorded",
        "readiness.recorded",
        "observation.recorded",
        "model_response.recorded",
        "readiness.recorded",
        "observation.recorded",
        "operation.failed",
    ]
    failed_event = next(
        event for event in final.events if event.type == "operation.failed"
    )
    assert failed_event.payload["reason"] == "repair_budget_exhausted"
