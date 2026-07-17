from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopExitKind, LoopPhase, Readiness, Turn
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.runtime import (
    ModelCallStatus,
    OperationRuntime,
    OperationSnapshot,
)

NOW = datetime(2026, 7, 16, 13, 0, tzinfo=timezone.utc)


class StaticContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        assert operation.operation.status is OperationStatus.RUNNING
        assert operation.loop_state.phase is LoopPhase.PREPARING_CONTEXT
        assert [event.type for event in operation.events] == [
            "trigger.received",
            "operation.created",
            "turn.created",
        ]
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    session_id=operation.operation.session_id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class AllowTextDomain:
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
    ) -> ActionProposal:
        raise AssertionError("text-only domain cannot validate tool calls")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only domain cannot project observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert operation.model_calls[-1].status is ModelCallStatus.COMPLETED
        assert operation.model_calls[-1].response is not None
        assert operation.events[-1].type == "model_response.recorded"
        return Readiness(
            allowed=True,
            code="ready.text_only",
            message="No external factual claim requires evidence.",
            evaluated_at=NOW,
        )


@pytest.mark.parametrize(
    ("trigger_kind", "session_id"),
    [
        pytest.param(TriggerKind.USER, None, id="user-sessionless"),
        pytest.param(TriggerKind.USER, "session-1", id="user-session"),
        pytest.param(TriggerKind.SCHEDULE, None, id="schedule"),
        pytest.param(TriggerKind.MONITOR, None, id="monitor"),
        pytest.param(TriggerKind.INTERNAL, None, id="internal"),
    ],
)
async def test_text_only_response_completes_from_committed_runtime_state(
    trigger_kind: TriggerKind,
    session_id: str | None,
) -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    response = ModelResponse(
        text="Hello from the generic loop.",
        finish_reason=FinishReason.STOP,
        usage=ModelUsage(input_tokens=7, output_tokens=6),
    )

    class InspectingMock(MockModelProvider):
        async def generate(self, request: ModelRequest) -> ModelResponse:
            before_io = await runtime.inspect(request.operation_id)
            assert before_io.operation.status is OperationStatus.RUNNING
            assert before_io.loop_state.phase is LoopPhase.AWAITING_MODEL
            assert before_io.turns[-1].model_request_id is not None
            assert before_io.model_calls[-1].status is ModelCallStatus.STARTED
            assert before_io.events[-2].type == "context.built"
            assert before_io.events[-1].type == "model_call.started"
            return await super().generate(request)

    provider = InspectingMock((response,))
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=AllowTextDomain(),
    )
    trigger = AgentTrigger(
        id=f"trigger-{trigger_kind.value}-{session_id or 'none'}",
        agent_id="agent-1",
        kind=trigger_kind,
        source_id=f"{trigger_kind.value}-1",
        session_id=session_id,
        payload={"message": "Say hello without using a tool."},
        created_at=NOW,
    )

    result = await loop.run(trigger)
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert result.reason == "completed"
    assert result.final_text == "Hello from the generic loop."
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert final.operation.final_text == result.final_text
    assert final.operation.session_id == session_id
    assert final.trigger.kind is trigger_kind
    assert final.loop_state.phase is LoopPhase.TERMINAL
    assert final.loop_state.turn_count == 1
    assert final.loop_state.input_tokens == 7
    assert final.loop_state.output_tokens == 6
    assert len(final.turns) == 1
    assert len(final.model_calls) == 1
    assert len(final.readiness) == 1
    assert final.observations == ()
    assert [event.type for event in final.events] == [
        "trigger.received",
        "operation.created",
        "turn.created",
        "context.built",
        "model_call.started",
        "model_response.recorded",
        "readiness.recorded",
        "operation.succeeded",
    ]
    assert not any(
        event.type.startswith(("task.", "evidence.")) for event in final.events
    )
    assert len(provider.requests) == 1
    provider.assert_consumed()


async def test_model_failure_is_committed_once_without_whole_loop_retry() -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = MockModelProvider((TimeoutError("provider timeout"),))
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=StaticContextBuilder(),
        domain=AllowTextDomain(),
    )
    trigger = AgentTrigger(
        id="trigger-model-failure",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"message": "This model call will fail."},
        created_at=NOW,
    )

    result = await loop.run(trigger)
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "model_provider_failure"
    assert final.operation.status is OperationStatus.FAILED
    assert final.operation.terminal_reason == "model_provider_failure"
    assert final.model_calls[-1].status is ModelCallStatus.FAILED
    assert final.model_calls[-1].error_code == "model_provider_failure"
    assert [event.type for event in final.events][-2:] == [
        "model_call.failed",
        "operation.failed",
    ]
    assert len(provider.requests) == 1
    provider.assert_consumed()
