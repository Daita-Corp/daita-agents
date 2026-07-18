from __future__ import annotations

from datetime import datetime, timezone

from daita._json import FrozenJsonObject
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopExitKind, Readiness, Turn
from daita.operations.checkpoints import ModelCallStatus, OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


class _TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Fail with one normalized provider error."),),
                ),
            ),
        )


class _NoActionDomain:
    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("a provider failure cannot reach action validation")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("a provider failure cannot produce evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        raise AssertionError("a provider failure cannot reach readiness")


async def test_loop_persists_only_normalized_provider_error_code() -> None:
    runtime = OperationRuntime(clock=lambda: NOW)
    provider = MockModelProvider(
        (
            ModelProviderError(
                ProviderErrorCode.RATE_LIMIT_ERROR,
                "provider-specific transport detail",
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=_TextContext(),
        domain=_NoActionDomain(),
    )
    trigger = AgentTrigger(
        id="trigger-normalized-provider-error",
        agent_id="agent-provider-error",
        kind=TriggerKind.USER,
        source_id="user-provider-error",
        payload={"message": "fail"},
        created_at=NOW,
    )

    result = await loop.run(trigger)
    snapshot = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == ProviderErrorCode.RATE_LIMIT_ERROR.value
    assert snapshot.operation.status is OperationStatus.FAILED
    assert (
        snapshot.operation.terminal_reason == ProviderErrorCode.RATE_LIMIT_ERROR.value
    )
    assert snapshot.model_calls[-1].status is ModelCallStatus.FAILED
    assert (
        snapshot.model_calls[-1].error_code == ProviderErrorCode.RATE_LIMIT_ERROR.value
    )
    error_payload = snapshot.events[-2].payload
    assert isinstance(error_payload, FrozenJsonObject)
    assert error_payload.to_dict() == {
        "error_code": ProviderErrorCode.RATE_LIMIT_ERROR.value,
        "model_call_id": snapshot.model_calls[-1].id,
    }
    assert "provider-specific transport detail" not in repr(snapshot)
    assert len(provider.requests) == 1
    provider.assert_consumed()
