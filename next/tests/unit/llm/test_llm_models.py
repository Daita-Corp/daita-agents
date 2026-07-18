from __future__ import annotations

from decimal import Decimal

import pytest

from daita._json import FrozenJsonObject, thaw_json
from daita.llm.errors import ModelProviderError, ProviderErrorCode
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
from daita.llm.protocols import ModelProvider


def test_tool_call_and_definition_are_provider_neutral_and_mutation_isolated() -> None:
    options: dict[str, object] = {"limit": 1}
    required = ["key"]
    arguments: dict[str, object] = {"key": "alpha", "options": options}
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": required,
    }

    call = ToolCall(
        id="call-1",
        name="fake.read",
        arguments=arguments,
        provider_call_id="provider-call-91",
    )
    definition = ToolDefinition(
        name="fake.read",
        description="Read a deterministic value.",
        input_schema=schema,
    )
    options["limit"] = 99
    required.append("unexpected")

    assert isinstance(call.arguments, FrozenJsonObject)
    assert thaw_json(call.arguments) == {
        "key": "alpha",
        "options": {"limit": 1},
    }
    assert call.id == "call-1"
    assert call.provider_call_id == "provider-call-91"
    assert isinstance(definition.input_schema, FrozenJsonObject)
    assert definition.input_schema.to_dict() == {
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "type": "object",
    }

    with pytest.raises(ValueError, match="provider_call_id"):
        ToolCall(
            id="call-2",
            name="fake.read",
            provider_call_id="   ",
        )


def test_provider_error_taxonomy_is_stable_and_typed() -> None:
    assert {code.value for code in ProviderErrorCode} == {
        "authentication_error",
        "rate_limit_error",
        "provider_unavailable",
        "model_not_found",
        "context_overflow",
        "invalid_request",
        "content_blocked",
        "timeout",
        "cancelled",
        "malformed_response",
    }

    error = ModelProviderError(
        ProviderErrorCode.RATE_LIMIT_ERROR,
        "The provider asked the adapter to slow down.",
    )
    assert error.code is ProviderErrorCode.RATE_LIMIT_ERROR
    assert str(error) == "The provider asked the adapter to slow down."

    with pytest.raises(TypeError, match="ProviderErrorCode"):
        ModelProviderError("rate_limit_error")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        ModelProviderError(ProviderErrorCode.TIMEOUT, " ")


def test_canonical_tool_exchange_keeps_operation_and_call_linkage() -> None:
    call = ToolCall(id="call-1", name="fake.read", arguments={"key": "alpha"})
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="op-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        content=(TextBlock("I will inspect that value."),),
        tool_calls=(call,),
    )
    tool = CanonicalMessage(
        agent_id="agent-1",
        operation_id="op-1",
        turn_id="turn-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id="call-1", output={"value": 42}),),
    )

    assert assistant.tool_calls == (call,)
    result = tool.content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.call_id == call.id
    assert isinstance(result.output, FrozenJsonObject)
    assert result.output.to_dict() == {"value": 42}


def test_assistant_provider_metadata_is_immutable_and_role_scoped() -> None:
    replay_items: list[dict[str, object]] = [
        {
            "id": "reasoning-1",
            "type": "reasoning",
            "encrypted_content": "encrypted-state",
        }
    ]
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="op-1",
        role=MessageRole.ASSISTANT,
        content=(TextBlock("Continuing after a tool call."),),
        provider_metadata={"openai_replay_items": replay_items},
    )
    replay_items[0]["encrypted_content"] = "mutated"

    assert isinstance(assistant.provider_metadata, FrozenJsonObject)
    assert thaw_json(assistant.provider_metadata) == {
        "openai_replay_items": [
            {
                "encrypted_content": "encrypted-state",
                "id": "reasoning-1",
                "type": "reasoning",
            }
        ]
    }

    with pytest.raises(ValueError, match="assistant"):
        CanonicalMessage(
            agent_id="agent-1",
            operation_id="op-1",
            role=MessageRole.USER,
            content=(TextBlock("hello"),),
            provider_metadata={"provider": "state"},
        )


def test_message_role_shape_is_strict() -> None:
    call = ToolCall(id="call-1", name="fake.read", arguments={})

    with pytest.raises(ValueError, match="assistant"):
        CanonicalMessage(
            agent_id="agent-1",
            operation_id="op-1",
            role=MessageRole.USER,
            content=(TextBlock("read alpha"),),
            tool_calls=(call,),
        )

    with pytest.raises(ValueError, match="tool-result"):
        CanonicalMessage(
            agent_id="agent-1",
            operation_id="op-1",
            role=MessageRole.USER,
            content=(ToolResultBlock(call_id="call-1", output={}),),
        )

    with pytest.raises(ValueError, match="content"):
        CanonicalMessage(
            agent_id="agent-1",
            operation_id="op-1",
            role=MessageRole.SYSTEM,
        )

    with pytest.raises(ValueError, match="Duplicate provider tool-call"):
        CanonicalMessage(
            agent_id="agent-1",
            operation_id="op-1",
            role=MessageRole.ASSISTANT,
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="fake.read",
                    provider_call_id="provider-call-1",
                ),
                ToolCall(
                    id="call-2",
                    name="fake.read",
                    provider_call_id="provider-call-1",
                ),
            ),
        )


def test_model_request_rejects_cross_operation_messages_and_duplicate_tools() -> None:
    message = CanonicalMessage(
        agent_id="agent-1",
        operation_id="another-op",
        role=MessageRole.USER,
        content=(TextBlock("hello"),),
    )
    tool = ToolDefinition(
        name="fake.read",
        description="Read.",
        input_schema={"type": "object"},
    )

    with pytest.raises(ValueError, match="operation"):
        ModelRequest(operation_id="op-1", turn_id="turn-1", messages=(message,))

    with pytest.raises(ValueError, match="Duplicate tool"):
        ModelRequest(
            operation_id="another-op",
            turn_id="turn-1",
            messages=(message,),
            tools=(tool, tool),
        )

    other_agent_message = CanonicalMessage(
        agent_id="agent-2",
        operation_id="another-op",
        role=MessageRole.USER,
        content=(TextBlock("hello from another agent"),),
    )
    with pytest.raises(ValueError, match="one agent"):
        ModelRequest(
            operation_id="another-op",
            turn_id="turn-1",
            messages=(message, other_agent_message),
        )


def test_response_is_strict_and_preserves_mixed_text_and_ordered_calls() -> None:
    first = ToolCall(id="call-1", name="fake.read", arguments={"key": "alpha"})
    second = ToolCall(id="call-2", name="fake.read", arguments={"key": "beta"})
    response = ModelResponse(
        text="I will read both values.",
        tool_calls=(first, second),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=ModelUsage(input_tokens=8, output_tokens=5),
    )

    assert response.tool_calls == (first, second)
    assert response.text == "I will read both values."

    with pytest.raises(ValueError, match="text or tool call"):
        ModelResponse(finish_reason=FinishReason.STOP)

    with pytest.raises(ValueError, match="Duplicate tool-call"):
        ModelResponse(
            tool_calls=(first, first),
            finish_reason=FinishReason.TOOL_CALLS,
        )

    with pytest.raises(ValueError, match="finish reason"):
        ModelResponse(tool_calls=(first,), finish_reason=FinishReason.STOP)

    with pytest.raises(ValueError, match="Duplicate provider tool-call"):
        ModelResponse(
            tool_calls=(
                ToolCall(
                    id="call-3",
                    name="fake.read",
                    provider_call_id="provider-call-1",
                ),
                ToolCall(
                    id="call-4",
                    name="fake.read",
                    provider_call_id="provider-call-1",
                ),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
        )


def test_usage_accounts_exact_decimal_cost_without_float_drift() -> None:
    usage = ModelUsage(
        input_tokens=10,
        output_tokens=4,
        reasoning_tokens=2,
        cache_read_tokens=3,
        estimated_cost_usd=Decimal("0.000123"),
    )

    assert usage.total_tokens == 14
    assert usage.reasoning_tokens == 2
    assert usage.estimated_cost_usd == Decimal("0.000123")

    with pytest.raises(TypeError, match="Decimal"):
        ModelUsage(estimated_cost_usd=0.1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative"):
        ModelUsage(input_tokens=-1)


async def test_model_provider_protocol_exposes_only_provider_id_and_generate() -> None:
    class StubProvider:
        @property
        def provider_id(self) -> str:
            return "stub"

        async def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                text=f"handled {request.operation_id}",
                finish_reason=FinishReason.STOP,
            )

    provider: ModelProvider = StubProvider()
    request = ModelRequest(
        operation_id="op-1",
        turn_id="turn-1",
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id="op-1",
                role=MessageRole.USER,
                content=(TextBlock("hello"),),
            ),
        ),
    )

    response = await provider.generate(request)

    assert provider.provider_id == "stub"
    assert response.text == "handled op-1"
    assert not hasattr(provider, "execute")
