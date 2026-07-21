from __future__ import annotations

import pytest

from daita.context import (
    ContextBlock,
    ContextKind,
    ContextMessageGroup,
    ContextProvenance,
    ContextTrust,
    RequiredContextOverflow,
    estimate_context_block_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
    select_context_blocks,
)
from daita.llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelProfile,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)


def _message(text: str, *, role: MessageRole = MessageRole.USER) -> CanonicalMessage:
    return CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        role=role,
        content=(TextBlock(text),),
    )


def _block(
    identity: str,
    text: str,
    *,
    priority: int,
    required: bool = False,
) -> ContextBlock:
    return ContextBlock(
        id=identity,
        owner="context.test",
        kind=ContextKind.OPERATION if required else ContextKind.MEMORY,
        trust=ContextTrust.TRUSTED_RUNTIME,
        provenance=(ContextProvenance(kind="test.fixture", reference_id=identity),),
        groups=(
            ContextMessageGroup(
                id=f"{identity}.group",
                messages=(_message(text),),
            ),
        ),
        priority=priority,
        required=required,
    )


def _profile(input_tokens: int, *, output_tokens: int = 20) -> ModelProfile:
    return ModelProfile(
        id="mock:budget",
        context_window_tokens=input_tokens + output_tokens,
        max_output_tokens=output_tokens,
    )


def test_approximate_estimator_is_deterministic_and_counts_canonical_json() -> None:
    short = _message("abcd")
    long = _message("abcd" * 40)

    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcde") == 2
    assert estimate_message_tokens(short) == estimate_message_tokens(short)
    assert estimate_message_tokens(long) > estimate_message_tokens(short)


def test_budget_reserves_output_and_tool_schema_before_selecting_blocks() -> None:
    required = _block("system.required", "must remain", priority=1, required=True)
    tools = (
        ToolDefinition(
            name="catalog_search",
            description="Search current catalog resources.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        ),
    )
    required_tokens = estimate_context_block_tokens(required)
    tool_tokens = estimate_tool_tokens(tools)
    selection = select_context_blocks(
        (required,),
        _profile(required_tokens + tool_tokens),
        tools=tools,
    )

    assert selection.blocks == (required,)
    assert selection.tool_tokens == tool_tokens
    assert selection.estimated_input_tokens == required_tokens + tool_tokens
    assert selection.output_reserve_tokens == 20
    assert selection.remaining_input_tokens == 0


def test_required_context_overflow_fails_before_returning_partial_projection() -> None:
    required = _block(
        "system.required",
        "x" * 400,
        priority=1,
        required=True,
    )
    required_tokens = estimate_context_block_tokens(required)

    with pytest.raises(RequiredContextOverflow) as captured:
        select_context_blocks((required,), _profile(required_tokens - 1))

    assert captured.value.required_tokens == required_tokens
    assert captured.value.available_tokens == required_tokens - 1
    assert captured.value.profile_id == "mock:budget"


def test_required_context_overflow_exposes_only_bounded_scalar_component_facts() -> (
    None
):
    error = RequiredContextOverflow(
        profile_id="mock:budget",
        required_tokens=29,
        available_tokens=20,
        tool_tokens=7,
        output_reserve_tokens=11,
        input_limit_tokens=27,
        required_system_tokens=3,
        required_routing_tokens=2,
        required_intent_tokens=4,
        current_operation_envelope_tokens=5,
        current_operation_body_tokens=6,
        minimum_session_tokens=9,
        projected_session_tokens=15,
        optional_omitted_tokens=8,
    )

    assert error.total_required_tokens == 36
    assert error.safe_facts == {
        "available_tokens": 20,
        "current_operation_body_tokens": 6,
        "current_operation_envelope_tokens": 5,
        "input_limit_tokens": 27,
        "minimum_session_tokens": 9,
        "optional_omitted_tokens": 8,
        "output_reserve_tokens": 11,
        "profile_id": "mock:budget",
        "projected_session_tokens": 15,
        "required_intent_tokens": 4,
        "required_routing_tokens": 2,
        "required_system_tokens": 3,
        "required_tokens": 29,
        "tool_tokens": 7,
        "total_required_tokens": 36,
    }
    mutated = error.safe_facts
    mutated["profile_id"] = "mock:changed"
    assert error.profile_id == "mock:budget"

    with pytest.raises(ValueError, match="non-negative integer"):
        RequiredContextOverflow(
            profile_id="mock:budget",
            required_tokens=1,
            available_tokens=0,
            tool_tokens=0,
            output_reserve_tokens=1,
            required_system_tokens=False,
        )


def test_optional_blocks_select_by_priority_but_keep_declaration_order() -> None:
    required = _block("system.required", "required", priority=0, required=True)
    low = _block("memory.low", "low" * 20, priority=10)
    high = _block("catalog.high", "high" * 20, priority=100)
    capacity = estimate_context_block_tokens(required) + estimate_context_block_tokens(
        high
    )

    selection = select_context_blocks(
        (required, low, high),
        _profile(capacity),
    )

    assert selection.blocks == (required, high)
    assert selection.omitted_block_ids == (low.id,)
    assert selection.messages == required.messages + high.messages


def test_budget_never_splits_an_indivisible_tool_call_result_group() -> None:
    call = ToolCall(id="call-1", name="data_read", arguments={"resource_id": "r1"})
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(call,),
    )
    result = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        role=MessageRole.TOOL,
        content=(ToolResultBlock(call_id=call.id, output={"value": "x" * 100}),),
    )
    exchange = ContextBlock(
        id="operation.exchange",
        owner="operations",
        kind=ContextKind.SESSION_RECENT,
        trust=ContextTrust.TRUSTED_RUNTIME,
        provenance=(ContextProvenance(kind="runtime.turn", reference_id="turn-1"),),
        groups=(
            ContextMessageGroup(
                id="operation.exchange.group",
                messages=(assistant, result),
            ),
        ),
        priority=100,
    )
    fallback = _block("memory.fallback", "small", priority=1)
    fallback_tokens = estimate_context_block_tokens(fallback)

    selection = select_context_blocks(
        (exchange, fallback),
        _profile(fallback_tokens),
    )

    assert selection.blocks == (fallback,)
    assert selection.omitted_block_ids == (exchange.id,)
    assert assistant not in selection.messages
    assert result not in selection.messages


def test_configured_input_threshold_and_duplicate_ids_fail_closed() -> None:
    block = _block("system.required", "required", priority=0, required=True)
    tokens = estimate_context_block_tokens(block)
    profile = _profile(tokens + 100)

    selection = select_context_blocks(
        (block,),
        profile,
        max_input_tokens=tokens,
        output_reserve_tokens=10,
    )
    assert selection.input_limit_tokens == tokens

    with pytest.raises(ValueError, match="unique"):
        select_context_blocks((block, block), profile)
