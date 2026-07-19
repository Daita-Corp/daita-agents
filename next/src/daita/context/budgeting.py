"""Deterministic approximate token budgeting for canonical context blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .._json import canonical_json
from ..llm.models import (
    CanonicalMessage,
    ModelProfile,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from .models import ContextBlock, ContextMessageGroup

APPROXIMATE_CHARACTERS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 4
_TOOL_OVERHEAD_TOKENS = 4
_GROUP_OVERHEAD_TOKENS = 1


class ContextBudgetError(RuntimeError):
    """Base failure for a context selection that cannot be made safely."""


class RequiredContextOverflow(ContextBudgetError):
    """Raised when required context cannot fit before model I/O."""

    def __init__(
        self,
        *,
        profile_id: str,
        required_tokens: int,
        available_tokens: int,
        tool_tokens: int,
        output_reserve_tokens: int,
    ) -> None:
        self.profile_id = profile_id
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        self.tool_tokens = tool_tokens
        self.output_reserve_tokens = output_reserve_tokens
        super().__init__(
            f"required context needs {required_tokens} tokens but model profile "
            f"{profile_id} has {available_tokens} available after tool and output "
            "reserves"
        )


@dataclass(frozen=True, slots=True)
class BudgetedContextBlock:
    block: ContextBlock
    estimated_tokens: int
    group_estimates: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block, ContextBlock):
            raise TypeError("budgeted block must contain a ContextBlock")
        if (
            not isinstance(self.estimated_tokens, int)
            or isinstance(self.estimated_tokens, bool)
            or self.estimated_tokens < 1
        ):
            raise ValueError("budgeted block token estimate must be positive")
        group_estimates = tuple(self.group_estimates)
        if len(group_estimates) != len(self.block.groups) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in group_estimates
        ):
            raise ValueError(
                "budgeted block requires one positive estimate per message group"
            )
        expected = sum(group_estimates)
        if expected != self.estimated_tokens:
            raise ValueError("budgeted block total must equal its group estimates")
        object.__setattr__(self, "group_estimates", group_estimates)


@dataclass(frozen=True, slots=True)
class ContextBudgetSelection:
    profile_id: str
    input_limit_tokens: int
    output_reserve_tokens: int
    tool_tokens: int
    selected: tuple[BudgetedContextBlock, ...]
    omitted_block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("selection profile_id must be non-empty")
        for value, field_name in (
            (self.input_limit_tokens, "input_limit_tokens"),
            (self.output_reserve_tokens, "output_reserve_tokens"),
            (self.tool_tokens, "tool_tokens"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        selected = tuple(self.selected)
        if any(not isinstance(item, BudgetedContextBlock) for item in selected):
            raise TypeError("selection must contain BudgetedContextBlock records")
        selected_ids = [item.block.id for item in selected]
        omitted_ids = tuple(self.omitted_block_ids)
        if any(not isinstance(value, str) or not value for value in omitted_ids):
            raise ValueError("omitted block IDs must be non-empty strings")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected block IDs must be unique")
        if len(omitted_ids) != len(set(omitted_ids)):
            raise ValueError("omitted block IDs must be unique")
        if set(selected_ids) & set(omitted_ids):
            raise ValueError("a context block cannot be selected and omitted")
        if self.estimated_input_tokens > self.input_limit_tokens:
            raise ValueError("selected context exceeds its input token limit")
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "omitted_block_ids", omitted_ids)

    @property
    def context_tokens(self) -> int:
        return sum(item.estimated_tokens for item in self.selected)

    @property
    def estimated_input_tokens(self) -> int:
        return self.tool_tokens + self.context_tokens

    @property
    def remaining_input_tokens(self) -> int:
        return self.input_limit_tokens - self.estimated_input_tokens

    @property
    def blocks(self) -> tuple[ContextBlock, ...]:
        return tuple(item.block for item in self.selected)

    @property
    def messages(self) -> tuple[CanonicalMessage, ...]:
        return tuple(message for block in self.blocks for message in block.messages)


def estimate_text_tokens(value: str) -> int:
    """Conservatively estimate text tokens with a stable four-character rule."""

    if not isinstance(value, str):
        raise TypeError("token estimation requires text")
    if not value:
        return 0
    return max(1, math.ceil(len(value) / APPROXIMATE_CHARACTERS_PER_TOKEN))


def estimate_message_tokens(message: CanonicalMessage) -> int:
    if not isinstance(message, CanonicalMessage):
        raise TypeError("message token estimation requires a CanonicalMessage")
    return _MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(
        canonical_json(_message_data(message))
    )


def estimate_message_group_tokens(group: ContextMessageGroup) -> int:
    if not isinstance(group, ContextMessageGroup):
        raise TypeError("message-group token estimation requires a ContextMessageGroup")
    return _GROUP_OVERHEAD_TOKENS + sum(
        estimate_message_tokens(message) for message in group.messages
    )


def estimate_context_block_tokens(block: ContextBlock) -> int:
    if not isinstance(block, ContextBlock):
        raise TypeError("block token estimation requires a ContextBlock")
    return sum(estimate_message_group_tokens(group) for group in block.groups)


def estimate_tool_tokens(tools: Sequence[ToolDefinition]) -> int:
    if isinstance(tools, (str, bytes)):
        raise TypeError("tool token estimation requires a sequence")
    normalized = tuple(tools)
    if any(not isinstance(tool, ToolDefinition) for tool in normalized):
        raise TypeError("tool token estimation requires ToolDefinition records")
    return sum(
        _TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(
            canonical_json(
                {
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "name": tool.name,
                }
            )
        )
        for tool in normalized
    )


def select_context_blocks(
    blocks: Sequence[ContextBlock],
    profile: ModelProfile,
    *,
    tools: Sequence[ToolDefinition] = (),
    output_reserve_tokens: int | None = None,
    max_input_tokens: int | None = None,
) -> ContextBudgetSelection:
    """Select complete blocks by requirement, priority, then declaration order.

    Selection never splits a block or any message group within it. The returned
    blocks retain declaration order so canonical transcript ordering is stable.
    """

    if isinstance(blocks, (str, bytes)):
        raise TypeError("context blocks must be a sequence")
    normalized = tuple(blocks)
    if not normalized:
        raise ValueError("context selection requires at least one block")
    if any(not isinstance(block, ContextBlock) for block in normalized):
        raise TypeError("context selection requires ContextBlock records")
    block_ids = [block.id for block in normalized]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("context block IDs must be unique")
    if not isinstance(profile, ModelProfile):
        raise TypeError("context selection requires a ModelProfile")

    reserve = (
        profile.max_output_tokens
        if output_reserve_tokens is None
        else _positive_integer(output_reserve_tokens, "output_reserve_tokens")
    )
    if reserve >= profile.context_window_tokens:
        raise ValueError("output reserve must leave positive model input capacity")
    profile_input_limit = profile.context_window_tokens - reserve
    if max_input_tokens is None:
        input_limit = profile_input_limit
    else:
        configured_limit = _positive_integer(max_input_tokens, "max_input_tokens")
        input_limit = min(profile_input_limit, configured_limit)

    tool_tokens = estimate_tool_tokens(tools)
    available = input_limit - tool_tokens
    estimates = tuple(_budgeted(block) for block in normalized)
    required_tokens = sum(
        item.estimated_tokens for item in estimates if item.block.required
    )
    if available < 0 or required_tokens > available:
        raise RequiredContextOverflow(
            profile_id=profile.id,
            required_tokens=required_tokens,
            available_tokens=max(0, available),
            tool_tokens=tool_tokens,
            output_reserve_tokens=reserve,
        )

    selected_indexes = {
        index for index, item in enumerate(estimates) if item.block.required
    }
    used = required_tokens
    optional_indexes = sorted(
        (index for index, item in enumerate(estimates) if not item.block.required),
        key=lambda index: (-estimates[index].block.priority, index),
    )
    for index in optional_indexes:
        estimate = estimates[index].estimated_tokens
        if used + estimate <= available:
            selected_indexes.add(index)
            used += estimate

    selected = tuple(
        estimate
        for index, estimate in enumerate(estimates)
        if index in selected_indexes
    )
    omitted = tuple(
        block.id
        for index, block in enumerate(normalized)
        if index not in selected_indexes
    )
    return ContextBudgetSelection(
        profile_id=profile.id,
        input_limit_tokens=input_limit,
        output_reserve_tokens=reserve,
        tool_tokens=tool_tokens,
        selected=selected,
        omitted_block_ids=omitted,
    )


def _budgeted(block: ContextBlock) -> BudgetedContextBlock:
    estimates = tuple(estimate_message_group_tokens(group) for group in block.groups)
    return BudgetedContextBlock(
        block=block,
        estimated_tokens=sum(estimates),
        group_estimates=estimates,
    )


def _positive_integer(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _tool_call_data(call: ToolCall) -> dict[str, object]:
    return {
        "arguments": call.arguments,
        "id": call.id,
        "name": call.name,
        "provider_call_id": call.provider_call_id,
    }


def _message_data(message: CanonicalMessage) -> dict[str, object]:
    content: list[dict[str, object]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"text": block.text, "type": "text"})
        elif isinstance(block, ToolResultBlock):
            content.append(
                {
                    "call_id": block.call_id,
                    "is_error": block.is_error,
                    "output": block.output,
                    "type": "tool_result",
                }
            )
        else:  # pragma: no cover - CanonicalMessage rejects unknown blocks.
            raise TypeError("unsupported canonical content block")
    return {
        "agent_id": message.agent_id,
        "content": content,
        "operation_id": message.operation_id,
        "provider_metadata": message.provider_metadata,
        "role": message.role.value,
        "session_id": message.session_id,
        "tool_calls": [_tool_call_data(call) for call in message.tool_calls],
        "turn_id": message.turn_id,
    }


__all__ = [
    "APPROXIMATE_CHARACTERS_PER_TOKEN",
    "BudgetedContextBlock",
    "ContextBudgetError",
    "ContextBudgetSelection",
    "RequiredContextOverflow",
    "estimate_context_block_tokens",
    "estimate_message_group_tokens",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "estimate_tool_tokens",
    "select_context_blocks",
]
