"""Canonical provider-neutral model records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from .._json import FrozenJsonObject


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str

    def __post_init__(self) -> None:
        _required_text(self.text, "text")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.id, "tool-call id")
        _required_text(self.name, "tool-call name")
        object.__setattr__(
            self, "arguments", FrozenJsonObject.from_mapping(self.arguments)
        )


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    call_id: str
    output: Mapping[str, object] = field(default_factory=dict)
    is_error: bool = False

    def __post_init__(self) -> None:
        _required_text(self.call_id, "tool-result call_id")
        if not isinstance(self.is_error, bool):
            raise TypeError("tool-result is_error must be a boolean")
        object.__setattr__(self, "output", FrozenJsonObject.from_mapping(self.output))


ContentBlock = TextBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        _required_text(self.name, "tool name")
        _required_text(self.description, "tool description")
        object.__setattr__(
            self, "input_schema", FrozenJsonObject.from_mapping(self.input_schema)
        )


@dataclass(frozen=True, slots=True)
class CanonicalMessage:
    agent_id: str
    operation_id: str
    role: MessageRole
    content: tuple[ContentBlock, ...] = ()
    turn_id: str | None = None
    session_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "message agent_id")
        _required_text(self.operation_id, "message operation_id")
        if not isinstance(self.role, MessageRole):
            raise TypeError("message role must be a MessageRole")
        if self.turn_id is not None:
            _required_text(self.turn_id, "message turn_id")
        if self.session_id is not None:
            _required_text(self.session_id, "message session_id")

        content = tuple(self.content)
        tool_calls = tuple(self.tool_calls)
        if not content and not tool_calls:
            raise ValueError("message must contain content or assistant tool calls")
        if any(
            not isinstance(block, (TextBlock, ToolResultBlock)) for block in content
        ):
            raise TypeError("message content contains an unsupported block")
        if any(not isinstance(call, ToolCall) for call in tool_calls):
            raise TypeError("message tool_calls must contain ToolCall records")
        if tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only an assistant message may contain tool calls")

        has_tool_result = any(isinstance(block, ToolResultBlock) for block in content)
        if self.role is MessageRole.TOOL:
            if not has_tool_result or any(
                not isinstance(block, ToolResultBlock) for block in content
            ):
                raise ValueError("a tool message must contain only tool-result blocks")
        elif has_tool_result:
            raise ValueError("tool-result blocks require the tool message role")

        call_ids = [call.id for call in tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Duplicate tool-call IDs in one message")

        object.__setattr__(self, "content", content)
        object.__setattr__(self, "tool_calls", tool_calls)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Normalized usage; reasoning and cache counts are subsets, not totals."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        token_fields = (
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in token_fields
        ):
            raise ValueError("token counts must be non-negative integers")
        if not isinstance(self.estimated_cost_usd, Decimal):
            raise TypeError("estimated_cost_usd must be a Decimal")
        if not self.estimated_cost_usd.is_finite() or self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be a finite non-negative Decimal")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelRequest:
    operation_id: str
    turn_id: str
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.operation_id, "model-request operation_id")
        _required_text(self.turn_id, "model-request turn_id")
        messages = tuple(self.messages)
        tools = tuple(self.tools)
        if not messages:
            raise ValueError("model request must contain at least one message")
        if any(not isinstance(message, CanonicalMessage) for message in messages):
            raise TypeError("model request messages must be canonical messages")
        if any(message.operation_id != self.operation_id for message in messages):
            raise ValueError("model request messages must belong to its operation")
        if len({message.agent_id for message in messages}) != 1:
            raise ValueError("model request messages must belong to one agent")
        if any(not isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("model request tools must be tool definitions")
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Duplicate tool definitions in one model request")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    finish_reason: FinishReason
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    provider_response_id: str | None = None
    provider_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.finish_reason, FinishReason):
            raise TypeError("finish_reason must be a FinishReason")
        if self.text is not None:
            _required_text(self.text, "model-response text")
        tool_calls = tuple(self.tool_calls)
        if any(not isinstance(call, ToolCall) for call in tool_calls):
            raise TypeError("model-response tool_calls must be ToolCall records")
        if self.text is None and not tool_calls:
            raise ValueError("model response must contain text or tool calls")
        call_ids = [call.id for call in tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Duplicate tool-call IDs in one model response")
        if tool_calls and self.finish_reason is not FinishReason.TOOL_CALLS:
            raise ValueError("tool calls require the tool_calls finish reason")
        if not tool_calls and self.finish_reason is FinishReason.TOOL_CALLS:
            raise ValueError("tool_calls finish reason requires at least one tool call")
        if not isinstance(self.usage, ModelUsage):
            raise TypeError("usage must be a ModelUsage record")
        if self.provider_response_id is not None:
            _required_text(self.provider_response_id, "provider_response_id")
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(
            self,
            "provider_metadata",
            FrozenJsonObject.from_mapping(self.provider_metadata),
        )
