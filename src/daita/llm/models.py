"""Canonical provider-neutral model records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import re

from .._json import FrozenJsonObject, canonical_json
from .pricing import CostEstimate


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


_PROVIDER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Capabilities and hard token limits for one canonical model identity.

    Profiles are provider-neutral configuration. Provider adapters translate
    requests; they do not own context budgeting or routing policy.
    """

    id: str
    context_window_tokens: int
    max_output_tokens: int
    supports_tools: bool = False
    supports_parallel_tools: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = False
    supports_reasoning: bool = False
    supports_vision: bool = False
    supports_documents: bool = False
    supports_prompt_caching: bool = False
    supports_native_continuation: bool = False
    data_routing_classification: str = "standard"
    available: bool = True
    healthy: bool = True

    def __post_init__(self) -> None:
        _required_text(self.id, "model-profile id")
        if len(self.id) > 256 or any(character.isspace() for character in self.id):
            raise ValueError(
                "model-profile id must be a bounded provider:model identity"
            )
        provider, separator, model = self.id.partition(":")
        if not separator or not _PROVIDER_ID.fullmatch(provider) or not model:
            raise ValueError("model-profile id must use canonical provider:model form")
        for token_value, field_name in (
            (self.context_window_tokens, "context_window_tokens"),
            (self.max_output_tokens, "max_output_tokens"),
        ):
            if (
                not isinstance(token_value, int)
                or isinstance(token_value, bool)
                or token_value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError(
                "max_output_tokens must leave positive model input capacity"
            )
        for field_name in (
            "supports_tools",
            "supports_parallel_tools",
            "supports_structured_output",
            "supports_streaming",
            "supports_reasoning",
            "supports_vision",
            "supports_documents",
            "supports_prompt_caching",
            "supports_native_continuation",
            "available",
            "healthy",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if self.supports_parallel_tools and not self.supports_tools:
            raise ValueError("parallel tool support requires native tool support")
        _required_text(
            self.data_routing_classification,
            "data_routing_classification",
        )
        if len(self.data_routing_classification) > 64:
            raise ValueError("data_routing_classification must be bounded")

    @property
    def maximum_input_tokens(self) -> int:
        """Input capacity after reserving the model's full output allowance."""

        return self.context_window_tokens - self.max_output_tokens


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


class ModelSensitivity(str, Enum):
    """Authoritative data-routing sensitivity attached before provider I/O."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def routing_rank(self) -> int:
        return (
            ModelSensitivity.PUBLIC,
            ModelSensitivity.INTERNAL,
            ModelSensitivity.CONFIDENTIAL,
            ModelSensitivity.RESTRICTED,
        ).index(self)


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
    provider_call_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "tool-call id")
        _required_text(self.name, "tool-call name")
        if self.provider_call_id is not None:
            _required_text(self.provider_call_id, "tool-call provider_call_id")
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
    role: MessageRole
    content: tuple[ContentBlock, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    provider_id: str | None = None
    provider_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("message role must be a MessageRole")
        if self.provider_id is not None:
            _required_text(self.provider_id, "message provider_id")

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
        provider_metadata = FrozenJsonObject.from_mapping(self.provider_metadata)
        if (
            provider_metadata or self.provider_id is not None
        ) and self.role is not MessageRole.ASSISTANT:
            raise ValueError(
                "only an assistant message may contain provider identity or metadata"
            )

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
        provider_call_ids = [
            call.provider_call_id
            for call in tool_calls
            if call.provider_call_id is not None
        ]
        if len(provider_call_ids) != len(set(provider_call_ids)):
            raise ValueError("Duplicate provider tool-call IDs in one message")

        object.__setattr__(self, "content", content)
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(self, "provider_metadata", provider_metadata)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Normalized usage; reasoning and cache counts are subsets, not totals."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_estimate: CostEstimate = field(default_factory=CostEstimate.unavailable)

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
        if not isinstance(self.cost_estimate, CostEstimate):
            raise TypeError("cost_estimate must be a CostEstimate")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    response_schema: Mapping[str, object] | None = None
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL
    allow_parallel_tool_calls: bool | None = None

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        tools = tuple(self.tools)
        response_schema = (
            None
            if self.response_schema is None
            else FrozenJsonObject.from_mapping(self.response_schema)
        )
        if not messages:
            raise ValueError("model request must contain at least one message")
        if any(not isinstance(message, CanonicalMessage) for message in messages):
            raise TypeError("model request messages must be canonical messages")
        if any(not isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("model request tools must be tool definitions")
        tool_names = [tool.name for tool in tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Duplicate tool definitions in one model request")
        if response_schema is not None:
            if not response_schema:
                raise ValueError("model-request response_schema cannot be empty")
            if len(canonical_json(response_schema)) > 64 * 1_024:
                raise ValueError(
                    "model-request response_schema exceeds its character bound"
                )
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("model-request sensitivity must be ModelSensitivity")
        if self.allow_parallel_tool_calls is not None and not isinstance(
            self.allow_parallel_tool_calls,
            bool,
        ):
            raise TypeError(
                "model-request allow_parallel_tool_calls must be bool or None"
            )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "response_schema", response_schema)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    finish_reason: FinishReason
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    request_input_tokens: int | None = None
    provider_id: str | None = None
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
        provider_call_ids = [
            call.provider_call_id
            for call in tool_calls
            if call.provider_call_id is not None
        ]
        if len(provider_call_ids) != len(set(provider_call_ids)):
            raise ValueError("Duplicate provider tool-call IDs in one model response")
        if tool_calls and self.finish_reason is not FinishReason.TOOL_CALLS:
            raise ValueError("tool calls require the tool_calls finish reason")
        if not tool_calls and self.finish_reason is FinishReason.TOOL_CALLS:
            raise ValueError("tool_calls finish reason requires at least one tool call")
        if not isinstance(self.usage, ModelUsage):
            raise TypeError("usage must be a ModelUsage record")
        request_input_tokens = self.request_input_tokens
        if request_input_tokens is None:
            request_input_tokens = self.usage.input_tokens
        if (
            not isinstance(request_input_tokens, int)
            or isinstance(request_input_tokens, bool)
            or request_input_tokens < 0
        ):
            raise ValueError("request_input_tokens must be a non-negative integer")
        if self.provider_response_id is not None:
            _required_text(self.provider_response_id, "provider_response_id")
        if self.provider_id is not None:
            _required_text(self.provider_id, "model-response provider_id")
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(self, "request_input_tokens", request_input_tokens)
        object.__setattr__(
            self,
            "provider_metadata",
            FrozenJsonObject.from_mapping(self.provider_metadata),
        )


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    """One ordered non-empty text fragment from a streaming provider."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("model text delta must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ModelToolCallDelta:
    """One ordered fragment of a provider tool call's JSON arguments."""

    index: int
    arguments_delta: str
    id: str | None = None
    name: str | None = None
    provider_call_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 0
        ):
            raise ValueError("tool-call delta index must be non-negative")
        if not isinstance(self.arguments_delta, str):
            raise TypeError("tool-call arguments_delta must be a string")
        for value, field_name in (
            (self.id, "tool-call delta id"),
            (self.name, "tool-call delta name"),
            (self.provider_call_id, "tool-call delta provider_call_id"),
        ):
            if value is not None:
                _required_text(value, field_name)


@dataclass(frozen=True, slots=True)
class ModelStreamCompleted:
    """Terminal stream event carrying the same canonical response as generate."""

    response: ModelResponse

    def __post_init__(self) -> None:
        if not isinstance(self.response, ModelResponse):
            raise TypeError("stream completion response must be ModelResponse")


ModelStreamEvent = ModelTextDelta | ModelToolCallDelta | ModelStreamCompleted
