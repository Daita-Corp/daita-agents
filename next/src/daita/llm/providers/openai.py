"""OpenAI Responses adapter for the provider-neutral model boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
import json
from typing import Protocol, cast
from uuid import uuid4

from ..._json import FrozenJsonObject, canonical_json, thaw_json
from ..errors import ModelProviderError, ProviderErrorCode
from ..models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)


class _ResponsesResource(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _ResponsesResource: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class OpenAIResponsesProvider:
    """Translate canonical requests to the OpenAI Responses API only."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: _OpenAIClient | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        self.model = model
        self._api_key = api_key
        self._client = client
        self._id_factory = _new_id if id_factory is None else id_factory

    @property
    def provider_id(self) -> str:
        return f"openai:{self.model}"

    @property
    def client(self) -> _OpenAIClient:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise ImportError(
                    "openai is required. Install with: "
                    "pip install 'daita-agents[openai]'"
                ) from error
            self._client = cast(_OpenAIClient, AsyncOpenAI(api_key=self._api_key))
        return self._client

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        arguments: dict[str, object] = {
            "model": self.model,
            "input": _response_input(request.messages),
            "include": ["reasoning.encrypted_content"],
            "store": False,
        }
        if request.tools:
            arguments["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": FrozenJsonObject.from_mapping(
                        tool.input_schema
                    ).to_dict(),
                    "strict": False,
                }
                for tool in request.tools
            ]
        try:
            response = await self.client.responses.create(**arguments)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error) from error
        try:
            return self._decode_response(response)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI returned a malformed response",
            ) from error

    def _decode_response(self, response: object) -> ModelResponse:
        status = _optional_text(_field(response, "status", None), "response status")
        if status == "failed":
            failure = _field(response, "error", None)
            code = _optional_text(_field(failure, "code", None), "response error code")
            raise ModelProviderError(
                _code_from_provider_value(code),
                "OpenAI reported a failed response",
            )

        output = _field(response, "output", ())
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise ValueError("response output must be a sequence")

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        replay_items: list[dict[str, object]] = []
        canonical_ids: set[str] = set()
        for item in output:
            item_type = _required_text(_field(item, "type"), "response item type")
            if item_type == "function_call":
                provider_call_id = _required_text(
                    _field(item, "call_id"), "provider call_id"
                )
                name = _required_text(_field(item, "name"), "function name")
                encoded_arguments = _required_text(
                    _field(item, "arguments"), "function arguments"
                )
                decoded_arguments = json.loads(encoded_arguments)
                if not isinstance(decoded_arguments, dict):
                    raise ValueError("function arguments must decode to an object")
                canonical_id = self._id_factory("call")
                if canonical_id in canonical_ids:
                    raise ValueError("id_factory returned a duplicate call ID")
                canonical_ids.add(canonical_id)
                calls.append(
                    ToolCall(
                        id=canonical_id,
                        provider_call_id=provider_call_id,
                        name=name,
                        arguments=decoded_arguments,
                    )
                )
            elif item_type == "message":
                text_parts.extend(_message_text(item))
            elif item_type == "reasoning":
                replay_items.append(_plain_provider_item(item))
            elif item_type == "computer_tool_call":
                continue

        fallback_value = _field(response, "output_text", None)
        if fallback_value is not None and not isinstance(fallback_value, str):
            raise ValueError("response output_text must be text")
        fallback_text = (
            None
            if fallback_value is None or not fallback_value.strip()
            else fallback_value
        )
        text = "\n".join(part for part in text_parts if part.strip()).strip()
        if not text and fallback_text is not None:
            text = fallback_text.strip()
        normalized_text = text or None

        if calls:
            finish_reason = FinishReason.TOOL_CALLS
        elif normalized_text is not None and status == "incomplete":
            finish_reason = FinishReason.LENGTH
        elif normalized_text is not None:
            finish_reason = FinishReason.STOP
        else:
            raise ValueError("response contains neither text nor function calls")

        response_id = _optional_text(_field(response, "id", None), "response id")
        provider_metadata: dict[str, object] = {}
        if replay_items:
            provider_metadata["openai_replay_items"] = replay_items
        return ModelResponse(
            finish_reason=finish_reason,
            text=normalized_text,
            tool_calls=tuple(calls),
            usage=_decode_usage(_field(response, "usage", None)),
            provider_response_id=response_id,
            provider_metadata=provider_metadata,
        )


OpenAIProvider = OpenAIResponsesProvider


def _response_input(messages: tuple[CanonicalMessage, ...]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    provider_call_ids: dict[str, str] = {}
    for message in messages:
        metadata = FrozenJsonObject.from_mapping(message.provider_metadata)
        replay_value = metadata.get("openai_replay_items")
        if replay_value is not None:
            replay_items = thaw_json(replay_value)
            if not isinstance(replay_items, list):
                raise ModelProviderError(
                    ProviderErrorCode.INVALID_REQUEST,
                    "OpenAI replay metadata must contain JSON objects",
                )
            decoded_replay_items: list[dict[str, object]] = []
            for replay_item in replay_items:
                if not isinstance(replay_item, dict):
                    raise ModelProviderError(
                        ProviderErrorCode.INVALID_REQUEST,
                        "OpenAI replay metadata must contain JSON objects",
                    )
                decoded_replay_items.append(replay_item)
            items.extend(decoded_replay_items)
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if text:
            items.append({"role": message.role.value, "content": text})
        if message.role is MessageRole.ASSISTANT:
            for call in message.tool_calls:
                provider_call_id = call.provider_call_id or call.id
                provider_call_ids[call.id] = provider_call_id
                items.append(
                    {
                        "type": "function_call",
                        "call_id": provider_call_id,
                        "name": call.name,
                        "arguments": canonical_json(call.arguments),
                    }
                )
        if message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": provider_call_ids.get(block.call_id, block.call_id),
                        "output": canonical_json(
                            {"is_error": block.is_error, "output": block.output}
                        ),
                    }
                )
    if not items:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no OpenAI input items",
        )
    return items


def _plain_provider_item(item: object) -> dict[str, object]:
    if isinstance(item, FrozenJsonObject):
        return item.to_dict()
    if isinstance(item, Mapping):
        return FrozenJsonObject.from_mapping(item).to_dict()
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        dumped_value = model_dump(mode="json", exclude_none=True)
        if isinstance(dumped_value, Mapping):
            return FrozenJsonObject.from_mapping(dumped_value).to_dict()
    provider_item: dict[str, object] = {}
    for name in ("id", "type", "summary", "status", "encrypted_content"):
        field = getattr(item, name, _MISSING)
        if field is not _MISSING and field is not None:
            provider_item[name] = field
    if provider_item.get("type") != "reasoning":
        raise ValueError("reasoning replay item is malformed")
    return FrozenJsonObject.from_mapping(provider_item).to_dict()


def _message_text(item: object) -> list[str]:
    content = _field(item, "content", ())
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise ValueError("message content must be a sequence")
    text: list[str] = []
    for part in content:
        part_type = _required_text(_field(part, "type"), "message part type")
        if part_type == "output_text":
            text.append(_required_text(_field(part, "text"), "output text"))
        elif part_type == "refusal":
            raise ModelProviderError(
                ProviderErrorCode.CONTENT_BLOCKED,
                "OpenAI refused the response",
            )
    return text


def _decode_usage(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage()
    input_details = _field(value, "input_tokens_details", None)
    output_details = _field(value, "output_tokens_details", None)
    return ModelUsage(
        input_tokens=_nonnegative_int(_field(value, "input_tokens", 0), "input tokens"),
        output_tokens=_nonnegative_int(
            _field(value, "output_tokens", 0), "output tokens"
        ),
        reasoning_tokens=_nonnegative_int(
            _field(output_details, "reasoning_tokens", 0), "reasoning tokens"
        ),
        cache_read_tokens=_nonnegative_int(
            _field(input_details, "cached_tokens", 0), "cached tokens"
        ),
        estimated_cost_usd=Decimal("0"),
    )


def _normalize_error(error: Exception) -> ModelProviderError:
    status_value = _field(error, "status_code", None)
    status = (
        status_value
        if isinstance(status_value, int) and not isinstance(status_value, bool)
        else None
    )
    code = _optional_text(_field(error, "code", None), "provider error code")
    name = type(error).__name__.lower()
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name:
        normalized = ProviderErrorCode.TIMEOUT
    elif status in {401, 403} or "authentication" in name or "permission" in name:
        normalized = ProviderErrorCode.AUTHENTICATION_ERROR
    elif status == 429 or "ratelimit" in name or "rate_limit" in name:
        normalized = ProviderErrorCode.RATE_LIMIT_ERROR
    elif status == 404 or code in {"model_not_found", "unknown_model"}:
        normalized = ProviderErrorCode.MODEL_NOT_FOUND
    elif code in {"context_length_exceeded", "context_window_exceeded"}:
        normalized = ProviderErrorCode.CONTEXT_OVERFLOW
    elif code in {"content_policy_violation", "content_blocked"}:
        normalized = ProviderErrorCode.CONTENT_BLOCKED
    elif isinstance(error, ConnectionError) or status is not None and status >= 500:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    elif status is not None and 400 <= status < 500:
        normalized = ProviderErrorCode.INVALID_REQUEST
    else:
        normalized = ProviderErrorCode.PROVIDER_UNAVAILABLE
    return ModelProviderError(normalized, f"OpenAI request failed: {normalized.value}")


def _code_from_provider_value(value: str | None) -> ProviderErrorCode:
    if value in {"context_length_exceeded", "context_window_exceeded"}:
        return ProviderErrorCode.CONTEXT_OVERFLOW
    if value in {"content_policy_violation", "content_blocked"}:
        return ProviderErrorCode.CONTENT_BLOCKED
    if value in {"model_not_found", "unknown_model"}:
        return ProviderErrorCode.MODEL_NOT_FOUND
    return ProviderErrorCode.PROVIDER_UNAVAILABLE


_MISSING = object()


def _field(value: object, name: str, default: object = _MISSING) -> object:
    if value is None:
        if default is not _MISSING:
            return default
        raise KeyError(name)
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _MISSING:
        return default
    raise KeyError(name)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = ["OpenAIProvider", "OpenAIResponsesProvider"]
