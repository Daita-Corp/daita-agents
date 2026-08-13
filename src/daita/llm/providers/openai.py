"""OpenAI Responses adapter for the provider-neutral model boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol, cast
from uuid import uuid4

from ..._installation import repair_guidance
from ..._json import FrozenJsonObject, canonical_json, thaw_json
from ..errors import ModelProviderError, ProviderErrorCode, detached_provider_error
from ..models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelToolCallDelta,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from ..pricing import (
    BillableQuantity,
    CostEstimate,
    PricingSchedule,
    calculate_cost_estimate,
    has_complete_pricing_coverage,
    load_bundled_pricing_schedules,
    validate_pricing_schedules,
)


class _ResponsesResource(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _ResponsesResource: ...


class _ResponseOutputOverride:
    """Retain terminal response metadata while supplying streamed output items."""

    __slots__ = ("_response", "output")

    def __init__(self, response: object, output: tuple[object, ...]) -> None:
        self._response = response
        self.output = output

    def __getattr__(self, name: str) -> object:
        if isinstance(self._response, Mapping) and name in self._response:
            return self._response[name]
        return getattr(self._response, name)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OpenAIResponsesProvider:
    """Translate canonical requests to the OpenAI Responses API only."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_output_tokens: int | None = None,
        client: _OpenAIClient | None = None,
        id_factory: Callable[[str], str] | None = None,
        service_tier: str = "default",
        region: str = "global",
        pricing_schedules: Iterable[PricingSchedule] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if max_output_tokens is not None and (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if service_tier != "default":
            raise ValueError("only the default OpenAI service tier is admitted")
        if region != "global":
            raise ValueError("only the global OpenAI endpoint is admitted")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.model = model.strip()
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens
        self._client = client
        self._id_factory = _new_id if id_factory is None else id_factory
        self._service_tier = service_tier
        self._region = region
        self._pricing_schedules = (
            load_bundled_pricing_schedules()
            if pricing_schedules is None
            else validate_pricing_schedules(pricing_schedules)
        )
        self._clock = clock

    @property
    def provider_id(self) -> str:
        return f"openai:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return has_complete_pricing_coverage(
            self._pricing_schedules,
            provider="openai",
            model=self.model,
            endpoint="responses",
            requested_at=self._clock(),
            qualifiers={
                "service_tier": self._service_tier,
                "region": self._region,
            },
            required_metrics=(
                "input_uncached_tokens",
                "input_cache_read_tokens",
                "input_cache_write_tokens",
                "output_tokens",
            ),
            usage_range_metric="request_input_tokens",
        )

    @property
    def client(self) -> _OpenAIClient:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise ImportError(
                    "Daita's OpenAI runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from error
            self._client = cast(_OpenAIClient, AsyncOpenAI(api_key=self._api_key))
        return self._client

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        failure: ModelProviderError | None = None
        try:
            return await self._generate(request)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError as error:
            failure = error
        except Exception:
            failure = ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI provider boundary failed",
            )
        if failure is None:
            raise AssertionError("OpenAI provider failed without an error")
        raise detached_provider_error(failure)

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        arguments = self._request_arguments(request)
        requested_at = self._clock()
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
            return self._decode_response(response, requested_at=requested_at)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI returned a malformed response",
            ) from error

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Translate ordered Responses API events into canonical stream events."""

        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        failure: ModelProviderError | None = None
        try:
            async for event in self._stream(request):
                yield event
            return
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError as error:
            failure = error
        except Exception:
            failure = ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI provider boundary failed",
            )
        if failure is None:
            raise AssertionError("OpenAI provider failed without an error")
        raise detached_provider_error(failure)

    async def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        arguments = self._request_arguments(request)
        arguments["stream"] = True
        requested_at = self._clock()
        try:
            stream = await self.client.responses.create(**arguments)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error) from error

        canonical_ids_by_index: dict[int, str] = {}
        canonical_ids_by_provider_call_id: dict[str, str] = {}
        provider_call_ids_by_index: dict[int, str] = {}
        names_by_index: dict[int, str] = {}
        completed_items_by_index: dict[int, object] = {}
        allocated_ids: set[str] = set()
        completed = False
        try:
            async for event in cast(AsyncIterator[object], stream):
                event_type = _required_text(_field(event, "type"), "stream event type")
                if event_type == "response.output_text.delta":
                    delta = _stream_fragment(_field(event, "delta"), "text delta")
                    if delta:
                        yield ModelTextDelta(delta)
                elif event_type == "response.output_item.added":
                    item = _field(event, "item")
                    if _field(item, "type", None) != "function_call":
                        continue
                    index = _nonnegative_int(
                        _field(event, "output_index"), "output index"
                    )
                    provider_call_id = _optional_stream_identity(
                        _field(item, "call_id", None), "provider call_id"
                    )
                    name = _optional_stream_identity(
                        _field(item, "name", None), "function name"
                    )
                    canonical_id = canonical_ids_by_index.get(index)
                    if canonical_id is None:
                        canonical_id = self._id_factory("call")
                        if canonical_id in allocated_ids:
                            raise ValueError("id_factory returned a duplicate call ID")
                        allocated_ids.add(canonical_id)
                        canonical_ids_by_index[index] = canonical_id
                    if provider_call_id is not None:
                        canonical_ids_by_provider_call_id[provider_call_id] = (
                            canonical_id
                        )
                        provider_call_ids_by_index[index] = provider_call_id
                    if name is not None:
                        names_by_index[index] = name
                    yield ModelToolCallDelta(
                        index=index,
                        arguments_delta="",
                        id=canonical_id,
                        name=name,
                        provider_call_id=provider_call_id,
                    )
                elif event_type == "response.function_call_arguments.delta":
                    index = _nonnegative_int(
                        _field(event, "output_index"), "output index"
                    )
                    arguments_delta = _stream_fragment(
                        _field(event, "delta"), "function arguments delta"
                    )
                    if not arguments_delta:
                        continue
                    canonical_id = canonical_ids_by_index.get(index)
                    if canonical_id is None:
                        canonical_id = self._id_factory("call")
                        if canonical_id in allocated_ids:
                            raise ValueError("id_factory returned a duplicate call ID")
                        allocated_ids.add(canonical_id)
                        canonical_ids_by_index[index] = canonical_id
                    yield ModelToolCallDelta(
                        index=index,
                        arguments_delta=arguments_delta,
                        id=canonical_id,
                        name=names_by_index.get(index),
                        provider_call_id=provider_call_ids_by_index.get(index),
                    )
                elif event_type == "response.output_item.done":
                    index = _nonnegative_int(
                        _field(event, "output_index"), "output index"
                    )
                    completed_items_by_index[index] = _field(event, "item")
                elif event_type in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    native_response = _field(event, "response")
                    native_output = _field(native_response, "output", ())
                    if (
                        isinstance(native_output, Sequence)
                        and not isinstance(native_output, (str, bytes))
                        and not native_output
                        and completed_items_by_index
                    ):
                        native_response = _ResponseOutputOverride(
                            native_response,
                            tuple(
                                completed_items_by_index[index]
                                for index in sorted(completed_items_by_index)
                            ),
                        )
                    response = self._decode_response(
                        native_response,
                        requested_at=requested_at,
                        canonical_ids_by_index=canonical_ids_by_index,
                        canonical_ids_by_provider_call_id=(
                            canonical_ids_by_provider_call_id
                        ),
                    )
                    yield ModelStreamCompleted(response)
                    completed = True
                    return
                elif event_type == "error":
                    code = _optional_text(
                        _field(event, "code", None), "stream error code"
                    )
                    raise ModelProviderError(
                        _code_from_provider_value(code),
                        "OpenAI stream failed",
                    )
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI returned a malformed stream",
            ) from error
        except Exception as error:
            raise _normalize_error(error) from error
        if not completed:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OpenAI stream ended without a terminal response",
            )

    def _request_arguments(self, request: ModelRequest) -> dict[str, object]:
        arguments: dict[str, object] = {
            "model": self.model,
            "input": _response_input(request.messages, self.provider_id),
            "include": ["reasoning.encrypted_content"],
            "service_tier": self._service_tier,
            "store": False,
        }
        if self._max_output_tokens is not None:
            arguments["max_output_tokens"] = self._max_output_tokens
        if request.allow_parallel_tool_calls is not None:
            arguments["parallel_tool_calls"] = request.allow_parallel_tool_calls
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
        if request.response_schema is not None:
            arguments["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "daita_response",
                    "schema": FrozenJsonObject.from_mapping(
                        request.response_schema
                    ).to_dict(),
                    "strict": True,
                }
            }
        return arguments

    def _decode_response(
        self,
        response: object,
        *,
        requested_at: datetime | None = None,
        canonical_ids_by_index: Mapping[int, str] | None = None,
        canonical_ids_by_provider_call_id: Mapping[str, str] | None = None,
    ) -> ModelResponse:
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
        index_ids = {} if canonical_ids_by_index is None else canonical_ids_by_index
        provider_ids = (
            {}
            if canonical_ids_by_provider_call_id is None
            else canonical_ids_by_provider_call_id
        )
        for output_index, item in enumerate(output):
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
                canonical_id = provider_ids.get(
                    provider_call_id,
                    index_ids.get(output_index),
                )
                if canonical_id is None:
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
            incomplete_details = _field(response, "incomplete_details", None)
            incomplete_reason = _optional_text(
                _field(incomplete_details, "reason", None),
                "incomplete reason",
            )
            if status == "incomplete" and incomplete_reason == "max_output_tokens":
                raise ModelProviderError(
                    ProviderErrorCode.OUTPUT_LIMIT,
                    "OpenAI exhausted the output token limit",
                )
            raise ValueError("response contains neither text nor function calls")

        response_id = _optional_text(_field(response, "id", None), "response id")
        response_model = _required_text(
            _field(response, "model"),
            "response model",
        )
        service_tier = _optional_text(
            _field(response, "service_tier", None),
            "response service tier",
        )
        provider_metadata: dict[str, object] = {}
        if replay_items:
            provider_metadata["openai_replay_items"] = replay_items
        provider_metadata["pricing_dimensions"] = {
            "response_model": response_model,
            "service_tier": service_tier,
            "region": self._region,
        }
        usage_value = _field(response, "usage", None)
        usage = _decode_usage(usage_value)
        if usage_value is not None:
            if not _has_complete_billing_dimensions(usage_value):
                usage = replace(
                    usage,
                    cost_estimate=CostEstimate.unavailable(
                        "billing_dimensions_incomplete"
                    ),
                )
            else:
                qualifiers = (
                    {"service_tier": service_tier, "region": self._region}
                    if service_tier is not None
                    else {"region": self._region}
                )
                usage = replace(
                    usage,
                    cost_estimate=calculate_cost_estimate(
                        self._pricing_schedules,
                        provider="openai",
                        model=response_model,
                        endpoint="responses",
                        requested_at=requested_at or self._clock(),
                        qualifiers=qualifiers,
                        usage_values={
                            "request_input_tokens": Decimal(usage.input_tokens)
                        },
                        quantities=_billable_quantities(usage),
                    ),
                )
        return ModelResponse(
            finish_reason=finish_reason,
            text=normalized_text,
            tool_calls=tuple(calls),
            usage=usage,
            provider_id=self.provider_id,
            provider_response_id=response_id,
            provider_metadata=provider_metadata,
        )


OpenAIProvider = OpenAIResponsesProvider


def _response_input(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    provider_call_ids: dict[str, str] = {}
    for message in messages:
        same_origin = message.provider_id == provider_id
        if same_origin:
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
                provider_call_id = (
                    call.provider_call_id if same_origin else None
                ) or call.id
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
            value = _stream_fragment(_field(part, "text"), "output text")
            if value.strip():
                text.append(value)
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
    input_tokens = _nonnegative_int(
        _field(value, "input_tokens"),
        "input tokens",
    )
    output_tokens = _nonnegative_int(
        _field(value, "output_tokens"),
        "output tokens",
    )
    reasoning_tokens = _nonnegative_int(
        _field(output_details, "reasoning_tokens", 0),
        "reasoning tokens",
    )
    cache_read_tokens = _nonnegative_int(
        _field(input_details, "cached_tokens", 0),
        "cached tokens",
    )
    cache_write_tokens = _nonnegative_int(
        _field(input_details, "cache_write_tokens", 0),
        "cache write tokens",
    )
    if cache_read_tokens + cache_write_tokens > input_tokens:
        raise ValueError("OpenAI cache token subsets exceed total input tokens")
    if reasoning_tokens > output_tokens:
        raise ValueError("OpenAI reasoning tokens exceed total output tokens")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _has_complete_billing_dimensions(value: object) -> bool:
    input_details = _field(value, "input_tokens_details", None)
    output_details = _field(value, "output_tokens_details", None)
    return (
        input_details is not None
        and output_details is not None
        and _field(input_details, "cached_tokens", _MISSING) is not _MISSING
        and _field(input_details, "cache_write_tokens", _MISSING) is not _MISSING
        and _field(output_details, "reasoning_tokens", _MISSING) is not _MISSING
    )


def _billable_quantities(usage: ModelUsage) -> tuple[BillableQuantity, ...]:
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0 or usage.reasoning_tokens > usage.output_tokens:
        raise ValueError("OpenAI usage counters are internally inconsistent")
    return (
        BillableQuantity(
            "input_uncached_tokens",
            Decimal(uncached),
            "token",
        ),
        BillableQuantity(
            "input_cache_read_tokens",
            Decimal(usage.cache_read_tokens),
            "token",
        ),
        BillableQuantity(
            "input_cache_write_tokens",
            Decimal(usage.cache_write_tokens),
            "token",
        ),
        BillableQuantity(
            "output_tokens",
            Decimal(usage.output_tokens),
            "token",
        ),
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
    if (
        isinstance(error, (asyncio.TimeoutError, TimeoutError))
        or status == 408
        or "timeout" in name
    ):
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
    if value in {"authentication_error", "invalid_api_key", "permission_denied"}:
        return ProviderErrorCode.AUTHENTICATION_ERROR
    if value in {"rate_limit_error", "rate_limit_exceeded"}:
        return ProviderErrorCode.RATE_LIMIT_ERROR
    if value in {"context_length_exceeded", "context_window_exceeded"}:
        return ProviderErrorCode.CONTEXT_OVERFLOW
    if value in {"content_policy_violation", "content_blocked"}:
        return ProviderErrorCode.CONTENT_BLOCKED
    if value in {"model_not_found", "unknown_model"}:
        return ProviderErrorCode.MODEL_NOT_FOUND
    if value in {"invalid_request", "invalid_request_error"}:
        return ProviderErrorCode.INVALID_REQUEST
    if value in {"timeout", "request_timeout"}:
        return ProviderErrorCode.TIMEOUT
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


def _stream_fragment(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    return value


def _optional_stream_identity(value: object, label: str) -> str | None:
    fragment = _stream_fragment(value, label) if value is not None else ""
    return fragment if fragment.strip() else None


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = ["OpenAIProvider", "OpenAIResponsesProvider"]
