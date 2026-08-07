"""Scoped OpenAI-compatible Chat Completions model adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
import ipaddress
import json
import re
from typing import Protocol, cast
from urllib.parse import urlsplit
from uuid import uuid4

from ..._json import FrozenJsonObject, canonical_json
from ..._installation import repair_guidance
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
    CostBasis,
    PricingQualifier,
    PricingSchedule,
    calculate_cost_estimate,
    has_complete_pricing_coverage,
    validate_pricing_schedules,
)

_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_CONTINUATION_KEY = "openai_compatible_continuation"


class _CompletionsResource(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _ChatResource(Protocol):
    @property
    def completions(self) -> _CompletionsResource: ...


class _OpenAICompatibleClient(Protocol):
    @property
    def chat(self) -> _ChatResource: ...


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class _StreamedToolCall:
    canonical_id: str
    provider_call_id: str | None = None
    name: str | None = None
    argument_fragments: list[str] = field(default_factory=list)


class OpenAICompatibleProvider:
    """Translate canonical requests to one explicitly configured chat endpoint."""

    def __init__(
        self,
        model: str,
        *,
        provider: str,
        base_url: str,
        api_key: str | None = None,
        max_tokens: int = 1_024,
        client: _OpenAICompatibleClient | None = None,
        id_factory: Callable[[str], str] | None = None,
        pricing_schedules: Iterable[PricingSchedule] = (),
        pricing_qualifiers: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(provider, str) or not _PROVIDER_NAME.fullmatch(provider):
            raise ValueError("provider must be a canonical provider name")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if id_factory is not None and not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if pricing_qualifiers is not None and not isinstance(
            pricing_qualifiers, Mapping
        ):
            raise TypeError("pricing_qualifiers must be a mapping or None")
        if pricing_qualifiers is not None and len(pricing_qualifiers) > 16:
            raise ValueError("pricing_qualifiers exceed their bound")
        admitted_schedules = validate_pricing_schedules(pricing_schedules)
        if any(
            schedule.basis is not CostBasis.CONFIGURED_CONTRACT
            for schedule in admitted_schedules
        ):
            raise ValueError(
                "compatible endpoint schedules must use configured_contract"
            )
        qualifier_values = {} if pricing_qualifiers is None else pricing_qualifiers
        admitted_qualifiers = tuple(
            PricingQualifier(name, value) for name, value in qualifier_values.items()
        )
        self.model = model.strip()
        self.provider = provider
        self.base_url = _validate_base_url(base_url)
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = client
        self._id_factory = _new_id if id_factory is None else id_factory
        self._pricing_schedules = admitted_schedules
        self._pricing_qualifiers = admitted_qualifiers
        self._clock = clock

    @property
    def provider_id(self) -> str:
        return f"{self.provider}:{self.model}"

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        return has_complete_pricing_coverage(
            self._pricing_schedules,
            provider=self.provider,
            model=self.model,
            endpoint="chat_completions",
            requested_at=self._clock(),
            qualifiers=self._pricing_qualifiers,
            required_metrics=(
                "input_uncached_tokens",
                "input_cache_read_tokens",
                "input_cache_write_tokens",
                "output_tokens",
            ),
            usage_range_metric="request_input_tokens",
        )

    @property
    def client(self) -> _OpenAICompatibleClient:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise ImportError(
                    "Daita's OpenAI-compatible runtime dependency is unavailable. "
                    f"{repair_guidance()}"
                ) from error
            self._client = cast(
                _OpenAICompatibleClient,
                AsyncOpenAI(api_key=self._api_key, base_url=self.base_url),
            )
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
                f"{self.provider} provider boundary failed",
            )
        if failure is None:
            raise AssertionError("compatible provider failed without an error")
        raise detached_provider_error(failure)

    async def _generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        arguments = self._request_arguments(request)
        requested_at = self._clock()
        try:
            response = await self.client.chat.completions.create(**arguments)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error, self.provider) from error
        try:
            return self._decode_response(response, requested_at=requested_at)
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"{self.provider} returned a malformed response",
            ) from error

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
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
                f"{self.provider} provider boundary failed",
            )
        if failure is None:
            raise AssertionError("compatible provider failed without an error")
        raise detached_provider_error(failure)

    async def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        arguments = self._request_arguments(request)
        arguments["stream"] = True
        arguments["stream_options"] = {"include_usage": True}
        requested_at = self._clock()
        try:
            raw_stream = await self.client.chat.completions.create(**arguments)
        except asyncio.CancelledError:
            raise
        except ImportError:
            raise
        except ModelProviderError:
            raise
        except Exception as error:
            raise _normalize_error(error, self.provider) from error
        iterator_method = getattr(raw_stream, "__aiter__", None)
        if not callable(iterator_method):
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"{self.provider} returned a malformed stream",
            )

        text_fragments: list[str] = []
        tool_states: dict[int, _StreamedToolCall] = {}
        finish_reason: str | None = None
        usage_value: object | None = None
        response_id: str | None = None
        response_model: str | None = None
        service_tier: str | None = None
        iterator = cast(AsyncIterator[object], iterator_method())
        while True:
            try:
                chunk = await anext(iterator)
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                raise
            except ModelProviderError:
                raise
            except Exception as error:
                raise _normalize_error(error, self.provider) from error
            try:
                chunk_id = _optional_text(_field(chunk, "id", None), "chunk id")
                if chunk_id is not None:
                    if response_id is not None and response_id != chunk_id:
                        raise ValueError("stream response ID changed")
                    response_id = chunk_id
                chunk_model = _optional_text(
                    _field(chunk, "model", None),
                    "stream response model",
                )
                if chunk_model is not None:
                    if response_model is not None and response_model != chunk_model:
                        raise ValueError("stream response model changed")
                    response_model = chunk_model
                chunk_service_tier = _optional_text(
                    _field(chunk, "service_tier", None),
                    "stream service tier",
                )
                if chunk_service_tier is not None:
                    if service_tier is not None and service_tier != chunk_service_tier:
                        raise ValueError("stream service tier changed")
                    service_tier = chunk_service_tier
                chunk_usage = _field(chunk, "usage", None)
                if chunk_usage is not None:
                    usage_value = chunk_usage
                choices = _sequence(
                    _field(chunk, "choices", ()),
                    "stream choices",
                )
                if len(choices) > 1:
                    raise ValueError("stream must contain at most one choice")
                if not choices:
                    # OpenAI-compatible servers may emit metadata-only or
                    # heartbeat chunks. Terminal validity is checked below.
                    continue
                choice = choices[0]
                choice_index = _field(choice, "index", 0)
                if choice_index != 0:
                    raise ValueError("stream choice index must be zero")
                delta = _field(choice, "delta")
                refusal = _optional_text(
                    _field(delta, "refusal", None),
                    "stream refusal",
                )
                if refusal is not None:
                    raise ModelProviderError(
                        ProviderErrorCode.CONTENT_BLOCKED,
                        f"{self.provider} blocked the response",
                    )
                content = _field(delta, "content", None)
                if content is not None:
                    if not isinstance(content, str):
                        raise ValueError("stream content must be text")
                    text_fragments.append(content)
                    if content:
                        yield ModelTextDelta(content)
                raw_calls = _field(delta, "tool_calls", ())
                if raw_calls is None:
                    raw_calls = ()
                for item in _sequence(raw_calls, "stream tool calls"):
                    index = _nonnegative_int(
                        _field(item, "index"),
                        "stream tool index",
                    )
                    state = tool_states.get(index)
                    is_first = state is None
                    if state is None:
                        state = _StreamedToolCall(self._id_factory("call"))
                        tool_states[index] = state
                    native_id = _optional_text(
                        _field(item, "id", None),
                        "stream provider call ID",
                    )
                    if native_id is not None:
                        if (
                            state.provider_call_id is not None
                            and state.provider_call_id != native_id
                        ):
                            raise ValueError("stream provider call ID changed")
                        state.provider_call_id = native_id
                    function = _field(item, "function", None)
                    name: str | None = None
                    argument_delta = ""
                    if function is not None:
                        name = _optional_text(
                            _field(function, "name", None),
                            "stream tool name",
                        )
                        if name is not None:
                            if state.name is not None and state.name != name:
                                raise ValueError("stream tool name changed")
                            state.name = name
                        raw_arguments = _field(function, "arguments", None)
                        if raw_arguments is not None:
                            if not isinstance(raw_arguments, str):
                                raise ValueError("stream tool arguments must be text")
                            argument_delta = raw_arguments
                    state.argument_fragments.append(argument_delta)
                    yield ModelToolCallDelta(
                        index=index,
                        arguments_delta=argument_delta,
                        id=state.canonical_id if is_first else None,
                        name=name,
                        provider_call_id=native_id,
                    )
                native_finish = _field(choice, "finish_reason", None)
                if native_finish is not None:
                    decoded_finish = _required_text(
                        native_finish,
                        "stream finish reason",
                    )
                    if finish_reason is not None and finish_reason != decoded_finish:
                        raise ValueError("stream finish reason changed")
                    finish_reason = decoded_finish
            except asyncio.CancelledError:
                raise
            except ModelProviderError:
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ModelProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    f"{self.provider} returned a malformed stream",
                ) from error

        try:
            if finish_reason is None:
                raise ValueError("stream ended without a finish reason")
            if finish_reason == "content_filter":
                raise ModelProviderError(
                    ProviderErrorCode.CONTENT_BLOCKED,
                    f"{self.provider} blocked the response",
                )
            canonical_finish = _finish_reason(finish_reason)
            if sorted(tool_states) != list(range(len(tool_states))):
                raise ValueError("stream tool indexes must be contiguous")
            calls: list[ToolCall] = []
            for index in sorted(tool_states):
                state = tool_states[index]
                if state.provider_call_id is None or state.name is None:
                    raise ValueError("stream tool call is missing identity")
                encoded_arguments = "".join(state.argument_fragments)
                arguments_value = (
                    {} if not encoded_arguments else json.loads(encoded_arguments)
                )
                if not isinstance(arguments_value, dict):
                    raise ValueError("stream tool arguments must decode to an object")
                calls.append(
                    ToolCall(
                        id=state.canonical_id,
                        provider_call_id=state.provider_call_id,
                        name=state.name,
                        arguments=arguments_value,
                    )
                )
            if calls and canonical_finish is not FinishReason.TOOL_CALLS:
                raise ValueError("stream tool calls require tool_calls finish")
            text = "".join(text_fragments).strip() or None
            if not calls and text is None and canonical_finish is FinishReason.LENGTH:
                raise ModelProviderError(
                    ProviderErrorCode.OUTPUT_LIMIT,
                    f"{self.provider} exhausted the output token limit",
                )
            response = ModelResponse(
                finish_reason=canonical_finish,
                text=text,
                tool_calls=tuple(calls),
                usage=self._decode_priced_usage(
                    usage_value,
                    response_model=response_model,
                    service_tier=service_tier,
                    requested_at=requested_at,
                ),
                provider_id=self.provider_id,
                provider_response_id=response_id,
                provider_metadata={
                    _CONTINUATION_KEY: {"provider_id": self.provider_id},
                    "pricing_dimensions": {
                        "response_model": response_model,
                        "service_tier": service_tier,
                    },
                },
            )
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"{self.provider} returned a malformed stream",
            ) from error
        yield ModelStreamCompleted(response)

    def _request_arguments(self, request: ModelRequest) -> dict[str, object]:
        try:
            arguments: dict[str, object] = {
                "max_tokens": self._max_tokens,
                "messages": _chat_messages(request.messages, self.provider_id),
                "model": self.model,
            }
            if request.allow_parallel_tool_calls is not None:
                arguments["parallel_tool_calls"] = request.allow_parallel_tool_calls
            if request.tools:
                arguments["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": FrozenJsonObject.from_mapping(
                                tool.input_schema
                            ).to_dict(),
                        },
                    }
                    for tool in request.tools
                ]
            if request.response_schema is not None:
                arguments["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "daita_response",
                        "strict": True,
                        "schema": FrozenJsonObject.from_mapping(
                            request.response_schema
                        ).to_dict(),
                    },
                }
            return arguments
        except ModelProviderError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ModelProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "canonical request cannot be translated for compatible chat",
            ) from error

    def _decode_response(
        self,
        response: object,
        *,
        requested_at: datetime | None = None,
    ) -> ModelResponse:
        choices = _sequence(_field(response, "choices"), "response choices")
        if len(choices) != 1:
            raise ValueError("response must contain exactly one choice")
        choice = choices[0]
        message = _field(choice, "message")
        refusal = _optional_text(_field(message, "refusal", None), "refusal")
        if refusal is not None:
            raise ModelProviderError(
                ProviderErrorCode.CONTENT_BLOCKED,
                f"{self.provider} blocked the response",
            )
        content = _optional_text(_field(message, "content", None), "content")
        raw_calls = _field(message, "tool_calls", ())
        if raw_calls is None:
            raw_calls = ()
        tool_calls = _sequence(raw_calls, "message tool calls")
        calls: list[ToolCall] = []
        canonical_ids: set[str] = set()
        for item in tool_calls:
            if _required_text(_field(item, "type", "function"), "tool type") != (
                "function"
            ):
                raise ValueError("only function tool calls are supported")
            function = _field(item, "function")
            encoded_arguments = _required_text(
                _field(function, "arguments"),
                "tool arguments",
            )
            decoded_arguments = json.loads(encoded_arguments)
            if not isinstance(decoded_arguments, dict):
                raise ValueError("tool arguments must decode to an object")
            canonical_id = self._id_factory("call")
            if canonical_id in canonical_ids:
                raise ValueError("id_factory returned a duplicate call ID")
            canonical_ids.add(canonical_id)
            calls.append(
                ToolCall(
                    id=canonical_id,
                    provider_call_id=_required_text(
                        _field(item, "id"),
                        "provider tool-call id",
                    ),
                    name=_required_text(_field(function, "name"), "tool name"),
                    arguments=decoded_arguments,
                )
            )
        native_finish = _required_text(
            _field(choice, "finish_reason"),
            "finish reason",
        )
        if native_finish == "content_filter":
            raise ModelProviderError(
                ProviderErrorCode.CONTENT_BLOCKED,
                f"{self.provider} blocked the response",
            )
        finish_reason = _finish_reason(native_finish)
        if not calls and content is None and finish_reason is FinishReason.LENGTH:
            raise ModelProviderError(
                ProviderErrorCode.OUTPUT_LIMIT,
                f"{self.provider} exhausted the output token limit",
            )
        response_id = _optional_text(_field(response, "id", None), "response id")
        response_model = _optional_text(
            _field(response, "model", None),
            "response model",
        )
        service_tier = _optional_text(
            _field(response, "service_tier", None),
            "response service tier",
        )
        return ModelResponse(
            finish_reason=finish_reason,
            text=content,
            tool_calls=tuple(calls),
            usage=self._decode_priced_usage(
                _field(response, "usage", None),
                response_model=response_model,
                service_tier=service_tier,
                requested_at=requested_at or self._clock(),
            ),
            provider_id=self.provider_id,
            provider_response_id=response_id,
            provider_metadata={
                _CONTINUATION_KEY: {"provider_id": self.provider_id},
                "pricing_dimensions": {
                    "response_model": response_model,
                    "service_tier": service_tier,
                },
            },
        )

    def _decode_usage(self, value: object) -> ModelUsage:
        """Decode compatible usage; fixed providers may refine native semantics."""

        return _decode_usage(value)

    def _decode_priced_usage(
        self,
        value: object,
        *,
        response_model: str | None,
        service_tier: str | None,
        requested_at: datetime,
    ) -> ModelUsage:
        usage = self._decode_usage(value)
        if value is None or not self._pricing_schedules:
            return usage
        qualifiers = {item.name: item.value for item in self._pricing_qualifiers}
        if service_tier is not None and "service_tier" in qualifiers:
            qualifiers["service_tier"] = service_tier
        return replace(
            usage,
            cost_estimate=calculate_cost_estimate(
                self._pricing_schedules,
                provider=self.provider,
                model=response_model or self.model,
                endpoint="chat_completions",
                requested_at=requested_at,
                qualifiers=qualifiers,
                usage_values={"request_input_tokens": Decimal(usage.input_tokens)},
                quantities=_billable_quantities(usage),
            ),
        )


def _chat_messages(
    messages: tuple[CanonicalMessage, ...],
    provider_id: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    call_ids: dict[str, str] = {}
    for message in messages:
        text = "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
        if message.role is MessageRole.ASSISTANT:
            same_origin = _same_origin(message, provider_id)
            native_calls: list[dict[str, object]] = []
            for call in message.tool_calls:
                native_id = (
                    call.provider_call_id
                    if same_origin and call.provider_call_id is not None
                    else call.id
                )
                call_ids[call.id] = native_id
                native_calls.append(
                    {
                        "id": native_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": canonical_json(call.arguments),
                        },
                    }
                )
            native_message: dict[str, object] = {
                "role": "assistant",
                "content": text or None,
            }
            if native_calls:
                native_message["tool_calls"] = native_calls
            result.append(native_message)
            continue
        if message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise ValueError("tool message contains a non-tool result")
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_ids.get(block.call_id, block.call_id),
                        "content": canonical_json(
                            {"is_error": block.is_error, "output": block.output}
                        ),
                    }
                )
            continue
        if not text:
            raise ValueError("canonical text message produced no content")
        result.append({"role": message.role.value, "content": text})
    if not result:
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "canonical request produced no compatible chat messages",
        )
    return result


def _same_origin(message: CanonicalMessage, provider_id: str) -> bool:
    if message.provider_id != provider_id:
        return False
    continuation = message.provider_metadata.get(_CONTINUATION_KEY)
    if continuation is None:
        return True
    if (
        not isinstance(continuation, Mapping)
        or continuation.get("provider_id") != provider_id
    ):
        raise ModelProviderError(
            ProviderErrorCode.INVALID_REQUEST,
            "compatible continuation origin does not match canonical provider",
        )
    return True


def _decode_usage(value: object) -> ModelUsage:
    if value is None:
        return ModelUsage()
    prompt_details = _field(value, "prompt_tokens_details", None)
    completion_details = _field(value, "completion_tokens_details", None)
    input_tokens = _usage_int(
        _field(value, "prompt_tokens"),
        "prompt tokens",
    )
    output_tokens = _usage_int(
        _field(value, "completion_tokens"),
        "completion tokens",
    )
    reasoning_tokens = _usage_int(
        _field(completion_details, "reasoning_tokens", 0),
        "reasoning tokens",
    )
    cache_read_tokens = _usage_int(
        _field(prompt_details, "cached_tokens", 0),
        "cached tokens",
    )
    cache_write_tokens = _usage_int(
        _field(prompt_details, "cache_write_tokens", 0),
        "cache write tokens",
    )
    if cache_read_tokens + cache_write_tokens > input_tokens:
        raise ValueError("compatible cache token subsets exceed total prompt tokens")
    if reasoning_tokens > output_tokens:
        raise ValueError("compatible reasoning tokens exceed completion tokens")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _billable_quantities(usage: ModelUsage) -> tuple[BillableQuantity, ...]:
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0 or usage.reasoning_tokens > usage.output_tokens:
        raise ValueError("compatible usage counters are internally inconsistent")
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


def _finish_reason(value: str) -> FinishReason:
    try:
        return {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALLS,
            "length": FinishReason.LENGTH,
        }[value]
    except KeyError as error:
        raise ValueError("unknown finish reason") from error


def _normalize_error(error: Exception, provider: str) -> ModelProviderError:
    status_value = _lenient_field(error, "status_code")
    status = (
        status_value
        if isinstance(status_value, int) and not isinstance(status_value, bool)
        else None
    )
    code_value = _lenient_field(error, "code")
    code = code_value if isinstance(code_value, str) else None
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
    return ModelProviderError(
        normalized,
        f"{provider} request failed: {normalized.value}",
    )


def _validate_base_url(value: str, *, loopback_only: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("base_url must be a normalized absolute URL")
    parse_failed = False
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        parse_failed = True
    if parse_failed:
        raise ValueError("base_url must be a valid absolute URL")
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an uncredentialed HTTP(S) endpoint")
    loopback = _is_loopback_host(parsed.hostname)
    if parsed.scheme == "http" and not loopback:
        raise ValueError("base_url permits HTTP only for a loopback endpoint")
    if loopback_only and not loopback:
        raise ValueError("base_url must use a loopback endpoint")
    return value.rstrip("/")


def _is_loopback_host(value: str) -> bool:
    if value.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


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


def _lenient_field(value: object, name: str) -> object | None:
    try:
        return _field(value, name, None)
    except Exception:
        return None


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


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


def _usage_int(value: object, label: str) -> int:
    if value is None:
        return 0
    return _nonnegative_int(value, label)


__all__ = ["OpenAICompatibleProvider"]
