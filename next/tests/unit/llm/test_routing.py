from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from decimal import Decimal
import traceback

import pytest

from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelRouteAttemptOutcome,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRegistry,
    ModelRouter,
)


def _profile(
    provider_id: str,
    *,
    tools: bool = True,
    structured: bool = True,
    context: int = 10_000,
    streaming: bool = False,
) -> ModelProfile:
    return ModelProfile(
        id=provider_id,
        context_window_tokens=context,
        max_output_tokens=1_000,
        supports_tools=tools,
        supports_parallel_tools=tools,
        supports_structured_output=structured,
        supports_streaming=streaming,
        input_cost_per_million_usd=Decimal("2"),
        output_cost_per_million_usd=Decimal("8"),
    )


def _registration(
    provider: MockModelProvider,
    *,
    allowed: frozenset[ModelSensitivity] = frozenset(
        {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
    ),
    tools: bool = True,
    structured: bool = True,
    context: int = 10_000,
    streaming: bool = False,
) -> ModelProviderRegistration:
    return ModelProviderRegistration(
        provider=provider,
        profile=_profile(
            provider.provider_id,
            tools=tools,
            structured=structured,
            context=context,
            streaming=streaming,
        ),
        allowed_sensitivities=allowed,
    )


def _request(
    *,
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
    messages: tuple[CanonicalMessage, ...] | None = None,
    tools: tuple[ToolDefinition, ...] = (),
    estimated_input_tokens: int = 100,
    allow_parallel_tool_calls: bool | None = None,
) -> ModelRequest:
    return ModelRequest(
        operation_id="operation-1",
        turn_id="turn-2",
        messages=messages
        or (
            CanonicalMessage(
                agent_id="agent-1",
                operation_id="operation-1",
                turn_id="turn-2",
                role=MessageRole.USER,
                content=(TextBlock("continue"),),
            ),
        ),
        tools=tools,
        sensitivity=sensitivity,
        allow_parallel_tool_calls=allow_parallel_tool_calls,
        context_selection={
            "schema_version": 1,
            "estimated_input_tokens": estimated_input_tokens,
        },
    )


def _response(text: str, *, provider_id: str | None = None) -> ModelResponse:
    return ModelResponse(
        text=text,
        finish_reason=FinishReason.STOP,
        usage=ModelUsage(input_tokens=10, output_tokens=5),
        provider_id=provider_id,
    )


class StreamingScriptProvider:
    def __init__(
        self,
        provider_id: str,
        *scripts: tuple[ModelStreamEvent | BaseException, ...],
        supports_policy: bool = True,
    ) -> None:
        self.provider_id = provider_id
        self._scripts = list(scripts)
        self.requests: list[ModelRequest] = []
        self._supports_policy = supports_policy

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return self._supports_policy

    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("streaming route must not call generate")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if not self._scripts:
            raise AssertionError("stream script exhausted")
        for item in self._scripts.pop(0):
            if isinstance(item, BaseException):
                raise item
            yield item


class PolicyScriptProvider(MockModelProvider):
    def __init__(
        self,
        script: Iterable[ModelResponse | Exception],
        *,
        provider_id: str,
        supports_policy: object,
    ) -> None:
        super().__init__(script, provider_id=provider_id)
        self._supports_policy = supports_policy

    def supports_request_policy(self, request: ModelRequest) -> bool:
        if isinstance(self._supports_policy, BaseException):
            raise self._supports_policy
        return self._supports_policy  # type: ignore[return-value]


def _stream_registration(
    provider: StreamingScriptProvider,
    *,
    allowed: frozenset[ModelSensitivity] = frozenset(
        {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
    ),
) -> ModelProviderRegistration:
    return ModelProviderRegistration(
        provider=provider,
        profile=_profile(provider.provider_id, streaming=True),
        allowed_sensitivities=allowed,
    )


async def test_primary_route_records_attempt_and_profile_owned_decimal_cost() -> None:
    provider = MockModelProvider((_response("primary"),), provider_id="openai:main")
    route = ModelRouter(
        _registration(provider),
        monotonic_clock=iter((1_000_000, 6_000_000)).__next__,
    )

    response = await route.generate(_request())

    assert route.provider_id.startswith("router:")
    assert route.profile.id == route.provider_id
    assert response.provider_id == "openai:main"
    assert response.usage.estimated_cost_usd == Decimal("0.000060")
    assert response.routing is not None
    assert response.routing.route_id == route.provider_id
    assert response.routing.selected_provider_id == "openai:main"
    assert response.routing.attempts[0].latency_ms == 5
    assert response.routing.attempts[0].outcome is ModelRouteAttemptOutcome.SUCCEEDED
    provider.assert_consumed()


async def test_transient_failure_falls_back_and_strips_foreign_native_state() -> None:
    primary = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.RATE_LIMIT_ERROR),),
        provider_id="openai:main",
    )
    fallback = MockModelProvider((_response("fallback"),), provider_id="anthropic:alt")
    assistant = CanonicalMessage(
        agent_id="agent-1",
        operation_id="operation-1",
        turn_id="turn-1",
        role=MessageRole.ASSISTANT,
        tool_calls=(
            ToolCall(
                id="canonical-call",
                provider_call_id="openai-call",
                name="fake.read",
                arguments={"key": "a"},
            ),
        ),
        provider_id="openai:main",
        provider_metadata={"openai_replay_items": [{"type": "reasoning"}]},
    )
    route = ModelRouter(
        _registration(primary),
        (_registration(fallback),),
        monotonic_clock=iter((0, 1_000_000, 2_000_000, 5_000_000)).__next__,
    )

    response = await route.generate(_request(messages=(assistant,)))

    assert response.provider_id == "anthropic:alt"
    assert response.routing is not None
    assert [item.provider_id for item in response.routing.attempts] == [
        "openai:main",
        "anthropic:alt",
    ]
    assert response.routing.attempts[0].error_code == "rate_limit_error"
    portable = fallback.requests[0].messages[0]
    assert portable.provider_id is None
    assert not portable.provider_metadata
    assert portable.tool_calls[0].id == "canonical-call"
    assert portable.tool_calls[0].provider_call_id is None


async def test_normalized_timeout_is_retryable_and_reaches_approved_fallback() -> None:
    primary = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.TIMEOUT),),
        provider_id="openai:main",
    )
    fallback = MockModelProvider((_response("fallback"),), provider_id="mock:alt")
    route = ModelRouter(_registration(primary), (_registration(fallback),))

    response = await route.generate(_request())

    assert response.provider_id == "mock:alt"
    assert response.routing is not None
    assert tuple(
        (attempt.provider_id, attempt.error_code)
        for attempt in response.routing.attempts
    ) == (
        ("openai:main", ProviderErrorCode.TIMEOUT.value),
        ("mock:alt", None),
    )


async def test_nonretryable_failure_never_invokes_fallback() -> None:
    primary = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.AUTHENTICATION_ERROR),),
        provider_id="openai:main",
    )
    fallback = MockModelProvider((_response("must not run"),), provider_id="mock:alt")
    route = ModelRouter(_registration(primary), (_registration(fallback),))

    with pytest.raises(ModelProviderError) as captured:
        await route.generate(_request())

    assert captured.value.code is ProviderErrorCode.AUTHENTICATION_ERROR
    assert captured.value.routing is not None
    assert len(captured.value.routing.attempts) == 1
    assert fallback.requests == ()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        (
            ModelProviderError(
                ProviderErrorCode.AUTHENTICATION_ERROR,
                "Authorization: Bearer sk-route-secret",
            ),
            ProviderErrorCode.AUTHENTICATION_ERROR,
        ),
        (
            RuntimeError("Authorization: Bearer sk-route-secret"),
            ProviderErrorCode.MALFORMED_RESPONSE,
        ),
    ),
)
async def test_router_terminal_errors_drop_provider_exception_diagnostics(
    failure: Exception,
    expected_code: ProviderErrorCode,
) -> None:
    primary = MockModelProvider((failure,), provider_id="openai:main")
    route = ModelRouter(_registration(primary))

    with pytest.raises(ModelProviderError) as captured:
        await route.generate(_request())

    assert captured.value.code is expected_code
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    formatted = "".join(traceback.format_exception(captured.value))
    assert "sk-route-secret" not in formatted


async def test_sensitive_fallback_requires_an_explicit_destination_grant() -> None:
    primary = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.TIMEOUT),),
        provider_id="local:primary",
    )
    fallback = MockModelProvider(
        (_response("must not leak"),), provider_id="remote:alt"
    )
    route = ModelRouter(
        _registration(
            primary,
            allowed=frozenset({ModelSensitivity.RESTRICTED}),
        ),
        (
            _registration(
                fallback,
                allowed=frozenset({ModelSensitivity.PUBLIC}),
            ),
        ),
    )

    with pytest.raises(ModelProviderError) as captured:
        await route.generate(_request(sensitivity=ModelSensitivity.RESTRICTED))

    assert captured.value.code is ProviderErrorCode.TIMEOUT
    assert fallback.requests == ()
    assert captured.value.routing is not None
    assert tuple(item.provider_id for item in captured.value.routing.attempts) == (
        "local:primary",
    )


async def test_capability_and_context_requirements_are_checked_before_io() -> None:
    unsuitable = MockModelProvider((_response("no"),), provider_id="mock:small")
    suitable = MockModelProvider((_response("yes"),), provider_id="mock:large")
    route = ModelRouter(
        _registration(unsuitable, tools=False, context=1_500),
        (_registration(suitable, context=10_000),),
    )
    tool = ToolDefinition(
        name="fake.read",
        description="Read.",
        input_schema={"type": "object"},
    )

    response = await route.generate(
        _request(tools=(tool,), estimated_input_tokens=2_000)
    )

    assert response.provider_id == "mock:large"
    assert unsuitable.requests == ()
    assert len(suitable.requests) == 1


async def test_request_policy_skips_unsupported_primary_for_supporting_fallback() -> (
    None
):
    primary = PolicyScriptProvider(
        (_response("must not run"),),
        provider_id="mock:unsupported",
        supports_policy=False,
    )
    fallback = PolicyScriptProvider(
        (_response("supported"),),
        provider_id="mock:supported",
        supports_policy=True,
    )
    route = ModelRouter(_registration(primary), (_registration(fallback),))
    request = _request(allow_parallel_tool_calls=False)

    assert route.supports_request_policy(request) is True
    response = await route.generate(request)

    assert response.provider_id == "mock:supported"
    assert primary.requests == ()
    assert fallback.requests == (request,)


@pytest.mark.parametrize(
    "unsupported_result",
    (False, None, 1, RuntimeError("policy check failed")),
)
async def test_all_request_policy_ineligible_routes_fail_before_io(
    unsupported_result: object,
) -> None:
    first = PolicyScriptProvider(
        (_response("must not run"),),
        provider_id="mock:first",
        supports_policy=unsupported_result,
    )
    second = PolicyScriptProvider(
        (_response("must not run"),),
        provider_id="mock:second",
        supports_policy=False,
    )
    route = ModelRouter(_registration(first), (_registration(second),))
    request = _request(allow_parallel_tool_calls=False)

    assert route.supports_request_policy(request) is False
    with pytest.raises(ModelProviderError) as captured:
        await route.generate(request)

    assert captured.value.code is ProviderErrorCode.INVALID_REQUEST
    assert captured.value.routing is not None
    assert captured.value.routing.attempts == ()
    assert first.requests == ()
    assert second.requests == ()


async def test_missing_context_estimate_fails_closed_before_provider_io() -> None:
    provider = MockModelProvider((_response("must not run"),), provider_id="mock:main")
    route = ModelRouter(_registration(provider))
    request = replace(_request(), context_selection={})

    with pytest.raises(ModelProviderError) as captured:
        await route.generate(request)

    assert captured.value.code is ProviderErrorCode.INVALID_REQUEST
    assert captured.value.routing is not None
    assert captured.value.routing.attempts == ()
    assert provider.requests == ()


async def test_cancellation_propagates_without_retry_or_fallback() -> None:
    class CancellingProvider:
        provider_id = "mock:main"

        def supports_request_policy(self, request: ModelRequest) -> bool:
            return True

        async def generate(self, request: ModelRequest) -> ModelResponse:
            raise asyncio.CancelledError

    primary = CancellingProvider()
    fallback = MockModelProvider((_response("no"),), provider_id="mock:alt")
    primary_registration = ModelProviderRegistration(
        provider=primary,
        profile=_profile(primary.provider_id),
        allowed_sensitivities=frozenset({ModelSensitivity.INTERNAL}),
    )
    route = ModelRouter(primary_registration, (_registration(fallback),))

    with pytest.raises(asyncio.CancelledError):
        await route.generate(_request())

    assert fallback.requests == ()


async def test_stream_falls_back_before_first_delta_and_records_route() -> None:
    primary = StreamingScriptProvider(
        "openai:main",
        (ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),),
    )
    fallback = StreamingScriptProvider(
        "anthropic:alt",
        (
            ModelTextDelta("fallback"),
            ModelStreamCompleted(_response("fallback")),
        ),
    )
    route = ModelRouter(
        _stream_registration(primary),
        (_stream_registration(fallback),),
        monotonic_clock=iter((0, 1_000_000, 2_000_000, 5_000_000)).__next__,
    )

    events = [event async for event in route.stream(_request())]

    assert events[0] == ModelTextDelta("fallback")
    terminal = events[1]
    assert isinstance(terminal, ModelStreamCompleted)
    assert terminal.response.provider_id == "anthropic:alt"
    assert terminal.response.usage.estimated_cost_usd == Decimal("0.000060")
    assert terminal.response.routing is not None
    assert tuple(
        (attempt.provider_id, attempt.outcome)
        for attempt in terminal.response.routing.attempts
    ) == (
        ("openai:main", ModelRouteAttemptOutcome.FAILED),
        ("anthropic:alt", ModelRouteAttemptOutcome.SUCCEEDED),
    )


async def test_stream_never_falls_back_after_a_delta_has_escaped() -> None:
    primary = StreamingScriptProvider(
        "openai:main",
        (
            ModelTextDelta("partial"),
            ModelProviderError(ProviderErrorCode.TIMEOUT),
        ),
    )
    fallback = StreamingScriptProvider(
        "anthropic:alt",
        (ModelStreamCompleted(_response("must not run")),),
    )
    route = ModelRouter(
        _stream_registration(primary),
        (_stream_registration(fallback),),
    )
    observed: list[ModelStreamEvent] = []

    with pytest.raises(ModelProviderError) as captured:
        async for event in route.stream(_request()):
            observed.append(event)

    assert observed == [ModelTextDelta("partial")]
    assert captured.value.code is ProviderErrorCode.TIMEOUT
    assert captured.value.routing is not None
    assert tuple(
        attempt.provider_id for attempt in captured.value.routing.attempts
    ) == ("openai:main",)
    assert fallback.requests == []


async def test_stream_cancellation_propagates_without_fallback() -> None:
    primary = StreamingScriptProvider(
        "openai:main",
        (asyncio.CancelledError(),),
    )
    fallback = StreamingScriptProvider(
        "anthropic:alt",
        (ModelStreamCompleted(_response("must not run")),),
    )
    route = ModelRouter(
        _stream_registration(primary),
        (_stream_registration(fallback),),
    )

    with pytest.raises(asyncio.CancelledError):
        _ = [event async for event in route.stream(_request())]

    assert fallback.requests == []


async def test_streaming_capability_is_checked_before_provider_io() -> None:
    nonstreaming = MockModelProvider((_response("no"),), provider_id="mock:no-stream")
    fallback = StreamingScriptProvider(
        "mock:stream",
        (ModelStreamCompleted(_response("yes")),),
    )
    route = ModelRouter(
        _registration(nonstreaming),
        (_stream_registration(fallback),),
    )

    events = [event async for event in route.stream(_request())]

    assert nonstreaming.requests == ()
    assert len(fallback.requests) == 1
    assert isinstance(events[-1], ModelStreamCompleted)


async def test_stream_request_policy_uses_supporting_fallback_before_io() -> None:
    primary = StreamingScriptProvider(
        "mock:unsupported",
        (ModelStreamCompleted(_response("must not run")),),
        supports_policy=False,
    )
    fallback = StreamingScriptProvider(
        "mock:supported",
        (ModelStreamCompleted(_response("supported")),),
    )
    route = ModelRouter(
        _stream_registration(primary),
        (_stream_registration(fallback),),
    )

    events = [
        event async for event in route.stream(_request(allow_parallel_tool_calls=False))
    ]

    assert primary.requests == []
    assert len(fallback.requests) == 1
    assert isinstance(events[-1], ModelStreamCompleted)


async def test_stream_all_request_policy_ineligible_fails_before_io() -> None:
    provider = StreamingScriptProvider(
        "mock:unsupported",
        (ModelStreamCompleted(_response("must not run")),),
        supports_policy=False,
    )
    route = ModelRouter(_stream_registration(provider))

    with pytest.raises(ModelProviderError) as captured:
        _ = [
            event
            async for event in route.stream(_request(allow_parallel_tool_calls=False))
        ]

    assert captured.value.code is ProviderErrorCode.INVALID_REQUEST
    assert captured.value.routing is not None
    assert captured.value.routing.attempts == ()
    assert provider.requests == []


def test_route_identity_binds_policy_profile_order_and_retry_configuration() -> None:
    first = MockModelProvider((), provider_id="mock:first")
    second = MockModelProvider((), provider_id="mock:second")
    primary = _registration(first)
    fallback = _registration(second)

    route = ModelRouter(primary, (fallback,))
    changed_order = ModelRouter(fallback, (primary,))
    changed_retry = ModelRouter(primary, (fallback,), max_attempts_per_provider=2)
    changed_policy = ModelRouter(
        _registration(
            first,
            allowed=frozenset({ModelSensitivity.PUBLIC}),
        ),
        (fallback,),
    )

    assert (
        len(
            {
                route.provider_id,
                changed_order.provider_id,
                changed_retry.provider_id,
                changed_policy.provider_id,
            }
        )
        == 4
    )


def test_registry_is_explicit_immutable_and_rejects_duplicate_ids() -> None:
    provider = MockModelProvider((), provider_id="mock:one")
    registration = _registration(provider)
    registry = ModelRegistry((registration,))

    assert registry.get("mock:one") is registration
    assert registry.registrations == (registration,)
    with pytest.raises(KeyError, match="unknown model provider"):
        registry.get("mock:missing")
    with pytest.raises(ValueError, match="unique"):
        ModelRegistry((registration, registration))


def test_provider_registration_requires_exact_profile_and_explicit_policy() -> None:
    provider = MockModelProvider((), provider_id="mock:one")

    with pytest.raises(ValueError, match="match provider_id"):
        ModelProviderRegistration(
            provider=provider,
            profile=_profile("mock:other"),
            allowed_sensitivities=frozenset({ModelSensitivity.INTERNAL}),
        )
    with pytest.raises(ValueError, match="explicit sensitivity"):
        ModelProviderRegistration(
            provider=provider,
            profile=_profile("mock:one"),
            allowed_sensitivities=frozenset(),
        )
