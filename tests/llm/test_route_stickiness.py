"""Component-owned tests split from ``test_kernel_contracts.py``."""

from __future__ import annotations

from tests.support.kernel import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    MockModelProvider,
    ModelProviderRegistration,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelSensitivity,
    RetryPolicy,
    TextBlock,
    ToolCall,
    ToolDefinition,
    pytest,
)


async def test_successful_fallback_provider_is_sticky_for_run():
    second = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall(id="one", name="lookup"),),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        ),
        provider_id="mock:stage-a-second",
    )
    # Mock providers accept normalized provider failures in their script.
    from daita.llm.errors import ModelProviderError, ProviderErrorCode

    first = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),),
        provider_id="mock:stage-a-first",
    )

    def registration(provider):
        return ModelProviderRegistration(
            provider=provider,
            profile=provider.model_profile,
            allowed_sensitivities=frozenset(ModelSensitivity),
        )

    router = ModelRouter(
        (registration(first), registration(second)),
        retry_policy=RetryPolicy(
            max_attempts_per_candidate=1, max_total_attempts=2, backoff_seconds=0
        ),
    )
    route = router.begin_run(ModelSensitivity.INTERNAL)
    request = ModelRequest(
        messages=(CanonicalMessage(role=MessageRole.USER, content=(TextBlock("q"),)),),
        tools=(
            ToolDefinition(
                name="lookup",
                description="Lookup.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
        sensitivity=ModelSensitivity.INTERNAL,
    )

    await router.generate_for_run(route, request)
    await router.generate_for_run(route, request)

    assert route.selected_provider_id == second.provider_id
    assert len(first.requests) == 1
    assert len(second.requests) == 2


async def test_selected_route_rejects_raised_sensitivity_without_new_fallback():
    from daita.llm.errors import ModelProviderError, ProviderErrorCode

    first = MockModelProvider(
        (ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),),
        provider_id="mock:raised-first",
    )
    selected = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="selected"),),
        provider_id="mock:raised-selected",
    )
    later = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="must not run"),),
        provider_id="mock:raised-later",
    )

    def registration(provider, allowed):
        return ModelProviderRegistration(
            provider=provider,
            profile=provider.model_profile,
            allowed_sensitivities=frozenset(allowed),
        )

    router = ModelRouter(
        (
            registration(first, ModelSensitivity),
            registration(
                selected,
                (ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL),
            ),
            registration(later, ModelSensitivity),
        ),
        retry_policy=RetryPolicy(
            max_attempts_per_candidate=1, max_total_attempts=2, backoff_seconds=0
        ),
    )
    route = router.begin_run(ModelSensitivity.INTERNAL)
    internal = ModelRequest(
        messages=(CanonicalMessage(role=MessageRole.USER, content=(TextBlock("q"),)),),
        sensitivity=ModelSensitivity.INTERNAL,
    )
    await router.generate_for_run(route, internal)

    confidential = ModelRequest(
        messages=internal.messages,
        sensitivity=ModelSensitivity.CONFIDENTIAL,
    )
    with pytest.raises(ModelProviderError) as captured:
        await router.generate_for_run(route, confidential)

    assert captured.value.code is ProviderErrorCode.INVALID_REQUEST
    assert route.selected_provider_id == selected.provider_id
    assert later.requests == ()
