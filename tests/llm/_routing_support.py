"""Shared helpers extracted from ``test_routing.py``."""

from __future__ import annotations

from decimal import Decimal

from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import (
    ModelProviderRegistration,
    ModelRouter,
    RetryPolicy,
)
from daita.loop import AgentLoop, InMemoryTranscriptStore
from tests.support.loop import NOW, ScriptedTools, TranscriptContext


def request():
    return ModelRequest(
        messages=(
            CanonicalMessage(role=MessageRole.USER, content=(TextBlock("hello"),)),
        )
    )


def registration(provider, *, streaming=False, tools=False, allowed_sensitivities=None):
    return ModelProviderRegistration(
        provider=provider,
        profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=10_000,
            max_output_tokens=1_000,
            supports_streaming=streaming,
            supports_tools=tools,
        ),
        allowed_sensitivities=(
            frozenset({ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL})
            if allowed_sensitivities is None
            else allowed_sensitivities
        ),
    )


def measured_usage(input_tokens, output_tokens, cost):
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_estimate=CostEstimate.complete(Decimal(cost)),
    )


def scripted_attempts(stream, outcomes):
    """Script one item per physical attempt, as a stream or a direct response."""

    if not stream:
        return MockModelProvider(outcomes, complete_pricing=True)
    return MockStreamingModelProvider(
        tuple(
            item if isinstance(item, Exception) else (ModelStreamCompleted(item),)
            for item in outcomes
        ),
        complete_pricing=True,
    )


def routed_loop(provider, stream, *, observer=None):
    async def no_backoff(delay):
        del delay

    router = ModelRouter(
        (registration(provider, streaming=stream, tools=True),),
        retry_policy=RetryPolicy(max_attempts_per_candidate=2, max_total_attempts=3),
        sleep=no_backoff,
    )
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=router,
        context_builder=TranscriptContext(),
        tools=ScriptedTools(
            {"lookup-1": ToolResultBlock(call_id="lookup-1", output={"value": 1})}
        ),
        transcripts=store,
        observer=observer,
        clock=lambda: NOW,
        stream_model_calls=stream,
    )
    return router, store, loop


def lookup_response(usage):
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id="lookup-1", name="lookup", arguments={}),),
        usage=usage,
    )
