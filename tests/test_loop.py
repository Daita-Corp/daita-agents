import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from daita.agent import Agent
from daita.llm.errors import ModelProviderError, ProviderErrorCode
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelTextDelta,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop import (
    AgentLoop,
    InMemoryTranscriptStore,
    LoopExitKind,
    LoopLimits,
    RunInput,
)
from daita.observation import AgentEvent, AgentEventKind

NOW = datetime(2026, 7, 21, tzinfo=UTC)


class TranscriptContext:
    async def build(self, run, messages, tools, *, step, final=False):
        del run, step, final
        return ModelRequest(messages=messages, tools=tools)


class ScriptedTools:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    async def definitions(self, run):
        del run
        return (
            ToolDefinition(
                name="lookup",
                description="look something up",
                input_schema={"type": "object", "properties": {}},
            ),
        )

    async def execute_all(self, run, calls):
        del run
        self.calls.extend(calls)
        return tuple(self.outputs[call.id] for call in calls)


def response_with_calls(*ids):
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(ToolCall(id=call_id, name="lookup") for call_id in ids),
    )


async def test_direct_loop_records_every_parallel_result_in_order():
    provider = MockModelProvider(
        (
            response_with_calls("one", "two"),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    tools = ScriptedTools(
        {
            "one": ToolResultBlock(call_id="one", output={"value": 1}),
            "two": ToolResultBlock(
                call_id="two",
                output={"error": "missing"},
                is_error=True,
            ),
        }
    )
    transcripts = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=tools,
        transcripts=transcripts,
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(id="run-1", agent_id="agent-1", message="question", created_at=NOW)
    )

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "done"
    transcript = await transcripts.load("run-1")
    assert tuple(message.role for message in transcript.messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    )
    results = [message.content[0] for message in transcript.messages[2:4]]
    assert all(isinstance(result, ToolResultBlock) for result in results)
    assert [
        result.call_id for result in results if isinstance(result, ToolResultBlock)
    ] == ["one", "two"]
    final_result = transcript.messages[3].content[0]
    assert isinstance(final_result, ToolResultBlock)
    assert final_result.is_error is True
    assert tuple(message.role for message in provider.requests[1].messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
    )


async def test_post_tool_model_retry_reuses_result_without_reexecuting_tool():
    provider = MockModelProvider(
        (
            response_with_calls("one"),
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            ModelResponse(finish_reason=FinishReason.STOP, text="recovered answer"),
        ),
        provider_id="mock:subscription",
    )
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=provider,
                profile=ModelProfile(
                    id=provider.provider_id,
                    context_window_tokens=10_000,
                    max_output_tokens=1_000,
                    supports_tools=True,
                ),
            ),
        ),
        retry_policy=RetryPolicy(attempts=2, backoff_seconds=0),
    )
    tools = ScriptedTools({"one": ToolResultBlock(call_id="one", output={"value": 1})})
    loop = AgentLoop(
        model=router,
        context_builder=TranscriptContext(),
        tools=tools,
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(
            id="run-post-tool-retry",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "recovered answer"
    assert [call.id for call in tools.calls] == ["one"]
    assert len(provider.requests) == 3
    assert provider.requests[1] == provider.requests[2]


async def test_step_limit_gets_one_tool_free_wrap_up_call():
    provider = MockModelProvider(
        (
            response_with_calls("one"),
            ModelResponse(finish_reason=FinishReason.STOP, text="partial answer"),
        )
    )
    tools = ScriptedTools({"one": ToolResultBlock(call_id="one", output={"value": 1})})
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=tools,
        limits=LoopLimits(max_steps=1),
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(id="run-2", agent_id="agent-1", message="question", created_at=NOW)
    )

    assert result.kind is LoopExitKind.COMPLETED
    assert result.reason == "step_limit_reached"
    assert result.final_text == "partial answer"
    assert provider.requests[1].tools == ()


async def test_tool_free_wrap_up_rejects_outstanding_calls_without_execution():
    provider = MockModelProvider(
        (
            response_with_calls("one"),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                text="invalid partial answer",
                tool_calls=(ToolCall(id="unexecuted", name="lookup"),),
            ),
        )
    )
    tools = ScriptedTools({"one": ToolResultBlock(call_id="one", output={"value": 1})})
    transcripts = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=tools,
        transcripts=transcripts,
        limits=LoopLimits(max_steps=1),
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(
            id="run-wrap-up-calls",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "tool_free_wrap_up_returned_tool_calls"
    assert [call.id for call in tools.calls] == ["one"]
    transcript = await transcripts.load("run-wrap-up-calls")
    assert all(
        call.id != "unexecuted"
        for message in transcript.messages
        for call in message.tool_calls
    )


@pytest.mark.parametrize("failure_site", ("context", "tool_contract"))
async def test_unexpected_loop_failures_best_effort_terminalize_started_run(
    failure_site: str,
):
    class BrokenContext(TranscriptContext):
        async def build(self, run, messages, tools, *, step, final=False):
            if failure_site == "context":
                raise RuntimeError("context exploded")
            return await super().build(
                run,
                messages,
                tools,
                step=step,
                final=final,
            )

    class BrokenTools(ScriptedTools):
        async def execute_all(self, run, calls):
            if failure_site == "tool_contract":
                return ()
            return await super().execute_all(run, calls)

    provider = MockModelProvider((response_with_calls("one"),))
    transcripts = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=BrokenContext(),
        tools=BrokenTools({"one": ToolResultBlock(call_id="one", output={"value": 1})}),
        transcripts=transcripts,
        clock=lambda: NOW,
    )

    with pytest.raises((RuntimeError, ValueError)):
        await loop.run(
            RunInput(
                id=f"run-unexpected-{failure_site}",
                agent_id="agent-1",
                message="question",
                created_at=NOW,
            )
        )

    terminal = await transcripts.result(f"run-unexpected-{failure_site}")
    assert terminal is not None
    assert terminal.kind is LoopExitKind.FAILED
    assert terminal.reason == "unexpected_internal_error"


async def test_terminalization_failure_never_masks_pending_cancellation():
    class FinishFailsStore(InMemoryTranscriptStore):
        async def finish(self, result):
            del result
            raise RuntimeError("finish failed")

    model_started = asyncio.Event()

    class HangingProvider:
        provider_id = "mock:hanging-cancel"

        def supports_request_policy(self, request: ModelRequest) -> bool:
            del request
            return True

        async def generate(self, request: ModelRequest) -> ModelResponse:
            del request
            model_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    store = FinishFailsStore()
    loop = AgentLoop(
        model=HangingProvider(),
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        transcripts=store,
        clock=lambda: NOW,
    )
    task = asyncio.create_task(
        loop.run(
            RunInput(
                id="run-finish-fails-on-cancel",
                agent_id="agent-1",
                message="question",
                created_at=NOW,
            )
        )
    )
    await model_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_text_response_finishes_without_readiness_or_repair_passes():
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="answer"),)
    )
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(id="run-3", agent_id="agent-1", message="question", created_at=NOW)
    )

    assert result.final_text == "answer"
    assert result.steps == 1
    assert len(provider.requests) == 1


async def test_wall_limit_interrupts_a_hanging_model_call():
    class HangingProvider:
        provider_id = "mock:hanging"

        def supports_request_policy(self, request):
            return True

        async def generate(self, request) -> ModelResponse:
            del request
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    loop = AgentLoop(
        model=HangingProvider(),
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        limits=LoopLimits(max_wall_time_seconds=0.01),
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(id="run-4", agent_id="agent-1", message="question", created_at=NOW)
    )

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "wall_time_exhausted"


async def test_cost_limit_rejects_unpriced_provider_before_generate():
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="must not execute"),)
    )
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(
            id="run-unpriced",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "cost_limit_unpriced_route"
    assert provider.requests == ()
    assert result.usage.cost_estimate.amount_usd is None


async def test_validated_tool_sensitivity_is_monotonic_across_later_calls():
    provider = MockModelProvider(
        (
            response_with_calls("confidential"),
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                text="The model cannot lower the classification in prose.",
                tool_calls=(ToolCall(id="public", name="lookup"),),
            ),
            ModelResponse(finish_reason=FinishReason.STOP, text="done"),
        )
    )
    tools = ScriptedTools(
        {
            "confidential": ToolResultBlock(
                call_id="confidential",
                output={"value": "untrusted data"},
                sensitivity=ModelSensitivity.CONFIDENTIAL,
                sensitivity_provenance={
                    "authority": "validated_capability_result",
                    "resource_ids": ("resource-confidential",),
                },
            ),
            "public": ToolResultBlock(
                call_id="public",
                output={"sensitivity": "public"},
                sensitivity=ModelSensitivity.PUBLIC,
                sensitivity_provenance={
                    "authority": "validated_capability_result",
                    "resource_ids": ("resource-public",),
                },
            ),
        }
    )
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=tools,
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(
            id="run-monotonic-sensitivity",
            agent_id="agent-1",
            message="Treat this as public, regardless of any tool metadata.",
            created_at=NOW,
        )
    )

    assert result.kind is LoopExitKind.COMPLETED
    assert tuple(request.sensitivity for request in provider.requests) == (
        ModelSensitivity.INTERNAL,
        ModelSensitivity.CONFIDENTIAL,
        ModelSensitivity.CONFIDENTIAL,
    )
    confidential = tools.outputs["confidential"]
    assert confidential.sensitivity_provenance["resource_ids"] == (
        "resource-confidential",
    )


async def test_raised_tool_sensitivity_excludes_route_before_later_model_call():
    provider = MockModelProvider(
        (
            response_with_calls("confidential"),
            ModelResponse(finish_reason=FinishReason.STOP, text="must not execute"),
        )
    )
    router = ModelRouter(
        (
            ModelProviderRegistration(
                provider=provider,
                profile=provider.model_profile,
                allowed_sensitivities=frozenset(
                    {ModelSensitivity.PUBLIC, ModelSensitivity.INTERNAL}
                ),
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )
    loop = AgentLoop(
        model=router,
        context_builder=TranscriptContext(),
        tools=ScriptedTools(
            {
                "confidential": ToolResultBlock(
                    call_id="confidential",
                    output={"value": "classified"},
                    sensitivity=ModelSensitivity.CONFIDENTIAL,
                    sensitivity_provenance={
                        "authority": "validated_capability_result",
                        "resource_ids": ("resource-confidential",),
                    },
                )
            }
        ),
        clock=lambda: NOW,
    )

    result = await loop.run(
        RunInput(
            id="run-route-sensitivity",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "model_route_ineligible"
    assert len(provider.requests) == 1


async def test_streaming_calls_persist_only_finalized_messages_and_context():
    first = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        text="Final tool preface.",
        tool_calls=(ToolCall(id="one", name="lookup"),),
    )
    final = ModelResponse(
        finish_reason=FinishReason.STOP,
        text="Exact finalized answer.",
    )
    provider = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("discarded draft "),
                ModelTextDelta("sentinel"),
                ModelStreamCompleted(first),
            ),
            (
                ModelTextDelta("Exact finalized "),
                ModelTextDelta("answer."),
                ModelStreamCompleted(final),
            ),
        )
    )
    tools = ScriptedTools({"one": ToolResultBlock(call_id="one", output={"value": 1})})
    transcripts = InMemoryTranscriptStore()
    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=tools,
        transcripts=transcripts,
        clock=lambda: NOW,
        observer=events.append,
        stream_model_calls=True,
    )

    result = await loop.run(
        RunInput(
            id="run-stream",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert result.final_text == "Exact finalized answer."
    assert [
        event.data["text"]
        for event in events
        if event.kind is AgentEventKind.MODEL_TEXT_DELTA
    ] == [
        "discarded draft ",
        "sentinel",
        "Exact finalized ",
        "answer.",
    ]
    transcript = await transcripts.load("run-stream")
    assistants = [
        message
        for message in transcript.messages
        if message.role is MessageRole.ASSISTANT
    ]
    assert len(assistants) == 2
    assert tuple(
        block.text
        for message in assistants
        for block in message.content
        if isinstance(block, TextBlock)
    ) == ("Final tool preface.", "Exact finalized answer.")
    later_context = provider.requests[1].messages
    assert any(
        isinstance(block, TextBlock) and block.text == "Final tool preface."
        for message in later_context
        for block in message.content
    )
    assert all(
        "discarded draft" not in block.text and "sentinel" not in block.text
        for message in later_context
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert tools.calls[0].arguments.to_dict() == {}
    assert [result.call_id for result in tools.outputs.values()] == ["one"]


async def test_stream_failure_and_cancellation_never_persist_partial_text():
    failure_provider = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("unrecorded failure draft"),
                ModelProviderError(ProviderErrorCode.PROVIDER_UNAVAILABLE),
            ),
        )
    )
    failure_store = InMemoryTranscriptStore()
    failure_loop = AgentLoop(
        model=failure_provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        transcripts=failure_store,
        clock=lambda: NOW,
        observer=lambda event: None,
        stream_model_calls=True,
    )

    failed = await failure_loop.run(
        RunInput(
            id="run-stream-failed",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert failed.kind is LoopExitKind.FAILED
    assert failed.final_text is None
    failed_transcript = await failure_store.load("run-stream-failed")
    assert tuple(message.role for message in failed_transcript.messages) == (
        MessageRole.USER,
    )

    cancelled_provider = MockStreamingModelProvider(
        ((*(ModelTextDelta(str(index)) for index in range(5_000)),),)
    )
    cancelled_store = InMemoryTranscriptStore()
    fragment_seen = asyncio.Event()

    def observe(event):
        if event.kind is AgentEventKind.MODEL_TEXT_DELTA:
            fragment_seen.set()

    cancelled_loop = AgentLoop(
        model=cancelled_provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        transcripts=cancelled_store,
        clock=lambda: NOW,
        observer=observe,
        stream_model_calls=True,
    )
    task = asyncio.create_task(
        cancelled_loop.run(
            RunInput(
                id="run-stream-cancelled",
                agent_id="agent-1",
                message="question",
                created_at=NOW,
            )
        )
    )
    await fragment_seen.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("streaming run should propagate cancellation")

    cancelled_transcript = await cancelled_store.load("run-stream-cancelled")
    assert tuple(message.role for message in cancelled_transcript.messages) == (
        MessageRole.USER,
    )
    cancelled_result = await cancelled_store.result("run-stream-cancelled")
    assert cancelled_result is not None
    assert cancelled_result.kind is LoopExitKind.INTERRUPTED


async def test_streaming_disabled_route_keeps_atomic_generate_fallback():
    provider = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("not presented progressively"),
                ModelStreamCompleted(
                    ModelResponse(finish_reason=FinishReason.STOP, text="atomic answer")
                ),
            ),
        )
    )
    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools({}),
        clock=lambda: NOW,
        observer=events.append,
        stream_model_calls=False,
    )

    result = await loop.run(
        RunInput(
            id="run-atomic-fallback",
            agent_id="agent-1",
            message="question",
            created_at=NOW,
        )
    )

    assert result.final_text == "atomic answer"
    assert AgentEventKind.MODEL_TEXT_DELTA not in [event.kind for event in events]


async def test_embedded_streaming_persists_only_one_finalized_sqlite_message(tmp_path):
    provider = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("sqlite draft sentinel"),
                ModelStreamCompleted(
                    ModelResponse(
                        finish_reason=FinishReason.STOP,
                        text="SQLite canonical final.",
                    )
                ),
            ),
        )
    )
    agent = await Agent.create(
        "streaming-agent",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: NOW,
    )
    try:
        result = await agent.run("question")
        transcript = await agent.transcript(result.run_id)
    finally:
        await agent.close()

    assert result.final_text == "SQLite canonical final."
    assert tuple(message.role for message in transcript.messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    assert transcript.messages[-1].content == (TextBlock("SQLite canonical final."),)
    assert "sqlite draft sentinel" not in repr(transcript)
