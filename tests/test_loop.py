import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop import (
    AgentLoop,
    InMemoryTranscriptStore,
    LoopExitKind,
    LoopLimits,
    RunInput,
)

NOW = datetime(2026, 7, 21, tzinfo=timezone.utc)


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
