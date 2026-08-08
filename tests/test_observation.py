import asyncio
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from daita._json import FrozenJsonObject
from daita.agent import Agent
from daita.llm.errors import (
    ContextWindowExceeded,
    ModelProviderError,
    ProviderErrorCode,
)
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.pricing import (
    CostBasis,
    CostEstimate,
    provider_reported_cost_estimate,
)
from daita.llm.providers.mock import MockModelProvider, MockStreamingModelProvider
from daita.loop import (
    AgentLoop,
    InMemoryTranscriptStore,
    LoopExitKind,
    LoopLimits,
    RunInput,
)
from daita.observation import AgentEvent, AgentEventKind

NOW = datetime(2026, 7, 21, tzinfo=timezone.utc)


class TranscriptContext:
    async def build(self, run, messages, tools, *, step, final=False):
        del run, step, final
        return ModelRequest(messages=messages, tools=tools)


class OverflowContext:
    async def build(self, run, messages, tools, *, step, final=False):
        del run, messages, tools, step, final
        raise ContextWindowExceeded


class ScriptedTools:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}

    async def definitions(self, run):
        del run
        return (
            ToolDefinition(
                name="lookup",
                description="read data",
                input_schema={"type": "object", "properties": {}},
            ),
        )

    async def execute_all(self, run, calls):
        del run
        return tuple(self.outputs[call.id] for call in calls)


class OrderingStore(InMemoryTranscriptStore):
    def __init__(self):
        super().__init__()
        self.actions = []

    async def start(self, run):
        transcript = await super().start(run)
        self.actions.append("persisted:start")
        return transcript

    async def append(self, run_id, message):
        await super().append(run_id, message)
        self.actions.append(f"persisted:{message.role.value}")

    async def finish(self, result):
        await super().finish(result)
        self.actions.append(f"persisted:finish:{result.kind.value}")


def _run(run_id="run-observed", conversation_id="conversation-observed"):
    return RunInput(
        id=run_id,
        agent_id="agent-observed",
        message="raw user sentinel",
        created_at=NOW,
        conversation_id=conversation_id,
    )


def _stop(text="raw assistant sentinel", *, usage=None):
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        text=text,
        usage=usage or ModelUsage(),
        provider_id="mock:observed",
        provider_metadata={"raw_metadata": "provider secret sentinel"},
    )


def _call():
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="lookup",
                arguments={"query": "raw argument sentinel"},
            ),
        ),
        usage=ModelUsage(input_tokens=3, output_tokens=2),
        provider_id="mock:observed",
    )


def _kinds(events):
    return tuple(event.kind for event in events)


def test_event_contract_is_immutable_bounded_and_deeply_frozen():
    assert tuple(AgentEventKind) == (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_TEXT_DELTA,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.APPROVAL_REQUESTED,
        AgentEventKind.APPROVAL_DECIDED,
        AgentEventKind.TOOL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
    )
    event = AgentEvent(
        kind=AgentEventKind.RUN_COMPLETED,
        occurred_at=NOW,
        run_id="run-1",
        conversation_id="conversation-1",
        data=FrozenJsonObject.from_mapping(
            {"duration_ms": 0, "nested": {"values": ["safe"]}}
        ),
    )
    assert isinstance(event.data["nested"], FrozenJsonObject)
    assert event.data.to_dict() == {
        "duration_ms": 0,
        "nested": {"values": ["safe"]},
    }
    with pytest.raises(FrozenInstanceError):
        setattr(event, "run_id", "changed")

    invalid: tuple[Callable[[], AgentEvent], ...] = (
        lambda: replace(event, occurred_at=datetime(2026, 7, 21)),
        lambda: replace(event, run_id=""),
        lambda: replace(event, conversation_id="x" * 257),
        lambda: replace(
            event,
            data=FrozenJsonObject.from_mapping({"value": "x" * 1_025}),
        ),
        lambda: replace(
            event,
            data=FrozenJsonObject.from_mapping({"duration_ms": -1}),
        ),
        lambda: replace(
            event,
            data=FrozenJsonObject.from_mapping({"duration_ms": 1.5}),
        ),
        lambda: replace(
            event,
            data=FrozenJsonObject.from_mapping({"duration_ms": []}),
        ),
        lambda: replace(
            event,
            data=FrozenJsonObject.from_mapping({"provider_id": 1}),
        ),
        lambda: replace(
            event,
            data=FrozenJsonObject.from_mapping({"model_call_index": 0}),
        ),
    )
    for construct in invalid:
        with pytest.raises(ValueError):
            construct()


async def test_text_run_order_payloads_and_durable_boundaries():
    usage = ModelUsage(
        input_tokens=7,
        output_tokens=5,
        reasoning_tokens=2,
        cache_read_tokens=3,
        cache_write_tokens=1,
        cost_estimate=CostEstimate.complete(
            Decimal("0.0100"),
            basis=CostBasis.PUBLIC_LIST,
            rate_schedule_id="test-schedule-2026-07",
        ),
    )
    store = OrderingStore()
    events: list[AgentEvent] = []

    def observe(event: AgentEvent) -> None:
        events.append(event)
        store.actions.append(f"event:{event.kind.value}")

    loop = AgentLoop(
        model=MockModelProvider((_stop(usage=usage),)),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        transcripts=store,
        clock=lambda: NOW,
        observer=observe,
    )
    result = await loop.run(_run())

    assert result.kind is LoopExitKind.COMPLETED
    assert _kinds(events) == (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
    )
    assert store.actions == [
        "persisted:start",
        "event:run.started",
        "persisted:user",
        "persisted:assistant",
        "event:model.completed",
        "persisted:finish:completed",
        "event:run.completed",
    ]
    assert all(event.run_id == "run-observed" for event in events)
    assert all(event.conversation_id == "conversation-observed" for event in events)
    assert events[0].data.to_dict() == {"agent_id": "agent-observed"}
    model_data = events[1].data.to_dict()
    assert set(model_data) == {
        "provider_id",
        "model_call_index",
        "has_text",
        "has_tool_calls",
        "duration_ms",
        "input_tokens",
        "context_input_tokens",
        "output_tokens",
    }
    assert model_data["provider_id"] == "mock:observed"
    assert model_data["model_call_index"] == 1
    assert model_data["has_text"] is True
    assert model_data["has_tool_calls"] is False
    assert isinstance(model_data["duration_ms"], int)
    assert model_data["duration_ms"] >= 0
    assert model_data["input_tokens"] == 7
    assert model_data["context_input_tokens"] == 7
    assert model_data["output_tokens"] == 5
    completed_data = events[2].data.to_dict()
    assert completed_data == {
        "exit_kind": "completed",
        "reason": "completed",
        "steps": 1,
        "duration_ms": completed_data["duration_ms"],
        "input_tokens": 7,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "cache_read_tokens": 3,
        "cache_write_tokens": 1,
        "total_tokens": 12,
        "cost_status": "complete",
        "cost_amount_usd": "0.01",
        "cost_basis": "public_list",
        "cost_rate_schedule_id": "test-schedule-2026-07",
        "cost_code": None,
        "cost_display": "$0.01 estimated at public list rates",
    }
    assert isinstance(completed_data["duration_ms"], int)
    assert completed_data["duration_ms"] >= 0


@pytest.mark.parametrize(
    ("estimate", "expected"),
    (
        (
            provider_reported_cost_estimate(
                Decimal("0"),
                currency="USD",
                unit="request",
            ),
            {
                "cost_status": "complete",
                "cost_amount_usd": "0",
                "cost_basis": "provider_reported",
                "cost_rate_schedule_id": None,
                "cost_code": None,
                "cost_display": ("provider API charge $0; local compute not estimated"),
            },
        ),
        (
            CostEstimate.partial(
                Decimal("0.12"),
                code="unpriced_attempt",
                basis=CostBasis.PUBLIC_LIST,
                rate_schedule_id="public:test",
            ),
            {
                "cost_status": "partial",
                "cost_amount_usd": "0.12",
                "cost_basis": "public_list",
                "cost_rate_schedule_id": "public:test",
                "cost_code": "unpriced_attempt",
                "cost_display": "≥$0.12 estimated; some attempts were unpriced",
            },
        ),
        (
            CostEstimate.unavailable("pricing_schedule_unavailable"),
            {
                "cost_status": "unavailable",
                "cost_amount_usd": None,
                "cost_basis": None,
                "cost_rate_schedule_id": None,
                "cost_code": "pricing_schedule_unavailable",
                "cost_display": "cost unavailable",
            },
        ),
    ),
)
async def test_run_observation_projects_bounded_cost_semantics(estimate, expected):
    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=MockModelProvider(
            (
                _stop(
                    usage=ModelUsage(
                        input_tokens=1,
                        output_tokens=1,
                        cost_estimate=estimate,
                    )
                ),
            )
        ),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        observer=events.append,
        clock=lambda: NOW,
    )

    await loop.run(_run(run_id=f"run-{estimate.status.value}"))

    data = events[-1].data.to_dict()
    assert {key: data[key] for key in expected} == expected
    assert "estimated_cost_usd" not in data
    assert all(
        not isinstance(value, str) or len(value) <= 256 for value in expected.values()
    )


async def test_tool_run_has_only_loop_level_events_and_no_raw_content():
    events: list[AgentEvent] = []
    provider = MockModelProvider((_call(), _stop()))
    tools = ScriptedTools(
        {
            "call-1": ToolResultBlock(
                call_id="call-1",
                output={"rows": ["raw result sentinel"]},
            )
        }
    )
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=tools,
        observer=events.append,
        clock=lambda: NOW,
    )

    result = await loop.run(_run())

    assert result.kind is LoopExitKind.COMPLETED
    assert _kinds(events) == (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
    )
    assert not {
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.TOOL_COMPLETED,
        AgentEventKind.APPROVAL_REQUESTED,
        AgentEventKind.APPROVAL_DECIDED,
    }.intersection(_kinds(events))
    rendered = repr(tuple(event.data.to_dict() for event in events))
    for sentinel in (
        "raw user sentinel",
        "raw assistant sentinel",
        "raw argument sentinel",
        "raw result sentinel",
        "provider secret sentinel",
    ):
        assert sentinel not in rendered


async def test_observer_exceptions_do_not_change_result_or_transcript():
    def broken_observer(event: AgentEvent) -> None:
        del event
        raise RuntimeError("observer failure")

    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=MockModelProvider((_stop("answer"),)),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        transcripts=store,
        observer=broken_observer,
        clock=lambda: NOW,
    )

    result = await loop.run(_run())
    transcript = await store.load(result.run_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "answer"
    assert tuple(message.role for message in transcript.messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    assert await store.result(result.run_id) == result


async def test_stream_fragments_are_bounded_and_observer_failure_is_non_directive():
    observed: list[AgentEvent] = []

    def broken_after_recording(event: AgentEvent) -> None:
        observed.append(event)
        if event.kind is AgentEventKind.MODEL_TEXT_DELTA:
            raise RuntimeError("fragment observer failure")

    final = _stop("canonical final")
    provider = MockStreamingModelProvider(
        (
            (
                ModelTextDelta("x" * 2_050),
                ModelStreamCompleted(final),
            ),
        )
    )
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        transcripts=store,
        observer=broken_after_recording,
        clock=lambda: NOW,
        stream_model_calls=True,
    )

    result = await loop.run(_run(run_id="run-stream-observed"))

    fragment_lengths = []
    for event in observed:
        if event.kind is not AgentEventKind.MODEL_TEXT_DELTA:
            continue
        fragment = event.data["text"]
        assert isinstance(fragment, str)
        fragment_lengths.append(len(fragment))
    assert fragment_lengths == [1_024, 1_024, 2]
    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "canonical final"
    transcript = await store.load(result.run_id)
    assert tuple(message.role for message in transcript.messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    assert all("x" * 16 not in repr(message) for message in transcript.messages)


async def test_failed_start_emits_nothing():
    class FailedStartStore(InMemoryTranscriptStore):
        async def start(self, run):
            del run
            raise RuntimeError("start failed")

    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=MockModelProvider((_stop(),)),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        transcripts=FailedStartStore(),
        observer=events.append,
    )

    with pytest.raises(RuntimeError, match="start failed"):
        await loop.run(_run())
    assert events == []


async def test_cancellation_persists_then_emits_interrupted_completion():
    entered = asyncio.Event()

    class HangingProvider:
        provider_id = "mock:hanging"

        def supports_request_policy(self, request: ModelRequest) -> bool:
            del request
            return True

        async def generate(self, request: ModelRequest) -> ModelResponse:
            del request
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    store = OrderingStore()
    events: list[AgentEvent] = []

    def observe(event: AgentEvent) -> None:
        events.append(event)
        store.actions.append(f"event:{event.kind.value}")

    loop = AgentLoop(
        model=HangingProvider(),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        transcripts=store,
        observer=observe,
        clock=lambda: NOW,
    )
    task = asyncio.create_task(loop.run(_run("run-cancel")))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    persisted = await store.result("run-cancel")
    assert persisted is not None
    assert persisted.kind is LoopExitKind.INTERRUPTED
    assert _kinds(events) == (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.RUN_COMPLETED,
    )
    assert events[-1].data["exit_kind"] == "interrupted"
    assert store.actions[-2:] == [
        "persisted:finish:interrupted",
        "event:run.completed",
    ]


async def test_context_overflow_persists_one_completion_without_model_event():
    store = OrderingStore()
    events: list[AgentEvent] = []

    def observe(event: AgentEvent) -> None:
        events.append(event)
        store.actions.append(f"event:{event.kind.value}")

    loop = AgentLoop(
        model=MockModelProvider((_stop("must not run"),)),
        context_builder=OverflowContext(),
        tools=ScriptedTools(),
        transcripts=store,
        observer=observe,
        clock=lambda: NOW,
    )
    result = await loop.run(_run("run-overflow"))

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "context_window_exceeded"
    assert _kinds(events) == (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.RUN_COMPLETED,
    )
    assert store.actions[-2:] == [
        "persisted:finish:failed",
        "event:run.completed",
    ]


async def test_handled_provider_error_emits_one_completion():
    events: list[AgentEvent] = []
    error = ModelProviderError(ProviderErrorCode.INVALID_REQUEST)
    loop = AgentLoop(
        model=MockModelProvider((error,)),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        observer=events.append,
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-provider-error"))

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "invalid_request"
    assert _kinds(events).count(AgentEventKind.RUN_COMPLETED) == 1
    assert AgentEventKind.MODEL_COMPLETED not in _kinds(events)


async def test_outer_step_limit_emits_one_completion():
    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=MockModelProvider((_call(), _stop("bounded answer"))),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(
            {"call-1": ToolResultBlock(call_id="call-1", output={"value": 1})}
        ),
        limits=LoopLimits(max_steps=1),
        observer=events.append,
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-step-limit"))

    assert result.kind is LoopExitKind.COMPLETED
    assert result.reason == "step_limit_reached"
    assert _kinds(events).count(AgentEventKind.RUN_COMPLETED) == 1


async def test_outer_wall_limit_failure_emits_one_completion():
    class HangingProvider:
        provider_id = "mock:wall-limit"

        def supports_request_policy(self, request: ModelRequest) -> bool:
            del request
            return True

        async def generate(self, request: ModelRequest) -> ModelResponse:
            del request
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    events: list[AgentEvent] = []
    loop = AgentLoop(
        model=HangingProvider(),
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        limits=LoopLimits(max_wall_time_seconds=0.01),
        observer=events.append,
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-wall-limit"))

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "wall_time_exhausted"
    assert _kinds(events).count(AgentEventKind.RUN_COMPLETED) == 1
    assert AgentEventKind.MODEL_COMPLETED not in _kinds(events)


async def test_no_observer_preserves_existing_behavior():
    provider = MockModelProvider((_stop("unchanged"),))
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=provider,
        context_builder=TranscriptContext(),
        tools=ScriptedTools(),
        transcripts=store,
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-unobserved"))

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text == "unchanged"
    assert await store.result(result.run_id) == result
    assert len(provider.requests) == 1


async def test_public_create_and_open_inject_the_observer(tmp_path):
    created_events: list[AgentEvent] = []
    first_provider = MockModelProvider((_stop("created"),))
    profile = ModelProfile(
        id=first_provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "observed",
        root=tmp_path,
        model=first_provider,
        model_profile=profile,
        observer=created_events.append,
    )
    try:
        await agent.run("create path")
    finally:
        await agent.close()

    opened_events: list[AgentEvent] = []
    second_provider = MockModelProvider((_stop("opened"),))
    reopened = await Agent.open(
        "observed",
        root=tmp_path,
        model=second_provider,
        model_profile=profile,
        observer=opened_events.append,
    )
    try:
        await reopened.run("open path")
    finally:
        await reopened.close()

    expected = (
        AgentEventKind.RUN_STARTED,
        AgentEventKind.MODEL_COMPLETED,
        AgentEventKind.RUN_COMPLETED,
    )
    assert _kinds(created_events) == expected
    assert _kinds(opened_events) == expected
