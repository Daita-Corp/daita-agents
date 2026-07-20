from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3

import pytest

from daita import Agent, SQLiteSource
from daita._json import FrozenJsonObject
from daita.agent import AgentHomeError
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import ActionProposal, Evidence, Observation


class _TextContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tools == ()
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock(message),),
                ),
            ),
        )


class _TextDomain:
    def tool_views(self, operation: OperationSnapshot) -> tuple[ToolDefinition, ...]:
        return ()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only domain has no actions")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only domain has no observations")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        return Readiness(
            allowed=True,
            code="ready",
            message="Text is ready.",
            evaluated_at=operation.operation.updated_at,
        )


class _GatedProvider:
    provider_id = "mock:gated-public-stream"

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.calls == 2:
            self.entered.set()
            await self.release.wait()
        return ModelResponse(
            text=f"answer-{self.calls}",
            finish_reason=FinishReason.STOP,
        )


class _BlockingProvider:
    provider_id = "mock:cancel-public-stream"

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_public_stream_yields_only_the_new_operations_committed_events(
    tmp_path: Path,
) -> None:
    provider = _GatedProvider()
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=_TextContext(),
        domain=_TextDomain(),
    )
    try:
        previous = await agent.run("seed event history")
        stream = agent.stream("stream this operation", session_id="stream-session")
        first = await asyncio.wait_for(anext(stream), timeout=1)
        await asyncio.wait_for(provider.entered.wait(), timeout=1)

        assert first["type"] == "trigger.received"
        assert first["operation_id"] != previous.operation_id
        assert first["session_id"] == "stream-session"
        assert first["payload"] == FrozenJsonObject.from_mapping({})

        provider.release.set()
        events = [first, *[event async for event in stream]]
        operation_id = first["operation_id"]
        assert isinstance(operation_id, str)
        assert {event["operation_id"] for event in events} == {operation_id}
        assert events[-1]["type"] == "operation.succeeded"

        snapshot = await agent.inspect(operation_id)
        assert tuple(event["id"] for event in events) == tuple(
            event.id for event in snapshot.events
        )
    finally:
        provider.release.set()
        await agent.close()


async def test_closing_public_stream_cancels_and_checkpoints_its_run(
    tmp_path: Path,
) -> None:
    provider = _BlockingProvider()
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=_TextContext(),
        domain=_TextDomain(),
    )
    try:
        stream = agent.stream("cancel after the first event")
        first = await asyncio.wait_for(anext(stream), timeout=1)
        await asyncio.wait_for(provider.entered.wait(), timeout=1)
        operation_id = first["operation_id"]
        assert isinstance(operation_id, str)

        await stream.aclose()

        snapshot = await agent.inspect(operation_id)
        assert snapshot.operation.status.value == "interrupted"
        assert snapshot.events[-1].type == "operation.interrupted"
    finally:
        await agent.close()


async def test_public_event_read_and_subscription_never_expose_canonical_payloads(
    tmp_path: Path,
) -> None:
    provider = _GatedProvider()
    agent = await Agent.create(
        "atlas",
        root=tmp_path,
        model=provider,
        context_builder=_TextContext(),
        domain=_TextDomain(),
    )
    subscription = agent.subscribe_events()
    try:
        run_task = asyncio.create_task(agent.run("project this event history"))
        first = await asyncio.wait_for(anext(subscription), timeout=1)
        result = await asyncio.wait_for(run_task, timeout=1)

        assert first["type"] == "trigger.received"
        assert first["operation_id"] == result.operation_id
        assert first["payload"] == FrozenJsonObject.from_mapping({})
        assert not hasattr(first, "event")

        history = await agent.events()
        assert history
        assert {event["operation_id"] for event in history} == {result.operation_id}
        assert all(
            event["payload"] == FrozenJsonObject.from_mapping({}) for event in history
        )
    finally:
        await subscription.aclose()
        await agent.close()


async def test_public_detach_persists_the_one_way_source_transition(
    tmp_path: Path,
) -> None:
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE values_for_test(value INTEGER)")

    agent = await Agent.create("atlas", root=tmp_path)
    registration = await agent.attach(SQLiteSource(database))
    detached = await agent.detach(registration.id)
    assert detached.id == registration.id
    assert not detached.active
    assert detached.detached_at is not None
    await agent.close()

    reopened = await Agent.open("atlas", root=tmp_path)
    try:
        with pytest.raises(AgentHomeError, match="conflicts with existing source"):
            await reopened.attach(SQLiteSource(database))
    finally:
        await reopened.close()
