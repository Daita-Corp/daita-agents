from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest

from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.loop.driver import InMemoryTranscriptStore
from daita.loop.models import ConversationRun, LoopExit, LoopExitKind, RunInput
from daita.loop.session import (
    RunCancellationToken,
    RunSession,
    RunSessionOptions,
)
from daita.loop.transcripts import ConversationPredecessor, RunSessionWriter

NOW = datetime(2026, 9, 16, tzinfo=UTC)


def _run(run_id: str, *, conversation_id: str = "conversation") -> RunInput:
    return RunInput(
        id=run_id,
        agent_id="agent",
        conversation_id=conversation_id,
        message=f"message for {run_id}",
        created_at=NOW,
    )


def _failed(run: RunInput) -> LoopExit:
    return LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id or run.id,
        kind=LoopExitKind.FAILED,
        reason="fixture_failure",
        created_at=NOW,
    )


def _message(text: str) -> CanonicalMessage:
    return CanonicalMessage(role=MessageRole.USER, content=(TextBlock(text),))


async def test_run_session_is_immutable_single_use_and_isolated():
    store = InMemoryTranscriptStore()
    first_run = _run("run-one", conversation_id="one")
    second_run = _run("run-two", conversation_id="two")
    first_options = RunSessionOptions(
        files_only=True,
        retained_skill_bindings=(("analysis", "sha256:first"),),
        one_time_artifact_destinations=("destination-one",),
    )
    second_options = RunSessionOptions(
        explicit_learning=True,
        retained_skill_bindings=(("review", "sha256:second"),),
        one_time_artifact_destinations=("destination-two",),
    )
    first = RunSession(
        run=first_run,
        writer=RunSessionWriter(store, first_run),
        absolute_deadline=100.0,
        cancellation=RunCancellationToken(),
        options=first_options,
    ).with_prepared(object())
    second = RunSession(
        run=second_run,
        writer=RunSessionWriter(store, second_run),
        absolute_deadline=100.0,
        cancellation=RunCancellationToken(),
        options=second_options,
    ).with_prepared(object())

    assert first.writer is not second.writer
    assert first.prepared is not second.prepared
    assert first.options is first_options
    assert second.options is second_options
    assert first.options.one_time_artifact_destinations == ("destination-one",)
    assert second.options.one_time_artifact_destinations == ("destination-two",)
    with pytest.raises(FrozenInstanceError):
        first.options.files_only = False  # type: ignore[misc]

    first.consume()
    assert first.used
    assert not second.used
    with pytest.raises(RuntimeError, match="single-use"):
        first.consume()
    second.consume()


async def test_run_session_writer_rejects_duplicate_out_of_order_and_terminal_writes():
    store = InMemoryTranscriptStore()
    run = _run("run-writer")
    writer = RunSessionWriter(store, run)

    await writer.start()
    with pytest.raises(ValueError, match="out of order"):
        await writer.append(_message("wrong"), position=1)
    await writer.append(_message("first"), position=0)
    with pytest.raises(ValueError, match="out of order"):
        await writer.append(_message("duplicate"), position=0)

    duplicate = RunSessionWriter(store, run)
    with pytest.raises(ValueError, match="already exists"):
        await duplicate.start()

    foreign = _failed(_run("run-foreign"))
    with pytest.raises(ValueError, match="foreign result"):
        await writer.finish(foreign)
    await writer.finish(_failed(run))
    with pytest.raises(RuntimeError, match="not appendable"):
        await writer.append(_message("after terminal"), position=1)
    with pytest.raises(RuntimeError, match="not terminalizable"):
        await writer.finish(_failed(run))


async def test_writer_start_binds_exact_terminal_conversation_predecessor():
    store = InMemoryTranscriptStore()
    first_run = _run("run-first")
    first_writer = RunSessionWriter(store, first_run)
    await first_writer.start()
    await first_writer.append(_message("first"), position=0)
    first_result = _failed(first_run)
    await first_writer.finish(first_result)
    predecessor = ConversationPredecessor.from_run(
        ConversationRun(
            turn_index=0,
            transcript=await store.load(first_run.id),
            result=first_result,
        )
    )

    stale = replace(predecessor, revision="sha256:" + "0" * 64)
    stale_run = _run("run-stale")
    with pytest.raises(ValueError, match="predecessor changed"):
        await RunSessionWriter(store, stale_run, predecessor=stale).start()

    second_run = _run("run-second")
    second_writer = RunSessionWriter(
        store,
        second_run,
        predecessor=predecessor,
    )
    await second_writer.start()

    competing_run = _run("run-competing")
    with pytest.raises(ValueError, match="predecessor changed"):
        await RunSessionWriter(
            store,
            competing_run,
            predecessor=predecessor,
        ).start()
