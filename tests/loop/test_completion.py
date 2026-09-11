"""Component-owned tests split from ``test_kernel_contracts.py``."""

from __future__ import annotations

from tests.support.kernel import (
    NOW,
    AgentLoop,
    CanonicalMessage,
    FinishReason,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    Mapping,
    MessageRole,
    MockModelProvider,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolCall,
    ToolResultBlock,
    Transcript,
    _CancellationResistantReadExecutor,
    _Context,
    _error,
    _InterruptibleSideEffect,
    _NoTools,
    _run,
    _runtime,
    asyncio,
    cast,
    execute_projected,
    pytest,
    validate_completed_transcript,
)


@pytest.mark.parametrize(
    ("finish_reason", "expected_reason"),
    (
        (FinishReason.LENGTH, "model_output_limit"),
        (FinishReason.CONTENT_FILTER, "content_filtered"),
        (FinishReason.ERROR, "model_response_error"),
    ),
)
async def test_nonterminal_finish_reasons_never_complete_normally(
    finish_reason: FinishReason,
    expected_reason: str,
):
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=MockModelProvider(
            (ModelResponse(finish_reason=finish_reason, text="partial sentinel"),)
        ),
        context_builder=_Context(),
        tools=_NoTools(),
        transcripts=store,
        clock=lambda: NOW,
    )

    result = await loop.run(_run(f"run-{finish_reason.value}"))

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == expected_reason
    assert result.final_text is None
    transcript = await store.load(result.run_id)
    assert transcript.messages[-1].content == (TextBlock("partial sentinel"),)


def test_completed_transcript_validator_requires_one_ordered_result_per_call():
    run = _run("run-invalid-completed-transcript")
    call_one = ToolCall(id="one", name="lookup")
    call_two = ToolCall(id="two", name="lookup")
    messages = (
        CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),)),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(call_one, call_two),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(ToolResultBlock(call_id="two", output={"value": 2}),),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(ToolResultBlock(call_id="one", output={"value": 1}),),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("answer"),),
        ),
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id or run.id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        final_text="answer",
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="ordered tool result"):
        validate_completed_transcript(Transcript(run=run, messages=messages), result)


async def test_in_memory_completion_is_one_atomic_store_operation():
    class _AtomicSpyStore(InMemoryTranscriptStore):
        def __init__(self):
            super().__init__()
            self.actions: list[str] = []

        async def append(self, run_id, message):
            self.actions.append(f"append:{message.role.value}")
            await super().append(run_id, message)

        async def complete(self, result, final_message):
            self.actions.append("complete")
            await super().complete(result, final_message)

    store = _AtomicSpyStore()
    loop = AgentLoop(
        model=MockModelProvider(
            (ModelResponse(finish_reason=FinishReason.STOP, text="answer"),)
        ),
        context_builder=_Context(),
        tools=_NoTools(),
        transcripts=store,
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-atomic-store-operation"))

    assert result.kind is LoopExitKind.COMPLETED
    assert store.actions == ["append:user", "complete"]
    transcript = await store.load(result.run_id)
    validate_completed_transcript(transcript, result)


async def test_cancelled_batch_keeps_known_side_effect_and_marks_later_call_unstarted():
    side_effect = _InterruptibleSideEffect()
    runtime = _runtime(side_effect)
    calls = (
        ToolCall(id="write", name="stage_a_write"),
        ToolCall(id="later-read", name="stage_a_read"),
    )
    task = asyncio.create_task(
        execute_projected(
            runtime,
            _run("run-batch-known"),
            calls,
            sensitivity=ModelSensitivity.INTERNAL,
        )
    )
    await asyncio.wait_for(side_effect.started.wait(), timeout=1)

    task.cancel()
    await asyncio.sleep(0)
    side_effect.release.set()
    outcome = await asyncio.wait_for(task, timeout=1)

    assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
    assert outcome.outcome_certainty is ToolBatchCertainty.DEFINITE
    results = outcome.ordered_results
    assert tuple(result.call_id for result in results) == ("write", "later-read")
    assert not results[0].is_error
    assert _error(results[1])["code"] == "tool_call_not_started"
    details = cast(Mapping[str, object], _error(results[1])["details"])
    assert details["execution_state"] == "not_started"


async def test_uncertain_side_effect_wait_is_bounded_and_records_outcome_unknown():
    side_effect = _InterruptibleSideEffect(ignore_worker_cancellation=True)
    runtime = _runtime(side_effect, recovery_timeout=0.01)
    calls = (
        ToolCall(id="write", name="stage_a_write"),
        ToolCall(id="later-read", name="stage_a_read"),
    )
    task = asyncio.create_task(
        execute_projected(
            runtime,
            _run("run-batch-unknown"),
            calls,
            sensitivity=ModelSensitivity.INTERNAL,
        )
    )
    await asyncio.wait_for(side_effect.started.wait(), timeout=1)

    task.cancel()
    outcome = await asyncio.wait_for(task, timeout=0.5)

    assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
    assert outcome.outcome_certainty is ToolBatchCertainty.OUTCOME_UNKNOWN
    assert _error(outcome.ordered_results[0])["code"] == "outcome_unknown"
    assert _error(outcome.ordered_results[1])["code"] == "tool_call_not_started"
    side_effect.release.set()
    await asyncio.sleep(0)


async def test_cancellation_resistant_read_has_a_bounded_settlement_wait():
    read = _CancellationResistantReadExecutor()
    runtime = _runtime(
        _InterruptibleSideEffect(),
        read_executor=read,
        recovery_timeout=0.01,
    )
    task = asyncio.create_task(
        execute_projected(
            runtime,
            _run("run-bounded-read-cancellation"),
            (ToolCall(id="read", name="stage_a_read", arguments={"query": "x"}),),
            sensitivity=ModelSensitivity.INTERNAL,
        )
    )
    await asyncio.wait_for(read.started.wait(), timeout=1)

    task.cancel()
    outcome = await asyncio.wait_for(task, timeout=0.5)

    assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
    assert outcome.outcome_certainty is ToolBatchCertainty.DEFINITE
    assert _error(outcome.ordered_results[0])["code"] == "tool_call_interrupted"
    read.release.set()
    await asyncio.sleep(0)


async def test_loop_persists_complete_interrupted_batch_before_cancellation_escapes():
    side_effect = _InterruptibleSideEffect()
    runtime = _runtime(side_effect)
    calls = (
        ToolCall(id="write", name="stage_a_write"),
        ToolCall(id="later-read", name="stage_a_read"),
    )
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=MockModelProvider(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="load-write",
                            name="toolbox_load",
                            arguments={"tool_names": ["stage_a_write"]},
                        ),
                    ),
                ),
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=calls,
                ),
            )
        ),
        context_builder=_Context(),
        tools=runtime,
        transcripts=store,
        clock=lambda: NOW,
    )
    task = asyncio.create_task(loop.run(_run("run-loop-batch-cancelled")))
    await asyncio.wait_for(side_effect.started.wait(), timeout=1)

    task.cancel()
    await asyncio.sleep(0)
    side_effect.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    transcript = await store.load("run-loop-batch-cancelled")
    assert tuple(message.role for message in transcript.messages) == (
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
    )
    tool_results = tuple(
        cast(ToolResultBlock, message.content[0])
        for message in transcript.messages
        if message.role is MessageRole.TOOL
    )
    assert tuple(result.call_id for result in tool_results) == (
        "load-write",
        "write",
        "later-read",
    )
    terminal = await store.result("run-loop-batch-cancelled")
    assert terminal is not None
    assert terminal.kind is LoopExitKind.INTERRUPTED
