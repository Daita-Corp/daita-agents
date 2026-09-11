"""Component-owned tests split from ``test_kernel_contracts.py``."""

from __future__ import annotations

from tests.support.kernel import (
    CATALOG_SEARCH_CAPABILITY_ID,
    NOW,
    AgentLoop,
    ContextToolProjectionAdapter,
    FinishReason,
    InMemoryTranscriptStore,
    LoopExitKind,
    LoopLimits,
    MessageRole,
    MockModelProvider,
    ModelResponse,
    ModelSensitivity,
    ToolBatchOutcome,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    _ConcurrentReadExecutor,
    _Context,
    _error,
    _FailingReadExecutor,
    _InterruptibleSideEffect,
    _NoTools,
    _PayloadReadExecutor,
    _run,
    _runtime,
    execute_projected,
    pytest,
)


async def test_tool_call_response_bound_rejects_batch_before_execution():
    calls = tuple(ToolCall(id=f"call-{index}", name="lookup") for index in range(2))
    store = InMemoryTranscriptStore()
    loop = AgentLoop(
        model=MockModelProvider(
            (ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls),)
        ),
        context_builder=_Context(),
        tools=_NoTools(),
        transcripts=store,
        limits=LoopLimits(
            max_tool_calls_per_response=1,
            max_tool_calls_per_run=1,
        ),
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-tool-call-response-bound"))

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "tool_calls_per_response_exceeded"
    assert tuple(
        message.role for message in (await store.load(result.run_id)).messages
    ) == (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL, MessageRole.TOOL)


async def test_tool_call_run_bound_counts_across_responses():
    class _CountingTools:
        def __init__(self) -> None:
            self.calls: list[ToolCall] = []
            self._projection = ContextToolProjectionAdapter(
                (
                    ToolDefinition(
                        name="lookup",
                        description="Lookup.",
                        input_schema={"type": "object", "properties": {}},
                    ),
                )
            )

        async def prepare_run(self, run):
            return await self._projection.prepare_run(run)

        def project(self, catalog, messages):
            return self._projection.project(catalog, messages)

        async def execute_all(self, run, calls, *, projection, messages, sensitivity):
            del run, projection, messages, sensitivity
            self.calls.extend(calls)
            return ToolBatchOutcome(
                tuple(
                    ToolResultBlock(call_id=call.id, output={"value": call.id})
                    for call in calls
                )
            )

    tools = _CountingTools()
    loop = AgentLoop(
        model=MockModelProvider(
            (
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(ToolCall(id="first", name="lookup"),),
                ),
                ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(ToolCall(id="second", name="lookup"),),
                ),
            )
        ),
        context_builder=_Context(),
        tools=tools,
        limits=LoopLimits(
            max_tool_calls_per_response=1,
            max_tool_calls_per_run=1,
        ),
        clock=lambda: NOW,
    )

    result = await loop.run(_run("run-tool-call-run-bound"))

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "tool_calls_per_run_exceeded"
    assert [call.id for call in tools.calls] == ["first"]


async def test_runtime_binds_classification_and_provenance_to_every_success():
    outcome = await execute_projected(
        _runtime(_InterruptibleSideEffect()),
        _run("run-runtime-classification"),
        (
            ToolCall(
                id="read",
                name="stage_a_read",
                arguments={"query": "value"},
            ),
        ),
        sensitivity=ModelSensitivity.INTERNAL,
    )

    result = outcome.ordered_results[0]
    assert result.sensitivity is ModelSensitivity.INTERNAL
    assert result.sensitivity_provenance["authority"] == "test_static_domain"
    assert result.sensitivity_provenance["capability_id"] == (
        CATALOG_SEARCH_CAPABILITY_ID
    )


async def test_read_concurrency_and_per_source_pressure_are_bounded():
    executor = _ConcurrentReadExecutor()
    runtime = _runtime(
        _InterruptibleSideEffect(),
        read_executor=executor,
        limits=LoopLimits(
            max_parallel_reads=4,
            max_parallel_reads_per_source=1,
        ),
    )
    calls = tuple(
        ToolCall(
            id=f"read-{index}",
            name="stage_a_read",
            arguments={"query": f"value-{index}"},
        )
        for index in range(4)
    )

    outcome = await execute_projected(
        runtime,
        _run("run-read-pressure"),
        calls,
        sensitivity=ModelSensitivity.INTERNAL,
    )

    assert all(not result.is_error for result in outcome.ordered_results)
    assert executor.maximum_active == 1


@pytest.mark.parametrize(
    ("payload", "limits", "expected_code"),
    (
        (
            "x" * 500,
            LoopLimits(max_tool_result_bytes=128),
            "tool_result_too_large",
        ),
        (
            {"one": {"two": {"three": {"four": "value"}}}},
            LoopLimits(max_tool_result_depth=4),
            "tool_result_too_deep",
        ),
    ),
)
async def test_tool_result_bytes_and_depth_fail_with_structured_bounds(
    payload: object,
    limits: LoopLimits,
    expected_code: str,
):
    runtime = _runtime(
        _InterruptibleSideEffect(),
        read_executor=_PayloadReadExecutor(payload),
        limits=limits,
    )
    outcome = await execute_projected(
        runtime,
        _run(f"run-{expected_code}"),
        (
            ToolCall(
                id="read",
                name="stage_a_read",
                arguments={"query": "value"},
            ),
        ),
        sensitivity=ModelSensitivity.INTERNAL,
    )

    assert outcome.ordered_results[0].is_error
    assert _error(outcome.ordered_results[0])["code"] == expected_code


async def test_unexpected_executor_failure_is_normalized_and_redacted():
    runtime = _runtime(
        _InterruptibleSideEffect(),
        read_executor=_FailingReadExecutor(),
    )
    outcome = await execute_projected(
        runtime,
        _run("run-redacted-executor-failure"),
        (
            ToolCall(
                id="read",
                name="stage_a_read",
                arguments={"query": "value"},
            ),
        ),
        sensitivity=ModelSensitivity.INTERNAL,
    )

    assert _error(outcome.ordered_results[0])["code"] == "tool_execution_failed"
    assert "SECRET EXECUTOR DIAGNOSTIC" not in repr(outcome.ordered_results[0])
