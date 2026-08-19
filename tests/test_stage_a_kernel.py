from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

import pytest

from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    Capability,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from daita.catalog.capabilities import CATALOG_SEARCH_CAPABILITY_ID
from daita.domains.data.controller import DataToolRuntime
from daita.domains.data.context import DataContextBuilder, _estimate_input_tokens
from daita.llm.errors import ContextEvidencePressureExceeded
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.routing import ModelProviderRegistration, ModelRouter, RetryPolicy
from daita.loop import (
    AgentLoop,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    LoopLimits,
    RunInput,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
    Transcript,
)
from daita.loop.models import validate_completed_transcript
from daita.memory.capabilities import MEMORY_SET_CAPABILITY_ID
from daita.storage.sqlite import SQLiteStateStore

NOW = datetime(2026, 8, 18, tzinfo=UTC)


def _error(result: ToolResultBlock) -> Mapping[str, object]:
    return cast(Mapping[str, object], result.output["error"])


class _Context:
    async def prepare(self, run, messages, tools):
        del run
        return messages[:-1], tools

    def project(
        self,
        snapshot,
        messages,
        *,
        step,
        final=False,
        previous_request_input_tokens=None,
    ):
        del step, previous_request_input_tokens
        static, tools = snapshot
        return ModelRequest(
            messages=(*static, *messages),
            tools=() if final else tools,
        )


class _NoTools:
    async def definitions(self, run):
        del run
        return ()

    async def execute_all(self, run, calls):
        del run
        assert calls == ()
        return ToolBatchOutcome(())


class _RuntimeCatalog:
    async def source_routing_facts(self, agent_id, source_ids=()):
        del agent_id, source_ids
        return ()


class _SnapshotCatalog:
    def __init__(self) -> None:
        self.context_reads = 0
        self.sensitivity_reads = 0
        self.revision = "one"

    async def admitted_model_sensitivity(self, agent_id, source_ids=()):
        del agent_id, source_ids
        self.sensitivity_reads += 1
        return ModelSensitivity.PUBLIC

    async def catalog_context(
        self,
        agent_id,
        query,
        *,
        limit,
        source_ids=(),
        resource_ids=(),
    ):
        del agent_id, query, limit, source_ids, resource_ids
        self.context_reads += 1
        return FrozenJsonObject.from_mapping(
            {
                "resources": (
                    {
                        "resource_id": "resource-snapshot",
                        "revision": self.revision,
                    },
                ),
                "total_matches": 1,
                "truncated": False,
                "trust_classification": "untrusted_external_data",
            }
        )


class _ReadExecutor:
    executor_id = "stage-a.read"

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return ToolOutput(kind="stage-a.read-result", data={"call": request.call_id})


class _ConcurrentReadExecutor(_ReadExecutor):
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return await super().execute(request)


class _PayloadReadExecutor(_ReadExecutor):
    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def execute(self, request: ToolExecution) -> ToolOutput:
        return ToolOutput(
            kind="stage-a.read-result",
            data={"call": request.call_id, "payload": self.payload},
        )


class _FailingReadExecutor(_ReadExecutor):
    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        raise RuntimeError("SECRET EXECUTOR DIAGNOSTIC")


class _CancellationResistantReadExecutor(_ReadExecutor):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue
        return await super().execute(request)


class _InterruptibleSideEffect:
    executor_id = "stage-a.write"

    def __init__(self, *, ignore_worker_cancellation: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.ignore_worker_cancellation = ignore_worker_cancellation

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        del request
        return FrozenJsonObject.from_mapping({"fingerprint": "current"})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.ignore_worker_cancellation:
                raise
            await self.release.wait()
        return ToolOutput(kind="stage-a.write-result", data={"call": request.call_id})


def _runtime(
    side_effect: _InterruptibleSideEffect,
    *,
    recovery_timeout: float = 1.0,
    read_executor: _ReadExecutor | None = None,
    limits: LoopLimits = LoopLimits(),
) -> DataToolRuntime:
    resolved_read = read_executor or _ReadExecutor()
    read = Capability(
        id=CATALOG_SEARCH_CAPABILITY_ID,
        description="Stage A read.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_kind="stage-a.read-result",
        output_schema={
            "type": "object",
            "properties": {
                "call": {"type": "string"},
                "payload": {},
            },
            "required": ["call"],
        },
        executor_id=resolved_read.executor_id,
    )
    write = Capability(
        id=MEMORY_SET_CAPABILITY_ID,
        description="Stage A side effect.",
        input_schema={"type": "object", "properties": {}},
        output_kind="stage-a.write-result",
        output_schema={
            "type": "object",
            "properties": {"call": {"type": "string"}},
            "required": ["call"],
        },
        executor_id=side_effect.executor_id,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
    )

    async def approve(_request):
        return ApprovalDecision.APPROVE

    return DataToolRuntime(
        CapabilityRegistry(
            capabilities=(read, write),
            executors=(resolved_read, side_effect),
            tool_views=(
                ToolView(
                    name="stage_a_read",
                    capability_id=read.id,
                    description=read.description,
                ),
                ToolView(
                    name="stage_a_write",
                    capability_id=write.id,
                    description=write.description,
                ),
            ),
        ),
        _RuntimeCatalog(),  # type: ignore[arg-type]
        approval_handler=approve,
        limits=limits,
        side_effect_recovery_timeout_seconds=recovery_timeout,
    )


def _run(run_id: str) -> RunInput:
    return RunInput(
        id=run_id,
        agent_id="agent-stage-a",
        message="question",
        created_at=NOW,
        conversation_id="conversation-stage-a",
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


async def test_sqlite_atomic_completion_rolls_back_both_values_on_encode_failure(
    tmp_path,
    monkeypatch,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    run = _run("run-sqlite-atomic-failure")
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    final = CanonicalMessage(
        role=MessageRole.ASSISTANT,
        content=(TextBlock("answer"),),
    )
    result = LoopExit(
        run_id=run.id,
        conversation_id=run.conversation_id or run.id,
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        final_text="answer",
        created_at=NOW,
    )
    await store.start(run)
    await store.append(run.id, user)

    def fail_encode(_result):
        raise RuntimeError("injected terminal encoding failure")

    monkeypatch.setattr("daita.storage.sqlite.encode_loop_exit", fail_encode)
    try:
        with pytest.raises(RuntimeError, match="injected terminal encoding failure"):
            await store.complete(result, final)

        assert (await store.load(run.id)).messages == (user,)
        assert await store.result(run.id) is None
    finally:
        await store.close()


async def test_cancelled_batch_keeps_known_side_effect_and_marks_later_call_unstarted():
    side_effect = _InterruptibleSideEffect()
    runtime = _runtime(side_effect)
    calls = (
        ToolCall(id="write", name="stage_a_write"),
        ToolCall(id="later-read", name="stage_a_read"),
    )
    task = asyncio.create_task(runtime.execute_all(_run("run-batch-known"), calls))
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
    task = asyncio.create_task(runtime.execute_all(_run("run-batch-unknown"), calls))
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
        runtime.execute_all(
            _run("run-bounded-read-cancellation"),
            (ToolCall(id="read", name="stage_a_read", arguments={"query": "x"}),),
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
        MessageRole.TOOL,
    )
    tool_results = tuple(
        cast(ToolResultBlock, message.content[0])
        for message in transcript.messages
        if message.role is MessageRole.TOOL
    )
    assert tuple(result.call_id for result in tool_results) == (
        "write",
        "later-read",
    )
    terminal = await store.result("run-loop-batch-cancelled")
    assert terminal is not None
    assert terminal.kind is LoopExitKind.INTERRUPTED


async def test_run_context_snapshot_is_prepared_once_and_aggregates_results():
    catalog = _SnapshotCatalog()
    profile = ModelProfile(
        id="mock:stage-a-context",
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    builder = DataContextBuilder(catalog, profile=profile)
    run = RunInput(
        id="run-context-snapshot",
        agent_id="agent-stage-a",
        message="question",
        created_at=NOW,
        conversation_id="conversation-stage-a-context",
        source_id="source-stage-a",
    )
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    tool = ToolDefinition(
        name="lookup",
        description="Lookup one value.",
        input_schema={"type": "object", "properties": {}},
    )

    snapshot = await builder.prepare(run, (user,), (tool,))
    assert not hasattr(builder, "build")
    first = builder.project(snapshot, (user,), step=1)
    catalog.revision = "changed-after-prepare"
    call = ToolCall(id="classified", name="lookup")
    current = (
        user,
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=call.id,
                    output={"value": "classified"},
                    sensitivity=ModelSensitivity.CONFIDENTIAL,
                    sensitivity_provenance={
                        "authority": "validated_capability_result",
                        "resource_ids": ("resource-snapshot",),
                    },
                ),
            ),
        ),
    )
    second = builder.project(snapshot, current, step=2)

    assert catalog.context_reads == 1
    assert catalog.sensitivity_reads == 1
    assert snapshot.static_context_sha256 == (
        second.sensitivity_provenance["static_context_sha256"]
    )
    assert first.messages[0] == second.messages[0]
    assert "changed-after-prepare" not in repr(second.messages[0])
    assert second.sensitivity is ModelSensitivity.CONFIDENTIAL
    initial_provenance = cast(
        Mapping[str, object],
        second.sensitivity_provenance["initial_sensitivity_provenance"],
    )
    assert initial_provenance["authority"] == "run_context_snapshot"
    assert initial_provenance["source_ids"] == ("source-stage-a",)
    assert initial_provenance["static_context_sha256"] == (
        snapshot.static_context_sha256
    )
    classified_results = cast(
        tuple[Mapping[str, object], ...],
        second.sensitivity_provenance["classified_results"],
    )
    assert classified_results[0]["call_id"] == "classified"
    final = builder.project(snapshot, current, step=3, final=True)
    assert final.tools == ()
    assert final.messages[0] == snapshot.final_static_messages[0]
    assert "execution step limit has been reached" in repr(final.messages[0])
    assert "execution step limit has been reached" not in repr(first.messages[0])


async def test_context_owner_rejects_cumulative_evidence_pressure_explicitly():
    builder = DataContextBuilder(
        _SnapshotCatalog(),
        profile=ModelProfile(
            id="mock:stage-a-pressure",
            context_window_tokens=32_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        max_context_evidence_bytes=64,
    )
    run = _run("run-context-pressure")
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    snapshot = await builder.prepare(run, (user,), ())
    result = ToolResultBlock(
        call_id="large",
        output={"rows": "x" * 200},
        sensitivity=ModelSensitivity.INTERNAL,
        sensitivity_provenance={"authority": "test"},
    )

    with pytest.raises(ContextEvidencePressureExceeded):
        builder.project(
            snapshot,
            (
                user,
                CanonicalMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=(ToolCall(id="large", name="lookup"),),
                ),
                CanonicalMessage(role=MessageRole.TOOL, content=(result,)),
            ),
            step=2,
        )


def test_token_estimate_is_conservative_accounting_not_raw_byte_count():
    request = ModelRequest(
        messages=(
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("a" * 3_000),),
            ),
        )
    )
    estimate = _estimate_input_tokens(request)

    assert estimate > len(("a" * 3_000).encode("utf-8"))


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
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
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
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
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
    ) == (MessageRole.USER,)


async def test_tool_call_run_bound_counts_across_responses():
    class _CountingTools:
        def __init__(self) -> None:
            self.calls: list[ToolCall] = []

        async def definitions(self, run):
            del run
            return (
                ToolDefinition(
                    name="lookup",
                    description="Lookup.",
                    input_schema={"type": "object", "properties": {}},
                ),
            )

        async def execute_all(self, run, calls):
            del run
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
    outcome = await _runtime(_InterruptibleSideEffect()).execute_all(
        _run("run-runtime-classification"),
        (
            ToolCall(
                id="read",
                name="stage_a_read",
                arguments={"query": "value"},
            ),
        ),
    )

    result = outcome.ordered_results[0]
    assert result.sensitivity is ModelSensitivity.INTERNAL
    assert result.sensitivity_provenance["authority"] == (
        "conservative_runtime_default"
    )
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

    outcome = await runtime.execute_all(_run("run-read-pressure"), calls)

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
    outcome = await runtime.execute_all(
        _run(f"run-{expected_code}"),
        (
            ToolCall(
                id="read",
                name="stage_a_read",
                arguments={"query": "value"},
            ),
        ),
    )

    assert outcome.ordered_results[0].is_error
    assert _error(outcome.ordered_results[0])["code"] == expected_code


async def test_unexpected_executor_failure_is_normalized_and_redacted():
    runtime = _runtime(
        _InterruptibleSideEffect(),
        read_executor=_FailingReadExecutor(),
    )
    outcome = await runtime.execute_all(
        _run("run-redacted-executor-failure"),
        (
            ToolCall(
                id="read",
                name="stage_a_read",
                arguments={"query": "value"},
            ),
        ),
    )

    assert _error(outcome.ordered_results[0])["code"] == "tool_execution_failed"
    assert "SECRET EXECUTOR DIAGNOSTIC" not in repr(outcome.ordered_results[0])
