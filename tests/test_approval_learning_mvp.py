import asyncio
import threading
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import cast

import pytest

from daita import Agent
from daita._json import FrozenJsonObject
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalRequest,
    Capability,
    SideEffectExecutor,
    ToolExecution,
)
from daita.domains.data.controller import DataToolRuntime
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import RunInput
from daita.memory import MEMORY_MAX_CHARACTERS, USER_MAX_CHARACTERS
from daita.memory.capabilities import (
    MEMORY_SET_CAPABILITY_ID,
    MEMORY_SET_EXECUTOR_ID,
    MEMORY_SET_OUTPUT_KIND,
    MEMORY_SET_TOOL_NAME,
)
from daita.observation import AgentEvent, AgentEventKind
from daita.skills.capabilities import (
    SKILL_DELETE_CAPABILITY_ID,
    SKILL_DELETE_EXECUTOR_ID,
    SKILL_DELETE_OUTPUT_KIND,
    SKILL_DELETE_TOOL_NAME,
    SKILL_SAVE_CAPABILITY_ID,
    SKILL_SAVE_EXECUTOR_ID,
    SKILL_SAVE_OUTPUT_KIND,
    SKILL_SAVE_TOOL_NAME,
)

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


def _stop(text: str = "done") -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _call(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


def _memory_call(
    call_id: str = "write",
    *,
    target: str = "memory",
    content: str = "replacement",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=MEMORY_SET_TOOL_NAME,
        arguments={"target": target, "content": content},
    )


def _skill_save_call(
    call_id: str = "skill-save",
    *,
    name: str = "reusable-workflow",
    description: str = "Apply one reusable workflow.",
    instructions: str = "Follow the verified steps and report assumptions.",
    expected_sha256: str | None = None,
) -> ToolCall:
    arguments = {
        "name": name,
        "description": description,
        "instructions": instructions,
    }
    if expected_sha256 is not None:
        arguments["expected_sha256"] = expected_sha256
    return ToolCall(
        id=call_id,
        name=SKILL_SAVE_TOOL_NAME,
        arguments=arguments,
    )


def _skill_delete_call(
    call_id: str = "skill-delete",
    *,
    name: str = "reusable-workflow",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=SKILL_DELETE_TOOL_NAME,
        arguments={"name": name},
    )


def _run(agent: Agent, run_id: str = "approval-run") -> RunInput:
    return RunInput(
        id=run_id,
        agent_id=agent.id,
        message="test",
        created_at=NOW,
        conversation_id="approval-conversation",
    )


def _runtime(agent: Agent) -> DataToolRuntime:
    loop = agent._embedded._loop
    assert loop is not None
    return cast(DataToolRuntime, loop._tools)


async def _execute(agent: Agent, *calls: ToolCall):
    return await _runtime(agent).execute_all(_run(agent), calls)


async def _skill_digest(agent: Agent, name: str) -> str:
    skill, digest = await agent._embedded._skill_store.read_skill_with_digest(name)
    assert skill is not None
    return digest


def _error_code(result: ToolResultBlock) -> str:
    error = result.output["error"]
    assert isinstance(error, Mapping)
    code = error["code"]
    assert isinstance(code, str)
    return code


def _tool_results(provider: MockModelProvider) -> tuple[ToolResultBlock, ...]:
    return tuple(
        block
        for request in provider.requests
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    )


def _system_text(request: ModelRequest) -> str:
    return "\n".join(
        block.text
        for message in request.messages
        if message.role is MessageRole.SYSTEM
        for block in message.content
        if isinstance(block, TextBlock)
    )


def _tool_event_kinds(events: list[AgentEvent]) -> tuple[AgentEventKind, ...]:
    return tuple(
        event.kind
        for event in events
        if event.kind
        in {
            AgentEventKind.TOOL_STARTED,
            AgentEventKind.APPROVAL_REQUESTED,
            AgentEventKind.APPROVAL_DECIDED,
            AgentEventKind.TOOL_COMPLETED,
        }
    )


async def _agent(
    tmp_path,
    name: str,
    *,
    approval_handler=None,
    observer=None,
    responses: tuple[ModelResponse, ...] = (_stop(),),
) -> Agent:
    provider = MockModelProvider(responses)
    return await Agent.create(
        name,
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approval_handler,
        observer=observer,
        clock=lambda: NOW,
    )


def test_approval_records_and_write_invariants_are_exact_and_frozen():
    request = ApprovalRequest(
        run_id="run",
        call_id="call",
        tool_name=MEMORY_SET_TOOL_NAME,
        capability_id=MEMORY_SET_CAPABILITY_ID,
        arguments=FrozenJsonObject.from_mapping(
            {"target": "memory", "content": "exact"}
        ),
        reason="Approve exactly once.",
    )
    assert tuple(ApprovalDecision) == (
        ApprovalDecision.APPROVE,
        ApprovalDecision.DENY,
    )
    assert dict(request.arguments) == {"target": "memory", "content": "exact"}
    with pytest.raises(FrozenInstanceError):
        request.reason = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.arguments["content"] = "changed"  # type: ignore[index]

    with pytest.raises(ValueError, match="write tools must be side-effecting"):
        Capability(
            id="test.capability",
            description="test",
            input_schema={"type": "object", "properties": {}},
            output_kind="test.output",
            output_schema={"type": "object", "properties": {}},
            executor_id="test.executor",
            access_mode=AccessMode.WRITE,
            side_effecting=False,
        )
    with pytest.raises(ValueError, match="read tools cannot be"):
        Capability(
            id="test.capability",
            description="test",
            input_schema={"type": "object", "properties": {}},
            output_kind="test.output",
            output_schema={"type": "object", "properties": {}},
            executor_id="test.executor",
            access_mode=AccessMode.READ,
            side_effecting=True,
        )


async def test_memory_set_identity_projection_and_read_tools_never_ask_approval(
    tmp_path,
):
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "identity", approval_handler=approve)
    try:
        registry = agent._embedded._capabilities
        view, capability = registry.resolve_tool(MEMORY_SET_TOOL_NAME)
        resolved, executor = registry.resolve_execution(capability.id)
        assert (
            view.name,
            capability.id,
            executor.executor_id,
            capability.output_kind,
            capability.access_mode,
            capability.side_effecting,
        ) == (
            MEMORY_SET_TOOL_NAME,
            MEMORY_SET_CAPABILITY_ID,
            MEMORY_SET_EXECUTOR_ID,
            MEMORY_SET_OUTPUT_KIND,
            AccessMode.WRITE,
            True,
        )
        assert resolved == capability
        definitions = await _runtime(agent).definitions(_run(agent))
        assert tuple(item.name for item in definitions) == (
            "artifact_create_document",
            "artifact_set_export_location",
            "memory_set",
            "skill_delete",
            "skill_save",
            "skill_view",
        )

        read = ToolCall(id="read", name="skill_view", arguments={"name": "absent"})
        result = (await _execute(agent, read))[0]
        assert _error_code(result) == "skill_not_found"
        assert approvals == []
    finally:
        await agent.close()


async def test_missing_handler_and_denial_are_model_visible_and_do_not_mutate(
    tmp_path,
):
    agent = await _agent(tmp_path, "missing-handler")
    try:
        result = (await _execute(agent, _memory_call(content="blocked")))[0]
        assert _error_code(result) == "approval_required"
        assert await agent.read_memory() == ""
    finally:
        await agent.close()

    requests: list[ApprovalRequest] = []

    async def deny(request):
        requests.append(request)
        return ApprovalDecision.DENY

    agent = await _agent(tmp_path, "denied", approval_handler=deny)
    try:
        result = (await _execute(agent, _memory_call(content="denied")))[0]
        assert _error_code(result) == "approval_denied"
        assert await agent.read_memory() == ""
        assert len(requests) == 1
    finally:
        await agent.close()


async def test_approval_executes_the_exact_frozen_invocation_once(
    tmp_path, monkeypatch
):
    requests: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest):
        requests.append(request)
        with pytest.raises(TypeError):
            request.arguments["content"] = "callback mutation"  # type: ignore[index]
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "exact", approval_handler=approve)
    store = agent._embedded._memory_store
    original = store.replace_from_tool
    executions: list[tuple[str, str]] = []

    async def counted(target, content):
        executions.append((target, content))
        await original(target, content)

    monkeypatch.setattr(store, "replace_from_tool", counted)
    try:
        content = "exact replacement 日本語"
        result = (await _execute(agent, _memory_call(content=content)))[0]
        assert not result.is_error
        assert executions == [("memory", content)]
        assert len(requests) == 1
        assert requests[0].capability_id == MEMORY_SET_CAPABILITY_ID
        assert dict(requests[0].arguments) == {
            "target": "memory",
            "content": content,
        }
        assert await agent.read_memory() == content
    finally:
        await agent.close()


@pytest.mark.parametrize("returned", ("approve", True, None))
async def test_non_enum_approval_values_fail_closed(tmp_path, returned):
    async def invalid(request):
        del request
        return returned

    agent = await _agent(
        tmp_path,
        f"invalid-{type(returned).__name__}",
        approval_handler=invalid,
    )
    try:
        result = (await _execute(agent, _memory_call(content="blocked")))[0]
        assert _error_code(result) == "approval_failed"
        assert await agent.read_memory() == ""
    finally:
        await agent.close()


async def test_callback_exception_is_an_ordinary_tool_error_and_loop_continues(
    tmp_path,
):
    async def broken(request):
        del request
        raise RuntimeError("handler sentinel")

    provider = MockModelProvider((_call(_memory_call(content="blocked")), _stop("ok")))
    agent = await Agent.create(
        "callback-error",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=broken,
    )
    try:
        result = await agent.run("try learning")
        assert result.final_text == "ok"
        tool_result = _tool_results(provider)[0]
        assert _error_code(tool_result) == "approval_failed"
        assert "handler sentinel" not in repr(tool_result.output)
        assert await agent.read_memory() == ""
    finally:
        await agent.close()


async def test_oversized_content_is_rejected_before_approval(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "oversized", approval_handler=approve)
    try:
        schema_result = (
            await _execute(
                agent,
                _memory_call(content="x" * (MEMORY_MAX_CHARACTERS + 1)),
            )
        )[0]
        assert _error_code(schema_result) == "invalid_argument_value"
        preflight_result = (
            await _execute(
                agent,
                _memory_call(
                    call_id="user-too-long",
                    target="user",
                    content="x" * (USER_MAX_CHARACTERS + 1),
                ),
            )
        )[0]
        assert _error_code(preflight_result) == "memory_invalid_content"
        assert approvals == []
        assert await agent.read_memory() == ""
        assert await agent.read_user_profile() == ""
    finally:
        await agent.close()


async def test_tool_and_approval_event_sequences_are_exact_and_content_free(tmp_path):
    sentinel = "RAW_MEMORY_CONTENT_MUST_NOT_ENTER_EVENTS"

    async def approve(request):
        assert request.arguments["content"] == sentinel
        return ApprovalDecision.APPROVE

    events: list[AgentEvent] = []
    agent = await _agent(
        tmp_path,
        "event-success",
        approval_handler=approve,
        observer=events.append,
    )
    try:
        result = (await _execute(agent, _memory_call(content=sentinel)))[0]
        assert not result.is_error
    finally:
        await agent.close()

    assert _tool_event_kinds(events) == (
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.APPROVAL_REQUESTED,
        AgentEventKind.APPROVAL_DECIDED,
        AgentEventKind.TOOL_COMPLETED,
    )
    assert sentinel not in repr([event.data for event in events])
    assert dict(events[0].data) == {
        "call_id": "write",
        "tool_name": MEMORY_SET_TOOL_NAME,
        "capability_id": MEMORY_SET_CAPABILITY_ID,
    }
    assert events[1].data == events[0].data
    assert dict(events[2].data) == {"call_id": "write", "outcome": "approved"}
    completed = events[3].data
    assert completed["call_id"] == "write"
    assert completed["tool_name"] == MEMORY_SET_TOOL_NAME
    assert completed["success"] is True
    assert completed["error_code"] is None
    duration_ms = completed["duration_ms"]
    assert isinstance(duration_ms, int)
    assert duration_ms >= 0


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "unavailable",
            (AgentEventKind.TOOL_STARTED, AgentEventKind.TOOL_COMPLETED),
        ),
        (
            "preflight",
            (AgentEventKind.TOOL_STARTED, AgentEventKind.TOOL_COMPLETED),
        ),
        (
            "missing",
            (AgentEventKind.TOOL_STARTED, AgentEventKind.TOOL_COMPLETED),
        ),
        (
            "denied",
            (
                AgentEventKind.TOOL_STARTED,
                AgentEventKind.APPROVAL_REQUESTED,
                AgentEventKind.APPROVAL_DECIDED,
                AgentEventKind.TOOL_COMPLETED,
            ),
        ),
        (
            "failed",
            (
                AgentEventKind.TOOL_STARTED,
                AgentEventKind.APPROVAL_REQUESTED,
                AgentEventKind.APPROVAL_DECIDED,
                AgentEventKind.TOOL_COMPLETED,
            ),
        ),
    ),
)
async def test_error_event_subsequences_and_one_completion(tmp_path, case, expected):
    events: list[AgentEvent] = []

    async def decide(request):
        del request
        if case == "denied":
            return ApprovalDecision.DENY
        if case == "failed":
            raise RuntimeError("no")
        return ApprovalDecision.APPROVE

    handler = None if case == "missing" else decide
    agent = await _agent(
        tmp_path,
        f"events-{case}",
        approval_handler=handler,
        observer=events.append,
    )
    try:
        if case == "unavailable":
            call = ToolCall(id="write", name="unknown_write")
        elif case == "preflight":
            call = _memory_call(target="user", content="x" * (USER_MAX_CHARACTERS + 1))
        else:
            call = _memory_call(content=case)
        result = (await _execute(agent, call))[0]
        assert result.is_error
    finally:
        await agent.close()

    assert _tool_event_kinds(events) == expected
    assert _tool_event_kinds(events).count(AgentEventKind.TOOL_COMPLETED) == 1
    if case == "unavailable":
        assert "capability_id" not in events[0].data
    if case in {"denied", "failed"}:
        assert events[2].data["outcome"] == ("denied" if case == "denied" else "failed")


async def test_read_groups_are_parallel_and_side_effects_are_ordered_barriers(
    tmp_path,
    monkeypatch,
):
    async def approve(request):
        del request
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "barriers", approval_handler=approve)
    try:
        for name in ("first", "second", "later"):
            await agent.save_skill(name, name, "body")
        store = agent._embedded._skill_store
        memory_store = agent._embedded._memory_store
        original_read = store.read_skill_with_digest
        original_replace = memory_store.replace_from_tool
        initial_started: set[str] = set()
        release_reads = asyncio.Event()
        write_done = asyncio.Event()
        actions: list[str] = []

        async def controlled_read(name):
            actions.append(f"read-start:{name}")
            if name in {"first", "second"}:
                initial_started.add(name)
                if len(initial_started) == 2:
                    release_reads.set()
                await release_reads.wait()
            if name == "later":
                assert write_done.is_set()
            value = await original_read(name)
            actions.append(f"read-done:{name}")
            return value

        async def controlled_replace(target, content):
            assert {"read-done:first", "read-done:second"} <= set(actions)
            actions.append("write-start")
            await original_replace(target, content)
            actions.append("write-done")
            write_done.set()

        monkeypatch.setattr(store, "read_skill_with_digest", controlled_read)
        monkeypatch.setattr(memory_store, "replace_from_tool", controlled_replace)
        results = await asyncio.wait_for(
            _execute(
                agent,
                ToolCall(id="first", name="skill_view", arguments={"name": "first"}),
                ToolCall(id="second", name="skill_view", arguments={"name": "second"}),
                _memory_call(call_id="write", content="ordered"),
                ToolCall(id="later", name="skill_view", arguments={"name": "later"}),
            ),
            timeout=2,
        )
        assert initial_started == {"first", "second"}
        assert tuple(result.call_id for result in results) == (
            "first",
            "second",
            "write",
            "later",
        )
        assert actions.index("write-done") < actions.index("read-start:later")
    finally:
        await agent.close()


async def test_two_approved_replacements_are_sequential_and_keep_result_order(
    tmp_path,
    monkeypatch,
):
    async def approve(request):
        del request
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "sequential-writes", approval_handler=approve)
    store = agent._embedded._memory_store
    original = store.replace_from_tool
    active = 0
    maximum_active = 0
    executions: list[str] = []

    async def controlled(target, content):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        executions.append(content)
        await asyncio.sleep(0)
        await original(target, content)
        active -= 1

    monkeypatch.setattr(store, "replace_from_tool", controlled)
    try:
        results = await _execute(
            agent,
            _memory_call("one", content="one"),
            _memory_call("two", content="two"),
        )
        assert tuple(result.call_id for result in results) == ("one", "two")
        assert executions == ["one", "two"]
        assert maximum_active == 1
        assert await agent.read_memory() == "two"
    finally:
        await agent.close()


async def test_public_and_model_writes_share_the_exact_composed_mutation_lock(tmp_path):
    async def approve(request):
        del request
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "shared-lock", approval_handler=approve)
    try:
        runtime = _runtime(agent)
        lock = agent._embedded._mutation_lock
        assert runtime._mutation_lock is lock
        assert agent._embedded._memory_store._mutation_lock is lock
        assert agent._embedded._skill_store._mutation_lock is lock
        result = (await _execute(agent, _memory_call(content="model")))[0]
        assert not result.is_error
        await agent.set_memory("direct")
        assert await agent.read_memory() == "direct"
    finally:
        await agent.close()


async def test_locked_revalidation_is_immediately_before_execution(
    tmp_path, monkeypatch
):
    async def approve(request):
        del request
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "revalidate-order", approval_handler=approve)
    runtime = _runtime(agent)
    _, capability = agent._embedded._capabilities.resolve_tool(MEMORY_SET_TOOL_NAME)
    _, resolved_executor = agent._embedded._capabilities.resolve_execution(
        capability.id
    )
    executor = cast(SideEffectExecutor, resolved_executor)
    original_preflight = executor.preflight
    original_execute = executor.execute
    actions: list[tuple[str, bool]] = []

    async def observed_preflight(request):
        actions.append(("preflight", runtime._mutation_lock.locked()))
        return await original_preflight(request)

    async def observed_execute(request):
        actions.append(("execute", runtime._mutation_lock.locked()))
        return await original_execute(request)

    monkeypatch.setattr(executor, "preflight", observed_preflight)
    monkeypatch.setattr(executor, "execute", observed_execute)
    try:
        result = (await _execute(agent, _memory_call(content="validated")))[0]
        assert not result.is_error
        assert actions == [
            ("preflight", False),
            ("preflight", True),
            ("execute", True),
        ]
    finally:
        await agent.close()


async def test_direct_write_during_approval_returns_state_changed(tmp_path):
    agent: Agent

    async def approve(request):
        assert request.arguments["content"] == "stale-model-value"
        await agent.set_memory("newer-direct-value")
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "state-change", approval_handler=approve)
    try:
        await agent.set_memory("old-value")
        result = (await _execute(agent, _memory_call(content="stale-model-value")))[0]
        assert _error_code(result) == "state_changed"
        assert await agent.read_memory() == "newer-direct-value"
    finally:
        await agent.close()


async def test_cancellation_during_approval_propagates_without_decision_or_write(
    tmp_path,
):
    entered = asyncio.Event()
    events: list[AgentEvent] = []

    async def pending(request):
        del request
        entered.set()
        await asyncio.Event().wait()
        return ApprovalDecision.APPROVE

    agent = await _agent(
        tmp_path,
        "cancel-approval",
        approval_handler=pending,
        observer=events.append,
    )
    try:
        task = asyncio.create_task(_execute(agent, _memory_call(content="never")))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await agent.read_memory() == ""
    finally:
        await agent.close()

    assert _tool_event_kinds(events) == (
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.APPROVAL_REQUESTED,
    )


async def test_cancellation_after_atomic_replacement_starts_waits_for_outcome(
    tmp_path,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    events: list[AgentEvent] = []

    async def approve(request):
        del request
        return ApprovalDecision.APPROVE

    agent = await _agent(
        tmp_path,
        "cancel-mutation",
        approval_handler=approve,
        observer=events.append,
    )
    store = agent._embedded._memory_store
    original = store._write_sync

    def blocked_write(*args):
        started.set()
        assert release.wait(timeout=2)
        return original(*args)

    monkeypatch.setattr(store, "_write_sync", blocked_write)
    try:
        task = asyncio.create_task(_execute(agent, _memory_call(content="definite")))
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), timeout=3)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await agent.read_memory() == "definite"
    finally:
        release.set()
        await agent.close()

    assert _tool_event_kinds(events) == (
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.APPROVAL_REQUESTED,
        AgentEventKind.APPROVAL_DECIDED,
        AgentEventKind.TOOL_COMPLETED,
    )
    assert events[-1].data["success"] is True


async def test_approval_state_is_not_persisted_across_restart(tmp_path):
    async def approve(request):
        del request
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "restart", approval_handler=approve)
    try:
        result = (await _execute(agent, _memory_call(content="approved-once")))[0]
        assert not result.is_error
    finally:
        await agent.close()

    provider = MockModelProvider((_stop(),))
    reopened = await Agent.open(
        "restart",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        result = (await _execute(reopened, _memory_call(content="not-approved")))[0]
        assert _error_code(result) == "approval_required"
        assert await reopened.read_memory() == "approved-once"
    finally:
        await reopened.close()


async def test_explicit_correction_is_one_approved_foreground_memory_write(tmp_path):
    content = "Revenue means paid invoice subtotal, excluding voided invoices."
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider(
        (_call(_memory_call(content=content)), _stop("I saved the correction."))
    )
    agent = await Agent.create(
        "learn-correction",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        result = await agent.run(
            "Correction: revenue means paid invoice subtotal. Remember this."
        )
        assert result.final_text == "I saved the correction."
        assert await agent.read_memory() == content
        assert len(approvals) == 1
        assert approvals[0].arguments["content"] == content
        assert len(provider.requests) == 2
        assert content in _system_text(provider.requests[1])
        tool_result = _tool_results(provider)[0]
        assert tool_result.output["data"] == FrozenJsonObject.from_mapping(
            {"target": "memory", "replaced": True}
        )
        prompt = _system_text(provider.requests[0])
        assert "explicit durable definitions/preferences/corrections" in prompt
        assert "ordinary text ends run" in prompt
        assert "Approval card alone confirms" in prompt
        assert "never ask typed approval" in prompt
    finally:
        await agent.close()


async def test_explicit_reusable_workflow_is_one_approved_foreground_skill(tmp_path):
    approvals: list[ApprovalRequest] = []
    call = _skill_save_call(
        name="monthly-revenue",
        description="Calculate monthly revenue consistently.",
        instructions="Use paid invoice date. Exclude voided invoices.",
    )

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider((_call(call), _stop("I saved the workflow.")))
    agent = await Agent.create(
        "learn-workflow",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        result = await agent.run("Save this as a reusable workflow for future runs.")
        assert result.final_text == "I saved the workflow."
        skill = await agent.read_skill("monthly-revenue")
        assert skill is not None
        assert skill.instructions == "Use paid invoice date. Exclude voided invoices."
        assert len(approvals) == 1
        assert dict(approvals[0].arguments) == dict(call.arguments)
        assert len(provider.requests) == 2
        assert "- monthly-revenue: Calculate monthly revenue consistently.\n" in (
            _system_text(provider.requests[1])
        )
        tool_result = _tool_results(provider)[0]
        assert tool_result.output["data"] == FrozenJsonObject.from_mapping(
            {"name": "monthly-revenue", "changed": True}
        )
        prompt = _system_text(provider.requests[0])
        assert "SKILL.md=procedures" in prompt
        assert "Replace, do not duplicate" in prompt
    finally:
        await agent.close()


async def test_weak_learning_signal_stays_in_transcript_without_a_write(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    provider = MockModelProvider((_stop("That is the current result."),))
    agent = await Agent.create(
        "weak-learning-signal",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        result = await agent.run("Revenue happened to be 42 in this one result.")
        assert result.final_text == "That is the current result."
        assert await agent.read_memory() == ""
        assert await agent.read_user_profile() == ""
        assert await agent.list_skills() == ()
        assert approvals == []
        prompt = _system_text(provider.requests[0])
        assert "inference/one-offs are weak" in prompt
        assert "Never learn raw results" in prompt
        assert "Approval card alone confirms" in prompt
        assert "never ask typed approval" in prompt
    finally:
        await agent.close()


async def test_learning_tool_descriptions_route_documents_and_require_write_first(
    tmp_path,
):
    agent = await _agent(tmp_path, "learning-tool-routing")
    try:
        definitions = {
            definition.name: definition
            for definition in await _runtime(agent).definitions(_run(agent))
        }
        memory_description = definitions[MEMORY_SET_TOOL_NAME].description
        skill_view_description = definitions["skill_view"].description
        skill_save_description = definitions[SKILL_SAVE_TOOL_NAME].description

        assert "USER.md(target=user)=durable preferences" in memory_description
        assert "MEMORY.md(target=memory)=schema-independent" in (memory_description)
        assert "SKILL.md=procedures" in memory_description
        assert "Text ends run: call first" in memory_description
        assert "replace duplicates" in memory_description
        assert "sole approval card" in memory_description
        assert "current_sha256" in skill_view_description
        assert "reusable validated steps with use, verification, and failure" in (
            skill_save_description
        )
        assert "Text ends run: call first" in skill_save_description
        assert "expected_sha256" in skill_save_description
        assert "sole approval card" in skill_save_description
    finally:
        await agent.close()


async def test_loaded_skill_is_replaced_instead_of_duplicated(tmp_path):
    bootstrap = await Agent.create("replace-not-duplicate", root=tmp_path)
    try:
        await bootstrap.save_skill(
            "monthly-revenue",
            "Use for monthly booked-revenue reporting.",
            "Use the invoice date.",
        )
        expected_sha256 = await _skill_digest(bootstrap, "monthly-revenue")
    finally:
        await bootstrap.close()

    replacement = _skill_save_call(
        "replace",
        name="monthly-revenue",
        description=(
            "Use for monthly booked-revenue reporting, excluding voided invoices."
        ),
        instructions=(
            "Use the paid invoice date. Exclude voided invoices. Verify the month total."
        ),
        expected_sha256=expected_sha256,
    )
    provider = MockModelProvider(
        (
            _call(
                ToolCall(
                    id="view",
                    name="skill_view",
                    arguments={"name": "monthly-revenue"},
                )
            ),
            _call(replacement),
            _stop("I updated the existing workflow."),
        )
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.open(
        "replace-not-duplicate",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=approve,
    )
    try:
        result = await agent.run(
            "Correct the monthly revenue workflow: use paid date and exclude voids."
        )
        assert result.final_text == "I updated the existing workflow."
        assert tuple(summary.name for summary in await agent.list_skills()) == (
            "monthly-revenue",
        )
        current = await agent.read_skill("monthly-revenue")
        assert current is not None
        assert current.instructions == replacement.arguments["instructions"]
        assert len(approvals) == 1
        assert approvals[0].arguments["name"] == "monthly-revenue"
        assert approvals[0].arguments["expected_sha256"] == expected_sha256

        viewed = _tool_results(provider)[0]
        viewed_data = viewed.output["data"]
        assert isinstance(viewed_data, Mapping)
        assert viewed_data["current_sha256"] == expected_sha256
    finally:
        await agent.close()


async def test_skill_write_identities_use_the_existing_registry_and_runtime(tmp_path):
    agent = await _agent(tmp_path, "skill-identities")
    try:
        expected = {
            SKILL_SAVE_TOOL_NAME: (
                SKILL_SAVE_CAPABILITY_ID,
                SKILL_SAVE_EXECUTOR_ID,
                SKILL_SAVE_OUTPUT_KIND,
            ),
            SKILL_DELETE_TOOL_NAME: (
                SKILL_DELETE_CAPABILITY_ID,
                SKILL_DELETE_EXECUTOR_ID,
                SKILL_DELETE_OUTPUT_KIND,
            ),
        }
        registry = agent._embedded._capabilities
        for tool_name, identity in expected.items():
            view, capability = registry.resolve_tool(tool_name)
            resolved, executor = registry.resolve_execution(capability.id)
            assert view.capability_id == capability.id
            assert resolved == capability
            assert (
                capability.id,
                executor.executor_id,
                capability.output_kind,
            ) == identity
            assert capability.access_mode is AccessMode.WRITE
            assert capability.side_effecting is True
            assert callable(getattr(executor, "preflight", None))
    finally:
        await agent.close()


@pytest.mark.parametrize("operation", ("save", "delete"))
async def test_denied_skill_save_and_delete_never_mutate(tmp_path, operation):
    approvals: list[ApprovalRequest] = []

    async def deny(request):
        approvals.append(request)
        return ApprovalDecision.DENY

    agent = await _agent(tmp_path, f"deny-skill-{operation}", approval_handler=deny)
    try:
        await agent.save_skill("target", "Original", "Keep this exact skill.")
        if operation == "delete":
            call = _skill_delete_call(name="target")
        else:
            call = _skill_save_call(
                name="target",
                description="Denied replacement",
                instructions="This replacement must not be persisted.",
                expected_sha256=await _skill_digest(agent, "target"),
            )
        before = agent.home / "skills" / "target" / "SKILL.md"
        before_bytes = before.read_bytes() if before.exists() else None
        result = (await _execute(agent, call))[0]
        assert _error_code(result) == "approval_denied"
        after_bytes = before.read_bytes() if before.exists() else None
        assert after_bytes == before_bytes
        assert len(approvals) == 1
    finally:
        await agent.close()


async def test_blind_and_stale_skill_replacements_fail_before_approval(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(
        tmp_path,
        "blind-stale-replacement",
        approval_handler=approve,
    )
    try:
        await agent.save_skill("target", "Original", "Original instructions.")
        path = agent.home / "skills/target/SKILL.md"
        before = path.read_bytes()

        blind = (
            await _execute(
                agent,
                _skill_save_call(
                    "blind",
                    name="target",
                    instructions="Blind replacement.",
                ),
            )
        )[0]
        stale = (
            await _execute(
                agent,
                _skill_save_call(
                    "stale",
                    name="target",
                    instructions="Stale replacement.",
                    expected_sha256="0" * 64,
                ),
            )
        )[0]

        assert _error_code(blind) == "skill_expected_sha256_required"
        assert _error_code(stale) == "skill_stale_replacement"
        assert approvals == []
        assert path.read_bytes() == before
    finally:
        await agent.close()


async def test_approved_delete_removes_only_exact_slug_and_direct_delete_is_idempotent(
    tmp_path,
):
    async def approve(request):
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "exact-delete", approval_handler=approve)
    try:
        await agent.save_skill("target", "Target", "Delete only this skill.")
        await agent.save_skill("target-extra", "Other", "Preserve this skill.")
        result = (await _execute(agent, _skill_delete_call(name="target")))[0]
        assert not result.is_error
        assert result.output["data"] == FrozenJsonObject.from_mapping(
            {"name": "target", "deleted": True}
        )
        assert await agent.read_skill("target") is None
        assert await agent.read_skill("target-extra") is not None
        assert await agent.delete_skill("target-extra") is True
        assert await agent.delete_skill("target-extra") is False
    finally:
        await agent.close()


async def test_absent_model_delete_returns_not_found_without_approval(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "absent-delete", approval_handler=approve)
    try:
        result = (await _execute(agent, _skill_delete_call(name="absent")))[0]
        assert _error_code(result) == "skill_not_found"
        invalid = (await _execute(agent, _skill_delete_call(name="../escape")))[0]
        assert _error_code(invalid) == "skill_invalid_name"
        assert approvals == []
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("call", "expected"),
    (
        (_skill_save_call(name="../escape"), "skill_invalid_name"),
        (
            _skill_save_call(description=" not-trimmed"),
            "skill_invalid_document",
        ),
        (
            _skill_save_call(description="two\nlines"),
            "skill_invalid_document",
        ),
        (
            _skill_save_call(instructions="First\n## Instructions\nSecond"),
            "skill_invalid_document",
        ),
        (
            _skill_save_call(instructions="x" * 12_001),
            "invalid_argument_value",
        ),
    ),
)
async def test_invalid_skill_save_inputs_fail_before_approval(tmp_path, call, expected):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(
        tmp_path,
        f"invalid-skill-{expected}-{len(call.arguments)}",
        approval_handler=approve,
    )
    try:
        result = (await _execute(agent, call))[0]
        assert _error_code(result) == expected
        assert approvals == []
        assert await agent.list_skills() == ()
    finally:
        await agent.close()


async def test_skill_count_and_complete_index_limits_fail_before_approval(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    count_agent = await _agent(tmp_path, "skill-count-limit", approval_handler=approve)
    try:
        for index in range(32):
            await count_agent.save_skill(
                f"skill-{index:02d}",
                "d",
                "body",
            )
        result = (await _execute(count_agent, _skill_save_call(name="skill-overflow")))[
            0
        ]
        assert _error_code(result) == "skill_invalid_document"
        assert len(await count_agent.list_skills()) == 32
    finally:
        await count_agent.close()

    index_agent = await _agent(tmp_path, "skill-index-limit", approval_handler=approve)
    try:
        for index in range(15):
            await index_agent.save_skill(
                f"wide-{index:02d}",
                "d" * 240,
                "body",
            )
        result = (
            await _execute(
                index_agent,
                _skill_save_call(
                    name="wide-overflow",
                    description="d" * 240,
                ),
            )
        )[0]
        assert _error_code(result) == "skill_invalid_document"
        assert len(await index_agent.list_skills()) == 15
        assert approvals == []
    finally:
        await index_agent.close()


async def test_skill_preflight_fingerprints_document_state_and_complete_index(
    tmp_path,
):
    agent = await _agent(tmp_path, "skill-fingerprints")
    try:
        registry = agent._embedded._capabilities
        _, capability = registry.resolve_tool(SKILL_SAVE_TOOL_NAME)
        _, executor = registry.resolve_execution(capability.id)
        request = ApprovalRequest(
            run_id="run",
            call_id="call",
            tool_name=SKILL_SAVE_TOOL_NAME,
            capability_id=SKILL_SAVE_CAPABILITY_ID,
            arguments=FrozenJsonObject.from_mapping(
                dict(_skill_save_call(name="target").arguments)
            ),
            reason="inspect",
        )
        execution = ToolExecution(
            run_id=request.run_id,
            capability_id=request.capability_id,
            arguments=request.arguments,
        )
        initial = await cast(SideEffectExecutor, executor).preflight(execution)
        assert set(initial) == {
            "name",
            "exists",
            "current_sha256",
            "state_sha256",
            "index_sha256",
        }
        assert initial["exists"] is False
        await agent.save_skill("other", "Changes index.", "Other body.")
        index_changed = await cast(SideEffectExecutor, executor).preflight(execution)
        assert index_changed["index_sha256"] != initial["index_sha256"]
        assert index_changed["current_sha256"] == initial["current_sha256"]
        await agent.save_skill("target", "Existing target.", "Current body.")
        replacement_call = _skill_save_call(
            name="target",
            expected_sha256=await _skill_digest(agent, "target"),
        )
        replacement_execution = ToolExecution(
            run_id=request.run_id,
            capability_id=request.capability_id,
            arguments=FrozenJsonObject.from_mapping(dict(replacement_call.arguments)),
        )
        selected_changed = await cast(SideEffectExecutor, executor).preflight(
            replacement_execution
        )
        assert selected_changed["exists"] is True
        assert selected_changed["current_sha256"] != initial["current_sha256"]
        assert selected_changed["state_sha256"] != initial["state_sha256"]
        assert "Current body" not in repr(selected_changed)
    finally:
        await agent.close()


@pytest.mark.parametrize("change", ("replace", "remove"))
async def test_selected_skill_change_during_approval_returns_state_changed(
    tmp_path, change
):
    agent: Agent

    async def approve(request):
        assert request.arguments["name"] == "target"
        if change == "replace":
            await agent.save_skill("target", "Newer", "Newer direct content.")
        else:
            assert await agent.delete_skill("target") is True
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, f"selected-{change}", approval_handler=approve)
    try:
        await agent.save_skill("target", "Original", "Original content.")
        call = (
            _skill_save_call(
                name="target",
                description="Model",
                instructions="Stale model content.",
                expected_sha256=await _skill_digest(agent, "target"),
            )
            if change == "replace"
            else _skill_delete_call(name="target")
        )
        result = (await _execute(agent, call))[0]
        assert _error_code(result) == "state_changed"
        current = await agent.read_skill("target")
        if change == "replace":
            assert current is not None and current.description == "Newer"
        else:
            assert current is None
    finally:
        await agent.close()


async def test_aggregate_skill_index_change_during_approval_returns_state_changed(
    tmp_path,
):
    agent: Agent

    async def approve(request):
        assert request.arguments["name"] == "target"
        await agent.save_skill("other", "Changes the complete index.", "Other body.")
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "index-change", approval_handler=approve)
    try:
        result = (
            await _execute(
                agent,
                _skill_save_call(name="target", description="Target workflow."),
            )
        )[0]
        assert _error_code(result) == "state_changed"
        assert await agent.read_skill("target") is None
        assert await agent.read_skill("other") is not None
    finally:
        await agent.close()


async def test_approved_identical_save_reports_unchanged_without_replacement(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "identical-save", approval_handler=approve)
    try:
        arguments = {
            "name": "target",
            "description": "Same description.",
            "instructions": "Same instructions.",
        }
        assert await agent.save_skill(**arguments) is True
        expected_sha256 = await _skill_digest(agent, "target")
        path = agent.home / "skills" / "target" / "SKILL.md"
        before = path.stat()
        result = (
            await _execute(
                agent,
                _skill_save_call(
                    **arguments,
                    expected_sha256=expected_sha256,
                ),
            )
        )[0]
        after = path.stat()
        assert not result.is_error
        assert result.output["data"] == FrozenJsonObject.from_mapping(
            {"name": "target", "changed": False}
        )
        assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
        )
        assert len(approvals) == 1
    finally:
        await agent.close()


async def test_changed_skill_arguments_require_another_exact_callback(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "changed-arguments", approval_handler=approve)
    try:
        first = (
            await _execute(
                agent,
                _skill_save_call("one", name="target", instructions="First version."),
            )
        )[0]
        expected_sha256 = await _skill_digest(agent, "target")
        second = (
            await _execute(
                agent,
                _skill_save_call(
                    "two",
                    name="target",
                    instructions="Second version.",
                    expected_sha256=expected_sha256,
                ),
            )
        )[0]
        results = (first, second)
        assert all(not result.is_error for result in results)
        assert len(approvals) == 2
        assert approvals[0].arguments["instructions"] == "First version."
        assert approvals[1].arguments["instructions"] == "Second version."
        assert approvals[0].arguments is not approvals[1].arguments
        current = await agent.read_skill("target")
        assert current is not None and current.instructions == "Second version."
    finally:
        await agent.close()


async def test_skill_events_are_exact_and_exclude_all_knowledge_content(tmp_path):
    sentinel = "SECRET_DESCRIPTION_AND_INSTRUCTIONS"
    events: list[AgentEvent] = []

    async def approve(request):
        assert sentinel in repr(request.arguments)
        return ApprovalDecision.APPROVE

    agent = await _agent(
        tmp_path,
        "skill-events",
        approval_handler=approve,
        observer=events.append,
    )
    try:
        result = (
            await _execute(
                agent,
                _skill_save_call(
                    name="event-skill",
                    description=sentinel,
                    instructions=sentinel,
                ),
            )
        )[0]
        assert not result.is_error
    finally:
        await agent.close()

    assert _tool_event_kinds(events) == (
        AgentEventKind.TOOL_STARTED,
        AgentEventKind.APPROVAL_REQUESTED,
        AgentEventKind.APPROVAL_DECIDED,
        AgentEventKind.TOOL_COMPLETED,
    )
    assert sentinel not in repr([event.data for event in events])
    assert "event-skill" not in repr([event.data for event in events])


async def test_model_skill_write_uses_shared_lock_and_side_effect_barrier(
    tmp_path, monkeypatch
):
    async def approve(request):
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "skill-lock-barrier", approval_handler=approve)
    store = agent._embedded._skill_store
    runtime = _runtime(agent)
    original_save = store.save_from_tool
    original_delete = store.delete_from_tool
    actions: list[str] = []

    async def observed_save(name, description, instructions):
        assert runtime._mutation_lock is agent._embedded._mutation_lock
        assert runtime._mutation_lock is store._mutation_lock
        assert runtime._mutation_lock.locked()
        actions.append("save")
        return await original_save(name, description, instructions)

    async def observed_delete(name):
        assert runtime._mutation_lock.locked()
        assert actions == ["save"]
        actions.append("delete")
        return await original_delete(name)

    monkeypatch.setattr(store, "save_from_tool", observed_save)
    monkeypatch.setattr(store, "delete_from_tool", observed_delete)
    try:
        results = await _execute(
            agent,
            _skill_save_call(name="ordered"),
            _skill_delete_call(name="ordered"),
        )
        assert tuple(result.call_id for result in results) == (
            "skill-save",
            "skill-delete",
        )
        assert all(not result.is_error for result in results)
        assert actions == ["save", "delete"]
    finally:
        await agent.close()


async def test_skill_save_cancellation_before_mutation_never_writes(tmp_path):
    entered = asyncio.Event()

    async def pending(request):
        entered.set()
        await asyncio.Event().wait()
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "skill-cancel-before", approval_handler=pending)
    try:
        task = asyncio.create_task(_execute(agent, _skill_save_call(name="never")))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await agent.read_skill("never") is None
    finally:
        await agent.close()


async def test_skill_save_cancellation_after_atomic_mutation_starts_is_definite(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()

    async def approve(request):
        return ApprovalDecision.APPROVE

    agent = await _agent(tmp_path, "skill-cancel-after", approval_handler=approve)
    store = agent._embedded._skill_store
    original = store._save_sync

    def blocked_save(*args):
        started.set()
        assert release.wait(timeout=2)
        return original(*args)

    monkeypatch.setattr(store, "_save_sync", blocked_save)
    try:
        task = asyncio.create_task(_execute(agent, _skill_save_call(name="definite")))
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), timeout=3)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await agent.read_skill("definite") is not None
    finally:
        release.set()
        await agent.close()


async def test_reopen_starts_no_learning_work_and_persists_no_skill_approval(tmp_path):
    approvals: list[ApprovalRequest] = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision.APPROVE

    first_provider = MockModelProvider((_stop(),))
    agent = await Agent.create(
        "skill-restart",
        root=tmp_path,
        model=first_provider,
        model_profile=_profile(first_provider),
        approval_handler=approve,
    )
    try:
        result = (await _execute(agent, _skill_save_call(name="persisted")))[0]
        assert not result.is_error
        assert len(approvals) == 1
        assert first_provider.requests == ()
    finally:
        await agent.close()

    reopened_provider = MockModelProvider((_stop(),))
    reopened = await Agent.open(
        "skill-restart",
        root=tmp_path,
        model=reopened_provider,
        model_profile=_profile(reopened_provider),
    )
    try:
        assert reopened_provider.requests == ()
        assert await reopened.read_skill("persisted") is not None
        expected_sha256 = await _skill_digest(reopened, "persisted")
        result = (
            await _execute(
                reopened,
                _skill_save_call(
                    name="persisted",
                    instructions="Not approved.",
                    expected_sha256=expected_sha256,
                ),
            )
        )[0]
        assert _error_code(result) == "approval_required"
        current = await reopened.read_skill("persisted")
        assert current is not None
        assert current.instructions != "Not approved."
    finally:
        await reopened.close()
