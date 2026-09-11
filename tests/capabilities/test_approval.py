"""Component-owned tests split from ``test_approval_learning_mvp.py``."""

from __future__ import annotations

from tests.support.approval_learning import (
    EAGER_LIMITS,
    MEMORY_MAX_CHARACTERS,
    MEMORY_SET_CAPABILITY_ID,
    MEMORY_SET_EXECUTOR_ID,
    MEMORY_SET_OUTPUT_KIND,
    MEMORY_SET_TOOL_NAME,
    USER_MAX_CHARACTERS,
    AccessMode,
    Agent,
    AgentEvent,
    AgentEventKind,
    ApprovalDecision,
    ApprovalRequest,
    Capability,
    FrozenInstanceError,
    FrozenJsonObject,
    MockModelProvider,
    OperationalEffect,
    SideEffectExecutor,
    ToolBatchInterruption,
    ToolCall,
    ToolExecution,
    _agent,
    _call,
    _error_code,
    _execute,
    _memory_call,
    _profile,
    _run,
    _runtime,
    _stop,
    _tool_event_kinds,
    _tool_results,
    asyncio,
    cast,
    pytest,
    threading,
    workspace_for,
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

    independent = Capability(
        id="test.capability",
        description="test",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.output",
        output_schema={"type": "object", "properties": {}},
        executor_id="test.executor",
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
    )
    assert independent.access_mode is AccessMode.NONE
    assert independent.operational_effect is OperationalEffect.CHANGE_ADVISORY_CONTEXT


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
            capability.operational_effect,
        ) == (
            MEMORY_SET_TOOL_NAME,
            MEMORY_SET_CAPABILITY_ID,
            MEMORY_SET_EXECUTOR_ID,
            MEMORY_SET_OUTPUT_KIND,
            AccessMode.NONE,
            OperationalEffect.CHANGE_ADVISORY_CONTEXT,
        )
        assert resolved == capability
        catalog = await _runtime(agent).prepare_run(_run(agent))
        names = tuple(item.view.name for item in catalog.entries)
        assert len(names) == len(set(names))
        assert {
            "artifact_create_document",
            "artifact_set_export_location",
            "memory_set",
            "skill_delete",
            "skill_save",
            "skill_view",
        } <= set(names)

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
    registry = agent._embedded._capabilities
    _, capability = registry.resolve_tool(MEMORY_SET_TOOL_NAME)
    _, executor = registry.resolve_execution(capability.id)
    original_execute = executor.execute
    original = store.replace_from_tool
    executions: list[tuple[str, str]] = []
    tool_executions: list[ToolExecution] = []

    async def capture_execution(request: ToolExecution):
        tool_executions.append(request)
        return await original_execute(request)

    async def counted(target, content, *, sensitivity):
        executions.append((target, content))
        await original(target, content, sensitivity=sensitivity)

    monkeypatch.setattr(executor, "execute", capture_execution)
    monkeypatch.setattr(store, "replace_from_tool", counted)
    try:
        content = "exact replacement 日本語"
        result = (await _execute(agent, _memory_call(content=content)))[0]
        assert not result.is_error
        assert executions == [("memory", content)]
        assert len(tool_executions) == 1
        assert tool_executions[0].call_id == "write"
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
        limits=EAGER_LIMITS,
        approval_handler=broken,
        workspace=workspace_for(tmp_path),
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
    tool_events = [
        event
        for event in events
        if event.data.get("tool_name") not in {"toolbox_search", "toolbox_load"}
    ]
    assert dict(tool_events[0].data) == {
        "call_id": "write",
        "tool_name": MEMORY_SET_TOOL_NAME,
        "capability_id": MEMORY_SET_CAPABILITY_ID,
        "toolbox_id": "knowledge",
        "load_mode": "on_demand",
        "provider_state": "loaded",
    }
    assert tool_events[1].data == tool_events[0].data
    assert dict(tool_events[2].data) == {
        "call_id": "write",
        "tool_name": MEMORY_SET_TOOL_NAME,
        "capability_id": MEMORY_SET_CAPABILITY_ID,
        "toolbox_id": "knowledge",
        "load_mode": "on_demand",
        "provider_state": "loaded",
        "outcome": "approved",
    }
    completed = tool_events[3].data
    assert completed["call_id"] == "write"
    assert completed["tool_name"] == MEMORY_SET_TOOL_NAME
    assert completed["capability_id"] == MEMORY_SET_CAPABILITY_ID
    assert completed["toolbox_id"] == "knowledge"
    assert completed["load_mode"] == "on_demand"
    assert completed["provider_state"] == "loaded"
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
        tool_events = [
            event
            for event in events
            if event.data.get("tool_name") not in {"toolbox_search", "toolbox_load"}
        ]
        assert tool_events[2].data["outcome"] == (
            "denied" if case == "denied" else "failed"
        )


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

        async def controlled_replace(target, content, *, sensitivity):
            assert {"read-done:first", "read-done:second"} <= set(actions)
            actions.append("write-start")
            await original_replace(target, content, sensitivity=sensitivity)
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

    async def controlled(target, content, *, sensitivity):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        executions.append(content)
        await asyncio.sleep(0)
        await original(target, content, sensitivity=sensitivity)
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
        outcome = await task
        assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
        assert _error_code(outcome.ordered_results[0]) == "tool_call_interrupted"
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
        outcome = await task
        assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
        assert not outcome.ordered_results[0].is_error
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
        limits=EAGER_LIMITS,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = (await _execute(reopened, _memory_call(content="not-approved")))[0]
        assert _error_code(result) == "approval_required"
        assert await reopened.read_memory() == "approved-once"
    finally:
        await reopened.close()
