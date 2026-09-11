"""Component-owned tests split from ``test_approval.py``."""

from __future__ import annotations

from tests.support.approval_learning import (
    EAGER_LIMITS,
    MEMORY_SET_TOOL_NAME,
    SKILL_DELETE_CAPABILITY_ID,
    SKILL_DELETE_EXECUTOR_ID,
    SKILL_DELETE_OUTPUT_KIND,
    SKILL_DELETE_TOOL_NAME,
    SKILL_SAVE_CAPABILITY_ID,
    SKILL_SAVE_EXECUTOR_ID,
    SKILL_SAVE_OUTPUT_KIND,
    SKILL_SAVE_TOOL_NAME,
    AccessMode,
    Agent,
    AgentEvent,
    AgentEventKind,
    ApprovalDecision,
    ApprovalRequest,
    FrozenJsonObject,
    Mapping,
    MockModelProvider,
    ModelSensitivity,
    OperationalEffect,
    SideEffectExecutor,
    ToolBatchInterruption,
    ToolCall,
    ToolExecution,
    _agent,
    _call,
    _error_code,
    _execute,
    _profile,
    _run,
    _runtime,
    _skill_delete_call,
    _skill_digest,
    _skill_save_call,
    _stop,
    _system_text,
    _tool_event_kinds,
    _tool_results,
    asyncio,
    cast,
    pytest,
    threading,
    workspace_for,
)


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
        limits=EAGER_LIMITS,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Save this as a reusable workflow for future runs.")
        assert result.final_text == "I saved the workflow."
        skill = await agent.read_skill("monthly-revenue")
        assert skill is not None
        assert skill.instructions == "Use paid invoice date. Exclude voided invoices."
        assert len(approvals) == 1
        assert dict(approvals[0].arguments) == dict(call.arguments)
        assert len(provider.logical_requests) == 2
        assert "- monthly-revenue: Calculate monthly revenue consistently.\n" not in (
            _system_text(provider.logical_requests[1])
        )
        first, second = provider.logical_requests
        assert (
            first.sensitivity_provenance["static_context_sha256"]
            == second.sensitivity_provenance["static_context_sha256"]
        )
        assert first.tools == second.tools
        assert (
            f"Remaining model requests including this one: {EAGER_LIMITS.max_steps - provider.requests.index(first)}."
            in _system_text(first)
        )
        assert (
            f"Remaining model requests including this one: {EAGER_LIMITS.max_steps - provider.requests.index(second)}."
            in _system_text(second)
        )
        transcript = await agent.transcript(result.run_id)
        assert second.messages[1:] == transcript.messages[:-1]
        tool_result = _tool_results(provider)[0]
        assert tool_result.output["data"] == FrozenJsonObject.from_mapping(
            {"name": "monthly-revenue", "changed": True}
        )
        prompt = _system_text(provider.logical_requests[0])
        assert "SKILL.md=procedures" in prompt
        assert "Replace, do not duplicate" in prompt
    finally:
        await agent.close()


async def test_learning_tool_descriptions_route_documents_and_require_write_first(
    tmp_path,
):
    agent = await _agent(tmp_path, "learning-tool-routing")
    try:
        runtime = _runtime(agent)
        catalog = await runtime.prepare_run(_run(agent))
        definitions = {
            entry.view.name: runtime._registry.tool_definition(entry.view.name)
            for entry in catalog.entries
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
    bootstrap = await Agent.create(
        "replace-not-duplicate", root=tmp_path, workspace=workspace_for(tmp_path)
    )
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
        limits=EAGER_LIMITS,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
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
            assert capability.access_mode is AccessMode.NONE
            assert (
                capability.operational_effect
                is OperationalEffect.CHANGE_ADVISORY_CONTEXT
            )
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
            call_id=request.call_id,
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
            call_id=replacement_call.id,
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
        assert (
            await agent.save_skill(
                arguments["name"],
                arguments["description"],
                arguments["instructions"],
                sensitivity=ModelSensitivity.INTERNAL,
            )
            is True
        )
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

    async def observed_save(name, description, instructions, *, sensitivity):
        assert runtime._mutation_lock is agent._embedded._mutation_lock
        assert runtime._mutation_lock is store._mutation_lock
        assert runtime._mutation_lock.locked()
        actions.append("save")
        return await original_save(
            name, description, instructions, sensitivity=sensitivity
        )

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
        outcome = await task
        assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
        assert _error_code(outcome.ordered_results[0]) == "tool_call_interrupted"
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
        outcome = await task
        assert outcome.interruption_kind is ToolBatchInterruption.CANCELLED
        assert not outcome.ordered_results[0].is_error
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
        limits=EAGER_LIMITS,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
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
        limits=EAGER_LIMITS,
        workspace=workspace_for(tmp_path),
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
