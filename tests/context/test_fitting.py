"""Component-owned tests split from ``test_conversations.py``."""

from __future__ import annotations

from tests.support.conversations import (
    _HISTORY_OMISSION_MARKER,
    _MAXIMUM_PRIOR_UTF8_BYTES,
    NOW,
    Agent,
    AgentContextBuilder,
    CanonicalMessage,
    CatalogSpy,
    LoopExitKind,
    Mapping,
    MessageRole,
    MockModelProvider,
    ModelProfile,
    ModelRequest,
    RunInput,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    _analytical_conversation_record,
    _conversation_record,
    _neutral_message,
    _prepared_request,
    _profile,
    _project_completed_history,
    _request_text,
    _simple_conversation_record,
    _stop,
    canonical_json,
    inspect,
    pytest,
    workspace_for,
)


def test_context_builder_exposes_only_fixed_absolute_history_bounds():
    assert "retain_messages" not in inspect.signature(AgentContextBuilder).parameters


def test_small_useful_query_turn_can_use_full_projection():
    record, _ = _analytical_conversation_record(oversized=False)
    projected = _project_completed_history((record,))
    results = [
        block
        for message in projected
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(results) == 1
    assert results[0].output["historical_projection"] == "full"
    data = results[0].output["data"]
    assert isinstance(data, Mapping)
    assert data["rows"][0]["region"] == "region-0"
    assert "catalog-snapshot-sentinel" not in repr(projected)


async def test_whole_request_budget_downgrades_full_turn_before_dropping_it():
    record, _ = _analytical_conversation_record(oversized=False)
    prior = _project_completed_history((record,))
    assert "'historical_projection', 'full'" in repr(prior)
    builder = AgentContextBuilder(
        CatalogSpy(),
        profile=ModelProfile(
            id="mock:full-downgrade",
            context_window_tokens=10_000,
            max_output_tokens=1_000,
            supports_tools=True,
        ),
    )
    current = "current-run-message-must-remain-complete"
    request = await _prepared_request(
        builder,
        RunInput(
            id="full-downgrade-run",
            agent_id="agent-history",
            message=current,
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (
            *prior,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(current),),
            ),
        ),
        (),
        step=1,
    )
    rendered = repr(request.messages)
    assert "'historical_projection', 'continuity'" in rendered
    assert "'historical_projection', 'full'" not in rendered
    assert "Analyze captured payments" in rendered
    assert "current-run-message-must-remain-complete" in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1


def test_oversized_terminal_answer_gets_deterministic_edge_projection():
    answer = "answer-beginning-sentinel-" + ("x" * 40_000) + "-answer-ending-sentinel"
    projected = _project_completed_history(
        (_simple_conversation_record(0, answer=answer),)
    )
    rendered = repr(projected)
    assert "history user 0" in rendered
    assert "answer-beginning-sentinel" in rendered
    assert "answer-ending-sentinel" in rendered
    assert f"original UTF-8 bytes: {len(answer.encode('utf-8'))}" in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1
    assert (
        len(
            canonical_json([_neutral_message(message) for message in projected]).encode(
                "utf-8"
            )
        )
        <= _MAXIMUM_PRIOR_UTF8_BYTES
    )


def test_completed_history_hard_bounds_keep_newest_whole_turns():
    ten_runs = tuple(_simple_conversation_record(index) for index in range(10))
    run_bounded = _project_completed_history(ten_runs)
    run_text = _request_text(ModelRequest(messages=run_bounded))
    assert "history user 0" not in run_text
    assert "history user 1" not in run_text
    assert "history user 2" in run_text
    assert "history user 9" in run_text
    run_marker = run_bounded[0].content[0]
    assert isinstance(run_marker, TextBlock)
    assert run_marker.text == _HISTORY_OMISSION_MARKER

    message_heavy = []
    for index in range(8):
        calls = tuple(
            ToolCall(id=f"{index}-{item}", name="lookup") for item in range(4)
        )
        messages = [
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(f"message-heavy user {index}"),),
            ),
            CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=calls),
            *(
                CanonicalMessage(
                    role=MessageRole.TOOL,
                    content=(ToolResultBlock(call_id=call.id, output={"ok": True}),),
                )
                for call in calls
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock(f"message-heavy answer {index}"),),
            ),
        ]
        message_heavy.append(_conversation_record(index, tuple(messages)))
    message_bounded = _project_completed_history(tuple(message_heavy))
    # Unknown future evidence fails closed, so all eight bounded user/answer
    # continuity pairs fit and the tool payloads are represented by one marker.
    assert len(message_bounded) == 17
    assert "message-heavy user 0" in repr(message_bounded)
    assert "message-heavy user 7" in repr(message_bounded)
    assert "name='lookup'" not in repr(message_bounded)

    byte_heavy = tuple(
        _simple_conversation_record(index, answer=str(index) + ("x" * 12_500))
        for index in range(2)
    )
    byte_bounded = _project_completed_history(byte_heavy)
    assert "history user 0" not in repr(byte_bounded)
    assert "history user 1" in repr(byte_bounded)
    byte_marker = byte_bounded[0].content[0]
    assert isinstance(byte_marker, TextBlock)
    assert byte_marker.text == _HISTORY_OMISSION_MARKER


async def test_final_budget_omits_oldest_turns_and_preserves_current_exchange():
    catalog = CatalogSpy()
    profile = ModelProfile(
        id="mock:budget",
        context_window_tokens=9_500,
        max_output_tokens=1_000,
        supports_tools=True,
    )
    builder = AgentContextBuilder(catalog, profile=profile)
    prior = _project_completed_history(
        tuple(
            _simple_conversation_record(index, answer=f"answer-{index}-" + "x" * 1_500)
            for index in range(3)
        )
    )
    current_call = ToolCall(id="current-call", name="lookup")
    current = (
        CanonicalMessage(
            role=MessageRole.USER,
            content=(TextBlock("current user sentinel"),),
        ),
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(current_call,)),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(ToolResultBlock(call_id=current_call.id, output={"value": 1}),),
        ),
    )
    request = await _prepared_request(
        builder,
        RunInput(
            id="budget-run",
            agent_id="agent-history",
            message="current user sentinel",
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (*prior, *current),
        (),
        step=2,
    )
    rendered = repr(request.messages)
    assert "history user 0" not in rendered
    assert "history user 2" in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1
    assert "current user sentinel" in rendered
    assert "current-call" in rendered
    roles = [message.role for message in request.messages]
    assert roles[-3:] == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]


async def test_context_fitting_retains_exact_current_anchor_and_updates_counts():
    source_id = "source:exact-current"
    resources = (
        {
            "kind": "table",
            "match_reasons": ("resource_name_exact_mention",),
            "name": "exact_current_target",
            "resource_id": "resource:exact-current",
            "revision": "sha256:" + ("a" * 64),
            "sensitivity": "internal",
            "source_id": source_id,
        },
        {
            "kind": "table",
            "match_reasons": ("resource_name_exact_mention",),
            "name": "prior_target_" + ("p" * 30_000),
            "resource_id": "resource:prior",
            "revision": "sha256:" + ("b" * 64),
            "sensitivity": "internal",
            "source_id": source_id,
        },
        {
            "kind": "table",
            "match_reasons": ("resource_name_contains",),
            "name": "broad_candidate_" + ("b" * 30_000),
            "resource_id": "resource:broad",
            "revision": "sha256:" + ("c" * 64),
            "sensitivity": "internal",
            "source_id": source_id,
        },
    )
    sources = (
        {
            "source_id": source_id,
            "source_revision": "catalog:sha256:" + ("d" * 64),
            "sync_id": "catalog-sync-current",
        },
    )
    catalog = CatalogSpy(resources, sources)
    builder = AgentContextBuilder(
        catalog,
        profile=ModelProfile(
            id="mock:catalog-fitting",
            context_window_tokens=20_000,
            max_output_tokens=1_000,
            supports_tools=True,
        ),
    )
    run = RunInput(
        id="catalog-fitting-run",
        agent_id="agent-history",
        message="Profile exact_current_target",
        created_at=NOW,
        conversation_id="catalog-fitting-conversation",
    )
    request = await _prepared_request(
        builder,
        run,
        (
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(run.message),),
            ),
        ),
        (),
        step=1,
    )
    system = request.messages[0].content[0]
    assert isinstance(system, TextBlock)
    assert "exact_current_target" in system.text
    assert "prior_target_" not in system.text
    assert "broad_candidate_" not in system.text
    assert '"returned_count":1' in system.text
    assert '"total_matches":3' in system.text
    assert '"truncated":true' in system.text


@pytest.mark.parametrize(
    "answer",
    ["Prior useful answer. " * 40, "前の回答。" * 50],
    ids=["ascii", "multibyte"],
)
async def test_discovery_pressure_preserves_newest_continuity_and_tool_exchange(answer):
    from daita.context import _estimate_input_tokens

    source_id = "source:budget"
    resources = tuple(
        {
            "kind": "table",
            "name": f"candidate_{index:02d}_" + "r" * 200,
            "resource_id": f"resource:{index}",
            "source_id": source_id,
            "revision": "sha256:" + "a" * 64,
            "sensitivity": "public",
            "match_reasons": ("unmatched_fallback",),
        }
        for index in range(24)
    )
    profile = ModelProfile(
        id="mock:discovery-budget",
        context_window_tokens=14000,
        max_output_tokens=1000,
        supports_tools=True,
    )
    builder = AgentContextBuilder(
        CatalogSpy(resources), profile=profile, catalog_limit=24
    )
    prior = _project_completed_history((_simple_conversation_record(7, answer=answer),))
    run = RunInput(
        id="run-discovery-budget",
        agent_id="agent-history",
        message="Continue with those findings.",
        created_at=NOW,
    )
    call = ToolCall(
        "current-query", "data_query", {"source_id": source_id, "sql": "SELECT 1"}
    )
    current = (
        run.start_message(),
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=call.id,
                    output={"rows": [{"value": "result" * 80}]},
                ),
            ),
        ),
    )
    tools = (
        ToolDefinition(
            name="data_query", description="Read data", input_schema={"type": "object"}
        ),
    )
    request = await _prepared_request(builder, run, (*prior, *current), tools, step=2)
    assert "history user 7" in repr(request.messages)
    assert answer in repr(request.messages)
    assert request.messages[-3:] == current
    assert _estimate_input_tokens(request) <= profile.maximum_input_tokens
    assert '"truncated":true' in repr(request.messages)


async def test_context_overflow_fails_before_provider_and_is_not_replayed(tmp_path):
    provider = MockModelProvider((_stop("must not run"),))
    tiny_profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=500,
        supports_tools=True,
    )
    agent = await Agent.create(
        "overflow",
        root=tmp_path,
        model=provider,
        model_profile=tiny_profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        failed = await agent.run("overflow sentinel " + ("x" * 80_000))
        assert failed.kind is LoopExitKind.FAILED
        assert failed.reason == "context_window_exceeded"
        assert provider.requests == ()
        records = await agent.conversation_runs(failed.conversation_id)
        assert records[0].result == failed
    finally:
        await agent.close()

    resumed_provider = MockModelProvider((_stop("recovered"),))
    reopened = await Agent.open(
        "overflow",
        root=tmp_path,
        model=resumed_provider,
        model_profile=_profile(resumed_provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        await reopened.run("clean follow-up", conversation_id=failed.conversation_id)
        assert "overflow sentinel" not in repr(resumed_provider.requests[0].messages)
    finally:
        await reopened.close()
