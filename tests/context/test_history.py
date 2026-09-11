"""Component-owned tests split from ``test_conversations.py``."""

from __future__ import annotations

from tests.support.conversations import (
    _HISTORY_OMISSION_MARKER,
    _MAXIMUM_PRIOR_UTF8_BYTES,
    NOW,
    Agent,
    AgentContextBuilder,
    AgentLoop,
    CanonicalMessage,
    CatalogSpy,
    EmbeddedAgent,
    FinishReason,
    FreshQueryTools,
    InMemoryTranscriptStore,
    LoopExit,
    LoopExitKind,
    Mapping,
    MessageRole,
    MockModelProvider,
    ModelProfile,
    ModelProviderError,
    ModelResponse,
    ProviderErrorCode,
    ReplayTools,
    RunInput,
    SQLiteStateStore,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    TranscriptContext,
    _analytical_conversation_record,
    _conversation_record,
    _neutral_message,
    _prepared_request,
    _profile,
    _project_completed_history,
    _request_text,
    _simple_conversation_record,
    _stop,
    _tool_response,
    canonical_json,
    pytest,
    workspace_for,
)


async def test_follow_up_uses_history_without_copying_it_into_new_transcript(tmp_path):
    provider = MockModelProvider((_stop("first answer"), _stop("follow-up answer")))
    agent = await Agent.create(
        "integrity",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("first user sentinel")
        follow_up = await agent.run(
            "follow-up user sentinel",
            conversation_id=first.conversation_id,
        )
        transcript = await agent.transcript(follow_up.run_id)

        assert _request_text(provider.requests[1]) == (
            "first user sentinel",
            "first answer",
            "follow-up user sentinel",
        )
        assert tuple(
            block.text
            for message in transcript.messages
            for block in message.content
            if isinstance(block, TextBlock)
        ) == ("follow-up user sentinel", "follow-up answer")
    finally:
        await agent.close()


async def test_only_completed_runs_are_eligible_for_follow_up_history(tmp_path):
    provider = MockModelProvider(
        (
            _stop("eligible completed answer"),
            ModelProviderError(ProviderErrorCode.TIMEOUT),
            _stop("final answer"),
        )
    )
    agent = await Agent.create(
        "eligibility",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("eligible completed user")
        failed = await agent.run(
            "ineligible failed user",
            conversation_id=first.conversation_id,
        )
        assert failed.kind is LoopExitKind.FAILED

        await agent.run(
            "final follow-up user",
            conversation_id=first.conversation_id,
        )
        final_request_text = _request_text(provider.requests[-1])
        assert "eligible completed user" in final_request_text
        assert "eligible completed answer" in final_request_text
        assert "ineligible failed user" not in final_request_text
    finally:
        await agent.close()


async def test_historical_tools_are_rewritten_redacted_and_provider_neutral(
    tmp_path,
):
    provider = MockModelProvider(
        (
            _tool_response(1),
            _stop("first answer"),
            _tool_response(2),
            _stop("second answer"),
            _stop("third answer"),
        )
    )
    agent = Agent(
        await EmbeddedAgent.create(
            "historical-tools",
            root=tmp_path,
            model=provider,
            model_profile=_profile(provider),
            tools=ReplayTools(),
            context_builder=TranscriptContext(),
            workspace=workspace_for(tmp_path),
        )
    )
    try:
        first = await agent.run("first tool turn")
        second = await agent.run(
            "second tool turn",
            conversation_id=first.conversation_id,
        )
        await agent.run("third turn", conversation_id=first.conversation_id)

        # Provider-native continuation remains intact inside the current run.
        current_assistant = next(
            message
            for message in provider.requests[1].messages
            if message.role is MessageRole.ASSISTANT and message.tool_calls
        )
        assert current_assistant.provider_id == "mock:scripted"
        continuation = current_assistant.provider_metadata["continuation"]
        assert isinstance(continuation, Mapping)
        assert continuation["run"] == 1
        assert all(
            call.provider_call_id is not None for call in current_assistant.tool_calls
        )

        historical_request = provider.requests[-1]
        historical_assistants = [
            message
            for message in historical_request.messages
            if message.role is MessageRole.ASSISTANT and message.tool_calls
        ]
        assert len(historical_assistants) == 2
        all_historical_ids = [
            call.id for message in historical_assistants for call in message.tool_calls
        ]
        assert [
            call.name
            for message in historical_assistants
            for call in message.tool_calls
        ] == ["skill_view", "skill_view"]
        assert all(call_id.startswith("hist_") for call_id in all_historical_ids)
        assert len(all_historical_ids) == len(set(all_historical_ids))
        assert all(message.provider_id is None for message in historical_assistants)
        assert all(not message.provider_metadata for message in historical_assistants)
        assert all(
            call.provider_call_id is None
            for message in historical_assistants
            for call in message.tool_calls
        )
        assert "memory_set" not in repr(historical_assistants)
        assert "skill_save" not in repr(historical_assistants)
        assert "skill_delete" not in repr(historical_assistants)

        historical_results = [
            block
            for message in historical_request.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert {block.call_id for block in historical_results} == set(
            all_historical_ids
        )
        assert "SECRET SKILL BODY" not in repr(historical_results)
        assert "[historical skill body redacted]" in repr(historical_results)

        # Durable per-run transcripts retain their exact canonical records.
        first_transcript = await agent.transcript(first.run_id)
        second_transcript = await agent.transcript(second.run_id)
        for transcript in (first_transcript, second_transcript):
            assistant = transcript.messages[1]
            assert assistant.tool_calls[0].id == "shared-memory_set"
            assert assistant.tool_calls[0].provider_call_id == "native-memory_set"
            assert assistant.provider_metadata
            assert "SECRET" in repr(assistant.tool_calls[0].arguments)
            assert "SECRET SKILL BODY" in repr(transcript.messages)
    finally:
        await agent.close()


async def test_oversized_analytical_history_keeps_compact_contract_and_requeries():
    record, durable_messages = _analytical_conversation_record(oversized=True)
    durable_payload = canonical_json(
        [_neutral_message(message) for message in durable_messages]
    )
    assert len(durable_payload.encode("utf-8")) > _MAXIMUM_PRIOR_UTF8_BYTES

    prior = _project_completed_history((record,))
    projected_payload = canonical_json([_neutral_message(message) for message in prior])
    assert len(projected_payload.encode("utf-8")) < (
        len(durable_payload.encode("utf-8")) // 3
    )
    assert len(projected_payload.encode("utf-8")) <= _MAXIMUM_PRIOR_UTF8_BYTES
    assert record.transcript.messages == durable_messages

    rendered = repr(prior)
    for contract in (
        "captured payments",
        "all dates",
        "grouped by region",
        "paid revenue",
        "paid order count",
        "AOV",
        "tax-exclusive merchandise revenue",
        "COGS",
        "gross margin",
        "enterprise customers",
        "overall results",
    ):
        assert contract in rendered
    assert "catalog-snapshot-sentinel" not in rendered
    assert "raw-row-sentinel" not in rendered
    assert "'rows'" not in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1

    historical_calls = [
        call
        for message in prior
        if message.role is MessageRole.ASSISTANT
        for call in message.tool_calls
    ]
    historical_results = [
        block
        for message in prior
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(historical_calls) == len(historical_results) == 1
    assert historical_calls[0].name == "data_query"
    assert historical_calls[0].id.startswith("hist_")
    assert historical_calls[0].provider_call_id is None
    assert historical_results[0].call_id == historical_calls[0].id
    receipt = historical_results[0].output
    assert receipt["kind"] == "data.query_result"
    assert receipt["historical_projection"] == "continuity"
    assert receipt["state"] == "success"
    data = receipt["data"]
    assert isinstance(data, Mapping)
    assert data["columns"] == (
        "region",
        "paid_revenue",
        "paid_order_count",
        "aov",
        "cogs",
        "gross_margin",
    )
    assert data["total_rows"] == 40
    assert data["returned_rows"] == 40
    assert data["truncated"] is False
    assert data["source_revision"] == "catalog:history"
    assert "rows" not in data

    follow_up = (
        "Now restrict that analysis to enterprise customers and compare it with "
        "the overall results."
    )
    catalog = CatalogSpy()
    profile = ModelProfile(
        id="mock:enterprise-follow-up",
        context_window_tokens=60_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    builder = AgentContextBuilder(catalog, profile=profile)
    request = await _prepared_request(
        builder,
        RunInput(
            id="enterprise-follow-up",
            agent_id="agent-history",
            message=follow_up,
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (
            *prior,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(follow_up),),
            ),
        ),
        (),
        step=1,
    )
    request_rendered = repr(request.messages)
    for contract in (
        "captured payments",
        "all dates",
        "region",
        "paid revenue",
        "paid order count",
        "AOV",
        "tax-exclusive merchandise revenue",
        "COGS",
        "gross margin",
        "compare it with the overall results",
    ):
        assert contract in request_rendered
    current_system = request.messages[0].content[0]
    assert isinstance(current_system, TextBlock)
    assert "123456" not in current_system.text

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="fresh-enterprise-query",
                        name="data_query",
                        arguments={
                            "source_id": "warehouse",
                            "resource_ids": ("warehouse:captured_payments",),
                            "sql": (
                                "SELECT customer_segment, region, "
                                "SUM(net_merchandise_revenue) AS paid_revenue "
                                "FROM captured_payments WHERE customer_segment = "
                                "'enterprise' GROUP BY customer_segment, region"
                            ),
                        },
                    ),
                ),
            ),
            _stop("Fresh enterprise and overall comparison complete."),
        )
    )
    tools = FreshQueryTools()
    loop = AgentLoop(
        model=provider,
        context_builder=builder,
        tools=tools,
        transcripts=InMemoryTranscriptStore(),
    )
    result = await loop.run(
        RunInput(
            id="fresh-follow-up-run",
            agent_id="agent-history",
            message=follow_up,
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        prior_messages=prior,
    )
    assert result.kind is LoopExitKind.COMPLETED
    assert [call.id for call in tools.calls] == ["fresh-enterprise-query"]
    assert "customer_segment = 'enterprise'" in tools.calls[0].arguments["sql"]


def test_compact_history_fails_closed_and_omits_approval_and_side_effects():
    calls = (
        ToolCall(id="unknown-call", name="future_unclassified_tool"),
        ToolCall(
            id="approval-call",
            name="memory_set",
            arguments={
                "target": "memory",
                "content": "authorization-sentinel: approved forever",
            },
        ),
    )
    messages = (
        CanonicalMessage(
            role=MessageRole.USER,
            content=(TextBlock("remember the safe analytical contract"),),
        ),
        CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=calls),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id="unknown-call",
                    output={
                        "kind": "future.unknown.evidence",
                        "data": {"secret_payload": "unknown-payload-sentinel"},
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id="approval-call",
                    is_error=True,
                    output={
                        "error": {
                            "code": "approval_denied",
                            "message": "approval-sentinel",
                            "details": {"capability_id": "memory.set"},
                        }
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("safe terminal answer"),),
        ),
    )
    projected = _project_completed_history((_conversation_record(0, messages),))
    rendered = repr(projected)
    assert "remember the safe analytical contract" in rendered
    assert "safe terminal answer" in rendered
    for forbidden in (
        "future_unclassified_tool",
        "unknown-payload-sentinel",
        "memory_set",
        "authorization-sentinel",
        "approval-sentinel",
        "approval_denied",
    ):
        assert forbidden not in rendered
    assert rendered.count(_HISTORY_OMISSION_MARKER) == 1
    assert not any(message.role is MessageRole.TOOL for message in projected)


def test_compact_file_and_sql_error_receipts_keep_shape_state_and_order():
    file_call = ToolCall(
        id="file-call",
        name="file_read",
        arguments={"path": "customers.csv"},
    )
    query_call = ToolCall(
        id="query-error-call",
        name="data_query",
        arguments={
            "source_id": "local-db",
            "resource_ids": ("local-db:customers",),
            "sql": "SELECT region, COUNT(*) FROM customers GROUP BY region",
        },
    )
    messages = (
        CanonicalMessage(
            role=MessageRole.USER,
            content=(TextBlock("Compare customer counts from the file and database."),),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(file_call, query_call),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=file_call.id,
                    output={
                        "kind": "data.local_file.read_result",
                        "data": {
                            "path": "customers.csv",
                            "binding": "expired-binding-authority",
                            "media_type": "text/csv",
                            "encoding": "utf-8",
                            "content": "raw-file-row-" + ("x" * 10_000),
                            "start_offset": 0,
                            "end_offset": 10_000,
                            "cursor": None,
                            "complete": True,
                            "physical_revision": "stat:history",
                            "content_sha256": "sha256:" + ("c" * 64),
                            "limitations": [],
                        },
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.TOOL,
            content=(
                ToolResultBlock(
                    call_id=query_call.id,
                    is_error=True,
                    output={
                        "error": {
                            "code": "sql_unknown_column",
                            "message": "unbounded-error-detail-sentinel",
                            "details": {"candidates": ["region_name"]},
                        }
                    },
                ),
            ),
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("The file shape is known; retry the database query."),),
        ),
    )
    projected = _project_completed_history((_conversation_record(0, messages),))
    calls = [
        call
        for message in projected
        if message.role is MessageRole.ASSISTANT
        for call in message.tool_calls
    ]
    results = [
        block
        for message in projected
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert [call.name for call in calls] == [
        "file_read",
        "data_query",
    ]
    assert [result.call_id for result in results] == [call.id for call in calls]
    file_receipt = results[0].output
    assert file_receipt["historical_projection"] == "continuity"
    assert file_receipt["state"] == "success"
    file_data = file_receipt["data"]
    assert isinstance(file_data, Mapping)
    assert file_data["path"] == "customers.csv"
    assert file_data["media_type"] == "text/csv"
    assert file_data["complete"] is True
    assert "content" not in file_data
    assert "binding" not in file_data
    assert "cursor" not in file_data
    error_receipt = results[1].output
    assert error_receipt["kind"] == "data.query_result"
    assert error_receipt["state"] == "error"
    projected_error = error_receipt["error"]
    assert isinstance(projected_error, Mapping)
    assert projected_error["code"] == "sql_unknown_column"
    assert "unbounded-error-detail-sentinel" not in repr(projected)
    assert repr(projected).count(_HISTORY_OMISSION_MARKER) == 1


async def test_catalog_queries_keep_current_and_most_recent_prior_user_separate():
    catalog = CatalogSpy()
    builder = AgentContextBuilder(
        catalog,
        profile=ModelProfile(
            id="mock:catalog-query",
            context_window_tokens=20_000,
            max_output_tokens=1_000,
            supports_tools=True,
        ),
    )
    prior = _project_completed_history(
        (
            _simple_conversation_record(0),
            _simple_conversation_record(1),
        )
    )
    request = await _prepared_request(
        builder,
        RunInput(
            id="referential-run",
            agent_id="agent-history",
            message="Now only EMEA",
            created_at=NOW,
            conversation_id="history-conversation",
        ),
        (
            *prior,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Now only EMEA"),),
            ),
        ),
        (),
        step=1,
    )
    assert catalog.queries == [("Now only EMEA", "history user 1")]
    system_text = request.messages[0].content[0]
    assert isinstance(system_text, TextBlock)
    assert (
        "fresh source/tool evidence outrank stale historical claims" in system_text.text
    )
    assert (
        "use its resource_id directly. Reuse authenticated current structure and "
        "values where sufficient" in system_text.text
    )


async def test_inspection_keeps_nonreplayable_runs_but_history_excludes_them(tmp_path):
    initial_provider = MockModelProvider((_stop("valid answer"),))
    agent = await Agent.create(
        "inspection",
        root=tmp_path,
        model=initial_provider,
        model_profile=_profile(initial_provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        first = await agent.run("valid completed user")
        agent_id = agent.id
        state_path = agent.home / "state.db"
    finally:
        await agent.close()

    store = await SQLiteStateStore.open(state_path)
    try:
        for offset, (sentinel, kind) in enumerate(
            (
                ("failed sentinel", LoopExitKind.FAILED),
                ("interrupted sentinel", LoopExitKind.INTERRUPTED),
                ("incomplete completed sentinel", LoopExitKind.COMPLETED),
            ),
            start=1,
        ):
            run = RunInput(
                id=f"manual-{offset}",
                agent_id=agent_id,
                message=sentinel,
                created_at=NOW,
                conversation_id=first.conversation_id,
            )
            await store.start(run)
            await store.append(
                run.id,
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=(TextBlock(sentinel),),
                ),
            )
            terminal = LoopExit(
                run_id=run.id,
                conversation_id=first.conversation_id,
                kind=kind,
                reason=kind.value,
                created_at=NOW,
                final_text=(
                    "claimed complete" if kind is LoopExitKind.COMPLETED else None
                ),
            )
            if kind is LoopExitKind.COMPLETED:
                with pytest.raises(
                    ValueError,
                    match="atomic transcript completion",
                ):
                    await store.finish(terminal)
            else:
                await store.finish(terminal)
        unfinished = RunInput(
            id="manual-unfinished",
            agent_id=agent_id,
            message="unfinished sentinel",
            created_at=NOW,
            conversation_id=first.conversation_id,
        )
        await store.start(unfinished)
        await store.append(
            unfinished.id,
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(unfinished.message),),
            ),
        )
    finally:
        await store.close()

    provider = MockModelProvider((_stop("follow-up answer"),))
    reopened = await Agent.open(
        "inspection",
        root=tmp_path,
        model=provider,
        model_profile=_profile(provider),
        workspace=workspace_for(tmp_path),
    )
    try:
        await reopened.run(
            "inspection follow-up", conversation_id=first.conversation_id
        )
        request = repr(provider.requests[0].messages)
        assert "valid completed user" in request
        for sentinel in (
            "failed sentinel",
            "interrupted sentinel",
            "incomplete completed sentinel",
            "unfinished sentinel",
        ):
            assert sentinel not in request
        records = await reopened.conversation_runs(first.conversation_id)
        assert [record.turn_index for record in records] == list(range(6))
        assert records[4].result is not None
        assert records[4].result.kind is LoopExitKind.INTERRUPTED
        assert records[4].result.reason == "previous_process_terminated"
        assert records[1].result is not None
        assert records[2].result is not None
        assert records[3].result is not None
        assert records[1].result.kind is LoopExitKind.FAILED
        assert records[2].result.kind is LoopExitKind.INTERRUPTED
        assert records[3].result.kind is LoopExitKind.INTERRUPTED
        assert records[3].result.reason == "previous_process_terminated"
    finally:
        await reopened.close()


async def test_historical_schema_slice_reuse_requires_current_matching_revisions():
    resource_id = "catalog-resource:sha256:" + ("a" * 64)
    source_id = "source:sha256:" + ("b" * 64)
    revision = "sha256:" + ("c" * 64)
    sync_id = "catalog-sync-current"
    source_revision = "catalog:sha256:" + ("d" * 64)
    schema_call = ToolCall(
        id="schema-history-call",
        name="catalog_schema",
        arguments={"resource_ids": (resource_id,)},
    )
    record = _conversation_record(
        0,
        (
            CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock("Plan the paid revenue query"),),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(schema_call,),
            ),
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id=schema_call.id,
                        output={
                            "kind": "catalog.schema_slice",
                            "data": {
                                "bounds": {"resources": 12},
                                "include_relationships": True,
                                "relationships": (),
                                "resources": (
                                    {
                                        "columns": (
                                            {
                                                "name": "paid_revenue",
                                                "nullable": False,
                                                "type": "NUMERIC",
                                            },
                                        ),
                                        "kind": "table",
                                        "name": "analytics.orders",
                                        "primary_key_fields": ("order_id",),
                                        "resource_id": resource_id,
                                        "revision": revision,
                                        "source_id": source_id,
                                        "structural_facts": {},
                                        "sync_id": sync_id,
                                        "unique_key_fields": (),
                                    },
                                ),
                                "sources": (
                                    {
                                        "source_id": source_id,
                                        "source_revision": source_revision,
                                        "sync_id": sync_id,
                                    },
                                ),
                                "total_matches": 1,
                                "truncation": {"resources": False},
                                "trust_classification": "untrusted_external_data",
                            },
                        },
                    ),
                ),
            ),
            CanonicalMessage(
                role=MessageRole.ASSISTANT,
                content=(TextBlock("Use paid_revenue from analytics.orders."),),
            ),
        ),
    )
    prior = _project_completed_history((record,))
    assert "catalog.schema_slice" in repr(prior)
    assert "paid_revenue" in repr(prior)

    current_resource = {
        "kind": "table",
        "name": "orders",
        "resource_id": resource_id,
        "revision": revision,
        "sensitivity": "internal",
        "source_id": source_id,
    }
    current_source = {
        "source_id": source_id,
        "source_revision": source_revision,
        "sync_id": sync_id,
    }
    profile = ModelProfile(
        id="mock:schema-history",
        context_window_tokens=60_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    run = RunInput(
        id="schema-history-follow-up",
        agent_id="agent-history",
        message="Now only EMEA",
        created_at=NOW,
        conversation_id="history-conversation",
    )
    current_user = CanonicalMessage(
        role=MessageRole.USER,
        content=(TextBlock(run.message),),
    )

    unchanged = await _prepared_request(
        AgentContextBuilder(
            CatalogSpy((current_resource,), (current_source,)),
            profile=profile,
        ),
        run,
        (*prior, current_user),
        (),
        step=1,
    )
    unchanged_text = repr(unchanged.messages)
    assert "catalog.schema_slice" in unchanged_text
    assert "paid_revenue" in unchanged_text

    changed_resource = {
        **current_resource,
        "revision": "sha256:" + ("e" * 64),
    }
    changed_source = {
        **current_source,
        "sync_id": "catalog-sync-refreshed",
        "source_revision": "catalog:sha256:" + ("f" * 64),
    }
    changed = await _prepared_request(
        AgentContextBuilder(
            CatalogSpy((changed_resource,), (changed_source,)),
            profile=profile,
        ),
        run,
        (*prior, current_user),
        (),
        step=1,
    )
    changed_text = repr(changed.messages)
    assert "catalog.schema_slice" not in changed_text
    assert "'name', 'paid_revenue'" not in changed_text
    assert _HISTORY_OMISSION_MARKER in changed_text

    missing_source = await _prepared_request(
        AgentContextBuilder(
            CatalogSpy((current_resource,), ()),
            profile=profile,
        ),
        run,
        (*prior, current_user),
        (),
        step=1,
    )
    missing_source_text = repr(missing_source.messages)
    assert "catalog.schema_slice" not in missing_source_text
    assert _HISTORY_OMISSION_MARKER in missing_source_text
