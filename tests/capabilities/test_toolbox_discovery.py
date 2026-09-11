"""Component-owned tests split from ``test_toolbox_contracts.py``."""

from __future__ import annotations

from tests.capabilities._toolbox_support import (
    Agent,
    FinishReason,
    LoopExitKind,
    LoopLimits,
    Mapping,
    MockModelProvider,
    ModelProfile,
    ModelResponse,
    ToolCall,
    ToolResultBlock,
    _append_results,
    _data,
    _error_code,
    _execute,
    _load,
    _run,
    _runtime,
    canonical_json,
    pytest,
    replace,
    sqlite3,
    workspace_for,
)


async def test_search_is_intent_only_deterministic_bounded_and_does_not_load() -> None:
    runtime, _, _, executors = _runtime()
    run = _run("run-search")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    definition = next(
        item
        for item in projection.provider_definitions
        if item.name == "toolbox_search"
    )
    properties = definition.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert set(properties) == {"query", "limit", "cursor"}
    assert definition.input_schema["required"] == ("query",)
    assert definition.input_schema["additionalProperties"] is False
    call = ToolCall(
        id="search",
        name="toolbox_search",
        arguments={
            "query": "on_demand_a",
            "limit": 1,
        },
    )
    first = (await _execute(runtime, run, projection, call)).ordered_results[0]
    second = (await _execute(runtime, run, projection, call)).ordered_results[0]
    assert first.output == second.output
    matches = _data(first)["matches"]
    assert isinstance(matches, tuple)
    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, Mapping)
    assert match["tool_name"] == "on_demand_a"
    assert match["toolbox_id"] == "artifacts"
    assert match["load_state"] == "on_demand"
    assert match["text_trust"] == "code"
    assert "domain_owner_id" not in match
    assert _data(first)["returned_count"] == 1
    assert _data(first)["truncated"] is True
    messages = _append_results((), (call,), (first,))
    assert runtime.project(catalog, messages).loaded_entries == ()
    assert first.capability_id is None
    assert first.executor_id is None
    assert all(executor.execute_calls == 0 for executor in executors)

    unavailable = ToolCall(id="unknown", name="unknown_tool")
    result = (await _execute(runtime, run, projection, unavailable)).ordered_results[0]
    assert _error_code(result) == "tool_not_available"


@pytest.mark.parametrize(
    "extra",
    (
        {"toolboxes": ["knowledge"]},
        {"data_access": "write"},
        {"operational_effect": "none"},
    ),
)
async def test_search_rejects_obsolete_filters_without_loading_or_execution(extra):
    runtime, _, _, executors = _runtime()
    run = _run("run-obsolete-filter")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    call = ToolCall(
        id="invalid-search",
        name="toolbox_search",
        arguments={"query": "effect_write", **extra},
    )
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]
    assert _error_code(result) == "toolbox_search_invalid"
    assert (
        runtime.project(catalog, _append_results((), (call,), (result,))).loaded_entries
        == ()
    )
    assert all(
        executor.execute_calls == executor.preflight_calls == 0
        for executor in executors
    )


@pytest.mark.parametrize(
    "arguments",
    (
        {"query": "bounded", "limit": 0},
        {"query": "bounded", "limit": 21},
        {"query": "bounded", "limit": True},
        {"query": ""},
        {"query": "x" * 513},
    ),
)
async def test_search_argument_bounds_are_validated_before_execution(arguments):
    runtime, _, _, executors = _runtime()
    run = _run("run-search-count")
    catalog = await runtime.prepare_run(run)
    result = (
        await _execute(
            runtime,
            run,
            runtime.project(catalog, ()),
            ToolCall(
                id="search-count",
                name="toolbox_search",
                arguments=arguments,
            ),
        )
    ).ordered_results[0]
    assert _error_code(result) == "toolbox_search_invalid"
    assert all(executor.execute_calls == 0 for executor in executors)


async def test_search_byte_bound_truncates_matches_or_fails_closed():
    runtime, _, _, _ = _runtime()
    run = _run("run-search-bytes")
    call = ToolCall(
        id="search-bytes", name="toolbox_search", arguments={"query": "bounded"}
    )
    catalog = await runtime.prepare_run(run)
    full = (
        await _execute(runtime, run, runtime.project(catalog, ()), call)
    ).ordered_results[0]
    full_data = _data(full)
    byte_limit = len(canonical_json(full_data).encode("utf-8")) - 1
    runtime, _, _, executors = _runtime(
        limits=replace(LoopLimits(), max_toolbox_search_result_bytes=byte_limit)
    )
    catalog = await runtime.prepare_run(run)
    bounded = (
        await _execute(runtime, run, runtime.project(catalog, ()), call)
    ).ordered_results[0]
    data = _data(bounded)
    assert len(canonical_json(data).encode("utf-8")) <= byte_limit
    assert data["truncated"] is True
    assert data["total_matches"] == full_data["total_matches"]
    returned = data["returned_count"]
    full_count = full_data["returned_count"]
    assert isinstance(returned, int) and isinstance(full_count, int)
    assert 0 < returned < full_count
    assert all(executor.execute_calls == 0 for executor in executors)

    runtime, _, _, executors = _runtime(
        limits=replace(LoopLimits(), max_toolbox_search_result_bytes=1)
    )
    catalog = await runtime.prepare_run(run)
    failed = (
        await _execute(runtime, run, runtime.project(catalog, ()), call)
    ).ordered_results[0]
    assert _error_code(failed) == "toolbox_search_limited"
    assert all(executor.execute_calls == 0 for executor in executors)


async def test_search_discovers_effects_across_toolboxes_but_cannot_approve_them():
    runtime, _, _, executors = _runtime()
    run = _run("run-search-effects")
    catalog = await runtime.prepare_run(run)
    call = ToolCall(
        id="search-all", name="toolbox_search", arguments={"query": "bounded"}
    )
    result = (
        await _execute(runtime, run, runtime.project(catalog, ()), call)
    ).ordered_results[0]
    matches = _data(result)["matches"]
    assert isinstance(matches, tuple)
    assert {match["toolbox_id"] for match in matches} == {
        "sources",
        "artifacts",
        "knowledge",
    }
    assert {match["tool_name"] for match in matches} == {
        entry.view.name for entry in catalog.entries
    }
    assert all(
        executor.execute_calls == executor.preflight_calls == 0
        for executor in executors
    )
    loaded, messages, projection = await _load(
        runtime,
        run,
        catalog,
        _append_results((), (call,), (result,)),
        ("effect_write",),
        call_id="load-effect",
    )
    assert not loaded.is_error
    denied = (
        await _execute(
            runtime,
            run,
            projection,
            ToolCall(id="effect", name="effect_write"),
            messages=messages,
        )
    ).ordered_results[0]
    assert _error_code(denied) == "approval_required"
    assert all(executor.execute_calls == 0 for executor in executors)


async def test_production_intent_search_distinguishes_current_competing_tools(tmp_path):
    database = tmp_path / "discovery.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sales(amount INTEGER)")
    cases = (
        ("remember metric", "semantic_save", 1),
        ("save definition", "semantic_save", 3),
        ("semantic_save", "semantic_save", 1),
        ("save semantic definition annotation catalog fields", "semantic_save", 1),
        ("knowledge memory semantic save preference definition", "semantic_save", 1),
        ("remember preference", "memory_set", 1),
        ("save reusable procedure", "skill_save", 1),
        ("create report", "artifact_create_document", 1),
        ("export findings csv", "artifact_create_tabular", 3),
        ("export all database rows csv", "data_export_tabular", 1),
        ("schedule recurring report", "routine_create", 1),
        ("set default export directory", "artifact_set_export_location", 1),
    )
    calls = tuple(
        ToolCall(
            id=f"intent-{index}", name="toolbox_search", arguments={"query": query}
        )
        for index, (query, _, _) in enumerate(cases)
    )
    provider = MockModelProvider(
        (
            ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls),
            ModelResponse(
                finish_reason=FinishReason.STOP, text="Discovery only; nothing saved."
            ),
        )
    )
    agent = await Agent.create(
        "intent-search",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
    )
    try:
        source = await agent.attach_sqlite(database)
        result = await agent.learn(
            "Remember the current sales metric definition.",
            source_scope_ids=(source.id,),
        )
        assert result.kind is LoopExitKind.COMPLETED
        transcript = await agent.transcript(result.run_id)
        blocks = {
            block.call_id: block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        for call, (query, expected, rank_bound) in zip(calls, cases, strict=True):
            block = blocks[call.id]
            assert not block.is_error, query
            matches = _data(block)["matches"]
            assert isinstance(matches, tuple)
            names = tuple(match["tool_name"] for match in matches)
            assert expected in names[:rank_bound], (query, names)
        assert await agent.list_semantic_annotations() == ()
        assert all(
            call.name == "toolbox_search"
            for message in transcript.messages
            for call in message.tool_calls
        )
    finally:
        await agent.close()
