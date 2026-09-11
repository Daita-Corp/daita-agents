"""Component-owned tests split from ``test_toolbox_contracts.py``."""

from __future__ import annotations

from tests.capabilities._toolbox_support import (
    Agent,
    CanonicalMessage,
    FinishReason,
    LoopExitKind,
    LoopLimits,
    MessageRole,
    MockModelProvider,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolboxAwareMockModelProvider,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    _append_results,
    _data,
    _error_code,
    _execute,
    _load,
    _run,
    _runtime,
    canonical_json,
    cast,
    pytest,
    replace,
    workspace_for,
)


async def test_toolbox_aware_mock_exposes_physical_and_logical_model_turns() -> None:
    desired = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id="ordinary", name="on_demand"),),
    )
    provider = ToolboxAwareMockModelProvider((desired,))
    user = CanonicalMessage(
        role=MessageRole.USER,
        content=(TextBlock("Use the on-demand tool."),),
    )
    control = ToolDefinition(
        name="toolbox_load",
        description="Load tools.",
        input_schema={"type": "object", "properties": {}},
    )
    initial = ModelRequest(messages=(user,), tools=(control,))

    load = await provider.generate(initial)

    assert tuple(call.name for call in load.tool_calls) == ("toolbox_load",)
    assert provider.requests == (initial,)
    assert provider.logical_requests == ()

    receipt = ToolResultBlock(
        call_id=load.tool_calls[0].id,
        output={
            "kind": "toolbox_load_receipt",
            "data": {"loaded_names": ["on_demand"]},
        },
    )
    projected = ModelRequest(
        messages=(
            user,
            CanonicalMessage(role=MessageRole.ASSISTANT, tool_calls=load.tool_calls),
            CanonicalMessage(role=MessageRole.TOOL, content=(receipt,)),
        ),
        tools=(
            control,
            ToolDefinition(
                name="on_demand",
                description="Run the ordinary on-demand tool.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
    )

    assert await provider.generate(projected) == desired
    assert provider.requests == (initial, projected)
    assert provider.logical_requests == (projected,)
    provider.assert_consumed()


async def test_load_is_atomic_transcript_verified_and_replaces_the_working_set() -> (
    None
):
    runtime, _, _, executors = _runtime()
    run = _run("run-load-replacement")
    catalog = await runtime.prepare_run(run)
    initial = runtime.project(catalog, ())

    mixed = ToolCall(
        id="mixed",
        name="toolbox_load",
        arguments={"tool_names": ["on_demand_a", "missing"]},
    )
    mixed_result = (await _execute(runtime, run, initial, mixed)).ordered_results[0]
    assert _error_code(mixed_result) == "toolbox_tool_not_available"
    mixed_messages = _append_results((), (mixed,), (mixed_result,))
    assert runtime.project(catalog, mixed_messages).loaded_entries == ()

    pinned = ToolCall(
        id="pinned",
        name="toolbox_load",
        arguments={"tool_names": ["pinned_read"]},
    )
    pinned_result = (await _execute(runtime, run, initial, pinned)).ordered_results[0]
    assert _error_code(pinned_result) == "toolbox_load_invalid"

    result_a, messages_a, projection_a = await _load(
        runtime,
        run,
        catalog,
        (),
        ("on_demand_a",),
        call_id="load-a",
    )
    assert not result_a.is_error
    assert tuple(item.view.name for item in projection_a.loaded_entries) == (
        "on_demand_a",
    )
    assert {item.name for item in projection_a.provider_definitions} == {
        "pinned_read",
        "on_demand_a",
        "toolbox_search",
        "toolbox_load",
        "toolbox_inspect",
    }
    assert runtime.project(catalog, messages_a) == projection_a
    receipt = _data(result_a)
    assert set(receipt) == {
        "activation_digest",
        "catalog_digest",
        "definition_bytes",
        "loaded_names",
        "run_id",
        "contracts",
    }
    assert all(
        internal_name not in canonical_json(receipt)
        for internal_name in ("domain_owner_id", "executor_id")
    )
    assert (
        cast(list[dict[str, object]], receipt["contracts"])[0]["capability_id"]
        == "test.toolbox.toolbox_test.on_demand_a"
    )

    ordinary_a = ToolCall(
        id="ordinary-a",
        name="on_demand_a",
        arguments={"value": "A"},
    )
    ordinary_result = (
        await _execute(runtime, run, projection_a, ordinary_a, messages=messages_a)
    ).ordered_results[0]
    assert not ordinary_result.is_error
    assert _data(ordinary_result)["value"] == "A"
    assert ordinary_result.capability_id == "test.toolbox.toolbox_test.on_demand_a"
    assert executors[1].execute_calls == 1

    result_b, messages_b, projection_b = await _load(
        runtime,
        run,
        catalog,
        messages_a,
        ("on_demand_b",),
        call_id="load-b",
    )
    assert not result_b.is_error
    assert tuple(item.view.name for item in projection_b.loaded_entries) == (
        "on_demand_b",
    )
    names_b = {item.name for item in projection_b.provider_definitions}
    assert "on_demand_b" in names_b
    assert "on_demand_a" not in names_b
    assert projection_b.activation_digest != projection_a.activation_digest
    assert projection_b.catalog_digest == projection_a.catalog_digest
    assert projection_b.registry_digest == projection_a.registry_digest
    with pytest.raises(ValueError, match="current toolbox transcript"):
        await _execute(runtime, run, projection_a, ordinary_a, messages=messages_b)

    new_run = _run("run-new")
    new_catalog = await runtime.prepare_run(new_run)
    assert runtime.project(new_catalog, ()).loaded_entries == ()
    assert runtime.project(new_catalog, messages_b).loaded_entries == ()


async def test_step_projection_owns_exact_currentness_validation() -> None:
    runtime, _, _, _ = _runtime()
    run = _run("run-projection-currentness")
    catalog = await runtime.prepare_run(run)
    _, messages, projection = await _load(
        runtime,
        run,
        catalog,
        (),
        ("on_demand_a",),
        call_id="load-currentness",
    )

    assert (
        projection.require_current(
            run_id=run.id,
            registry_digest=catalog.registry_digest,
            catalog_digest=catalog.catalog_digest,
            messages=messages,
        )
        is projection
    )
    mismatches: tuple[tuple[str, str, str, tuple[CanonicalMessage, ...]], ...] = (
        (
            "run-other",
            catalog.registry_digest,
            catalog.catalog_digest,
            messages,
        ),
        (
            run.id,
            "sha256:" + "0" * 64,
            catalog.catalog_digest,
            messages,
        ),
        (
            run.id,
            catalog.registry_digest,
            "sha256:" + "1" * 64,
            messages,
        ),
        (run.id, catalog.registry_digest, catalog.catalog_digest, ()),
    )
    for (
        expected_run,
        expected_registry,
        expected_catalog,
        current_messages,
    ) in mismatches:
        with pytest.raises(ValueError, match="current toolbox transcript"):
            projection.require_current(
                run_id=expected_run,
                registry_digest=expected_registry,
                catalog_digest=expected_catalog,
                messages=current_messages,
            )

    forged = replace(projection, provider_definitions=())
    with pytest.raises(ValueError, match="current toolbox transcript"):
        forged.require_current(
            run_id=run.id,
            registry_digest=catalog.registry_digest,
            catalog_digest=catalog.catalog_digest,
            messages=messages,
        )


async def test_forged_stale_and_cross_run_load_receipts_fail_closed() -> None:
    runtime, _, _, _ = _runtime()
    run = _run("run-forgery")
    catalog = await runtime.prepare_run(run)
    result, messages, loaded = await _load(
        runtime,
        run,
        catalog,
        (),
        ("on_demand_a",),
        call_id="load-forgery",
    )
    assert loaded.loaded_entries

    data = dict(_data(result))
    data["activation_digest"] = "sha256:" + "0" * 64
    forged = replace(
        result,
        output={"kind": "toolbox_load_receipt", "data": data},
    )
    forged_messages = _append_results(
        (),
        (
            ToolCall(
                id="load-forgery",
                name="toolbox_load",
                arguments={"tool_names": ["on_demand_a"]},
            ),
        ),
        (forged,),
    )
    assert runtime.project(catalog, forged_messages).loaded_entries == ()

    tampered = dict(_data(result))
    tampered["contracts"] = [
        {"capability_id": "unadmitted.effect", "requires_automation_grant": False}
    ]
    tampered_result = replace(
        result, output={"kind": "toolbox_load_receipt", "data": tampered}
    )
    tampered_messages = _append_results(
        (),
        (
            ToolCall(
                id="load-forgery",
                name="toolbox_load",
                arguments={"tool_names": ["on_demand_a"]},
            ),
        ),
        (tampered_result,),
    )
    assert runtime.project(catalog, tampered_messages).loaded_entries == ()

    stale_data = dict(_data(result))
    stale_data["definition_bytes"] = 0
    stale = replace(
        result,
        output={"kind": "toolbox_load_receipt", "data": stale_data},
    )
    stale_messages = _append_results(
        (),
        (
            ToolCall(
                id="load-forgery",
                name="toolbox_load",
                arguments={"tool_names": ["on_demand_a"]},
            ),
        ),
        (stale,),
    )
    assert runtime.project(catalog, stale_messages).loaded_entries == ()

    other_run = _run("run-other")
    other_catalog = await runtime.prepare_run(other_run)
    assert runtime.project(other_catalog, messages).loaded_entries == ()


async def test_replay_pairs_reused_call_ids_with_their_ordered_results() -> None:
    runtime, _, _, _ = _runtime()
    run = _run("run-reused-load-call-id")
    catalog = await runtime.prepare_run(run)
    _, messages_a, projection_a = await _load(
        runtime,
        run,
        catalog,
        (),
        ("on_demand_a",),
        call_id="reused-load",
    )
    _, messages_b, projection_b = await _load(
        runtime,
        run,
        catalog,
        messages_a,
        ("on_demand_b",),
        call_id="reused-load",
    )
    assert tuple(entry.view.name for entry in projection_a.loaded_entries) == (
        "on_demand_a",
    )
    assert tuple(entry.view.name for entry in projection_b.loaded_entries) == (
        "on_demand_b",
    )


async def test_only_one_load_succeeds_and_same_response_cannot_use_new_surface() -> (
    None
):
    runtime, _, _, executors = _runtime()
    run = _run("run-one-load")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    load_a = ToolCall(
        id="load-a",
        name="toolbox_load",
        arguments={"tool_names": ["on_demand_a"]},
    )
    load_b = ToolCall(
        id="load-b",
        name="toolbox_load",
        arguments={"tool_names": ["on_demand_b"]},
    )
    ordinary = ToolCall(
        id="same-response",
        name="on_demand_a",
        arguments={"value": "not-yet"},
    )
    outcome = await _execute(runtime, run, projection, load_a, load_b, ordinary)
    assert not outcome.ordered_results[0].is_error
    assert _error_code(outcome.ordered_results[1]) == "toolbox_load_invalid"
    assert _error_code(outcome.ordered_results[2]) == "tool_not_available"
    assert executors[1].execute_calls == 0


async def test_terminal_step_load_does_not_create_an_extra_model_request(
    tmp_path,
) -> None:
    profile = ModelProfile(
        id="mock:phase1-wrap-up",
        context_window_tokens=64_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="wrap-up-load",
                        name="toolbox_load",
                        arguments={"tool_names": ["artifact_create_document"]},
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The bounded run ended after preparing the requested tool.",
            ),
        ),
        provider_id=profile.id,
    )
    agent = await Agent.create(
        "phase1-toolbox-wrap-up",
        root=tmp_path,
        model=provider,
        model_profile=profile,
        limits=LoopLimits(max_steps=1),
        workspace=workspace_for(tmp_path),
    )
    try:
        result = await agent.run("Prepare a document tool.")
        assert result.kind is LoopExitKind.FAILED
        assert result.reason == "step_limit_reached"
        assert len(provider.requests) == 1
    finally:
        await agent.close()
