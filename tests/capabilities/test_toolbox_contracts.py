"""Component-owned tests split from ``test_stage_m3_tool_catalog.py``."""

from __future__ import annotations

from tests.capabilities._toolbox_support import (
    NOW,
    TOOLBOX_DEFINITIONS,
    AccessMode,
    Agent,
    Any,
    CanonicalMessage,
    Capability,
    CapabilityDeclarations,
    CapabilityRegistry,
    LoopLimits,
    Mapping,
    MessageRole,
    MockModelProvider,
    ModelProfile,
    OperationalEffect,
    RunInput,
    TextBlock,
    ToolboxDefinition,
    ToolboxId,
    ToolCall,
    ToolCatalogLimitExceeded,
    ToolLoadMode,
    ToolManifestLimitExceeded,
    ToolPresentation,
    ToolResultBlock,
    ToolSurfaceLimitExceeded,
    ToolTextTrust,
    ToolView,
    _append_results,
    _data,
    _error_code,
    _execute,
    _Executor,
    _load,
    _presentation,
    _run,
    _runtime,
    cast,
    pytest,
    replace,
    workspace_for,
)


def test_canonical_toolbox_records_are_closed_bounded_and_exact() -> None:
    assert tuple(item.id for item in TOOLBOX_DEFINITIONS) == tuple(ToolboxId)
    assert tuple(item.label for item in TOOLBOX_DEFINITIONS) == (
        "Files",
        "Sources",
        "Artifacts",
        "Knowledge",
        "Jobs",
        "Routines",
    )
    assert len({item.id for item in TOOLBOX_DEFINITIONS}) == 6
    assert len({item.label for item in TOOLBOX_DEFINITIONS}) == 6
    assert tuple(item.value for item in ToolLoadMode) == ("pinned", "on_demand")
    assert tuple(item.value for item in ToolTextTrust) == (
        "code",
        "admitted_untrusted",
    )

    with pytest.raises(ValueError, match="bounded single-line"):
        ToolboxDefinition(ToolboxId.FILES, "x" * 65, "summary")
    with pytest.raises(ValueError, match="too many keywords"):
        ToolPresentation(
            ToolboxId.FILES,
            ToolLoadMode.PINNED,
            ToolTextTrust.CODE,
            "summary",
            "use it",
            tuple(f"key{index}" for index in range(17)),
        )
    with pytest.raises(ValueError, match="reserved runtime control"):
        ToolView(
            "toolbox_load",
            "test.reserved",
            "reserved",
            _presentation("reserved", ToolboxId.SOURCES, ToolLoadMode.PINNED),
        )
    with pytest.raises(TypeError):
        ToolView("missing", "test.missing", "missing")  # type: ignore[call-arg]


def test_registry_digest_includes_presentation_and_effects_cannot_be_pinned() -> None:
    executor = _Executor("digest")
    capability = Capability(
        id="test.toolbox.digest",
        description="Digest test.",
        input_schema={"type": "object", "properties": {}},
        output_kind="test.toolbox.digest.output",
        output_schema={"type": "object", "properties": {}},
        executor_id=executor.executor_id,
        access_mode=AccessMode.READ,
    )

    def registry(load_mode: ToolLoadMode) -> CapabilityRegistry:
        declaration = CapabilityDeclarations(
            domain_owner_id="digest",
            capabilities=(capability,),
            executor_ids=(executor.executor_id,),
            tool_views=(
                ToolView(
                    "digest_tool",
                    capability.id,
                    capability.description,
                    _presentation("digest", ToolboxId.SOURCES, load_mode),
                ),
            ),
        )
        return CapabilityRegistry(declarations=(declaration,), executors=(executor,))

    assert (
        registry(ToolLoadMode.PINNED).digest != registry(ToolLoadMode.ON_DEMAND).digest
    )
    original_registry = registry(ToolLoadMode.ON_DEMAND)
    original_view, _ = original_registry.resolve_tool("digest_tool")
    updated_view = replace(
        original_view,
        presentation=replace(
            original_view.presentation,
            summary="Find the requested operation using user intent.",
            keywords=("intent", "operation"),
        ),
    )
    updated_registry = CapabilityRegistry(
        declarations=(
            CapabilityDeclarations(
                domain_owner_id="digest",
                capabilities=(capability,),
                executor_ids=(executor.executor_id,),
                tool_views=(updated_view,),
            ),
        ),
        executors=(executor,),
    )
    assert updated_registry.digest != original_registry.digest
    assert updated_registry.contract_digest(capability.id) == (
        original_registry.contract_digest(capability.id)
    )

    effect_executor = _Executor("pinned_effect", effectful=True)
    effect_capability = replace(
        capability,
        id="test.toolbox.pinned_effect",
        executor_id=effect_executor.executor_id,
        operational_effect=OperationalEffect.CHANGE_ADVISORY_CONTEXT,
    )
    declaration = CapabilityDeclarations(
        domain_owner_id="effect",
        capabilities=(effect_capability,),
        executor_ids=(effect_executor.executor_id,),
        tool_views=(
            ToolView(
                "pinned_effect",
                effect_capability.id,
                effect_capability.description,
                _presentation(
                    "pinned_effect", ToolboxId.KNOWLEDGE, ToolLoadMode.PINNED
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="effectful tool must be on demand"):
        CapabilityRegistry(declarations=(declaration,), executors=(effect_executor,))


async def test_production_inventory_has_exact_membership_and_loading_policy(
    tmp_path,
) -> None:
    agent = await Agent.create(
        "phase1-toolbox-inventory", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    try:
        registry = agent._embedded._capabilities
        assert registry.tool_names
        expected_pinned = {
            "artifact_list",
            "artifact_read",
            "catalog_inspect",
            "catalog_schema",
            "catalog_search",
            "data_query",
            "delivery_list",
            "distribution_destination_list",
            "file_read",
            "file_search",
            "job_inspect",
            "job_list",
            "job_read_results",
            "routine_list",
            "skill_view",
        }
        for name in registry.tool_names:
            view, capability, owner = registry.resolve_tool_owner(name)
            assert view.presentation.toolbox_id in ToolboxId
            assert view.presentation.load_mode in ToolLoadMode
            assert view.presentation.text_trust is ToolTextTrust.CODE
            assert owner
            if capability.operational_effect is not OperationalEffect.NONE:
                assert view.presentation.load_mode is ToolLoadMode.ON_DEMAND
            assert (view.presentation.load_mode is ToolLoadMode.PINNED) == (
                name in expected_pinned
            )
        assert {
            name
            for name in registry.tool_names
            if registry.resolve_tool(name)[0].presentation.toolbox_id
            is ToolboxId.ARTIFACTS
        } == {
            "artifact_convert",
            "artifact_create_document",
            "artifact_create_tabular",
            "artifact_edit_text",
            "artifact_list",
            "artifact_read",
            "artifact_save_local",
            "artifact_set_export_location",
            "artifact_snapshot_result",
            "data_export_tabular",
        }
        assert {
            name
            for name in registry.tool_names
            if registry.resolve_tool(name)[0].presentation.toolbox_id is ToolboxId.FILES
        } == {"file_query", "file_read", "file_search"}
        assert registry.resolve_tool("start_data_profile")[
            0
        ].presentation.toolbox_id is (ToolboxId.JOBS)
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "first,second",
    [
        ("artifact_create_document", "routine_create"),
        ("file_query", "artifact_edit_text"),
    ],
)
async def test_contract_schema_survives_replacement_in_real_request(
    tmp_path, first, second
):
    agent = await Agent.create(
        "contract-retention",
        root=tmp_path,
        model=MockModelProvider(()),
        model_profile=ModelProfile(
            id="mock:scripted",
            context_window_tokens=64000,
            max_output_tokens=2000,
            supports_tools=True,
        ),
        workspace=workspace_for(tmp_path),
    )
    try:
        runtime = agent._embedded._capability_runtime
        builder = agent._embedded._context_builder
        assert builder is not None
        run = replace(_run(), agent_id=agent.id)
        messages: tuple[CanonicalMessage, ...] = (run.start_message(),)
        catalog = await runtime.prepare_run(run)
        snapshot = await builder.prepare(run, messages, catalog)
        call = ToolCall("inspect-first", "toolbox_inspect", {"tool_name": first})
        inspected = await _execute(
            runtime, run, runtime.project(catalog, messages), call, messages=messages
        )
        assert not inspected.ordered_results[0].is_error
        messages = _append_results(messages, (call,), inspected.ordered_results)
        _, messages, _ = await _load(
            runtime, run, catalog, messages, (first,), call_id="load-first"
        )
        _, messages, projection = await _load(
            runtime, run, catalog, messages, (second,), call_id="replace-first"
        )
        request = builder.project(snapshot, messages, step=4, tool_context=projection)
        assert first not in {tool.name for tool in request.tools}
        retained = next(
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "inspect-first"
        )
        contract = cast(Mapping[str, Any], _data(retained))["value"]
        definition = runtime._registry.tool_definition(first)
        assert contract["input_schema"] == definition.input_schema
        assert contract["complete"] is True
        assert contract["input_schema_digest"]
        assert contract["contract_digest"]
    finally:
        await agent.close()


async def test_catalog_manifest_and_initial_projection_are_exact_and_bounded() -> None:
    runtime, _, domain, _ = _runtime()
    run = _run()
    catalog = await runtime.prepare_run(run)
    assert tuple(item.toolbox_id for item in catalog.toolbox_manifest) == (
        ToolboxId.SOURCES,
        ToolboxId.ARTIFACTS,
        ToolboxId.KNOWLEDGE,
    )
    counts = {
        item.toolbox_id: (item.pinned_count, item.on_demand_count)
        for item in catalog.toolbox_manifest
    }
    assert counts == {
        ToolboxId.SOURCES: (1, 0),
        ToolboxId.ARTIFACTS: (0, 1),
        ToolboxId.KNOWLEDGE: (0, 2),
    }
    source_manifest = next(
        item
        for item in catalog.toolbox_manifest
        if item.toolbox_id is ToolboxId.SOURCES
    )
    assert source_manifest.access_modes == (AccessMode.READ,)
    assert source_manifest.operational_effects == (OperationalEffect.NONE,)
    assert "update" not in source_manifest.summary
    initial = runtime.project(catalog, ())
    assert {item.name for item in initial.provider_definitions} == {
        "pinned_read",
        "toolbox_search",
        "toolbox_load",
        "toolbox_inspect",
    }
    assert tuple(item.view.name for item in initial.callable_entries) == (
        "pinned_read",
    )
    assert initial.loaded_entries == ()
    assert initial.catalog_digest == catalog.catalog_digest
    assert initial.registry_digest == catalog.registry_digest


async def test_pinned_loaded_manifest_catalog_and_search_limits_fail_closed() -> None:
    runtime, _, _, _ = _runtime(
        limits=replace(LoopLimits(), max_run_tool_catalog_entries=1)
    )
    with pytest.raises(ToolCatalogLimitExceeded):
        await runtime.prepare_run(_run("run-catalog-limit"))

    runtime, _, _, _ = _runtime(
        limits=replace(LoopLimits(), max_toolbox_manifest_bytes=1)
    )
    with pytest.raises(ToolManifestLimitExceeded):
        await runtime.prepare_run(_run("run-manifest-limit"))

    runtime, _, _, _ = _runtime(
        limits=replace(LoopLimits(), max_pinned_tool_definition_bytes=1)
    )
    with pytest.raises(ToolSurfaceLimitExceeded):
        await runtime.prepare_run(_run("run-pinned-limit"))

    limits = replace(LoopLimits(), max_loaded_tools=1)
    runtime, _, _, _ = _runtime(limits=limits)
    run = _run("run-loaded-count-limit")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    oversized = ToolCall(
        id="oversized-count",
        name="toolbox_load",
        arguments={"tool_names": ["on_demand_a", "on_demand_b"]},
    )
    result = (await _execute(runtime, run, projection, oversized)).ordered_results[0]
    assert _error_code(result) == "toolbox_load_limit_exceeded"

    runtime, _, _, _ = _runtime(
        limits=replace(LoopLimits(), max_loaded_tool_definition_bytes=1)
    )
    run = _run("run-loaded-byte-limit")
    catalog = await runtime.prepare_run(run)
    projection = runtime.project(catalog, ())
    result = (
        await _execute(
            runtime,
            run,
            projection,
            ToolCall(
                id="oversized-bytes",
                name="toolbox_load",
                arguments={"tool_names": ["on_demand_a"]},
            ),
        )
    ).ordered_results[0]
    assert _error_code(result) == "toolbox_load_limit_exceeded"


async def test_static_context_stays_frozen_while_provider_definitions_change(
    tmp_path,
) -> None:
    profile = ModelProfile(
        id="mock:phase1-context",
        context_window_tokens=64_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    agent = await Agent.create(
        "phase1-context-projection",
        root=tmp_path,
        model=MockModelProvider((), provider_id="mock:phase1-context"),
        model_profile=profile,
        workspace=workspace_for(tmp_path),
    )
    try:
        runtime = agent._embedded._capability_runtime
        builder = agent._embedded._context_builder
        assert builder is not None
        run = RunInput(
            id="run-context-projection",
            agent_id=agent.id,
            message="exercise evolving provider definitions",
            created_at=NOW,
            conversation_id="conversation-context-projection",
        )
        start = run.start_message()
        catalog = await runtime.prepare_run(run)
        snapshot = await builder.prepare(run, (start,), catalog)
        initial = runtime.project(catalog, (start,))
        request_initial = builder.project(
            snapshot,
            (start,),
            step=1,
            tool_context=initial,
        )

        result_a, messages_a, projection_a = await _load(
            runtime,
            run,
            catalog,
            (start,),
            ("artifact_create_document",),
            call_id="context-load-a",
        )
        assert not result_a.is_error
        request_a = builder.project(
            snapshot,
            messages_a,
            step=2,
            tool_context=projection_a,
        )

        def system_text(request):
            return "\n".join(
                block.text
                for message in request.messages
                if message.role is MessageRole.SYSTEM
                for block in message.content
                if isinstance(block, TextBlock)
            )

        # This files-only prepared scope has no catalog tools. Document creation
        # is loaded, while publication remains discoverable for a later step.
        assert "catalog_schema first" not in system_text(request_initial)
        assert "catalog_inspect gives" not in system_text(request_initial)
        assert "toolbox_load artifact_save_local" in system_text(request_a)
        assert "then invoke it from the next step" in system_text(request_a)
        assert "artifact_convert only converts" not in system_text(request_a)
        replacement_name = next(
            entry.view.name
            for entry in catalog.entries
            if entry.load_mode is ToolLoadMode.ON_DEMAND
            and entry.view.name != "artifact_create_document"
        )
        result_b, messages_b, projection_b = await _load(
            runtime,
            run,
            catalog,
            messages_a,
            (replacement_name,),
            call_id="context-load-b",
        )
        assert not result_b.is_error
        request_b = builder.project(
            snapshot,
            messages_b,
            step=3,
            tool_context=projection_b,
        )

        assert snapshot.static_context_sha256
        assert snapshot.catalog_digest == catalog.catalog_digest
        assert snapshot.registry_digest == catalog.registry_digest
        assert snapshot.initial_sensitivity is request_initial.sensitivity
        assert {item.name for item in request_initial.tools} != {
            item.name for item in request_a.tools
        }
        assert "artifact_create_document" in {item.name for item in request_a.tools}
        assert "artifact_create_document" not in {item.name for item in request_b.tools}
        assert replacement_name in {item.name for item in request_b.tools}
        assert "artifact_create_document for Markdown/TXT" not in system_text(request_b)
        assert projection_a.catalog_digest == projection_b.catalog_digest
        assert projection_a.activation_digest != projection_b.activation_digest

        with pytest.raises(ValueError, match="current toolbox transcript"):
            builder.project(
                snapshot,
                messages_b,
                step=4,
                tool_context=projection_a,
            )

        with pytest.raises(ValueError, match="projection differs"):
            builder.project(
                snapshot,
                messages_b,
                step=4,
                tool_context=replace(
                    projection_b,
                    provider_definitions=projection_a.provider_definitions,
                ),
            )
    finally:
        await agent.close()
