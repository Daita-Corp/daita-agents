"""Component-owned tests split from ``test_kernel_contracts.py``."""

from __future__ import annotations

from tests.support.kernel import (
    NOW,
    START_DATA_PROFILE_CAPABILITY_ID,
    AgentContextBuilder,
    CanonicalMessage,
    ContextEvidencePressureExceeded,
    ContextToolProjectionAdapter,
    Mapping,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelSensitivity,
    RunInput,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    _estimate_input_tokens,
    _run,
    _SnapshotCatalog,
    cast,
    pytest,
    replace,
)


async def test_run_context_snapshot_is_prepared_once_and_aggregates_results():
    catalog = _SnapshotCatalog()
    profile = ModelProfile(
        id="mock:stage-a-context",
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    builder = AgentContextBuilder(catalog, profile=profile)
    run = RunInput(
        id="run-context-snapshot",
        agent_id="agent-stage-a",
        message="question",
        created_at=NOW,
        conversation_id="conversation-stage-a-context",
        source_scope_ids=("source-snapshot",),
    )
    user = CanonicalMessage(role=MessageRole.USER, content=(TextBlock("question"),))
    tool = ToolDefinition(
        name="lookup",
        description="Lookup one value.",
        input_schema={"type": "object", "properties": {}},
    )

    projection = ContextToolProjectionAdapter((tool,))
    tool_catalog = await projection.prepare_run(run)
    step_projection = projection.project(tool_catalog, (user,))
    snapshot = await builder.prepare(run, (user,), tool_catalog)
    assert not hasattr(builder, "build")
    first = builder.project(
        snapshot,
        (user,),
        step=1,
        tool_context=step_projection,
    )
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
    second = builder.project(
        snapshot,
        current,
        step=2,
        tool_context=step_projection,
    )

    assert catalog.context_reads == 1
    assert catalog.sensitivity_reads == 1
    assert snapshot.static_context_sha256 == (
        second.sensitivity_provenance["static_context_sha256"]
    )
    assert first.messages[0] == second.messages[0]
    assert "changed-after-prepare" not in repr(second.messages[0])
    assert "Successful completion requires a bounded, non-empty" in repr(
        first.messages[0]
    )
    assert "durable-job start receipt is a handoff" not in repr(first.messages[0])
    assert second.sensitivity is ModelSensitivity.CONFIDENTIAL
    initial_provenance = cast(
        Mapping[str, object],
        second.sensitivity_provenance["initial_sensitivity_provenance"],
    )
    assert initial_provenance["authority"] == "run_context_snapshot"
    assert initial_provenance["source_ids"] == ("source-snapshot",)
    assert initial_provenance["static_context_sha256"] == (
        snapshot.static_context_sha256
    )
    classified_results = cast(
        tuple[Mapping[str, object], ...],
        second.sensitivity_provenance["classified_results"],
    )
    assert classified_results[0]["call_id"] == "classified"
    assert "execution step limit has been reached" not in repr(first.messages[0])
    assert "execution step limit has been reached" not in repr(second.messages[0])
    assert not hasattr(snapshot, "final_static_messages")


async def test_optional_context_uses_run_allowance_without_rejecting_mandatory_input():
    catalog = _SnapshotCatalog()
    profile = ModelProfile(
        id="mock:large-window",
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        supports_tools=True,
    )
    builder = AgentContextBuilder(catalog, profile=profile)
    run = replace(_run("run-context-budget"), source_scope_ids=("source-snapshot",))
    user = run.start_message()
    projection = ContextToolProjectionAdapter(())
    tools = await projection.prepare_run(run)
    wide = await builder.prepare(run, (user,), tools, max_total_tokens=100_000)
    narrow = await builder.prepare(run, (user,), tools, max_total_tokens=1_000)
    wide_request = builder.project(
        wide, (user,), step=1, tool_context=projection.project(tools, (user,))
    )
    narrow_request = builder.project(
        narrow,
        (user,),
        step=1,
        tool_context=projection.project(tools, (user,)),
        remaining_tokens=1,
    )
    assert '"returned_count":1' in repr(wide_request.messages[0])
    assert '"returned_count":0' in repr(narrow_request.messages[0])
    assert "Remaining cumulative run allowance: 1 tokens" in repr(
        narrow_request.messages[0]
    )
    assert narrow_request.messages[-1] == user
    assert wide.initial_sensitivity == narrow.initial_sensitivity
    assert (
        wide.initial_sensitivity_provenance["source_ids"]
        == narrow.initial_sensitivity_provenance["source_ids"]
    )
    # Context shaping cannot replace provider counting with a byte-based refusal.
    assert narrow_request.max_total_tokens is None


async def test_context_owns_durable_job_handoff_guidance() -> None:
    builder = AgentContextBuilder(
        _SnapshotCatalog(),
        profile=ModelProfile(
            id="mock:durable-handoff-context",
            context_window_tokens=32_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
    )
    run = RunInput(
        id="run-durable-handoff-context",
        agent_id="agent-stage-a",
        message="Profile the current table in the background.",
        created_at=NOW,
        conversation_id="conversation-durable-handoff-context",
        source_scope_ids=("source-snapshot",),
    )
    user = run.start_message()
    start_tool = ToolDefinition(
        name="start_data_profile",
        description="Start one durable data profile.",
        input_schema={"type": "object", "properties": {}},
    )
    projection = ContextToolProjectionAdapter(
        (start_tool,),
        capability_ids=(START_DATA_PROFILE_CAPABILITY_ID,),
    )
    catalog = await projection.prepare_run(run)
    request = builder.project(
        await builder.prepare(run, (user,), catalog),
        (user,),
        step=1,
        tool_context=projection.project(catalog, (user,)),
    )

    system = request.messages[0].content[0]
    assert isinstance(system, TextBlock)
    assert "durable-job start receipt is a handoff" in system.text
    assert "do not poll, list, inspect, read, or cancel" in system.text
    assert "code-owned terminal-job run" in system.text


async def test_context_owner_rejects_cumulative_evidence_pressure_explicitly():
    builder = AgentContextBuilder(
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
    projection = ContextToolProjectionAdapter(())
    tool_catalog = await projection.prepare_run(run)
    step_projection = projection.project(tool_catalog, (user,))
    snapshot = await builder.prepare(run, (user,), tool_catalog)
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
            tool_context=step_projection,
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
