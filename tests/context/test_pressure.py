"""Component-owned tests split from ``test_kernel_contracts.py``."""

from __future__ import annotations

from daita.capabilities import CapabilityRegistry
from daita.capability_runtime import CapabilityRuntime
from daita.domains.analysis.domain import AnalysisCapabilityDomain
from daita.hosting.execution_governor import RunAdmissionCoordinator
from daita.llm.errors import ContextWindowExceeded
from tests.support.kernel import (
    NOW,
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
    builder = AgentContextBuilder(
        catalog, profile=profile, user_timezone=lambda: "America/Chicago"
    )
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


async def test_step_status_follows_the_transcript_and_keeps_a_stable_prefix():
    catalog = _SnapshotCatalog()
    profile = ModelProfile(
        id="mock:step-status",
        context_window_tokens=200_000,
        max_output_tokens=4_096,
        supports_tools=True,
    )
    builder = AgentContextBuilder(catalog, profile=profile)
    run = replace(_run("run-step-status"), source_scope_ids=("source-snapshot",))
    user = run.start_message()
    projection = ContextToolProjectionAdapter(())
    tools = await projection.prepare_run(run)
    snapshot = await builder.prepare(run, (user,), tools, max_total_tokens=10_000)
    first, second = (
        builder.project(
            snapshot,
            (user,),
            step=step,
            tool_context=projection.project(tools, (user,)),
            remaining_tokens=remaining,
            remaining_steps=24 - step,
        )
        for step, remaining in ((1, 10_000), (2, 7_500))
    )
    # Budget status changes per step but never rewrites the cacheable prefix.
    assert first.messages[:-1] == second.messages[:-1] == (first.messages[0], user)
    assert "Token budget left" not in repr(first.messages[0])
    for request, remaining in ((first, 10_000), (second, 7_500)):
        status = request.messages[-1]
        assert status.role is MessageRole.SYSTEM
        assert f"{remaining} tokens" in repr(status)

    unbounded = builder.project(
        snapshot, (user,), step=1, tool_context=projection.project(tools, (user,))
    )
    assert unbounded.messages[1:] == (user,)
    assert "This run has a token budget" in repr(first.messages[0])
    assert "This run has a token budget" not in repr(unbounded.messages[0])


async def test_budget_pressure_forecast_uses_measured_output_not_profile_maximum():
    catalog = _SnapshotCatalog()
    profile = ModelProfile(
        id="mock:large-output",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        supports_tools=True,
    )
    builder = AgentContextBuilder(catalog, profile=profile)
    run = replace(_run("run-forecast"), source_scope_ids=("source-snapshot",))
    user = run.start_message()
    projection = ContextToolProjectionAdapter(())
    tools = await projection.prepare_run(run)
    snapshot = await builder.prepare(run, (user,), tools, max_total_tokens=100_000)

    def status(remaining: int) -> str:
        request = builder.project(
            snapshot,
            (user,),
            step=2,
            tool_context=projection.project(tools, (user,)),
            remaining_tokens=remaining,
            previous_request_input_tokens=2_000,
            request_input_growth_tokens=500,
            previous_response_output_tokens=150,
        )
        return repr(request.messages[-1])

    # 2 x 2,000 input + 3 x 500 growth + 2 x 2,048 output floor = 9,596 tokens.
    assert "9596 tokens, allowing 2048 output" in status(98_000)
    assert "Budget pressure" not in status(98_000)
    assert "Budget pressure" in status(9_000)


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
    assert "Token budget left for this run (not a routine ceiling): 1 tokens" in repr(
        narrow_request.messages[-1]
    )
    assert narrow_request.messages[-2] == user
    assert wide.initial_sensitivity == narrow.initial_sensitivity
    assert (
        wide.initial_sensitivity_provenance["source_ids"]
        == narrow.initial_sensitivity_provenance["source_ids"]
    )
    # Context shaping cannot replace provider counting with a byte-based refusal.
    assert narrow_request.max_total_tokens is None


@pytest.mark.parametrize("available", (False, True))
@pytest.mark.parametrize(
    "message,requested_reads",
    (
        ("Simulate a numerical system.", False),
        ("Use Python and tools.batch for independent relational reads.", True),
        ("Use tools.call for the read inside Python.", True),
    ),
)
async def test_python_guidance_follows_applicable_loaded_projection(
    monkeypatch, available, message, requested_reads
):
    monkeypatch.setattr("daita.domains.analysis.domain.available", lambda: available)
    domain = AnalysisCapabilityDomain(RunAdmissionCoordinator())
    runtime = CapabilityRuntime(
        CapabilityRegistry(declarations=(domain.declarations,), executors=(domain,)),
        (domain,),
    )
    builder = AgentContextBuilder(
        _SnapshotCatalog(),
        profile=ModelProfile(
            id="mock:python-context",
            context_window_tokens=32_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
    )
    run = replace(_run("python-context"), message=message, start=None)
    messages: tuple[CanonicalMessage, ...] = (run.start_message(),)
    catalog = await runtime.prepare_run(run)
    snapshot = await builder.prepare(run, messages, catalog)
    before = builder.project(
        snapshot, messages, step=1, tool_context=runtime.project(catalog, messages)
    )
    assert "python_execute" not in {tool.name for tool in before.tools}
    assert (
        "Python workspace:" not in cast(TextBlock, before.messages[0].content[0]).text
    )
    assert (
        "Python requested reads:" in cast(TextBlock, before.messages[0].content[0]).text
    ) is (available and requested_reads)

    call = ToolCall(
        id="load-python",
        name="toolbox_load",
        arguments={"tool_names": ("python_execute",)},
    )
    outcome = await runtime.execute_all(
        run,
        (call,),
        projection=runtime.project(catalog, messages),
        messages=messages,
        sensitivity=ModelSensitivity.PUBLIC,
    )
    result = outcome.ordered_results[0]
    assert result.is_error is (not available)
    messages = (
        *messages,
        CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
        CanonicalMessage(MessageRole.TOOL, content=(result,)),
    )
    request = builder.project(
        snapshot,
        messages,
        step=2,
        tool_context=runtime.project(catalog, messages),
    )
    callable_python = "python_execute" in {tool.name for tool in request.tools}
    assert callable_python is available
    system = cast(TextBlock, request.messages[0].content[0]).text
    assert ("Python workspace:" in system) is callable_python
    assert ("Python requested reads:" in system) is (available and requested_reads)
    assert not any(
        f"Python {procedure}:" in system
        for procedure in ("CSV", "database", "query results", "charts")
    )
    assert request.messages[-len(messages) :] == messages
    assert _estimate_input_tokens(request) <= snapshot.profile.maximum_input_tokens


@pytest.mark.parametrize(
    "message,capability_id,kind,data,is_error,expected",
    (
        ("Normalize text and validate JSON records.", None, None, {}, False, set()),
        ("Compute from sales.csv.", None, None, {}, False, {"CSV"}),
        (
            "Compute from a SQLite database join.",
            None,
            None,
            {},
            False,
            {"database", "query results"},
        ),
        ("Transform query results.", None, None, {}, False, {"query results"}),
        ("Plot a histogram from retained values.", None, None, {}, False, {"charts"}),
        (
            "Compute from this input.",
            "data.query",
            "data.query_result",
            {"columns": ["aliased_total"]},
            False,
            {"database", "query results"},
        ),
        (
            "Compute from this input.",
            "data.local_file.query",
            "data.local_file.query_result",
            {"format": "csv", "columns": ["aliased_total"]},
            False,
            {"CSV", "query results"},
        ),
        (
            "Compute from this input.",
            "data.local_file.read",
            "data.local_file.read_result",
            {"path": "sales.CSV"},
            False,
            {"CSV"},
        ),
        (
            "Compute from this input.",
            "data.local_file.read",
            "data.local_file.read_result",
            {"path": "notes.txt", "content": "CSV database query chart"},
            False,
            set(),
        ),
        (
            "Compute from this input.",
            "mcp.tool",
            "data.query_result",
            {"format": "csv", "columns": ["chart"]},
            False,
            set(),
        ),
        (
            "Compute from this input.",
            "data.query",
            "data.query_result",
            {},
            True,
            set(),
        ),
    ),
)
async def test_python_procedures_use_current_task_and_typed_metadata(
    message, capability_id, kind, data, is_error, expected
):
    builder = AgentContextBuilder(
        _SnapshotCatalog(),
        profile=ModelProfile(
            id="mock:python-procedures",
            context_window_tokens=32_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
    )
    ids = ("analysis.execute", "data.query", "data.local_file.query")
    projection = ContextToolProjectionAdapter(
        tuple(
            ToolDefinition(
                name=name,
                description="Read or compute.",
                input_schema={"type": "object"},
            )
            for name in ("python_execute", "data_query", "file_query")
        ),
        capability_ids=ids,
    )
    run = replace(_run("python-procedures"), message=message, start=None)
    user = run.start_message()
    catalog = await projection.prepare_run(run)
    # Earlier tasks must not add procedures to the current computation.
    history = (
        CanonicalMessage(
            MessageRole.USER,
            content=(
                TextBlock(
                    "Chart CSV database query results using tools.batch and tools.capture."
                ),
            ),
        ),
        CanonicalMessage(MessageRole.ASSISTANT, content=(TextBlock("Prior answer."),)),
    )
    snapshot = await builder.prepare(run, (*history, user), catalog)
    messages: tuple[CanonicalMessage, ...] = (user,)
    if capability_id is not None:
        call = ToolCall(id="captured-input", name="lookup")
        messages = (
            user,
            CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
            CanonicalMessage(
                MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id=call.id,
                        capability_id=capability_id,
                        executor_id=f"{capability_id}.executor",
                        output={"kind": kind, "data": data},
                        is_error=is_error,
                    ),
                ),
            ),
        )
    request = builder.project(
        snapshot,
        messages,
        step=2,
        tool_context=projection.project(catalog, messages),
    )
    system = cast(TextBlock, request.messages[0].content[0]).text
    assert "Python requested reads:" not in system
    # Check selected procedure categories, not their exact prose or a scripted
    # model's reasoning. Selection cannot add callable tools or alter evidence.
    assert {
        procedure
        for procedure in ("CSV", "database", "query results", "charts")
        if f"Python {procedure}:" in system
    } == expected
    assert {tool.name for tool in request.tools} == {
        "python_execute",
        "data_query",
        "file_query",
        "toolbox_inspect",
    }
    assert request.messages[-len(messages) :] == messages


async def test_python_guidance_is_counted_at_preparation_and_projection():
    profile = ModelProfile(
        id="mock:python-bounds",
        context_window_tokens=32_000,
        max_output_tokens=2_000,
        supports_tools=True,
    )
    run = replace(
        _run("python-bounds"),
        message="Compute CSV and database query results and plot a chart.",
        start=None,
    )
    messages = (run.start_message(),)
    projection = ContextToolProjectionAdapter(
        (
            ToolDefinition(
                name="python_execute",
                description="Compute.",
                input_schema={"type": "object"},
            ),
            ToolDefinition(
                name="data_query", description="Read.", input_schema={"type": "object"}
            ),
        ),
        capability_ids=("analysis.execute", "data.query"),
    )
    catalog = await projection.prepare_run(run)
    tools = projection.project(catalog, messages)
    builder = AgentContextBuilder(_SnapshotCatalog(), profile=profile)
    snapshot = await builder.prepare(run, messages, catalog)
    request = builder.project(snapshot, messages, step=1, tool_context=tools)
    size = _estimate_input_tokens(request)
    assert size <= profile.maximum_input_tokens
    # A prepared projection still checks the whole request, including guidance,
    # without truncating the current transcript or silently dropping procedures.
    bounded = replace(
        snapshot,
        profile=replace(
            profile, context_window_tokens=size + profile.max_output_tokens
        ),
    )
    assert builder.project(bounded, messages, step=1, tool_context=tools) == request
    with pytest.raises(ContextWindowExceeded):
        builder.project(
            replace(
                bounded,
                profile=replace(
                    bounded.profile,
                    context_window_tokens=bounded.profile.context_window_tokens - 1,
                ),
            ),
            messages,
            step=1,
            tool_context=tools,
        )
    small_builder = AgentContextBuilder(
        _SnapshotCatalog(), profile=replace(profile, context_window_tokens=6_000)
    )
    with pytest.raises(ContextWindowExceeded):
        await small_builder.prepare(run, messages, catalog)


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


@pytest.mark.parametrize("arrow", (False, True))
@pytest.mark.parametrize(
    "message,dataset_requested",
    (
        ("Compute statistics from a complete dataset.", True),
        ("Use Python to compute over every row of the connected table.", True),
        ("Use tools.capture('data_query', args, name='numbers').", True),
        ("Normalize retained JSON records.", False),
    ),
)
async def test_dataset_guidance_requires_exact_loaded_export_metadata(
    monkeypatch, arrow, message, dataset_requested
):
    from daita.capabilities import (
        CapabilityDeclarations,
        ToolExecution,
        ToolOutput,
    )
    from daita.domains.data.capabilities import data_query_capability_declarations
    from daita.domains.data.export_capabilities import (
        exact_export_capability_declarations,
        exact_export_outputs,
    )
    from tests.support.capability_runtime import StaticTestDomain

    monkeypatch.setattr("daita.domains.analysis.domain.available", lambda: True)
    analysis = AnalysisCapabilityDomain(RunAdmissionCoordinator())
    exports = exact_export_capability_declarations(
        adapter_ids=("sqlite", "postgresql"), allow_arrow=arrow
    )
    query = data_query_capability_declarations(
        file_outputs=exact_export_outputs(allow_arrow=arrow),
        file_capability_id=exports.capabilities[0].id,
    )
    declarations = CapabilityDeclarations(
        domain_owner_id="data",
        capabilities=(*query.capabilities, *exports.capabilities),
        executor_ids=(*query.executor_ids, *exports.executor_ids),
        tool_views=query.tool_views,
    )
    data = StaticTestDomain(
        declarations.capabilities, declarations.tool_views, domain_owner_id="data"
    )

    class Export:
        def __init__(self, executor_id: str) -> None:
            self.executor_id = executor_id

        async def execute(self, request: ToolExecution) -> ToolOutput:
            raise AssertionError("Guidance and loading cannot execute a query")

    runtime = CapabilityRuntime(
        CapabilityRegistry(
            declarations=(analysis.declarations, declarations),
            executors=(
                analysis,
                *(Export(item) for item in declarations.executor_ids),
            ),
        ),
        (analysis, data),
    )
    builder = AgentContextBuilder(
        _SnapshotCatalog(),
        profile=ModelProfile(
            id="mock:typed-dataset-context",
            context_window_tokens=32000,
            max_output_tokens=2000,
            supports_tools=True,
        ),
    )
    run = replace(_run("typed-dataset"), message=message, start=None)
    messages: tuple[CanonicalMessage, ...] = (run.start_message(),)
    catalog = await runtime.prepare_run(run)
    snapshot = await builder.prepare(run, messages, catalog)
    # data_query is pinned, so loading python_execute makes the export callable.
    for names in (("python_execute",),):
        call = ToolCall(
            id=f"load-{len(names)}",
            name="toolbox_load",
            arguments={"tool_names": names},
        )
        result = (
            await runtime.execute_all(
                run,
                (call,),
                projection=runtime.project(catalog, messages),
                messages=messages,
                sensitivity=ModelSensitivity.PUBLIC,
            )
        ).ordered_results[0]
        assert not result.is_error
        messages = (
            *messages,
            CanonicalMessage(MessageRole.ASSISTANT, tool_calls=(call,)),
            CanonicalMessage(MessageRole.TOOL, content=(result,)),
        )
        request = builder.project(
            snapshot,
            messages,
            step=len(names) + 1,
            tool_context=runtime.project(catalog, messages),
        )
        text = cast(TextBlock, request.messages[0].content[0]).text
        assert "tools.batch" in text and "in submission order" in text
        assert ("tools.capture(" in text) is (arrow and dataset_requested)
        assert request.messages[-len(messages) :] == messages
        assert _estimate_input_tokens(request) <= snapshot.profile.maximum_input_tokens
