"""Execution facts and recovery use the prepared, currently applicable surface."""

from dataclasses import replace
from typing import cast

import pytest

from daita.capabilities import CapabilityRegistry, ExecutionAdmissionPolicy
from daita.capability_runtime import CapabilityRuntime
from daita.llm.models import ToolCall
from tests.capabilities._toolbox_support import (
    _data,
    _execute,
    _run,
    _runtime,
)
from tests.support.capability_runtime import StaticTestDomain


@pytest.mark.parametrize("size", [4, 8, 38])
async def test_target_admission_reports_available_recovery_without_dispatch(size):
    _, _, original, executors = _runtime()
    capability = replace(
        original.declarations.capabilities[0],
        input_schema={
            "type": "object",
            "properties": {
                "resource_ids": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["resource_ids"],
        },
        execution_admission_policy=ExecutionAdmissionPolicy(
            "read", True, True, "resource_ids", 4, 16
        ),
    )
    domain = StaticTestDomain(
        (capability, *original.declarations.capabilities[1:]),
        original.declarations.tool_views,
        domain_owner_id=original.domain_owner_id,
    )
    registry = CapabilityRegistry(
        declarations=(domain.declarations,), executors=executors
    )
    runtime = CapabilityRuntime(registry, (domain,))
    run = _run()
    projection = runtime.project(await runtime.prepare_run(run), ())
    call = ToolCall(
        "query",
        original.declarations.tool_views[0].name,
        {"resource_ids": tuple(str(i) for i in range(size))},
    )
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]
    if size == 4:
        assert not result.is_error and executors[0].execute_calls == 1
    else:
        assert result.is_error and executors[0].execute_calls == 0
        details = result.output["error"]["details"]
        assert details["execution_disposition"] == "not_dispatched"
        assert details["foreground_max_targets"] == 4
        assert details["durable_available"] is False
        assert details["recovery"]["kind"] == "reduce_scope"
        assert "start_graph_job" not in str(result.output)
    search = ToolCall("search", "toolbox_search", {"query": call.name})
    metadata = _data(
        (await _execute(runtime, run, projection, search)).ordered_results[0]
    )
    limits = metadata["matches"][0]["execution_limits"]  # type: ignore[index]
    assert limits["inline_max_targets"] == 4 and limits["graph_max_targets"] == 16
    assert "inline_max_targets" in registry.tool_definition(call.name).description


@pytest.mark.parametrize("state", ["inapplicable", "revoked", "on_demand", "callable"])
async def test_durable_recovery_intersects_preparation_with_current_applicability(
    monkeypatch, state
):
    from daita.capabilities import ToolLoadMode

    _, _, original, executors = _runtime()
    capability = replace(
        original.declarations.capabilities[0],
        input_schema={
            "type": "object",
            "properties": {
                "resource_ids": {"type": "array", "items": {"type": "string"}}
            },
        },
        execution_admission_policy=ExecutionAdmissionPolicy(
            "read", True, True, "resource_ids", 4, 16
        ),
    )
    views = list(original.declarations.tool_views)
    views[2] = replace(
        views[2],
        name="start_graph_job",
        presentation=replace(
            views[2].presentation,
            load_mode=(
                ToolLoadMode.PINNED if state == "callable" else ToolLoadMode.ON_DEMAND
            ),
        ),
    )
    domain = StaticTestDomain(
        (capability, *original.declarations.capabilities[1:]),
        tuple(views),
        domain_owner_id=original.domain_owner_id,
    )
    enabled = state != "inapplicable"

    async def project(run):
        return tuple(v.name for v in views if enabled or v.name != "start_graph_job")

    monkeypatch.setattr(domain, "project", project)
    registry = CapabilityRegistry(
        declarations=(domain.declarations,), executors=executors
    )
    runtime = CapabilityRuntime(registry, (domain,))
    run = _run()
    projection = runtime.project(await runtime.prepare_run(run), ())
    if state == "revoked":
        enabled = False
    call = ToolCall(
        "oversized", views[0].name, {"resource_ids": tuple(str(i) for i in range(8))}
    )
    result = (await _execute(runtime, run, projection, call)).ordered_results[0]
    details = result.output["error"]["details"]
    available = state in {"on_demand", "callable"}
    assert details["durable_available"] is available
    assert executors[0].execute_calls == 0
    if available:
        assert details["recovery"]["tool_name"] == "start_graph_job"
        assert details["recovery"]["inspection_tool"] == "toolbox_inspect"
        assert details["recovery"]["load_tool"] == (
            "toolbox_load" if state == "on_demand" else None
        )
    else:
        assert details["recovery"]["kind"] == "reduce_scope"
        assert "start_graph_job" not in str(result.output)
    search = (
        await _execute(
            runtime,
            run,
            projection,
            ToolCall("current-search", "toolbox_search", {"query": "start_graph_job"}),
        )
    ).ordered_results[0]
    assert _data(search)["exact_tool_available"] is available


async def test_search_and_inspection_share_limits_and_mark_exact_absence():
    from tests.support.capability_runtime import execute_projected

    runtime, _, _, _ = _runtime()
    run = _run()
    result = (
        await execute_projected(
            runtime,
            run,
            (ToolCall("search", "toolbox_search", {"query": "start_graph_job"}),),
        )
    ).ordered_results[0]
    data = _data(result)
    assert data["exact_tool_available"] is False
    assert data["exact_tool_name"] == "start_graph_job"
    assert all(
        m["match_status"] == "unmatched_fallback"
        for m in cast(tuple[dict[str, object], ...], data["matches"])
    )
    # Source-supplied lookalike fields cannot create trusted progress guidance.
    from daita.llm.models import ToolResultBlock
    from daita.loop.session import RunSessionEvidence

    tracker = RunSessionEvidence()
    call = ToolCall("search", "toolbox_search", {"query": "start_graph_job"})
    tracker.observe_progress(call, result)
    tracker.observe_progress(call, result)
    assert len(tracker.progress_warnings) == 1
    untrusted = ToolResultBlock(
        "source",
        output={"kind": "data.query_result", "data": {"exact_tool_available": False}},
    )
    isolated = RunSessionEvidence()
    isolated.observe_progress(call, untrusted)
    isolated.observe_progress(call, untrusted)
    assert isolated.progress_warnings == ()
