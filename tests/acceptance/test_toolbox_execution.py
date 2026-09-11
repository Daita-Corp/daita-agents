"""Component-owned tests split from ``test_toolbox_contracts.py``."""

from __future__ import annotations

from tests.capabilities._toolbox_support import (
    ApprovalDecision,
    CapabilityRegistry,
    CapabilityRuntime,
    OperationalEffect,
    ToolboxId,
    ToolCall,
    ToolLoadMode,
    ToolTextTrust,
    _data,
    _declaration,
    _execute,
    _load,
    _run,
)


async def test_offline_cross_domain_conformance_matrix_preserves_dispatch_and_approval() -> (
    None
):
    source_domain, source_executors = _declaration(
        "source_owner",
        (
            (
                "source_read",
                ToolboxId.SOURCES,
                ToolLoadMode.PINNED,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
            (
                "artifact_build",
                ToolboxId.ARTIFACTS,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
        ),
    )
    knowledge_domain, knowledge_executors = _declaration(
        "knowledge_owner",
        (
            (
                "knowledge_read",
                ToolboxId.KNOWLEDGE,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
            (
                "knowledge_write",
                ToolboxId.KNOWLEDGE,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.CHANGE_ADVISORY_CONTEXT,
                ToolTextTrust.CODE,
            ),
        ),
    )
    job_domain, job_executors = _declaration(
        "job_owner",
        (
            (
                "job_list_matrix",
                ToolboxId.JOBS,
                ToolLoadMode.PINNED,
                OperationalEffect.NONE,
                ToolTextTrust.CODE,
            ),
            (
                "job_cancel_matrix",
                ToolboxId.JOBS,
                ToolLoadMode.ON_DEMAND,
                OperationalEffect.CANCEL_JOB,
                ToolTextTrust.CODE,
            ),
        ),
    )
    executors = (*source_executors, *knowledge_executors, *job_executors)
    registry = CapabilityRegistry(
        declarations=(
            source_domain.declarations,
            knowledge_domain.declarations,
            job_domain.declarations,
        ),
        executors=executors,
    )
    approvals: list[str] = []

    async def approve(request) -> ApprovalDecision:
        approvals.append(request.tool_name)
        return ApprovalDecision.APPROVE

    runtime = CapabilityRuntime(
        registry,
        (source_domain, knowledge_domain, job_domain),
        approval_handler=approve,
    )
    run = _run("run-cross-domain")
    catalog = await runtime.prepare_run(run)
    assert tuple(item.toolbox_id for item in catalog.toolbox_manifest) == (
        ToolboxId.SOURCES,
        ToolboxId.ARTIFACTS,
        ToolboxId.KNOWLEDGE,
        ToolboxId.JOBS,
    )
    assert {
        entry.toolbox_id
        for entry in catalog.entries
        if entry.domain_owner_id == "source_owner"
    } == {
        ToolboxId.SOURCES,
        ToolboxId.ARTIFACTS,
    }
    assert {
        entry.domain_owner_id
        for entry in catalog.entries
        if entry.toolbox_id is ToolboxId.KNOWLEDGE
    } == {"knowledge_owner"}

    search = ToolCall(
        id="matrix-search",
        name="toolbox_search",
        arguments={"query": "knowledge write"},
    )
    search_result = (
        await _execute(runtime, run, runtime.project(catalog, ()), search)
    ).ordered_results[0]
    matches = _data(search_result)["matches"]
    assert isinstance(matches, tuple)
    assert matches
    assert all("domain_owner_id" not in item for item in matches)

    load_result, messages, projection = await _load(
        runtime,
        run,
        catalog,
        (),
        ("artifact_build", "knowledge_read", "knowledge_write"),
        call_id="matrix-load",
    )
    assert not load_result.is_error
    calls = (
        ToolCall(id="artifact", name="artifact_build", arguments={"value": "a"}),
        ToolCall(id="knowledge", name="knowledge_read", arguments={"value": "k"}),
        ToolCall(id="write", name="knowledge_write", arguments={"value": "w"}),
    )
    outcome = await _execute(runtime, run, projection, *calls, messages=messages)
    assert all(not result.is_error for result in outcome.ordered_results)
    assert approvals == ["knowledge_write"]
    assert source_executors[1].execute_calls == 1
    assert knowledge_executors[0].execute_calls == 1
    assert knowledge_executors[1].execute_calls == 1
    assert knowledge_executors[1].preflight_calls == 2
    assert messages
