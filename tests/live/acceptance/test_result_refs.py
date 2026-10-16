"""L8: a real model cites earlier results by their visible result_ref.

Two bounded cases on a local SQLite probe home:

* foreground: a first turn queries a table summary and shows it; a second
  turn asks for a CSV of that result, which cites it without querying again;
* graph: a durable model worker runs its read and finishes with task_complete.

Pass condition: the right result is referenced on the first try, and no call is
rejected for an invalid result ref or invalid evidence.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import pytest

from daita import Agent, GraphState, LoopLimits
from daita.capabilities import RESERVED_TOOL_NAMES
from daita.domains.data import DATA_QUERY_EVIDENCE_KIND
from daita.jobs.graph.models import AttemptState, TaskRole
from daita.llm.models import (
    ModelProfile,
    ToolCall,
    ToolResultBlock,
    tool_result_refs,
)
from daita.llm.protocols import ManagedModelProvider
from daita.loop.models import LoopExitKind, Transcript
from tests.support.job_benchmarks import create_probe_home
from tests.support.live_billing import cost_cap, live_provider
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_RESULT_REFS"
_MODEL_ID = "DAITA_RESULT_REFS_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_RESULT_REFS_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_RESULT_REFS_LIVE_MAX_COST_USD"
_REPEATS = "DAITA_RESULT_REFS_LIVE_REPEATS"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})
_REF_REJECTIONS = frozenset(
    {
        "result_ref_invalid",
        "write_evidence_invalid",
        "task_result_evidence_invalid",
        "task_result_artifact_unverified",
        "artifact_evidence_invalid",
    }
)


def _repeats() -> int:
    raw = os.environ.get(_REPEATS, "1")
    if not raw.isdigit() or not 1 <= int(raw) <= 5:
        raise ValueError(f"{_REPEATS} must be an integer from 1 to 5")
    return int(raw)


pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 after authorizing two real-model cases per "
            f"repeat; each run is capped by {_MAX_COST} (default $0.15)"
        ),
    ),
    pytest.mark.parametrize("repeat", range(_repeats())),
]


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.15")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


def _live_provider() -> tuple[ModelProfile, ManagedModelProvider]:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    if model_id.partition(":")[0] not in _API_PROVIDERS:
        pytest.fail(f"{_MODEL_ID} must name an API-backed provider")

    def api_key() -> str:
        value = os.environ.get(_MODEL_KEY)
        if value is None or not value.strip():
            pytest.skip(f"{_MODEL_KEY} is unavailable for the authorized live case")
        return value

    return live_provider(model_id, api_key=api_key, max_output_tokens=2_048)


def _error_code(block: ToolResultBlock) -> str | None:
    error = block.output.get("error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def _assert_no_ref_rejections(transcript: Transcript) -> None:
    rejected = [
        (call.name, _error_code(result))
        for call, result in transcript.tool_pairs
        if result is not None and _error_code(result) in _REF_REJECTIONS
    ]
    assert not rejected, f"result refs were rejected: {rejected}"


def _successful(
    transcript: Transcript, name: str
) -> list[tuple[int, ToolCall, ToolResultBlock]]:
    return [
        (index, call, result)
        for index, (call, result) in enumerate(transcript.tool_pairs)
        if call.name == name and result is not None and not result.is_error
    ]


async def test_l8_csv_of_a_query_result_cites_that_result(
    tmp_path: Path, repeat: int
) -> None:
    profile, provider = _live_provider()
    home = await create_probe_home(tmp_path, f"l8-csv-{repeat}", distractor_tables=3)
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=profile,
        limits=LoopLimits(
            max_steps=10,
            max_total_tokens=100_000,
            max_estimated_cost_usd=cost_cap(_cost_limit()),
        ),
        workspace=workspace_for(home.root),
    )
    try:
        # Rows are shown first; the follow-up asks for a CSV of what was shown,
        # which should cite that result rather than query or export again.
        shown = await agent.run(
            "Use one query to count the rows in every table of this source.",
            source_scope_ids=(home.source_id,),
        )
        assert shown.kind is LoopExitKind.COMPLETED
        assert shown.conversation_id is not None
        result = await agent.run(
            "Make a CSV of that result.",
            conversation_id=shown.conversation_id,
            source_scope_ids=(home.source_id,),
        )
        assert result.kind is LoopExitKind.COMPLETED
        first = await agent._embedded._store.load(shown.run_id)
        transcript = await agent._embedded._store.load(result.run_id)
        _assert_no_ref_rejections(first)
        _assert_no_ref_rejections(transcript)
        queries = _successful(first, "data_query")
        assert queries, "the counts must come from a successful query"
        assert not [
            call for call, _result in transcript.tool_pairs if call.name == "data_query"
        ], "the CSV must not query again"
        artifact_calls = [
            (call, block)
            for call, block in transcript.tool_pairs
            if call.name == "artifact_create"
        ]
        assert len(artifact_calls) == 1, "the CSV must be created on the first try"
        artifact_call, artifact_result = artifact_calls[0]
        assert artifact_result is not None and not artifact_result.is_error
        data = artifact_result.output["data"]
        assert isinstance(data, Mapping)
        cited = tuple(cast(tuple[str, ...], data["evidence_call_ids"]))
        # The visible ref the model cited resolves to the query it ran earlier.
        assert cited == (queries[-1][1].id,)
        assert all(
            not str(ref).startswith(("call_", "toolu_"))
            for ref in cast(
                tuple[object, ...], artifact_call.arguments["evidence_refs"]
            )
        )
    finally:
        await agent.close()
        await provider.close()


async def _wait_terminal(agent: Agent, job_id: str) -> None:
    deadline = asyncio.get_running_loop().time() + 120
    while asyncio.get_running_loop().time() < deadline:
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.job.state in {
            GraphState.SUCCEEDED,
            GraphState.FAILED,
            GraphState.CANCELLED,
            GraphState.NEEDS_ATTENTION,
        }:
            assert inspection.job.state is GraphState.SUCCEEDED
            return
        await asyncio.sleep(0.1)
    raise AssertionError("live graph did not reach a terminal state")


async def test_l8_graph_task_complete_cites_its_read(
    tmp_path: Path, repeat: int
) -> None:
    profile, provider = _live_provider()
    home = await create_probe_home(tmp_path, f"l8-graph-{repeat}", distractor_tables=3)
    resources = tuple(sorted(home.resource_ids.items()))
    resource_ids = tuple(resource_id for _name, resource_id in resources)
    sql = "SELECT " + ", ".join(
        f"(SELECT COUNT(*) FROM {name}) AS count_{index}"
        for index, (name, _resource_id) in enumerate(resources)
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=profile,
        limits=LoopLimits(
            max_steps=8,
            max_total_tokens=40_000,
            max_estimated_cost_usd=cost_cap(_cost_limit()),
        ),
        workspace=workspace_for(home.root),
    )
    try:
        foreground = await agent.run(
            "Start exactly one durable graph. Load start_graph_job, then call it "
            "with objective 'Run the exact admitted relational read', outcome_contract "
            f"{{'required_result_kind':'{DATA_QUERY_EVIDENCE_KIND}'}}, deadline_seconds "
            "300, and initial_task containing capability_id 'data.query', arguments "
            f"source_id={home.source_id!r}, resource_ids={resource_ids!r}, "
            f"sql={sql!r}, parameters=(). Use expected_result_contract "
            f"{{'result_kind':'{DATA_QUERY_EVIDENCE_KIND}'}} and retained_references "
            f"source_ids=({home.source_id!r},), resource_ids={resource_ids!r}, "
            "connector_binding_ids=(). After the durable receipt, stop.",
            source_scope_ids=(home.source_id,),
        )
        assert foreground.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        assert len(jobs) == 1
        await _wait_terminal(agent, jobs[0].job_id)
        inspection = await agent.inspect_job(jobs[0].job_id)
        assert inspection is not None
        worker = next(task for task in inspection.tasks if task.role is TaskRole.WORKER)
        attempts = [
            item for item in inspection.attempts if item.task_id == worker.task_id
        ]
        assert [item.state for item in attempts] == [AttemptState.SUCCEEDED]
        transcript = await agent._embedded._store.load(attempts[0].run_id)
        _assert_no_ref_rejections(transcript)
        completions = [
            call
            for call, _result in transcript.tool_pairs
            if call.name == "task_complete"
        ]
        assert len(completions) == 1, "task_complete must succeed on the first try"
        reads = _successful(transcript, "data_query")
        assert reads, "the worker must run its read"
        task_result = await agent.read_job_result(jobs[0].job_id)
        assert task_result is not None
        # The worker cited its read by visible ref; the runtime bound the call.
        refs = completions[0].arguments.get("evidence_refs")
        assert isinstance(refs, tuple) and refs, "task_complete must cite evidence"
        cited = _cited_call_ids(transcript, completions[0])
        assert cited and cited <= {call.id for _index, call, _result in reads}
    finally:
        await agent.close()
        await provider.close()


def _cited_call_ids(transcript: Transcript, completion: ToolCall) -> set[str]:
    """Resolve the completion's refs the way the model saw them numbered."""

    refs = tool_result_refs(
        transcript.messages, uncitable_tool_names=RESERVED_TOOL_NAMES
    )
    resolved = set()
    for ref in cast(tuple[str, ...], completion.arguments["evidence_refs"]):
        call_id = refs.resolve(ref)
        assert call_id is not None, f"unresolvable ref {ref!r}"
        resolved.add(call_id)
    return resolved
