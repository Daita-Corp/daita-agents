"""Cost-bounded live smoke through the revision-2 public composition."""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import Agent, GraphState, LoopLimits, create_llm_provider
from daita.domains.data import DATA_QUERY_EVIDENCE_KIND
from daita.jobs.graph.models import AttemptState, TaskExecutionKind, TaskRole
from daita.llm.profiles import reviewed_model_profile
from daita.loop.models import LoopExitKind
from tests.support.job_benchmarks import create_probe_home
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_GRAPH_TASK"
_MODEL_ID = "DAITA_GRAPH_TASK_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_GRAPH_TASK_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_GRAPH_TASK_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one bounded "
            "live production graph smoke"
        ),
    ),
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
    raise AssertionError("live production graph did not reach a terminal state")


async def test_live_model_task_uses_public_revision_2_composition(
    tmp_path: Path,
) -> None:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(f"{_MODEL_ID} must name an API-backed provider")
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name a reviewed tool-capable model")
    api_key = os.environ.get(_MODEL_KEY)
    if api_key is None or not api_key.strip():
        pytest.skip(f"{_MODEL_KEY} is unavailable for the authorized live smoke")
    provider = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 2_048),
    )
    home = await create_probe_home(
        tmp_path,
        "live-production-graph",
        distractor_tables=3,
    )
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
            max_estimated_cost_usd=_cost_limit(),
        ),
        workspace=workspace_for(home.root),
    )
    try:
        status = await Agent.inspect_home(home.name, root=home.root)
        assert status.found_revision == status.current_revision == 2
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
        result = await agent.read_job_result(jobs[0].job_id)
        assert inspection is not None
        worker = next(task for task in inspection.tasks if task.role is TaskRole.WORKER)
        attempt = next(
            item for item in inspection.attempts if item.task_id == worker.task_id
        )
        assert worker.execution_kind is TaskExecutionKind.MODEL
        assert attempt.state is AttemptState.SUCCEEDED
        assert result is not None and result.result_kind == "graph.result_finalized"
        assert len(inspection.delivery_ids) == 1
    finally:
        await agent.close()
        await provider.close()
