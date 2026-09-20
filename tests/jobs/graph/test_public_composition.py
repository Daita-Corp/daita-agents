"""Production-composition proof for the current durable graph entry point."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from daita import Agent, GraphInspection, GraphState, LoopLimits
from daita._json import canonical_json
from daita.domains.data import (
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_EVIDENCE_KIND,
    DATA_QUERY_TOOL_NAME,
)
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind
from tests.support.job_benchmarks import create_probe_home
from tests.support.workspace import workspace_for

pytestmark = [pytest.mark.integration, pytest.mark.acceptance]


class _ProductionGraphProvider(MockModelProvider):
    """Route deterministic answers by the public request's authenticated shape."""

    def __init__(
        self,
        *,
        source_id: str,
        resources: tuple[tuple[str, str], ...],
    ) -> None:
        super().__init__((), provider_id="mock:production-graph", complete_pricing=True)
        self._source_id = source_id
        self._resource_ids = tuple(resource_id for _name, resource_id in resources)
        self._sql = "SELECT " + ", ".join(
            f"(SELECT COUNT(*) FROM {name}) AS count_{index}"
            for index, (name, _resource_id) in enumerate(resources)
        )
        self._captured: list[ModelRequest] = []

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._captured)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self._captured.append(request)
        tool_names = {item.name for item in request.tools}
        result_call_ids = {
            block.call_id
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        usage = ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0")))
        if "task_complete" in tool_names:
            if "graph-data-query" not in result_call_ids:
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="graph-data-query",
                            name=DATA_QUERY_TOOL_NAME,
                            arguments={
                                "source_id": self._source_id,
                                "resource_ids": self._resource_ids,
                                "sql": self._sql,
                                "parameters": (),
                            },
                        ),
                    ),
                    usage=usage,
                )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="graph-task-complete",
                        name="task_complete",
                        arguments={
                            "result_kind": DATA_QUERY_EVIDENCE_KIND,
                            "summary": "The exact bounded query completed.",
                            "payload": {"queried_resource_count": 5},
                            "evidence_call_ids": ("graph-data-query",),
                            "artifact_ids": (),
                            "residual_risk": None,
                            "downstream_constraints": {},
                        },
                    ),
                ),
                usage=usage,
            )
        if "start_graph_job" in tool_names:
            if "public-start-graph" in result_call_ids:
                return ModelResponse(
                    finish_reason=FinishReason.STOP,
                    text="The durable graph was admitted.",
                    usage=usage,
                )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="public-start-graph",
                        name="start_graph_job",
                        arguments={
                            "objective": "Count the rows in the exact admitted table.",
                            "outcome_contract": {
                                "required_result_kind": DATA_QUERY_EVIDENCE_KIND
                            },
                            "deadline_seconds": 300,
                            "initial_task": {
                                "capability_id": DATA_QUERY_CAPABILITY_ID,
                                "arguments": {
                                    "source_id": self._source_id,
                                    "resource_ids": self._resource_ids,
                                    "sql": self._sql,
                                    "parameters": (),
                                },
                                "expected_result_contract": {
                                    "result_kind": DATA_QUERY_EVIDENCE_KIND
                                },
                                "retained_references": {
                                    "source_ids": (self._source_id,),
                                    "resource_ids": self._resource_ids,
                                    "connector_binding_ids": (),
                                },
                            },
                        },
                    ),
                ),
                usage=usage,
            )
        assert "toolbox_load" in tool_names
        return ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="public-load-graph",
                    name="toolbox_load",
                    arguments={"tool_names": ("start_graph_job",)},
                ),
            ),
            usage=usage,
        )


async def _wait_terminal(agent: Agent, job_id: str) -> GraphInspection:
    deadline = asyncio.get_running_loop().time() + 10
    while asyncio.get_running_loop().time() < deadline:
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.job.state in {
            GraphState.SUCCEEDED,
            GraphState.FAILED,
            GraphState.CANCELLED,
            GraphState.NEEDS_ATTENTION,
        }:
            return inspection
        await asyncio.sleep(0.01)
    raise AssertionError("production graph did not reach a terminal state")


async def test_public_agent_executes_graph_model_task_in_revision_2_home(
    tmp_path: Path,
) -> None:
    home = await create_probe_home(
        tmp_path, "public-production-graph", distractor_tables=3
    )
    resources = tuple(sorted(home.resource_ids.items()))
    provider = _ProductionGraphProvider(
        source_id=home.source_id,
        resources=resources,
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        workspace=workspace_for(home.root),
    )
    try:
        status = await Agent.inspect_home(home.name, root=home.root)
        assert status.found_revision == status.current_revision == 2
        foreground = await agent.run(
            "Start a durable graph query over all five exact resources.",
            source_scope_ids=(home.source_id,),
        )
        assert foreground.kind is LoopExitKind.COMPLETED
        jobs = await agent.list_jobs()
        transcript = await agent.transcript(foreground.run_id)
        outputs = [
            dict(block.output)
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert len(jobs) == 1, canonical_json(outputs)
        inspection = await _wait_terminal(agent, jobs[0].job_id)
        result = await agent.read_job_result(jobs[0].job_id)
        diagnostic: list[dict[str, object]] = []
        for attempt in inspection.attempts:
            try:
                attempt_transcript = await agent.transcript(attempt.run_id)
            except KeyError:
                continue
            diagnostic.extend(
                dict(block.output)
                for message in attempt_transcript.messages
                for block in message.content
                if isinstance(block, ToolResultBlock)
            )
        assert inspection.job.state is GraphState.SUCCEEDED, canonical_json(
            {
                "diagnostic": diagnostic,
                "tasks": [
                    task.specification.digest_material() for task in inspection.tasks
                ],
            }
        )
        assert inspection.job.origin_run_id == foreground.run_id
        assert len(inspection.delivery_ids) == 1
        assert result is not None
        assert result.result_kind == "graph.result_finalized"
        assert result.payload["accepted_result_count"] == 1
        assert any(
            item.result_kind == DATA_QUERY_EVIDENCE_KIND for item in inspection.results
        )
        assert any(
            "task_complete" in {item.name for item in request.tools}
            for request in provider.requests
        )
    finally:
        await agent.close()
