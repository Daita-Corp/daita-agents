"""Production-composition proof for the current durable graph entry point."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from daita import (
    Agent,
    AgentConfig,
    GraphInspection,
    GraphState,
    HostCapacityLimits,
    LoopLimits,
)
from daita._json import canonical_json
from daita.domains.data import (
    DATA_QUERY_CAPABILITY_ID,
    DATA_QUERY_EVIDENCE_KIND,
    DATA_QUERY_TOOL_NAME,
)
from daita.jobs.graph.models import TaskRole
from daita.llm.models import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind
from daita.storage.home_migrations import CURRENT_HOME_REVISION
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
        count_operation: bool = False,
        selection_reference: bool = False,
        premature_completion: bool = False,
        child_reference_error: bool = False,
    ) -> None:
        super().__init__((), provider_id="mock:production-graph", complete_pricing=True)
        self._source_id = source_id
        self._resource_ids = tuple(resource_id for _name, resource_id in resources)
        self._sql = "SELECT " + ", ".join(
            f"(SELECT COUNT(*) FROM {name}) AS count_{index}"
            for index, (name, _resource_id) in enumerate(resources)
        )
        self._captured: list[ModelRequest] = []
        self._selection_reference = selection_reference
        self._premature_completion = premature_completion
        self._child_reference_error = child_reference_error
        self._capability_id = (
            "data.count_selection" if count_operation else DATA_QUERY_CAPABILITY_ID
        )
        self._tool_name = "data_count_plan" if count_operation else DATA_QUERY_TOOL_NAME
        self._result_kind = (
            "data.count_operation" if count_operation else DATA_QUERY_EVIDENCE_KIND
        )
        self._arguments = {
            "source_id": self._source_id,
            "resource_ids": self._resource_ids,
        }
        if not count_operation:
            self._arguments.update(sql=self._sql, parameters=())

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
            if self._tool_name not in tool_names:
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "graph-load-operation",
                            "toolbox_load",
                            {"tool_names": (self._tool_name,)},
                        ),
                    ),
                    usage=usage,
                )
            if (
                self._premature_completion
                and "premature-complete" not in result_call_ids
            ):
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "premature-complete",
                            "task_complete",
                            {
                                "result_kind": self._result_kind,
                                "summary": "Unsupported complete count claim.",
                                "payload": {"total_rows": 1331},
                            },
                        ),
                    ),
                    usage=usage,
                )
            if "premature-complete" in result_call_ids:
                rejected = next(
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "premature-complete"
                )
                assert (
                    rejected.is_error
                    and cast(Any, rejected.output)["error"]["code"]
                    == "task_result_evidence_missing"
                )
            if "graph-data-query" not in result_call_ids:
                arguments = self._arguments
                if self._capability_id == "data.count_selection":
                    context_text = next(
                        block.text
                        for message in request.messages
                        for block in message.content
                        if isinstance(block, TextBlock)
                        and "<code_owned_task_context>" in block.text
                    )
                    material = json.loads(
                        context_text.split("<code_owned_task_context>", 1)[1].split(
                            "</code_owned_task_context>", 1
                        )[0]
                    )
                    arguments = material["task_specification"][
                        "expected_result_contract"
                    ]["initial_call"]["arguments"]
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            id="graph-data-query",
                            name=self._tool_name,
                            arguments=arguments,
                        ),
                    ),
                    usage=usage,
                )
            failed = next(
                (
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "graph-data-query"
                    and block.is_error
                ),
                None,
            )
            if failed is not None:
                details = cast(Any, failed.output)["error"]["details"]
                assert details["failure_codes"] == ("catalog_source_stale",)
                assert details["counted_resources"] == 4
                assert details["failed_resources"] == 34
                assert details["total_rows"] is None
                recovery = details["recovery"]
                assert recovery["tool_name"] in tool_names
                assert recovery["arguments"]["kind"] == "source_or_contract_drift"
                if "partial-complete" not in result_call_ids:
                    return ModelResponse(
                        finish_reason=FinishReason.TOOL_CALLS,
                        tool_calls=(
                            ToolCall(
                                "partial-complete",
                                "task_complete",
                                {
                                    "result_kind": self._result_kind,
                                    "summary": "Unsupported complete count claim.",
                                    "payload": {"total_rows": 1331},
                                },
                            ),
                        ),
                        usage=usage,
                    )
                rejected = next(
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "partial-complete"
                )
                assert (
                    rejected.is_error
                    and cast(Any, rejected.output)["error"]["code"]
                    == "task_result_evidence_missing"
                )
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "graph-block-drift", "task_block", recovery["arguments"]
                        ),
                    ),
                    usage=usage,
                )
            if (
                self._child_reference_error
                and "child-reference-complete" not in result_call_ids
            ):
                counted = next(
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "graph-data-query"
                )
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            "child-reference-complete",
                            "task_complete",
                            {
                                "result_kind": self._result_kind,
                                "summary": "Exact count with invalid child references.",
                                "payload": {},
                                "evidence_call_ids": cast(Any, counted.output)["data"][
                                    "evidence_call_ids"
                                ],
                            },
                        ),
                    ),
                    usage=usage,
                )
            if "child-reference-complete" in result_call_ids:
                rejected = next(
                    block
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "child-reference-complete"
                )
                assert rejected.is_error
                assert (
                    cast(Any, rejected.output)["error"]["code"]
                    == "task_result_evidence_invalid"
                )
                assert (
                    cast(Any, rejected.output)["error"]["details"]["recovery"][
                        "tool_name"
                    ]
                    == "task_complete"
                )
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="graph-task-complete",
                        name="task_complete",
                        arguments={
                            "result_kind": self._result_kind,
                            "summary": "The exact bounded query completed.",
                            "payload": {
                                "queried_resource_count": len(self._resource_ids)
                            },
                            **(
                                {}
                                if self._capability_id == "data.count_selection"
                                else {"evidence_call_ids": ("graph-data-query",)}
                            ),
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
            initial_arguments = self._arguments
            references = {
                "source_ids": (self._source_id,),
                "resource_ids": self._resource_ids,
                "connector_binding_ids": (),
            }
            if self._selection_reference:
                selection = next(
                    cast(Any, block.output["data"])["selection"]["count_selection_ref"]
                    for message in request.messages
                    for block in message.content
                    if isinstance(block, ToolResultBlock)
                    and block.call_id == "graph-catalog-selection"
                )
                initial_arguments = {"selection_ref": selection}
                references = {}
            return ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="public-start-graph",
                        name="start_graph_job",
                        arguments={
                            "objective": "Count the rows in the exact admitted table.",
                            "outcome_contract": {
                                "required_result_kind": self._result_kind
                            },
                            "deadline_seconds": 300,
                            "initial_task": {
                                "capability_id": self._capability_id,
                                "arguments": initial_arguments,
                                "expected_result_contract": {
                                    "result_kind": self._result_kind
                                },
                                "retained_references": references,
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
                *(
                    (
                        ToolCall(
                            "graph-catalog-selection",
                            "catalog_schema",
                            {
                                "source_id": self._source_id,
                                "detail": "identity",
                                "limit": 50,
                            },
                        ),
                    )
                    if self._selection_reference
                    else ()
                ),
            ),
            usage=usage,
        )


async def _wait_terminal(agent: Agent, job_id: str) -> GraphInspection:
    # The expanded fixture exercises ten child reads and repeated contract
    # validation. Avoid saturating its state store with full inspection polls.
    deadline = asyncio.get_running_loop().time() + 30
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
        await asyncio.sleep(0.05)
    assert inspection is not None
    raise AssertionError(
        f"production graph did not reach a terminal state: {inspection.job.state}; "
        f"attempts={[a.state for a in inspection.attempts]}"
    )


@pytest.mark.parametrize(
    "count_operation,selection_reference,native_drift,premature_completion,child_reference_error",
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (True, True, False, False, False),
        (True, True, True, False, False),
        (True, True, False, True, False),
        (True, True, False, False, True),
    ],
)
async def test_public_agent_executes_graph_model_task_in_current_home(
    tmp_path: Path,
    count_operation: bool,
    selection_reference: bool,
    native_drift: bool,
    premature_completion: bool,
    child_reference_error: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.adapters.sqlite_query import SQLiteQueryBackend

    reads = []
    original_read = SQLiteQueryBackend.execute_read

    async def read(self, **kwargs):
        import sqlite3

        result = await original_read(self, **kwargs)
        reads.append(kwargs)
        if native_drift and len(reads) == 1:
            with sqlite3.connect(home.database) as connection:
                connection.execute(
                    'ALTER TABLE "'
                    + resources[0][0].replace('"', '""')
                    + '" ADD COLUMN native_drift INTEGER'
                )
        return result

    if native_drift:
        monkeypatch.setattr(SQLiteQueryBackend, "execute_read", read)
    home = await create_probe_home(
        tmp_path,
        "public-production-graph",
        distractor_tables=36 if count_operation else 3,
    )
    resources = tuple(sorted(home.resource_ids.items()))
    provider = _ProductionGraphProvider(
        source_id=home.source_id,
        resources=resources,
        count_operation=count_operation,
        selection_reference=selection_reference,
        premature_completion=premature_completion,
        child_reference_error=child_reference_error,
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
        assert status.found_revision == status.current_revision == CURRENT_HOME_REVISION
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
        if native_drift:
            worker = next(t for t in inspection.tasks if t.role is TaskRole.WORKER)
            assert inspection.job.state is GraphState.NEEDS_ATTENTION, canonical_json(
                diagnostic
            )
            assert not any(r.task_id == worker.task_id for r in inspection.results)
            attempt = next(
                a for a in inspection.attempts if a.task_id == worker.task_id
            )
            children = [
                r
                for r in await agent.analysis_evidence(attempt.run_id)
                if r.kind == "child"
            ]
            assert (
                len(children) == 10
                and sum(r.facts["status"] == "succeeded" for r in children) == 1
            )
            assert len(reads) == 1
            assert inspection.controls[-1].kind.value == "source_or_contract_drift"
            return
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
            item.result_kind == provider._result_kind for item in inspection.results
        )
        if count_operation:
            work = next(
                task for task in inspection.tasks if task.role.value == "worker"
            )
            assert set(work.specification.authority.resource_ids) == set(
                home.resource_ids.values()
            )
            assert {"data.count_selection", "data.count_rows"} <= set(
                work.specification.authority.capability_ids
            )
            attempt = next(a for a in inspection.attempts if a.task_id == work.task_id)
            accepted = next(r for r in inspection.results if r.task_id == work.task_id)
            assert accepted.verification["evidence_call_ids"] == ("graph-data-query",)
            evidence = await agent.analysis_evidence(attempt.run_id)
            children = [r for r in evidence if r.kind == "child"]
            assert len(children) == 10
            attempt_transcript = await agent.transcript(attempt.run_id)
            call, output = next(
                (c, r)
                for c, r in attempt_transcript.tool_pairs
                if c.name == "data_count_plan"
            )
            assert set(call.arguments) == {"plan_ref"}
            assert output is not None and not output.is_error
            data = cast(Mapping[str, object], output.output["data"])
            assert data["counted_resources"] == data["selected_resources"] == 38
            assert data["catalog_enumeration_complete"] is selection_reference
            assert all(
                r.facts["status"] == "succeeded"
                and len(
                    cast(
                        tuple[str, ...],
                        cast(Mapping[str, object], r.facts["arguments"])[
                            "resource_ids"
                        ],
                    )
                )
                <= 4
                for r in children
            )
        assert any(
            "task_complete" in {item.name for item in request.tools}
            for request in provider.requests
        )
    finally:
        await agent.close()


async def test_configured_graph_capacity_overlaps_four_model_attempts_and_preserves_root_limits(
    tmp_path,
):
    home = await create_probe_home(
        tmp_path, "parallel-public-graphs", distractor_tables=3
    )
    resources = tuple(sorted(home.resource_ids.items()))
    ready = asyncio.Event()
    release = asyncio.Event()

    class ParallelProvider(_ProductionGraphProvider):
        active = 0
        peak = 0

        async def generate(self, request):
            if "task_complete" in {tool.name for tool in request.tools}:
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == 4:
                    ready.set()
                try:
                    await release.wait()
                    return await super().generate(request)
                finally:
                    self.active -= 1
            return await super().generate(request)

    provider = ParallelProvider(source_id=home.source_id, resources=resources)
    config = AgentConfig(
        host_capacity=HostCapacityLimits(
            execution_capacity=8, provider_capacity=5, graph_parallelism=4
        )
    )
    agent = await Agent.open(
        home.name,
        root=home.root,
        model=provider,
        model_profile=provider.model_profile,
        config=config,
        limits=LoopLimits(max_estimated_cost_usd=Decimal("1")),
        workspace=workspace_for(home.root),
    )
    try:
        for _ in range(4):
            result = await agent.run(
                "Start a durable query graph.", source_scope_ids=(home.source_id,)
            )
            assert result.kind is LoopExitKind.COMPLETED
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert provider.active == 4
        jobs = await agent.list_jobs()
        assert len(jobs) == 4
        for job in jobs:
            inspection = await agent.inspect_job(job.job_id)
            assert inspection is not None
            active = [
                attempt
                for attempt in inspection.attempts
                if attempt.state.value in {"claimed", "running"}
            ]
            assert len(active) == 1
            assert len(active) <= inspection.job.specification.limits.max_parallelism
        release.set()
        terminals = await asyncio.gather(
            *(_wait_terminal(agent, job.job_id) for job in jobs)
        )
        assert all(
            inspection.job.state is GraphState.SUCCEEDED for inspection in terminals
        )
        assert provider.peak == 4
    finally:
        release.set()
        await agent.close()
    assert (
        agent.admission_diagnostics().active_leases
        == agent.admission_diagnostics().active_permits
        == 0
    )
