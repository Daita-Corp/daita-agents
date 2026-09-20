"""Opt-in live-model acceptance for the two Phase 8 graph-effect families."""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import (
    Agent,
    GraphInspection,
    GraphState,
    LoopLimits,
    MCPToolSelection,
    create_llm_provider,
)
from daita._json import canonical_json
from daita.capabilities import (
    AccessMode,
    AutomationEligibility,
    EffectEvidenceBasis,
    EffectOutcome,
    OperationalEffect,
)
from daita.jobs.graph.models import TaskRole
from daita.llm.models import ModelProfile, ModelSensitivity, ToolResultBlock
from daita.llm.profiles import reviewed_model_profile
from daita.llm.protocols import ManagedModelProvider
from daita.loop.models import LoopExitKind
from tests.support.graph_effects_live import create_native_graph_effect_fixture
from tests.support.mcp_actions import ActionFixture
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_GRAPH_EFFECTS"
_MODEL_ID = "DAITA_GRAPH_EFFECTS_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_GRAPH_EFFECTS_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_GRAPH_EFFECTS_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})
_TERMINAL_STATES = frozenset(
    {
        GraphState.SUCCEEDED,
        GraphState.FAILED,
        GraphState.CANCELLED,
        GraphState.NEEDS_ATTENTION,
    }
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing the bounded "
            "live-model graph-effect acceptance tests"
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


def _live_provider() -> tuple[ManagedModelProvider, ModelProfile, LoopLimits]:
    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(f"{_MODEL_ID} must name an API-backed provider")
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name a reviewed tool-capable model")
    provider_key_name = f"DAITA_GRAPH_EFFECTS_LIVE_{provider_name.upper()}_API_KEY"
    api_key = os.environ.get(provider_key_name) or os.environ.get(_MODEL_KEY)
    if api_key is None or not api_key.strip():
        pytest.skip(
            f"{provider_key_name} or {_MODEL_KEY} is unavailable for the authorized smoke"
        )
    provider = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 2_048),
    )
    return (
        provider,
        profile,
        LoopLimits(
            max_steps=12,
            max_total_tokens=40_000,
            max_estimated_cost_usd=_cost_limit(),
        ),
    )


async def _diagnostic(agent: Agent, inspection: GraphInspection) -> str:
    tool_results: list[object] = []
    for attempt in inspection.attempts:
        try:
            transcript = await agent.transcript(attempt.run_id)
        except KeyError:
            continue
        tool_results.extend(
            block.output
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
    return canonical_json(
        {
            "failure_code": inspection.job.failure_code,
            "state": inspection.job.state.value,
            "tasks": [
                {
                    "id": task.task_id,
                    "role": task.role.value,
                    "state": task.state.value,
                }
                for task in inspection.tasks
            ],
            "tool_results": tool_results,
        }
    )


async def _wait_for_success(agent: Agent, job_id: str) -> GraphInspection:
    deadline = asyncio.get_running_loop().time() + 180
    while asyncio.get_running_loop().time() < deadline:
        inspection = await agent.inspect_job(job_id)
        assert inspection is not None
        if inspection.job.state in _TERMINAL_STATES:
            assert inspection.job.state is GraphState.SUCCEEDED, await _diagnostic(
                agent, inspection
            )
            return inspection
        await asyncio.sleep(0.1)
    raise AssertionError("live-model graph effect did not reach a terminal state")


def _graph_prompt(graph_input: dict[str, object]) -> str:
    return (
        "This is a wire-contract acceptance test. Load the exact tool "
        "start_graph_job, call it exactly once with the JSON object below, and do "
        "not invoke the target effect in this foreground run. After the durable "
        "graph admission succeeds, give a concise confirmation and stop. JSON: "
        + canonical_json(graph_input)
    )


async def test_live_model_completes_preview_bound_native_graph_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, profile, limits = _live_provider()
    agent: Agent | None = None
    try:
        fixture = await create_native_graph_effect_fixture(
            tmp_path,
            monkeypatch,
            model=provider,
            model_profile=profile,
            limits=limits,
        )
        agent = fixture.agent
        graph_input: dict[str, object] = {
            "objective": "Apply the exact admitted company upsert once.",
            "outcome_contract": {"kind": "native_upsert_result"},
            "deadline_seconds": 600,
            "initial_task": {
                "capability_id": "data.upsert_rows",
                "arguments": fixture.intent,
                "expected_result_contract": {"result_kind": "native_upsert_result"},
                "retained_references": {
                    "source_ids": (fixture.source_id,),
                    "resource_ids": (fixture.resource_id,),
                    "connector_binding_ids": (),
                },
                "effect_grant": {"constraints": fixture.grant_constraints},
            },
        }
        foreground = await agent.run(
            _graph_prompt(graph_input),
            source_scope_ids=(fixture.source_id,),
        )
        assert foreground.kind is LoopExitKind.COMPLETED, foreground
        jobs = await agent.list_jobs(limit=5)
        assert len(jobs) == 1, foreground
        inspection = await _wait_for_success(agent, jobs[0].job_id)
        worker = next(task for task in inspection.tasks if task.role is TaskRole.WORKER)
        result = await agent.read_task_result(jobs[0].job_id, worker.task_id)
        assert result is not None and len(result.effect_receipt_ids) == 1
        assert result.downstream_constraints["effect_completion_evidence"] == (
            "adapter_verified",
        )
        assert result.residual_risk is None
        receipts = await agent._embedded._store.list_effect_receipts(
            agent.id, run_id=result.run_id
        )
        assert len(receipts) == 1
        assert receipts[0].outcome is EffectOutcome.SUCCEEDED
        assert receipts[0].evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
        expected_domain = fixture.expected_row["domain"]
        assert isinstance(expected_domain, str)
        assert (
            fixture.database.rows[expected_domain]["name"]
            == fixture.expected_row["name"]
        )
        assert (
            len(
                [
                    entry
                    for entry in fixture.database.log
                    if entry[0] == "fetch" and entry[1].startswith("INSERT")
                ]
            )
            == 1
        )
        assert len(fixture.approvals) == 1
        assert fixture.approvals[0].capability_id == "jobs.graph.start"
    finally:
        if agent is not None:
            await agent.close()
        await provider.close()


async def test_live_model_completes_exact_grant_mcp_graph_effect(
    tmp_path: Path,
) -> None:
    provider, profile, limits = _live_provider()
    action = ActionFixture(tmp_path)
    agent: Agent | None = None
    try:
        agent = await Agent.create(
            "live-mcp-graph-effect",
            root=tmp_path,
            workspace=workspace_for(tmp_path),
            model=provider,
            model_profile=profile,
            limits=limits,
            clock=lambda: action.clock,
            mcp_client_factory=action.factory,
            approval_handler=action.approve,
        )
        status = await agent.attach_mcp_server(
            endpoint=action.server.endpoint,
            selections=(
                MCPToolSelection(
                    "notify",
                    "notify",
                    "Send the reviewed notification.",
                    access_mode=AccessMode.NONE,
                    operational_effect=OperationalEffect.EXTERNAL_ACTION,
                    automation_eligibility=AutomationEligibility.AUTOMATION_DIRECT,
                ),
            ),
            maximum_outbound_sensitivity=ModelSensitivity.INTERNAL,
        )
        binding = status.binding
        tool = next(item for item in binding.tools if item.remote_name == "notify")
        await agent.close()
        agent = None
        agent = await Agent.open(
            "live-mcp-graph-effect",
            root=tmp_path,
            workspace=workspace_for(tmp_path),
            model=provider,
            model_profile=profile,
            limits=limits,
            clock=lambda: action.clock,
            mcp_client_factory=action.factory,
            approval_handler=action.approve,
        )
        arguments = {
            "destination": "fixed-room",
            "content": "Phase 8 graph status",
            "options": {"recipient": "release-owner"},
        }
        constraints = {
            "binding_id": binding.binding_id,
            "binding_revision": binding.revision,
            "remote_tool_name": tool.remote_name,
            "fixed_arguments": arguments,
            "variable_argument_names": (),
        }
        graph_input = {
            "objective": "Invoke the exact admitted notification once.",
            "outcome_contract": {"kind": "mcp_action_result"},
            "deadline_seconds": 600,
            "initial_task": {
                "capability_id": tool.capability_id,
                "arguments": arguments,
                "expected_result_contract": {"result_kind": "mcp_action_result"},
                "retained_references": {
                    "source_ids": (),
                    "resource_ids": (),
                    "connector_binding_ids": (binding.binding_id,),
                },
                "effect_grant": {"constraints": constraints},
            },
        }
        foreground = await agent.run(_graph_prompt(graph_input))
        assert foreground.kind is LoopExitKind.COMPLETED, foreground
        jobs = await agent.list_jobs(limit=5)
        assert len(jobs) == 1, foreground
        inspection = await _wait_for_success(agent, jobs[0].job_id)
        worker = next(task for task in inspection.tasks if task.role is TaskRole.WORKER)
        result = await agent.read_task_result(jobs[0].job_id, worker.task_id)
        assert result is not None and len(result.effect_receipt_ids) == 1
        assert result.downstream_constraints["effect_completion_evidence"] == (
            "server_reported_invocation_only",
        )
        assert result.residual_risk is not None
        assert result.summary == (
            "The MCP server reported that the exact action was invoked. "
            "Downstream business completion is not verified."
        )
        assert result.provenance["summary_authority"] == ("code_owned_effect_evidence")
        assert len(action.server.calls) == 1
        assert action.server.calls[0] == ("notify", arguments)
        receipts = await agent._embedded._store.list_effect_receipts(
            agent.id, run_id=result.run_id
        )
        assert len(receipts) == 1
        assert receipts[0].outcome is EffectOutcome.SUCCEEDED
        assert receipts[0].evidence_basis is EffectEvidenceBasis.SERVER_REPORTED
        assert len(action.approvals) == 1
        assert action.approvals[0].capability_id == "jobs.graph.start"
    finally:
        if agent is not None:
            await agent.close()
        await provider.close()
