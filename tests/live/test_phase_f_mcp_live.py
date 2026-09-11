"""Real-model Phase F effectiveness; MCP effects stay inside MockTransport.

Ten cases, at most twelve AgentLoop runs per model/repetition. No model
responses or routine proposals are scripted in this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from _phase_f_live_support import (
    AUTHORIZATION,
    CONTENT,
    COST_ENV,
    DESTINATION,
    NEXT_SLOT,
    RESEARCH_TOKEN,
    SOURCE,
    assert_action,
    assert_completed,
    evaluate,
    live_provider,
    model_ids,
    repeats,
    report_path,
)

from daita.capabilities import EffectEvidenceBasis
from daita.distribution.models import OutcomeState
from daita.llm.models import ToolResultBlock
from live.benchmarks._support import (
    RunCapture,
    assert_on_demand_invocation,
    results_for,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(AUTHORIZATION) != "1",
        reason=(
            f"set {AUTHORIZATION}=1 only after authorizing up to twelve live "
            f"runs per model/repetition, each capped by {COST_ENV}; MCP is simulated"
        ),
    ),
]


@pytest.fixture(params=model_ids())
def model_id(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=range(repeats()))
def repetition(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def evidence_path(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> Path:
    path = report_path(request.node.nodeid).resolve()
    record_property("phase_f_evidence", str(path))
    return path


@pytest.mark.parametrize("tool_count", [2, 32, 128])
@pytest.mark.parametrize("phrasing", [0, 1])
async def test_live_action_discovery_and_catalog_scale(
    tmp_path: Path,
    model_id: str,
    repetition: int,
    tool_count: int,
    phrasing: int,
    evidence_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    profile, provider = live_provider(model_id)
    async with evaluate(
        tmp_path,
        provider,
        profile,
        evidence_path,
        tool_count=tool_count,
        case_id=request.node.nodeid,
    ) as scenario:
        result, transcript = await scenario.run(scenario.action_prompt(phrasing))
        assert_completed(result, transcript)
        assert result.usage.total_tokens > 0
        capture = RunCapture(result, transcript, tuple(scenario.provider.requests))
        assert_on_demand_invocation(capture, scenario.action.local_name)
        await assert_action(scenario)
        assert [
            call for call in scenario.server.calls if call[0] == "notify_release"
        ] == [("notify_release", {"destination": DESTINATION, "content": CONTENT})]
        assert all(
            name == "notify_release" for name, _ in scenario.server.calls
        ), "The direct-notification task made an unnecessary connector call."
        assert (
            len(scenario.approvals) == 1 and scenario.approvals[0]["approved"] is True
        )
        scenario.check_answer(result, status="server_reported", research=False)


@pytest.mark.parametrize("fault", ["success", "tool_error", "disconnect"])
async def test_live_research_and_honest_action_evidence(
    tmp_path: Path,
    model_id: str,
    repetition: int,
    fault: str,
    evidence_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    profile, provider = live_provider(model_id)
    async with evaluate(
        tmp_path,
        provider,
        profile,
        evidence_path,
        fault=fault,
        case_id=request.node.nodeid,
    ) as scenario:
        result, transcript = await scenario.run(scenario.research_prompt())
        assert_completed(result, transcript)
        assert result.usage.total_tokens > 0
        await assert_action(scenario, uncertain=fault != "success")
        names = [name for name, _ in scenario.server.calls]
        assert names.index("research_release") < names.index("notify_release")
        content = next(
            args["content"]
            for name, args in scenario.server.calls
            if name == "notify_release"
        )
        assert (
            isinstance(content, str) and SOURCE in content and RESEARCH_TOKEN in content
        )
        action_attempts = [
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.name == scenario.action.local_name
        ]
        # Dispatch safety and model restraint are distinct checks. A runtime-blocked
        # model retry still fails the effectiveness evaluation and remains visible.
        assert len(action_attempts) == 1, action_attempts
        assert any(
            isinstance(block, ToolResultBlock)
            and not block.is_error
            and SOURCE in str(block.output)
            for message in transcript.messages
            for block in message.content
        )
        scenario.check_answer(
            result,
            status="server_reported" if fault == "success" else "uncertain",
            research=True,
        )
        if fault != "success":
            assert len(await scenario.agent.list_effects(unresolved_only=True)) == 1


async def test_live_model_authors_grant_and_runs_immediate_and_weekly_occurrences(
    tmp_path: Path,
    model_id: str,
    repetition: int,
    evidence_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    profile, provider = live_provider(model_id)
    async with evaluate(
        tmp_path,
        provider,
        profile,
        evidence_path,
        routine=True,
        case_id=request.node.nodeid,
    ) as scenario:
        creation, transcript = await scenario.run(scenario.routine_prompt())
        assert_completed(creation, transcript)
        if scenario.evaluation_profile == "user_flow":
            scenario.check_answer(creation, status="routine_saved", research=False)
        routines = await scenario.agent.list_routines()
        assert len(routines) == 1, creation.final_text
        inspection = await scenario.agent.inspect_routine(routines[0].routine_id)
        assert inspection is not None
        routine = inspection.routine
        assert len(routine.capability_grants) == 1
        assert scenario.grant_is_exact(routine.capability_grants[0].material())
        assert len(routine.outcome_contract.effect_requirements) == 1
        requirement = routine.outcome_contract.effect_requirements[0]
        assert requirement.capability_id == scenario.action.capability_id
        assert requirement.minimum_successful_calls == 1
        assert requirement.accepted_evidence_bases == frozenset(
            {EffectEvidenceBasis.SERVER_REPORTED}
        )
        assert routine.allowed_source_ids == routine.allowed_resource_ids == ()
        assert routine.skill_bindings == ()
        assert routine.eligible_model_routes == (model_id,)
        assert routine.outcome_contract.require_exact_source_bindings
        assert routine.outcome_contract.artifact_requirements == ()
        assert routine.maximum_consecutive_failures == 1

        immediate, immediate_transcript = await scenario.scheduled_result(1)
        assert_completed(immediate, immediate_transcript)
        scenario.check_answer(immediate, status="server_reported", research=True)
        await assert_action(scenario)

        scenario.clock = NEXT_SLOT
        scenario.agent._embedded._routine_supervisor.wake()
        weekly, weekly_transcript = await scenario.scheduled_result(2)
        assert_completed(weekly, weekly_transcript)
        scenario.check_answer(weekly, status="server_reported", research=True)
        await assert_action(scenario, count=2)
        for transcript in (immediate_transcript, weekly_transcript):
            assert results_for(transcript, scenario.research.local_name)
            calls = [
                call
                for message in transcript.messages
                for call in message.tool_calls
                if call.name == scenario.action.local_name
            ]
            assert len(calls) == 1
        assert (
            len(scenario.approvals) == 1 and scenario.approvals[0]["approved"] is True
        )
        assert scenario.conversation_id is not None
        for item in await scenario.agent.inbox(
            conversation_id=scenario.conversation_id
        ):
            delivery = await scenario.agent.inspect_delivery(item.delivery_id)
            assert delivery is not None
            assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        for name, arguments in scenario.server.calls:
            if name == "notify_release":
                assert SOURCE in str(arguments["content"]) and RESEARCH_TOKEN in str(
                    arguments["content"]
                )
