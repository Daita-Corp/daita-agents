"""Bounded real-model authoring comparisons; all connector effects are simulated."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from daita.capabilities import EffectEvidenceBasis
from tests.support.mcp_routine_harness import (
    assert_completed,
    evaluate,
    live_provider,
    report_path,
)

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_CONTRACT_QUALIFICATION") != "1",
        reason="requires explicit bounded contract qualification authorization",
    ),
]


@pytest.mark.parametrize("variant", ["familiar", "unfamiliar"])
@pytest.mark.parametrize("repetition", [0, 1])
async def test_model_authors_scheduled_contract_without_dispatch(
    tmp_path: Path,
    variant: str,
    repetition: int,
    request: pytest.FixtureRequest,
    record_property,
) -> None:
    """The shared harness and prompt are identical for baseline and candidate code."""
    del repetition
    model_id = os.environ.get("DAITA_CONTRACT_MODEL_ID", "openai:gpt-5.6-terra")
    profile, provider = live_provider(model_id)
    path = report_path(request.node.nodeid)
    record_property("contract_evidence", str(path.resolve()))
    names = (
        ("destination", "content")
        if variant == "familiar"
        else ("channel_ref", "body_text")
    )
    async with evaluate(
        tmp_path,
        provider,
        profile,
        path,
        case_id=request.node.nodeid,
        routine=True,
        run_immediately=False,
        action_argument_names=names,
    ) as scenario:
        creation, transcript = await scenario.run(scenario.routine_prompt())
        assert_completed(creation, transcript)
        scenario.check_answer(creation, status="routine_saved", research=False)
        routines = await scenario.agent.list_routines()
        assert len(routines) == 1, creation.final_text
        inspection = await scenario.agent.inspect_routine(routines[0].routine_id)
        assert inspection is not None
        routine = inspection.routine
        assert len(routine.capability_grants) == 1
        assert scenario.grant_is_exact(routine.capability_grants[0].material())
        assert routine.allowed_source_ids == routine.allowed_resource_ids == ()
        assert routine.skill_bindings == ()
        assert routine.eligible_model_routes == (model_id,)
        assert routine.maximum_consecutive_failures == 1
        assert len(routine.outcome_contract.effect_requirements) == 1
        requirement = routine.outcome_contract.effect_requirements[0]
        assert requirement.capability_id == scenario.action.capability_id
        assert requirement.minimum_successful_calls == 1
        assert requirement.accepted_evidence_bases == frozenset(
            {EffectEvidenceBasis.SERVER_REPORTED}
        )
        assert routine.outcome_contract.require_exact_source_bindings
        assert routine.outcome_contract.require_current_run_provenance
        assert routine.outcome_contract.artifact_requirements == ()
        assert (
            len(scenario.approvals) == 1 and scenario.approvals[0]["approved"] is True
        )
        assert scenario.server.calls == []
        assert await scenario.agent.list_effects() == ()
        assert inspection.recent_occurrences == ()
