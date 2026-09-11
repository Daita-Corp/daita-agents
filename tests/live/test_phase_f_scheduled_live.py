"""Independent live scheduled execution, with a typed owner-authorized fixture.

The original ten-case matrix remains unchanged. Only fixture authorization is
seeded here; both scheduled runs use the real model and production composition.
"""

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from _phase_f_live_support import (
    NEXT_SLOT,
    NOW,
    REPORT_INSTRUCTION,
    RESEARCH_TOKEN,
    SOURCE,
    assert_action,
    assert_completed,
    evaluate,
    live_provider,
    owner_routine_draft,
)

from daita.distribution.models import OutcomeState
from daita.llm.models import CanonicalMessage, MessageRole, ModelSensitivity, TextBlock
from daita.loop.models import LoopExit, LoopExitKind, RunInput

# Shared opt-in and collection parameters; no import executes a provider.
from .test_phase_f_mcp_live import (
    evidence_path as evidence_path,
    model_id as model_id,
    pytestmark as pytestmark,
    repetition as repetition,
)


async def test_live_owner_admitted_routine_runs_immediate_and_weekly(
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
        await admit_owner_routine(scenario)
        for count in (1, 2):
            if count == 2:
                scenario.clock = NEXT_SLOT
                scenario.agent._embedded._routine_supervisor.wake()
            result, transcript = await scenario.scheduled_result(count)
            assert_completed(result, transcript)
            scenario.check_answer(result, status="server_reported", research=True)
            await assert_action(scenario, count=count)
            assert result.usage.total_tokens > 0
        assert scenario.approvals == []  # exact typed owner authorization is standing
        assert scenario.conversation_id is not None
        for item in await scenario.agent.inbox(
            conversation_id=scenario.conversation_id
        ):
            delivery = await scenario.agent.inspect_delivery(item.delivery_id)
            assert delivery is not None
            assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        for name, arguments in scenario.server.calls:
            if name == "notify_release":
                assert SOURCE in str(arguments["content"])
                assert RESEARCH_TOKEN in str(arguments["content"])


async def admit_owner_routine(scenario):
    """Seed owner provenance, then use the ordinary public proposal/admission API.

    The seed is explicit offline fixture data, not a live-model answer or an
    evaluation of creation effectiveness. It incurs no model request or usage.
    """
    scenario.setup_mode = "owner_admitted_fixture"
    origin = RunInput(
        id=f"run-{uuid4().hex}",
        agent_id=scenario.agent.id,
        conversation_id=f"conversation-{uuid4().hex}",
        message="The owner authorizes the exact release-readiness routine fixture.",
        created_at=NOW,
        history_sensitivity=ModelSensitivity.INTERNAL,
    )
    assert origin.conversation_id is not None
    store = scenario.agent._embedded._store
    await store.start(origin)
    await store.append(origin.id, origin.start_message())
    await store.complete(
        LoopExit(
            run_id=origin.id,
            conversation_id=origin.conversation_id,
            created_at=NOW,
            kind=LoopExitKind.COMPLETED,
            reason="completed",
            final_text="Owner fixture authorization recorded.",
            sensitivity=ModelSensitivity.INTERNAL,
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("Owner fixture authorization recorded."),),
        ),
    )
    scenario.conversation_id = origin.conversation_id
    destinations = await scenario.agent.distribution_destinations(
        scenario.conversation_id,
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
    )
    draft = owner_routine_draft(scenario, origin.id, destinations[0].destination_id)
    if scenario.evaluation_profile == "strict":
        draft = replace(
            draft,
            authorized_instruction=draft.authorized_instruction
            + " "
            + REPORT_INSTRUCTION,
        )
    proposal = await scenario.agent.propose_routine(draft)
    stored = await scenario.agent.create_routine(proposal)
    assert len(stored.capability_grants) == 1
    assert scenario.grant_is_exact(stored.capability_grants[0].material())
    return stored
