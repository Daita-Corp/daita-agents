"""Shared helpers extracted from ``test_mcp_actions.py``."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from daita.llm.models import CanonicalMessage, MessageRole, ModelSensitivity, TextBlock
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from tests.support.mcp_routine_harness import (
    NOW,
    REPORT_INSTRUCTION,
    owner_routine_draft,
)


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
