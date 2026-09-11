"""Shared helpers extracted from ``test_model_write_acceptance.py``."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from daita import ScheduledRoutineDraft
from daita._json import FrozenJsonObject
from daita.llm.models import CanonicalMessage, MessageRole, TextBlock
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.routines.capabilities import _parsed_spec
from tests.support.postgresql_live import database as database
from tests.support.postgresql_live_harness import (
    FIXTURE_SENSITIVITY,
    NOW,
    REPORT_INSTRUCTION,
)


async def admit_owner_routine(scenario, owner_prompt):
    """Seed foreground owner provenance, then admit through the public Agent API.

    This follows the independent scheduled-live fixture pattern. The seeded
    transcript is explicitly fixture data, excluded from measured live usage.
    It establishes no model-authored creation or discovery effectiveness.
    """
    assert scenario.setup_mode == "owner_admitted_routine"
    origin = RunInput(
        id=f"run-{uuid4().hex}",
        agent_id=scenario.agent.id,
        conversation_id=f"conversation-{uuid4().hex}",
        message=owner_prompt,
        created_at=NOW,
        history_sensitivity=FIXTURE_SENSITIVITY,
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
            sensitivity=FIXTURE_SENSITIVITY,
        ),
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=(TextBlock("Owner fixture authorization recorded."),),
        ),
    )
    scenario.conversation_id = origin.conversation_id
    destination = (
        await scenario.agent.distribution_destinations(
            origin.conversation_id, sensitivity_ceiling=FIXTURE_SENSITIVITY
        )
    )[0]
    arguments = {
        **scenario.routine_contract,
        "title": "Owner-authorized company finding",
        "authorized_instruction": (
            "Maintain the owner-supplied finding in the exact admitted companies table: "
            "domain new.test, name New, evidence_url https://evidence.test/new.test. "
            "Obtain a current preview; upsert exactly this one row by domain, insert missing "
            "rows and update only name/evidence_url, preserve omitted notes and let PostgreSQL "
            "generate IDs. Invoke once even if unchanged. " + REPORT_INSTRUCTION
        ),
        "distribution_destination_id": destination.destination_id,
        "skill_names": [],
        "requested_capability_grants": [
            {
                "capability_id": "data.upsert_rows",
                "constraints": scenario.routine_grant,
                "max_calls_per_occurrence": 1,
            }
        ],
    }
    if scenario.profile == "user_flow":
        # This owner-authored natural fixture explicitly authorizes schema access
        # to the same frozen table. It is not inferred by routine admission.
        arguments["allowed_capability_ids"] = sorted(
            {*arguments["allowed_capability_ids"], "catalog.schema"}
        )
        arguments["authorized_instruction"] = (
            "Keep company new.test in our companies table with name New and evidence "
            "link https://evidence.test/new.test. Add it if missing, otherwise update "
            "its name and evidence link, leaving other details alone. Check and save "
            "it even when unchanged, and tell me what changed."
        )
    parsed = cast(
        dict[str, Any], dict(_parsed_spec(FrozenJsonObject.from_mapping(arguments)))
    )
    parsed.pop("basis_run_id")
    draft = ScheduledRoutineDraft(origin_run_id=origin.id, **parsed)
    proposal = await scenario.agent.propose_routine(draft)
    return await scenario.agent.create_routine(proposal)
