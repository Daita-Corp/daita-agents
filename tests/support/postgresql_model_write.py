"""Shared helpers extracted from ``test_model_write_acceptance.py``."""

from __future__ import annotations

from daita import CalendarDaySelector, CalendarSchedule, RoutineIntent
from tests.support.postgresql_live import database as database
from tests.support.postgresql_live_harness import REPORT_INSTRUCTION

# Immediately, then every Monday at 09:00 America/Chicago.
OWNER_SCHEDULE = CalendarSchedule(
    timezone="America/Chicago",
    hour=9,
    minute=0,
    day_selector=CalendarDaySelector.WEEKDAYS,
    weekdays=(1,),
)


async def admit_owner_routine(scenario, source_id):
    """Save the owner's routine through the public Agent API, with no model request.

    This follows the independent scheduled-live fixture pattern. It establishes
    no model-authored creation or discovery effectiveness. The routine reports
    to a new conversation of its own.
    """
    assert scenario.setup_mode == "owner_admitted_routine"
    prompt = (
        "Keep company new.test in our companies table with name New and evidence "
        "link https://evidence.test/new.test. Add it if missing, otherwise update "
        "its name and evidence link, leaving other details alone. Check and save "
        "it even when unchanged, and tell me what changed."
        if scenario.profile == "user_flow"
        else "Maintain the owner-supplied finding in the exact admitted companies table: "
        "domain new.test, name New, evidence_url https://evidence.test/new.test. "
        "Obtain a current preview; upsert exactly this one row by domain, insert missing "
        "rows and update only name/evidence_url, preserve omitted notes and let PostgreSQL "
        "generate IDs. Invoke once even if unchanged. " + REPORT_INSTRUCTION
    )
    routine = await scenario.agent.create_routine(
        RoutineIntent(
            title="Owner-authorized company finding",
            prompt=prompt,
            schedule=OWNER_SCHEDULE,
            sources=(source_id,),
            run_immediately=True,
        )
    )
    scenario.conversation_id = routine.conversation_id
    return routine
