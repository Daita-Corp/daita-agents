"""Shared helpers extracted from ``test_mcp_actions.py``."""

from __future__ import annotations

from daita import RoutineIntent
from tests.support.mcp_routine_harness import (
    REPORT_INSTRUCTION,
    owner_routine_arguments,
)


async def admit_owner_routine(scenario):
    """Save the owner's exact routine through the public API, with no model request.

    The routine reports to a new conversation of its own; no seeded run is needed.
    """
    scenario.setup_mode = "owner_admitted_fixture"
    arguments = owner_routine_arguments(scenario)
    if scenario.evaluation_profile == "strict":
        arguments["prompt"] = f"{arguments['prompt']} {REPORT_INSTRUCTION}"
    stored = await scenario.agent.create_routine(RoutineIntent.from_mapping(arguments))
    scenario.conversation_id = stored.conversation_id
    return stored
