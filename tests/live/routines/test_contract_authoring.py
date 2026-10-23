"""A real model saves the example routine with one routine_save call and one confirm."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    LoopLimits,
    OnceSchedule,
)
from daita.loop.models import LoopExitKind
from tests.support.live_billing import cost_cap, live_provider
from tests.support.workspace import workspace_for

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get("DAITA_RUN_CONTRACT_QUALIFICATION") != "1",
        reason="requires explicit bounded routine authoring authorization",
    ),
]

_REQUEST = (
    "Create a one-time routine 8 minutes from now that writes a two-sentence "
    "note on why data quality matters."
)


def _api_key() -> str:
    key = os.environ.get("DAITA_CONTRACT_LLM_API_KEY", "")
    if not key.strip():
        pytest.fail("DAITA_CONTRACT_LLM_API_KEY is required under API billing")
    return key


async def test_model_saves_example_routine_with_one_call_and_one_confirm(
    tmp_path: Path, record_property
) -> None:
    model_id = os.environ.get("DAITA_CONTRACT_MODEL_ID", "openai:gpt-5.6-terra")
    profile, provider = live_provider(
        model_id, api_key=_api_key, max_output_tokens=2_048
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    root = tmp_path / "root"
    started = datetime.now(UTC)
    agent = await Agent.create(
        "routine-authoring",
        root=root,
        model=provider,
        model_profile=profile,
        limits=LoopLimits(max_estimated_cost_usd=cost_cap(Decimal("0.50"))),
        workspace=workspace_for(root),
        approval_handler=approve,
    )
    try:
        result = await agent.run(_REQUEST)
        transcript = await agent.transcript(result.run_id)
        routines = await agent.list_routines()
        inspection = (
            await agent.inspect_routine(routines[0].routine_id) if routines else None
        )
    finally:
        await agent.close()

    calls = [call for message in transcript.messages for call in message.tool_calls]
    record_property("tool_calls", [call.name for call in calls])
    record_property("model_steps", result.steps)
    assert result.kind is LoopExitKind.COMPLETED, result.reason
    assert [call.name for call in calls if call.name == "routine_save"] == [
        "routine_save"
    ]
    (confirm,) = approvals
    assert confirm.capability_id == "routines.create"
    card = confirm.arguments
    assert set(card) == {"title", "schedule", "prompt", "sources"}
    assert str(card["schedule"]).startswith("Once at ")
    assert "data quality" in str(card["prompt"]).casefold()

    assert inspection is not None and len(routines) == 1
    routine = inspection.routine
    assert isinstance(routine.schedule, OnceSchedule)
    # Eight minutes after the request, counted from approval at creation.
    assert started + timedelta(minutes=7) <= routine.schedule.exact_at
    assert routine.schedule.exact_at <= datetime.now(UTC) + timedelta(minutes=9)
    assert "data quality" in routine.authorized_instruction.casefold()
    assert routine.source_ids == ()
    assert inspection.recent_occurrences == ()
