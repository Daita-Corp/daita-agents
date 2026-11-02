"""A routine request saves in one call and one confirm, and its result lands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from daita import (
    Agent,
    ApprovalDecision,
    ApprovalRequest,
    OnceSchedule,
    WorkColumn,
    WorkKind,
    connect,
)
from daita.distribution import OutcomeState
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.routines.temporal import system_user_timezone
from tests.support.host import StartHost, bounded, measured, scheduled_routine
from tests.support.toolbox_model import ToolboxAwareMockModelProvider
from tests.support.workspace import workspace_for

pytestmark = [pytest.mark.acceptance, pytest.mark.integration]

NOW = datetime(2026, 10, 22, 14, 0, tzinfo=UTC)
EXAMPLE = (
    "Create a one-time routine 8 minutes from now that writes a two-sentence "
    "note on why data quality matters."
)
TITLE = "Why data quality matters"
PROMPT = "Write a two-sentence note on why data quality matters."


def _respond(*calls: ToolCall, text: str | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=calls,
        text=text,
        usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
    )


async def test_the_example_request_saves_with_one_routine_save_and_one_confirm(
    tmp_path: Path,
) -> None:
    confirms: list[ApprovalRequest] = []

    async def confirm(request: ApprovalRequest) -> ApprovalDecision:
        confirms.append(request)
        return ApprovalDecision.APPROVE

    provider = ToolboxAwareMockModelProvider(
        (
            _respond(
                ToolCall(
                    "save",
                    "routine_save",
                    {
                        "title": TITLE,
                        "prompt": PROMPT,
                        "schedule": {"kind": "once", "after_seconds": 480},
                    },
                )
            ),
            _respond(text="Saved; it runs in 8 minutes."),
        ),
        provider_id="mock:example-routine",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "example-routine",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=32_000,
            max_output_tokens=2_000,
            supports_tools=True,
        ),
        clock=lambda: NOW,
        approval_handler=confirm,
    )
    try:
        result = await agent.run(EXAMPLE)
        transcript = await agent.transcript(result.run_id)
        saves = [
            call
            for message in transcript.messages
            for call in message.tool_calls
            if call.name == "routine_save"
        ]
        assert len(saves) == 1
        (request,) = confirms
        assert (request.tool_name, request.capability_id) == (
            "routine_save",
            "routines.create",
        )
        card = request.arguments
        assert set(card) == {"title", "schedule", "prompt", "sources"}
        assert (card["title"], card["prompt"]) == (TITLE, PROMPT)
        assert card["sources"] == "All current sources"
        zone = system_user_timezone()
        local = (NOW + timedelta(minutes=8)).astimezone(
            UTC if zone is None else ZoneInfo(zone)
        )
        assert card["schedule"] == (
            f"Once at {local.strftime('%a %Y-%m-%d %H:%M %Z')} (timed from approval)"
        )

        (summary,) = await agent.list_routines()
        inspection = await agent.inspect_routine(summary.routine_id)
        assert inspection is not None
        routine = inspection.routine
        assert routine.schedule == OnceSchedule(NOW + timedelta(seconds=480))
        assert routine.conversation_id == result.conversation_id
        assert (routine.title, routine.authorized_instruction) == (TITLE, PROMPT)
        assert routine.source_ids == ()
    finally:
        await agent.close()


async def test_an_occurrence_delivers_to_the_inbox_and_shows_in_work_done(
    start_host: StartHost,
) -> None:
    model = MockModelProvider(
        (measured("Seven invoices are paid."), measured("Seven are paid today."))
    )
    harness = await start_host(model=model)
    await harness.create_agent()
    agent = await connect("sales", cwd=harness.work(), root=harness.root)
    try:
        routine = await scheduled_routine(agent, due_in=timedelta(seconds=1))

        async def done() -> object:
            board = await agent.work_board()
            return next(
                (
                    card
                    for card in board.columns.get(WorkColumn.DONE, ())
                    if card.kind is WorkKind.ROUTINE_RUN
                    and card.routine_id == routine.routine_id
                ),
                None,
            )

        card = await bounded(done, timeout=30)
        assert card.delivery_id is not None
        inspection = await agent.inspect_delivery(card.delivery_id)
        assert inspection is not None
        delivery = inspection.delivery
        # The routine's conversation owns the Inbox the result lands in.
        assert delivery.conversation_id == routine.conversation_id
        assert delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        assert delivery.outcome.conclusion_preview == "Seven are paid today."
        model.assert_consumed()
    finally:
        await agent.close()
