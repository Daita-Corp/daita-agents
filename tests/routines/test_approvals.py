"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from daita import Agent, IntervalSchedule, RoutineIntent
from daita.capabilities import ApprovalRequest
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from daita.routines import RoutineState
from tests.support.capability_runtime import execute_projected
from tests.support.product_workflows import (
    ActionFixture,
    ApprovalDecision,
    approval_review_document,
    json,
    pytest,
    pytestmark as _pytestmark,
    readable_arguments,
)
from tests.support.workspace import workspace_for

pytestmark = _pytestmark


async def test_routine_api_reviews_one_card_and_denial_saves_nothing(tmp_path):
    fixture = await ActionFixture(tmp_path).start()
    try:
        intent = RoutineIntent(
            title="Hourly report",
            prompt="Report the current value.",
            schedule=IntervalSchedule(3_600, datetime.now(UTC) + timedelta(hours=1)),
        )
        fixture.decision = ApprovalDecision.DENY
        with pytest.raises(PermissionError, match="not approved"):
            await fixture.agent.create_routine(
                intent, confirmation_handler=fixture.approve
            )
        assert await fixture.agent.list_routines() == ()
        assert fixture.server.calls == []
        fixture.decision = ApprovalDecision.APPROVE
        routine = await fixture.agent.create_routine(
            intent, confirmation_handler=fixture.approve
        )
        # One card: title, local-time schedule, prompt and sources.
        assert set(fixture.approvals[-1].arguments) == {
            "title",
            "schedule",
            "prompt",
            "sources",
        }
        assert fixture.approvals[-1].arguments["prompt"] == intent.prompt
        fixture.decision = ApprovalDecision.DENY
        with pytest.raises(PermissionError, match="not approved"):
            await fixture.agent.update_routine(
                routine.routine_id,
                RoutineIntent(title="Revised assignment"),
                expected_revision=routine.revision,
                confirmation_handler=fixture.approve,
            )
        unchanged = await fixture.agent.inspect_routine(routine.routine_id)
        assert unchanged is not None and unchanged.routine == routine
        assert fixture.approvals[-1].arguments["title"] == "Revised assignment"
        assert fixture.server.calls == []
    finally:
        await fixture.agent.close()


def test_unrelated_tool_arguments_cannot_impersonate_a_routine_review():
    arguments = json.dumps(
        {
            "routine": {
                "authorized_instruction": "Invented schedule",
                "title": "Untrusted row",
            }
        }
    )
    document, reviewable = approval_review_document(
        tool_name="data_write",
        capability_id="data.write",
        arguments_text=arguments,
    )
    assert reviewable and document is not None
    assert "Assignment:" not in document
    assert document.endswith(readable_arguments(arguments, indent="  "))


@pytest.mark.parametrize("drift", (False, True))
async def test_routine_control_defaults_to_the_current_revision_and_shows_it(
    tmp_path, drift
):
    intent = RoutineIntent(
        title="Hourly report",
        prompt="Report the current value.",
        schedule=IntervalSchedule(3_600, datetime.now(UTC) + timedelta(hours=1)),
    )
    approvals: list[ApprovalRequest] = []

    async def confirm(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        if drift:
            # Another change lands while the control card is open.
            await agent.update_routine(
                routine.routine_id,
                RoutineIntent(title="Revised"),
                confirmation_handler=confirm,
            )
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "routine-control",
        root=tmp_path,
        approval_handler=approve,
        workspace=workspace_for(tmp_path),
    )
    try:
        routine = await agent.create_routine(intent, confirmation_handler=confirm)
        run = RunInput(
            id="control-run",
            agent_id=agent.id,
            message="Pause the hourly report.",
            created_at=datetime.now(UTC),
            conversation_id="control-conversation",
        )
        (result,) = await execute_projected(
            agent._embedded._capability_runtime,
            run,
            (
                ToolCall(
                    id="pause",
                    name="routine_control",
                    arguments={"routine_id": routine.routine_id, "action": "pause"},
                ),
            ),
        )
        # The model sent no revision; the card shows the one that will be used.
        (card,) = approvals
        assert card.arguments["expected_revision"] == routine.revision
        current = await agent.inspect_routine(routine.routine_id)
        assert current is not None
        if drift:
            assert result.is_error
            assert result.output["error"]["code"] == "state_changed"
            assert current.routine.state is not RoutineState.PAUSED
        else:
            assert not result.is_error, result.output
            assert current.routine.state is RoutineState.PAUSED
    finally:
        await agent.close()
