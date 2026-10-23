"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from daita import IntervalSchedule, RoutineIntent
from tests.support.product_workflows import (
    ActionFixture,
    ApprovalDecision,
    approval_review_document,
    json,
    pytest,
    pytestmark as _pytestmark,
)

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
        tool_name="data_update_rows",
        capability_id="data.update_rows",
        arguments_text=arguments,
    )
    assert reviewable and document is not None
    assert "Assignment:" not in document
    assert document.endswith(arguments)
