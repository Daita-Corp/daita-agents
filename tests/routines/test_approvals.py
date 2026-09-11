"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from tests.support.product_workflows import (
    ActionFixture,
    ApprovalDecision,
    FrozenJsonObject,
    approval_review_document,
    json,
    pytest,
    pytestmark as _pytestmark,
    replace,
)

pytestmark = _pytestmark


async def test_routine_control_api_reviews_exact_validated_revision_and_denial_saves_nothing(
    tmp_path,
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        draft = replace(await fixture.draft(), run_immediately=False)
        proposal = await fixture.agent.propose_routine(draft)
        fixture.decision = ApprovalDecision.DENY
        with pytest.raises(PermissionError, match="not approved"):
            await fixture.agent.create_routine(
                proposal, confirmation_handler=fixture.approve
            )
        assert await fixture.agent.list_routines() == ()
        assert fixture.server.calls == []
        fixture.decision = ApprovalDecision.APPROVE
        routine = await fixture.agent.create_routine(
            proposal, confirmation_handler=fixture.approve
        )
        assert fixture.approvals[-1].arguments["proposal"][
            "capability_grants"
        ] == tuple(
            FrozenJsonObject.from_mapping(grant.material())
            for grant in routine.capability_grants
        )
        assert (
            fixture.approvals[-1].arguments["authority"]["bindings"][0][
                "maximum_outbound_sensitivity"
            ]
            == "internal"
        )
        fixture.decision = ApprovalDecision.DENY
        with pytest.raises(PermissionError, match="not approved"):
            await fixture.agent.update_routine(
                routine.routine_id,
                expected_revision=routine.revision,
                draft=replace(draft, title="Revised assignment"),
                confirmation_handler=fixture.approve,
            )
        unchanged = await fixture.agent.inspect_routine(routine.routine_id)
        assert unchanged is not None and unchanged.routine == routine
        assert (
            fixture.approvals[-1].arguments["proposal"]["title"] == "Revised assignment"
        )
        assert (
            fixture.approvals[-1].arguments["proposal"]["revision"]
            == routine.revision + 1
        )
        assert fixture.server.calls == []
    finally:
        await fixture.agent.close()


async def test_permission_changes_cannot_be_granted_by_routine_approval(tmp_path):
    fixture = await ActionFixture(tmp_path).start()
    try:
        draft = await fixture.draft()
        proposal = await fixture.agent.propose_routine(draft)

        async def revoke_during_review(request):
            await fixture.agent.revoke_mcp_server(fixture.binding.binding_id)
            return ApprovalDecision.APPROVE

        with pytest.raises((ValueError, RuntimeError)):
            await fixture.agent.create_routine(
                proposal, confirmation_handler=revoke_during_review
            )
        assert await fixture.agent.list_routines() == ()
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
