"""Product controls exercise the same public authority and evidence boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import pytest
from _workspace_support import workspace_for
from test_mcp_actions import ActionFixture, response
from test_native_write_public import create_fixture
from textual.widgets import Input, OptionList, Static

from daita import (
    Agent,
    ApprovalDecision,
    EffectResolutionDecision,
    OnceSchedule,
    cli,
)
from daita._json import FrozenJsonObject
from daita.artifacts.models import ArtifactAuthorship
from daita.distribution import ArtifactRequirement, OutcomeState
from daita.llm.models import ModelSensitivity, ToolCall
from daita.tui.app import DaitaApp
from daita.tui.projection import approval_review_document, effect_receipt_mapping
from daita.tui.screens.effects import EffectsScreen
from daita.tui.screens.permissions import PermissionsScreen
from daita.tui.screens.routines import render_routine_inspection
from daita.tui.screens.selection import SelectionScreen
from daita.tui.widgets.approval import ApprovalPanel

pytestmark = pytest.mark.acceptance


def choose(app, identities, *, multi=False):
    screen = app.screen
    assert isinstance(screen, SelectionScreen)
    listing = screen.query_one("#picker-options", OptionList)
    for identity in identities:
        listing.highlighted = next(
            i
            for i in range(listing.option_count)
            if listing.get_option_at_index(i).id == identity
        )
        if multi:
            screen.action_toggle_selected()
    screen.action_confirm()


async def test_guided_upsert_authoring_applies_exact_preview(tmp_path, monkeypatch):
    (
        agent,
        _model,
        _db,
        _research,
        _binding,
        resource,
        constraints,
        _batch,
        _clock,
        _approvals,
    ) = await create_fixture(tmp_path, monkeypatch)
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(tmp_path))
    app.controller.agent = agent
    try:
        inspection = await agent.inspect_source_permissions(resource.source_id)
        choice = next(
            item for item in inspection.resources if item.resource_id == resource.id
        )
        assert choice.upsert_conflict_keys == (("domain",),)
        assert choice.generated_identity_columns == ("id",)
        before = inspection.state
        async with app.run_test(size=(110, 38)) as pilot:
            await app.push_screen(PermissionsScreen(source_id=resource.source_id))
            manager = app.screen
            manager.query_one("#perm-max-rows", Input).value = "7"
            assert await pilot.click("#perm-write")
            await pilot.pause()
            choose(app, (resource.id,))
            await pilot.pause()
            choose(app, ("upsert",))
            await pilot.pause()
            choose(app, ("0",))
            await pilot.pause()
            choose(app, ("domain", "name", "evidence_url"), multi=True)
            await pilot.pause()
            choose(app, ("allow",))
            await pilot.pause()
            choose(app, ("name", "evidence_url"), multi=True)
            await pilot.pause()
            assert app.screen is manager
            assert isinstance(manager, PermissionsScreen)
            preview = manager._preview
            assert preview is not None and manager._reviewable
            assert (
                await agent.inspect_source_permissions(resource.source_id)
            ).state == before
            assert preview.after.relational_write_scopes[0].max_rows == 7
            assert preview.after.relational_write_scopes[0].allowed_operations == (
                "upsert",
            )
            body = str(manager.query_one("#perm-body", Static).content)
            assert '"generated_identity_columns"' in body and '"max_rows": 7' in body
            assert await pilot.click("#perm-apply")
            await pilot.pause()
            assert (
                await agent.inspect_source_permissions(resource.source_id)
            ).state == preview.after
            app.exit(0)
    finally:
        await agent.close()


async def test_once_mcp_research_creates_sourced_briefing_and_inbox_while_host_open(
    tmp_path,
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        draft = await fixture.draft()
        draft = replace(
            draft,
            title="Competitor briefing",
            authorized_instruction="Research competitors using the admitted tool and create a sourced briefing with coverage limits.",
            schedule=OnceSchedule(fixture.clock + timedelta(hours=1)),
            run_immediately=False,
            allowed_capability_ids=(
                fixture.research.capability_id,
                "artifact.create_document",
            ),
            requested_capability_grants=(),
            outcome_contract=replace(
                draft.outcome_contract,
                effect_requirements=(),
                maximum_total_artifact_bytes=4096,
                artifact_requirements=(
                    ArtifactRequirement(
                        required=True,
                        minimum_count=1,
                        maximum_count=1,
                        allowed_media_types=("text/markdown",),
                        allowed_authorships=(
                            ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
                        ),
                        allowed_producer_capability_ids=("artifact.create_document",),
                        maximum_artifact_bytes=4096,
                        maximum_total_bytes=4096,
                        maximum_sensitivity=ModelSensitivity.INTERNAL,
                    ),
                ),
            ),
        )
        proposal = await fixture.agent.propose_routine(draft)
        routine = await fixture.agent.create_routine(
            proposal, confirmation_handler=fixture.approve
        )
        assert len(fixture.approvals) == 1
        request = fixture.approvals[0]
        exact = request.render_arguments_for_review()
        document, reviewable = approval_review_document(
            tool_name=request.tool_name,
            capability_id=request.capability_id,
            arguments_text=exact,
            reason=request.reason,
        )
        assert reviewable and exact is not None and document is not None
        assert document.endswith(exact)
        assert "Competitor briefing" in document and "outbound ceiling" in document
        assert "third-party service fees" in document
        assert fixture.server.calls == []
        await fixture.agent.close()
        fixture.clock += timedelta(hours=1)
        assert fixture.server.calls == []  # No host means no scheduled progress.
        fixture.model.steps = [
            response(
                ToolCall(
                    "load",
                    "toolbox_load",
                    {
                        "tool_names": (
                            fixture.research.local_name,
                            "artifact_create_document",
                        )
                    },
                )
            ),
            response(
                ToolCall(
                    "research", fixture.research.local_name, {"query": "competitors"}
                )
            ),
            response(
                ToolCall(
                    "briefing",
                    "artifact_create_document",
                    {
                        "format": "markdown",
                        "content": "# Competitor briefing\n[Research source](https://research.test/report) reports improved status. Coverage is limited; researched claims are not exhaustive.",
                    },
                )
            ),
            response(text="Created a sourced briefing; coverage is limited."),
        ]
        await fixture.reopen()
        delivery = await fixture.delivery()
        assert delivery is not None
        assert delivery.delivery.outcome.conclusion_state is OutcomeState.SUCCEEDED
        assert len(delivery.delivery.outcome.artifact_references) == 1
        assert [name for name, _ in fixture.server.calls] == ["research"]
        assert await fixture.agent.list_sources() == ()
        assert await fixture.agent.list_effects() == ()
        inspection = await fixture.agent.inspect_routine(routine.routine_id)
        assert inspection is not None
        rendered = render_routine_inspection(inspection).plain
        assert "completed" in rendered and "delivery_ids" in rendered
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize("surface", ("cli", "tui"))
@pytest.mark.parametrize("decision", tuple(EffectResolutionDecision))
async def test_receipt_recovery_controls_preserve_observation_and_dispatch_nothing(
    tmp_path, surface, decision
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        fixture.transport.mode = "disconnect"
        fixture.script()
        await fixture.agent.run("Send one reviewed test notification.")
        receipt = (await fixture.agent.list_effects(unresolved_only=True))[0]
        calls = list(fixture.server.calls)
        note = (
            "Investigated the remote system; record this exact decision without retry."
        )
        if surface == "cli":
            await fixture.agent.close()
            original_open = Agent.open

            async def opened(name, **kwargs):
                return await original_open(name, **{**fixture.kwargs(), **kwargs})

            args = cli.build_parser().parse_args(
                [
                    "--root",
                    str(fixture.root),
                    "--workspace",
                    str(workspace_for(tmp_path).root),
                    "effects",
                    "resolve",
                    "mcp-actions",
                    receipt.receipt_id,
                    "--expected-digest",
                    receipt.receipt_digest,
                    "--decision",
                    decision.value,
                    "--note",
                    note,
                ]
            )
            with (
                patch.object(Agent, "open", side_effect=opened),
                patch("builtins.input", return_value="y"),
            ):
                result = await cli._execute(args)
            assert isinstance(result, dict)
            assert isinstance(result["resolution"], dict)
            assert result["receipt_digest"] == receipt.receipt_digest
            assert result["resolution"]["decision"] == decision.value
            await fixture.reopen()
        else:
            app = DaitaApp(start_bootstrap=False, workspace=workspace_for(tmp_path))
            app.controller.agent = fixture.agent
            fixture.agent._embedded._approval_handler = app.handle_approval
            async with app.run_test(size=(100, 32)) as pilot:
                await app.push_screen(EffectsScreen(receipt_id=receipt.receipt_id))
                await pilot.pause()
                manager = app.screen
                assert isinstance(manager, EffectsScreen)
                assert receipt.receipt_digest in str(
                    manager.query_one("#effects-detail", Static).content
                )
                manager.query_one("#effects-note", Input).value = note
                assert await pilot.click("#effects-" + decision.value.replace("_", "-"))
                await pilot.pause()
                panel = manager.query_one(ApprovalPanel)
                assert panel.active
                assert "performs no action" in str(
                    panel.query_one("#approval-inline-text", Static).content
                )
                assert await pilot.click("#approval-inline-yes")
                await pilot.pause()
                assert not panel.active
                app.exit(0)
            fixture.agent = await Agent.open("mcp-actions", **fixture.kwargs())
        resolved = await fixture.agent.inspect_effect(receipt.receipt_id)
        assert resolved is not None and resolved.resolution is not None
        assert resolved.receipt_digest == receipt.receipt_digest
        assert resolved.outcome == receipt.outcome
        assert resolved.resolution.decision is decision
        assert fixture.server.calls == calls
        assert effect_receipt_mapping(resolved)["resolution"] is not None
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize("field", ("password", "api_key", "access_token"))
@pytest.mark.parametrize(
    "capability_id", ("mcp.tool", "data.update_rows", "data.upsert_rows")
)
def test_approval_rejects_secret_fields_without_confusing_token_budgets(
    field, capability_id
):
    safe = json.dumps(
        {
            "routine": {"per_run_max_tokens": 5000},
            "authorization_fingerprint": "sha256:" + "0" * 64,
        }
    )
    assert approval_review_document(
        tool_name="routine_create", capability_id="routines.create", arguments_text=safe
    )[1]
    unsafe = json.dumps({"nested": {field: "sensitive"}})
    assert not approval_review_document(
        tool_name="action", capability_id=capability_id, arguments_text=unsafe
    )[1]


@pytest.mark.parametrize("operation", ("update", "upsert"))
def test_native_review_sanitizes_preview_text_and_denies_oversized_details(operation):
    from daita.capabilities import MAX_APPROVAL_DOCUMENT_CHARACTERS, ApprovalRequest

    arguments = {
        "arguments": {"assignments": [{"column": "name", "value": "\x1b[2JNew"}]},
        "target": {"name": "companies\x1b[2J", "source_name": "Owner's database"},
        "preview": {"matched_rows": 1, "samples": []},
    }
    request = ApprovalRequest(
        "run",
        "call",
        f"data_{operation}_rows",
        f"data.{operation}_rows",
        FrozenJsonObject.from_mapping(arguments),
        "Exact review",
    )
    document, reviewable = approval_review_document(
        tool_name=request.tool_name,
        capability_id=request.capability_id,
        arguments_text=request.render_arguments_for_review(),
        reason=request.reason,
    )
    assert reviewable and document is not None and "\x1b" not in document
    assert "Exact validated details:" in document and "\\u001b" in document
    oversized = replace(
        request,
        arguments=FrozenJsonObject.from_mapping(
            {
                **arguments,
                "arguments": {"value": "x" * MAX_APPROVAL_DOCUMENT_CHARACTERS},
            }
        ),
    )
    assert oversized.render_arguments_for_review() is None
    assert not approval_review_document(
        tool_name=oversized.tool_name,
        capability_id=oversized.capability_id,
        arguments_text=oversized.render_arguments_for_review(),
        reason=oversized.reason,
    )[1]


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


async def test_background_refresh_preserves_foreground_approval_state(tmp_path):
    from daita.tui.screens.chat import ChatScreen

    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(tmp_path))
    agent = await Agent.create(
        "status-review", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    app.controller.agent = agent
    try:
        async with app.run_test(size=(110, 36)) as pilot:
            await app.push_screen(ChatScreen())
            await app._refresh_status(running=True, state="approval")
            await app.refresh_background_status(notify_new=False)
            assert "approval" in str(
                app.screen.query_one("#status-primary", Static).content
            )
            await app._refresh_status(running=False)
            assert "ready" in str(
                app.screen.query_one("#status-primary", Static).content
            )
            app.exit(0)
    finally:
        await agent.close()


async def test_exiting_during_recovery_review_cancels_without_resolution(tmp_path):
    import asyncio

    fixture = await ActionFixture(tmp_path).start()
    fixture.transport.mode = "disconnect"
    fixture.script()
    await fixture.agent.run("Send one test notification.")
    receipt = (await fixture.agent.list_effects(unresolved_only=True))[0]
    calls = list(fixture.server.calls)
    app = DaitaApp(start_bootstrap=False, workspace=workspace_for(tmp_path))
    app.controller.agent = fixture.agent
    fixture.agent._embedded._approval_handler = app.handle_approval
    try:
        async with asyncio.timeout(10):
            async with app.run_test(size=(100, 32)) as pilot:
                await app.push_screen(EffectsScreen(receipt_id=receipt.receipt_id))
                await pilot.pause()
                app.screen.query_one("#effects-note", Input).value = (
                    "Investigating only."
                )
                assert await pilot.click("#effects-allow-future-work")
                await pilot.pause()
                assert app.screen.query_one(ApprovalPanel).active
                app.exit(0)
        await fixture.reopen()
        assert await fixture.agent.inspect_effect(receipt.receipt_id) == receipt
        assert fixture.server.calls == calls
    finally:
        await fixture.agent.close()


@pytest.mark.parametrize("command", ("create", "update"))
async def test_cli_routine_mutations_request_human_confirmation_before_saving(
    tmp_path, command
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        draft = replace(await fixture.draft(), run_immediately=False)
        routine = None
        if command == "update":
            routine = await fixture.agent.create_routine(
                await fixture.agent.propose_routine(draft)
            )
        arguments = [
            "--root",
            str(fixture.root),
            "--workspace",
            str(workspace_for(tmp_path).root),
            "routines",
            command,
            "mcp-actions",
        ]
        if routine is not None:
            arguments += [routine.routine_id, str(routine.revision)]
        arguments += ["--spec", str(tmp_path / "reviewed.json")]
        args = cli.build_parser().parse_args(arguments)
        with (
            patch.object(Agent, "open", return_value=fixture.agent),
            patch.object(
                cli,
                "_routine_draft_from_file",
                return_value=replace(draft, title="CLI proposal"),
            ),
            patch("builtins.input", return_value="n") as confirm,
        ):
            with pytest.raises(PermissionError, match="not approved"):
                await cli._execute(args)
        confirm.assert_called_once()
        await fixture.reopen()
        if routine is None:
            assert await fixture.agent.list_routines() == ()
        else:
            unchanged = await fixture.agent.inspect_routine(routine.routine_id)
            assert unchanged is not None and unchanged.routine == routine
        assert fixture.server.calls == []
    finally:
        await fixture.agent.close()
