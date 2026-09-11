"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from tests.support.product_workflows import (
    ActionFixture,
    Agent,
    ApprovalPanel,
    DaitaApp,
    EffectResolutionDecision,
    EffectsScreen,
    Input,
    Static,
    cli,
    effect_receipt_mapping,
    patch,
    pytest,
    pytestmark,
    workspace_for,
)


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
