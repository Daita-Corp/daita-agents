"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

from tests.support.product_workflows import (
    ActionFixture,
    Agent,
    ApprovalPanel,
    DaitaApp,
    EffectsScreen,
    Input,
    Static,
    cli,
    patch,
    pytest,
    pytestmark,
    replace,
    workspace_for,
)


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
