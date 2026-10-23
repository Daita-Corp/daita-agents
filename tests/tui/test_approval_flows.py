"""Component-owned tests split from ``test_product_workflows.py``."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from daita import OnceSchedule, RoutineIntent
from tests.support.host import StartHost, host_workspace, remote_agent
from tests.support.product_workflows import (
    ActionFixture,
    Agent,
    DaitaApp,
    Static,
    cli,
    patch,
    pytest,
    pytestmark as _pytestmark,
    workspace_for,
)

pytestmark = _pytestmark


async def test_background_refresh_preserves_foreground_approval_state(
    start_host: StartHost,
):
    from daita.tui.screens.chat import ChatScreen

    harness = await start_host()
    app = DaitaApp(
        root=harness.root, start_bootstrap=False, workspace=host_workspace(harness)
    )
    agent = await remote_agent(harness, "status-review")
    app.controller.agent = agent
    try:
        async with app.run_test(size=(110, 36)):
            try:
                await app.push_screen(ChatScreen())
                await app._refresh_status(running=True, state="approval")
                await app.refresh_background_status(notify_new=False)
                assert "approval" in str(
                    app.screen.query_one("#status-primary", Static).content
                )
                await app._refresh_status(running=False)
                assert "approval" not in str(
                    app.screen.query_one("#status-primary", Static).content
                )
            finally:
                app.exit(0)
    finally:
        await agent.close()


@pytest.mark.parametrize("command", ("create", "update"))
async def test_cli_routine_mutations_request_human_confirmation_before_saving(
    tmp_path, command
):
    fixture = await ActionFixture(tmp_path).start()
    try:
        due = datetime.now(UTC) + timedelta(days=1)
        routine = None
        if command == "update":
            routine = await fixture.agent.create_routine(
                RoutineIntent(
                    title="Saved routine",
                    prompt="Report the latest result.",
                    schedule=OnceSchedule(due),
                )
            )
        spec = tmp_path / "reviewed.json"
        spec.write_text(
            json.dumps(
                {
                    "title": "CLI proposal",
                    "prompt": "Report the latest result.",
                    "schedule": {"kind": "once", "exact_at": due.isoformat()},
                }
            ),
            encoding="utf-8",
        )
        arguments = [
            "--no-host",
            "--root",
            str(fixture.root),
            "--workspace",
            str(workspace_for(tmp_path).root),
            "routines",
            command,
            "mcp-actions",
        ]
        if routine is not None:
            arguments += [routine.routine_id, "--expected-revision", "1"]
        arguments += ["--spec", str(spec)]
        args = cli.build_parser().parse_args(arguments)
        with (
            patch.object(Agent, "open", return_value=fixture.agent),
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
