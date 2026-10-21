"""Acceptance coverage for the one Textual interactive path."""

from __future__ import annotations

from pathlib import Path

import pytest

from daita import Agent
from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.tui.app import DaitaApp, run_daita_app
from daita.tui.screens.onboarding import AgentCreateScreen
from daita.tui.screens.selection import SelectionScreen
from tests.support.host import StartHost, host_workspace, remote_agent
from tests.support.workspace import workspace_for


async def test_empty_root_opens_create_screen_inside_textual(start_host: StartHost):
    harness = await start_host()
    app = DaitaApp(root=harness.root, workspace=host_workspace(harness))
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, AgentCreateScreen)
        assert app.controller.agent is None
        await pilot.press("escape")
        await pilot.pause()
    assert app.return_value == 0


async def test_requested_missing_agent_does_not_prompt_before_app(
    start_host: StartHost,
):
    harness = await start_host()
    app = DaitaApp(
        root=harness.root, agent_name="missing", workspace=host_workspace(harness)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        # Failure is handled inside the app, not by a pre-app selector.
        assert app.screen is not None
        app.exit(0)


async def test_run_terminal_application_constructs_textual_app(
    tmp_path: Path, monkeypatch
):
    from daita import terminal

    seen: dict[str, object] = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(terminal, "_load_textual_app", lambda: fake_run)
    code = await terminal.run_terminal_application(
        root=tmp_path,
        agent_name="atlas",
        conversation_id="conversation-1",
        workspace=workspace_for(tmp_path),
    )
    assert code == 0
    assert seen["root"] == tmp_path
    assert seen["agent_name"] == "atlas"
    assert seen["conversation_id"] == "conversation-1"


async def test_tui_fatal_error_preserves_exit_code_and_traceback(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from daita.tui import app as app_module

    try:
        raise RuntimeError("sentinel fatal UI error")
    except RuntimeError as error:
        captured_error = error

    class FailedApp:
        return_code = 1
        _startup_error = None
        _exception = captured_error

        async def run_async(self):
            return None

    monkeypatch.setattr(app_module, "DaitaApp", lambda **kwargs: FailedApp())
    code = await run_daita_app(root=tmp_path, workspace=workspace_for(tmp_path))
    output = capsys.readouterr()
    assert code == 1
    assert "Daita terminal UI failed:" in output.err
    assert "test_tui_fatal_error_preserves_exit_code_and_traceback" in output.err
    assert "RuntimeError: sentinel fatal UI error" in output.err


async def test_tui_nonzero_exit_without_exception_is_not_reported_as_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from daita.tui import app as app_module

    class FailedApp:
        return_code = 2
        _startup_error = None
        _exception = None

        async def run_async(self):
            return None

    monkeypatch.setattr(app_module, "DaitaApp", lambda **kwargs: FailedApp())
    code = await run_daita_app(root=tmp_path, workspace=workspace_for(tmp_path))
    output = capsys.readouterr()
    assert code == 2
    assert "Textual exit code: 2" in output.err


async def test_requested_conversation_is_validated_and_resumed(
    tmp_path: Path, monkeypatch, start_host: StartHost
):
    harness = await start_host()
    opened = await remote_agent(harness, "atlas")
    await opened.close()
    checked: list[str] = []

    # The host route passes every wire argument, such as caller_principal_id.
    async def conversation_exists(
        _agent: Agent, conversation_id: str, **_kwargs: object
    ) -> bool:
        checked.append(conversation_id)
        return True

    async def conversation_runs(_agent: Agent, _conversation_id: str, **_kwargs):
        return ()

    monkeypatch.setattr(Agent, "conversation_exists", conversation_exists)
    monkeypatch.setattr(Agent, "conversation_runs", conversation_runs)
    app = DaitaApp(
        root=harness.root,
        workspace=host_workspace(harness),
        agent_name="atlas",
        conversation_id="conversation-1",
        start_bootstrap=False,
    )

    async def ready() -> None:
        return None

    monkeypatch.setattr(app, "_ensure_ready", ready)
    try:
        await app._bootstrap_inner()
        assert app._startup_error is None
        assert app.controller.conversation_id == "conversation-1"
        assert checked == ["conversation-1"]
    finally:
        await app.controller.close()


async def test_unknown_requested_conversation_fails_startup(
    tmp_path: Path, monkeypatch, start_host: StartHost
):
    harness = await start_host()
    opened = await remote_agent(harness, "atlas")
    await opened.close()
    app = DaitaApp(
        root=harness.root,
        workspace=host_workspace(harness),
        agent_name="atlas",
        conversation_id="conversation-missing",
        start_bootstrap=False,
    )

    async def ready() -> None:
        return None

    monkeypatch.setattr(app, "_ensure_ready", ready)
    try:
        with pytest.raises(
            ValueError, match="unknown conversation for requested agent"
        ):
            await app._bootstrap_inner()
    finally:
        await app.controller.close()

    assert app.controller.conversation_id is None


async def test_incompatible_agent_state_returns_to_agent_picker(
    tmp_path: Path, monkeypatch, start_host: StartHost
):
    harness = await start_host()
    opened = await remote_agent(harness, "atlas")
    await opened.close()
    app = DaitaApp(
        root=harness.root,
        workspace=host_workspace(harness),
        agent_name="atlas",
        start_bootstrap=False,
    )

    async def incompatible(name: str, **_kwargs: object) -> None:
        raise StateCompatibilityError(
            StateCompatibilityCode.REVISION_UNSUPPORTED,
            tmp_path / name / "state.db",
            "unknown migration history",
            current_revision="4",
        )

    shown: list[SelectionScreen] = []

    async def await_modal(screen: SelectionScreen) -> None:
        shown.append(screen)
        return None

    monkeypatch.setattr(app.controller, "open_agent", incompatible)
    monkeypatch.setattr(app, "_await_modal", await_modal)
    try:
        await app._bootstrap_inner()
    finally:
        await app.controller.close()

    (picker,) = shown
    assert picker._initial_error == "atlas: unknown migration history"
    assert [(o.identity, o.description) for o in picker._options] == [
        ("atlas", "cannot open with this release")
    ]
    assert app._startup_error is None
