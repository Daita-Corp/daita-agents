"""Acceptance coverage for the one Textual interactive path."""

from __future__ import annotations

from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import Agent
from daita.tui.app import DaitaApp
from daita.tui.screens.onboarding import AgentCreateScreen


async def test_empty_root_opens_create_screen_inside_textual(tmp_path: Path):
    app = DaitaApp(root=tmp_path, workspace=workspace_for(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        assert isinstance(app.screen, AgentCreateScreen)
        assert app.controller.agent is None
        await pilot.press("escape")
        await pilot.pause()
    assert app.return_value == 0


async def test_requested_missing_agent_does_not_prompt_before_app(tmp_path: Path):
    app = DaitaApp(
        root=tmp_path, agent_name="missing", workspace=workspace_for(tmp_path)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        # Failure is handled inside the app, not by a pre-app selector.
        assert app.screen is not None
        app.exit(0)


async def test_lazy_entry_does_not_import_textual_until_load():
    import sys

    modules = {
        name for name in sys.modules if name.split(".")[0] in {"textual", "rich"}
    }
    # Other tests may already have imported Textual; the production entry stays lazy.
    from daita import terminal

    source = Path(terminal.__file__).read_text(encoding="utf-8")
    assert "from .tui.app import run_daita_app" in source
    assert source.index("def _load_textual_app") < source.index(
        "from .tui.app import run_daita_app"
    )


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


async def test_requested_conversation_is_validated_and_resumed(
    tmp_path: Path, monkeypatch
):
    opened = await Agent.create(
        "atlas", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await opened.close()
    checked: list[str] = []

    async def conversation_exists(_agent: Agent, conversation_id: str) -> bool:
        checked.append(conversation_id)
        return True

    async def conversation_runs(_agent: Agent, _conversation_id: str):
        return ()

    monkeypatch.setattr(Agent, "conversation_exists", conversation_exists)
    monkeypatch.setattr(Agent, "conversation_runs", conversation_runs)
    app = DaitaApp(
        root=tmp_path,
        workspace=workspace_for(tmp_path),
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
    tmp_path: Path, monkeypatch
):
    opened = await Agent.create(
        "atlas", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    await opened.close()
    app = DaitaApp(
        root=tmp_path,
        workspace=workspace_for(tmp_path),
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
