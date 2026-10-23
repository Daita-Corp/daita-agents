"""The one-time offer to keep the Daita host running after the first routine."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import pytest

from daita import connect
from daita.host import service
from daita.host.paths import host_paths
from daita.host.settings import read_setting, write_setting
from daita.llm.models import ToolCall
from daita.tui.app import DaitaApp
from daita.tui.screens.confirm import ConfirmScreen
from daita.tui.widgets.approval import ApprovalPanel
from tests.support.approval_learning import MockModelProvider, _call
from tests.support.host import (
    HostHarness,
    StartHost,
    bounded,
    host_workspace,
    measured,
)

pytestmark = pytest.mark.integration

SIZE = (110, 36)


class Destination(NamedTuple):
    conversation_id: str


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Where service.installed() looks, so this machine's own service is ignored.
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


async def test_the_offer_appears_once_and_a_no_is_remembered(
    start_host: StartHost, home: Path
) -> None:
    harness, model, destination = await _agent(start_host)
    app = _app(harness)
    async with app.run_test(size=SIZE):
        try:
            await _ready(app, destination)
            await _save_routine(app, model, destination, "Hourly count")
            offer = await bounded(lambda: _offer(app))
            assert "Keep Daita running in the background" in offer._message
            offer.action_decline()
            await bounded(lambda: _run_settled(app))
            assert read_setting(host_paths(harness.root), "background_offer") is False

            await _save_routine(app, model, destination, "Daily count")
            await bounded(lambda: _run_settled(app))
            assert _offer(app) is None
            assert {r.title for r in await app.controller.list_routines()} == {
                "Hourly count",
                "Daily count",
            }
        finally:
            app.exit(0)


async def test_yes_installs_the_service_and_is_not_asked_again(
    start_host: StartHost, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs: list[object] = []

    def install(paths: object) -> list[str]:
        installs.append(paths)
        return ["Daita host service installed.", "The Daita host is running."]

    # launchctl and systemctl never run in tests.
    monkeypatch.setattr(service, "install", install)
    harness, model, destination = await _agent(start_host)
    app = _app(harness)
    async with app.run_test(size=SIZE):
        try:
            await _ready(app, destination)
            await _save_routine(app, model, destination, "Hourly count")
            (await bounded(lambda: _offer(app))).action_accept()
            await bounded(lambda: installs)
            await bounded(lambda: _run_settled(app))
            assert installs == [host_paths(harness.root)]
            assert read_setting(host_paths(harness.root), "background_offer") is False
        finally:
            app.exit(0)


@pytest.mark.parametrize("reason", ["installed", "declined before"])
async def test_no_offer_when_installed_or_declined_before(
    start_host: StartHost, home: Path, reason: str
) -> None:
    harness, model, destination = await _agent(start_host)
    if reason == "installed":
        path = service.service_path(home=home)
        path.parent.mkdir(parents=True)
        path.write_text("installed", encoding="utf-8")
    else:
        write_setting(host_paths(harness.root), "background_offer", False)
    app = _app(harness)
    async with app.run_test(size=SIZE):
        try:
            await _ready(app, destination)
            await _save_routine(app, model, destination, "Hourly count")
            await bounded(lambda: _run_settled(app))
            await asyncio.sleep(0.1)
            assert _offer(app) is None
            assert len(await app.controller.list_routines()) == 1
        finally:
            app.exit(0)


async def _agent(
    start_host: StartHost,
) -> tuple[HostHarness, MockModelProvider, Destination]:
    """A host agent with one answered question, and that answer's destination."""

    model = MockModelProvider((measured("Seven invoices are paid."),))
    harness = await start_host(model=model)
    await harness.create_agent()
    agent = await connect("sales", cwd=harness.work(), root=harness.root)
    try:
        origin = await agent.run("How many invoices are paid?")
        assert origin.conversation_id is not None
    finally:
        await agent.close()
    return harness, model, Destination(origin.conversation_id)


def _app(harness: HostHarness) -> DaitaApp:
    return DaitaApp(
        root=harness.root, workspace=host_workspace(harness), agent_name="sales"
    )


async def _ready(app: DaitaApp, destination: Destination) -> None:
    await bounded(
        lambda: app.controller.agent is not None
        and all(worker.is_finished for worker in app.workers)
    )
    app.controller.conversation_id = destination.conversation_id


async def _save_routine(
    app: DaitaApp, model: MockModelProvider, destination: Destination, title: str
) -> None:
    """Run one turn whose model saves a routine, approving it when asked."""

    model.replace_script(
        (
            _call(_routine_save(title)),
            measured("Scheduled."),
        )
    )
    await app._start_run(f"Report the paid invoice count: {title}.")

    def settled_or_approved() -> bool:
        chat = app.chat()
        if chat is not None and chat.query_one(ApprovalPanel).active:
            chat.query_one(ApprovalPanel).action_approve()
        return app._run_task is not None and app._run_task.done() or bool(_offer(app))

    await bounded(settled_or_approved, timeout=20)


def _routine_save(title: str) -> ToolCall:
    now = datetime.now(UTC)
    return ToolCall(
        id=f"save-{title.split()[0].lower()}",
        name="routine_save",
        arguments={
            "title": title,
            "prompt": "Report the paid invoice count.",
            "schedule": {
                "kind": "interval",
                "interval_seconds": 3600,
                "anchor_at": (now + timedelta(hours=1)).isoformat(),
            },
        },
    )


def _offer(app: DaitaApp) -> ConfirmScreen | None:
    return app.screen if isinstance(app.screen, ConfirmScreen) else None


def _run_settled(app: DaitaApp) -> bool:
    return app._run_task is not None and app._run_task.done()
