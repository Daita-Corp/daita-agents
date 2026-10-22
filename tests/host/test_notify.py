"""Desktop notifications: only when nobody watches, rate limited, never data."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from daita import connect
from daita.host.interactions import InteractionKind
from daita.host.notify import Notifier
from daita.host.paths import HostPaths
from daita.host.settings import read_setting
from daita.work_board import WorkBoard, WorkCard, WorkColumn, WorkKind
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _stop,
)
from tests.support.host import (
    StartHost,
    bounded,
    finished_run,
    measured,
    pending_interaction,
    scheduled_routine,
)

SENTINEL = "sentinel-secret-41c9"
START = datetime(2026, 10, 21, 9, tzinfo=UTC)


class Sent:
    """A sender that records notifications instead of showing them."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, title: str, text: str) -> None:
        with self._lock:
            self.items.append((title, text))

    def texts(self) -> list[str]:
        with self._lock:
            return [text for _title, text in self.items]


def _remembering(*contents: str) -> MockModelProvider:
    script = []
    for index, content in enumerate(contents):
        script += [_call(_memory_call(f"write-{index}", content=content)), _stop()]
    return MockModelProvider(tuple(script))


@pytest.mark.integration
async def test_a_question_notifies_only_while_no_client_watches_and_holds_no_data(
    start_host: StartHost,
) -> None:
    sent = Sent()
    harness = await start_host(
        model=_remembering(f"password={SENTINEL}", SENTINEL),
        notify=sent,
        notify_interval=0.0,
    )
    await harness.create_agent()

    first = await harness.submit(
        message=f"Remember {SENTINEL}.", cwd=str(harness.work())
    )
    question = await pending_interaction(harness.http, "sales")
    await bounded(sent.texts)
    assert sent.items == [("Daita", "sales needs your approval")]
    assert all(SENTINEL not in text for text in sent.texts())
    await harness.http.post(
        f"/v1/agents/sales/interactions/{question['id']}", json={"decision": "deny"}
    )
    await finished_run(harness.http, "sales", first["run_ref"])

    async with harness.events("sales"):
        second = await harness.submit(
            message="Remember again.", cwd=str(harness.work())
        )
        question = await pending_interaction(harness.http, "sales")
        # The decision is made as the question opens; give a send time to land.
        await asyncio.sleep(0.3)
        await harness.http.post(
            f"/v1/agents/sales/interactions/{question['id']}", json={"decision": "deny"}
        )
        await finished_run(harness.http, "sales", second["run_ref"])
    assert len(sent.items) == 1


@pytest.mark.integration
async def test_a_routine_result_ready_while_nobody_watches_notifies(
    start_host: StartHost,
) -> None:
    sent = Sent()
    model = MockModelProvider((measured("Seven."), measured("Seven are paid.")))
    harness = await start_host(model=model, notify=sent, sweep_seconds=0.05)
    await harness.create_agent()
    agent = await connect("sales", cwd=harness.work(), root=harness.root)
    try:
        await scheduled_routine(agent, model.provider_id, due_in=timedelta(seconds=2))
    finally:
        await agent.close()

    await bounded(sent.texts, timeout=20)
    assert sent.texts() == ["sales: a result is ready"]
    model.assert_consumed()


async def test_the_rate_limit_coalesces_into_one_later_notification(
    tmp_path: Path,
) -> None:
    sent, now = Sent(), [0.0]
    notifier = _notifier(tmp_path, sent, clock=lambda: now[0])

    for _ in range(3):
        notifier.interaction("sales", InteractionKind.TOOL_APPROVAL)
    notifier.interaction("ops", InteractionKind.ROUTINE_CONFIRMATION)
    await bounded(lambda: len(sent.items) == 2)
    now[0] = 29.0
    notifier.flush()
    now[0] = 30.0
    notifier.flush()
    await bounded(lambda: len(sent.items) == 3)

    assert sent.texts() == [
        "sales needs your approval",
        "ops needs you to confirm a routine",
        "sales: 2 items need you",
    ]


async def test_a_client_attaching_drops_what_waited_for_the_interval(
    tmp_path: Path,
) -> None:
    sent, now, watching = Sent(), [0.0], {"sales": 0}
    notifier = _notifier(
        tmp_path, sent, clock=lambda: now[0], attached=watching.__getitem__
    )
    notifier.interaction("sales", InteractionKind.TOOL_APPROVAL)
    notifier.interaction("sales", InteractionKind.TOOL_APPROVAL)
    watching["sales"] = 1
    now[0] = 31.0
    notifier.flush()
    await asyncio.sleep(0.1)

    assert sent.texts() == ["sales needs your approval"]


@pytest.mark.parametrize("content", ["notifications = false\n", "notifications ="])
async def test_the_setting_or_a_broken_file_turns_notifications_off(
    tmp_path: Path, content: str
) -> None:
    sent = Sent()
    notifier = _notifier(tmp_path, sent)
    HostPaths(tmp_path).settings.write_text(content, encoding="utf-8")

    notifier.interaction("sales", InteractionKind.TOOL_APPROVAL)
    notifier.work("sales", _board(_card(WorkColumn.DONE, START)))
    await asyncio.sleep(0.1)

    assert sent.items == []


async def test_only_cards_new_since_start_or_last_look_notify(tmp_path: Path) -> None:
    sent = Sent()
    notifier = _notifier(tmp_path, sent, interval=0.0)
    old = _card(WorkColumn.DONE, START - timedelta(minutes=1), card_id="old")
    needs_you = _card(WorkColumn.NEEDS_YOU, START, card_id="job-1")

    notifier.work("sales", _board(old, needs_you))
    notifier.work("sales", _board(old, needs_you))
    await bounded(lambda: sent.items)
    await asyncio.sleep(0.1)

    assert sent.texts() == ["sales: an item needs you"]


def _notifier(
    root: Path,
    send: Sent,
    *,
    clock: Callable[[], float] = lambda: 0.0,
    attached: Callable[[str], int] = lambda _agent: 0,
    interval: float = 30.0,
) -> Notifier:
    paths = HostPaths(root)
    return Notifier(
        attached=attached,
        enabled=lambda: read_setting(paths, "notifications"),
        send=send,
        interval=interval,
        clock=clock,
        now=lambda: START,
    )


def _card(column: WorkColumn, updated_at: datetime, card_id: str = "d-1") -> WorkCard:
    return WorkCard(
        kind=WorkKind.ROUTINE_RUN,
        id=card_id,
        column=column,
        title=f"Report {SENTINEL}",
        status="succeeded",
        summary=f"The value is {SENTINEL}.",
        updated_at=updated_at,
    )


def _board(*cards: WorkCard) -> WorkBoard:
    return WorkBoard(
        columns={
            column: tuple(card for card in cards if card.column is column)
            for column in WorkColumn
        },
        truncated={column: 0 for column in WorkColumn},
        generated_at=START,
    )
