"""Event history: sequenced, bounded, replayable, and reset when a cursor is lost."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from daita._json import FrozenJsonObject
from daita.host.codec import decode
from daita.host.events import EventBus
from daita.host.protocol import EventType
from daita.observation import AgentEvent, AgentEventKind
from tests.host.conftest import StartHost
from tests.support.approval_learning import MockModelProvider, _stop
from tests.support.host import EventStream, finished_run

AGENT = "sales"


def _delta(text: str) -> AgentEvent:
    return AgentEvent(
        kind=AgentEventKind.MODEL_TEXT_DELTA,
        occurred_at=datetime(2026, 10, 20, 9, tzinfo=UTC),
        run_id="run-1",
        conversation_id="conversation-1",
        data=FrozenJsonObject.from_mapping({"text": text}),
    )


async def _take(stream: AsyncIterator[tuple[int, str, str]], count: int) -> list:
    async with asyncio.timeout(5):
        return [await anext(stream) for _ in range(count)]


@pytest.mark.unit
async def test_replay_starts_after_the_cursor_then_goes_live() -> None:
    bus = EventBus()
    for state in ("queued", "running", "completed"):
        bus.publish(AGENT, EventType.RUN_STATE, {"state": state})
    first = bus.head(AGENT) - 2
    stream = bus.subscribe(AGENT, after=first)

    replayed = await _take(stream, 2)
    bus.publish(AGENT, EventType.WORK_CHANGED, {})
    (live,) = await _take(stream, 1)

    assert [seq for seq, _event, _text in replayed] == [first + 1, first + 2]
    assert [json.loads(text)["seq"] for _seq, _event, text in replayed] == [
        first + 1,
        first + 2,
    ]
    assert live[:2] == (first + 3, "work_changed")
    assert bus.attached(AGENT) == 1
    await stream.aclose()
    assert bus.attached(AGENT) == 0


@pytest.mark.unit
async def test_pressure_merges_text_deltas_and_never_drops_other_events() -> None:
    bus = EventBus(max_events=4)
    bus.publish(AGENT, EventType.RUN_STATE, {"state": "running"})
    before = bus.head(AGENT) - 1
    for text in ("EMEA ", "leads ", "today."):
        bus.publish(AGENT, EventType.AGENT_EVENT, _delta(text))
    bus.publish(AGENT, EventType.RUN_STATE, {"state": "waiting_for_input"})
    bus.publish(AGENT, EventType.RUN_STATE, {"state": "running"})

    replay = await _take(bus.subscribe(AGENT, after=before), 4)
    (reset,) = await _take(bus.subscribe(AGENT, after=before + 2), 1)

    assert [event for _seq, event, _text in replay] == [
        "run_state",
        "agent_event",
        "run_state",
        "run_state",
    ]
    merged = decode(json.loads(replay[1][2])["data"], AgentEvent)
    assert merged.data["text"] == "EMEA leads today."
    # A client that already saw "EMEA " would see it twice, so it resets.
    assert reset[1] == "reset"

    # The remaining delta has no neighbour to merge with: it is dropped, never
    # one of the run states.
    bus.publish(AGENT, EventType.RUN_STATE, {"state": "completed"})
    after_delta = await _take(bus.subscribe(AGENT, after=before + 4), 3)
    assert [event for _seq, event, _text in after_delta] == ["run_state"] * 3
    (lost,) = await _take(bus.subscribe(AGENT, after=before), 1)
    assert lost[1] == "reset"


@pytest.mark.unit
async def test_events_older_than_the_age_bound_are_evicted() -> None:
    now = [0.0]
    bus = EventBus(max_age_seconds=900, clock=lambda: now[0])
    bus.publish(AGENT, EventType.RUN_STATE, {"state": "running"})
    old = bus.head(AGENT)
    now[0] = 901.0
    bus.publish(AGENT, EventType.RUN_STATE, {"state": "completed"})

    (reset,) = await _take(bus.subscribe(AGENT, after=old - 1), 1)
    (kept,) = await _take(bus.subscribe(AGENT, after=old), 1)

    assert reset[1] == "reset"
    assert kept[:2] == (old + 1, "run_state")


@pytest.mark.unit
async def test_a_restarted_host_never_reuses_a_sequence_a_client_saw() -> None:
    before = EventBus()
    before.publish(AGENT, EventType.RUN_STATE, {"state": "running"})
    after = EventBus()

    (record,) = await _take(after.subscribe(AGENT, after=before.head(AGENT)), 1)

    assert after.head(AGENT) > before.head(AGENT) or record[1] == "reset"
    assert record[1] == "reset"


@pytest.mark.integration
async def test_clients_replay_from_their_last_sequence_or_are_told_to_reset(
    start_host: StartHost,
) -> None:
    harness = await start_host(
        model=MockModelProvider((_stop("EMEA leads."),)), heartbeat_seconds=1
    )
    await harness.create_agent()
    async with harness.events(AGENT) as live:
        run = await harness.submit(message="Which region leads?")
        await finished_run(harness.http, AGENT, run["run_ref"])
        await live.next("run_state", lambda data: data["state"] == "completed")
    middle = live.records[len(live.records) // 2][0]

    async with EventStream(harness.http, AGENT, after=middle) as replay:
        expected = [record for record in live.records if record[0] > middle]
        for record in expected:
            assert await replay.next() == record
        _seq, _event, heartbeat = await replay.next("heartbeat", timeout=5)
    async with EventStream(harness.http, AGENT, after=0) as stale:
        seq, event, _data = await stale.next()

    assert heartbeat is None
    assert event == "reset" and seq == harness.host.bus.head(AGENT)
