"""Host-owned runs: streamed, exclusive per conversation, cancelled only explicitly."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from daita.loop.models import LoopExit, LoopExitKind
from daita.observation import AgentEventKind
from tests.host.conftest import StartHost
from tests.support.approval_learning import (
    MockModelProvider,
    _call,
    _memory_call,
    _stop,
)
from tests.support.host import (
    client,
    error_code,
    finished_run,
    pending_interaction,
    value,
)

pytestmark = pytest.mark.integration

NOTE = "Revenue means paid invoice subtotal."


def _remembering(*answers: str) -> MockModelProvider:
    """A model that asks to write memory (needing approval), then answers."""

    script = []
    for index, answer in enumerate(answers):
        script += [_call(_memory_call(f"write-{index}", content=NOTE)), _stop(answer)]
    return MockModelProvider(tuple(script))


async def test_a_run_streams_its_events_and_returns_its_loop_exit(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=MockModelProvider((_stop("EMEA leads."),)))
    await harness.create_agent()

    async with harness.events("sales") as events:
        run = await harness.submit(message="Which region leads?")
        assert run["state"] == "queued"
        _seq, _event, started = await events.next(
            "run_state", lambda data: data["state"] == "running"
        )
        _seq, _event, done = await events.next(
            "run_state", lambda data: data["state"] == "completed"
        )

    assert started["run_ref"] == run["run_ref"]
    assert isinstance(done["exit"], LoopExit)
    assert done["exit"].final_text == "EMEA leads."
    kinds = [data.kind for _s, event, data in events.records if event == "agent_event"]
    assert kinds[0] is AgentEventKind.RUN_STARTED
    assert kinds[-1] is AgentEventKind.RUN_COMPLETED
    sequences = [seq for seq, _event, _data in events.records]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)
    by_run_id = value(await harness.http.get(f"/v1/agents/sales/runs/{done['run_id']}"))
    assert by_run_id["run_ref"] == run["run_ref"] and by_run_id["exit"] == done["exit"]


async def test_a_client_disconnecting_mid_run_never_cancels_the_run(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    first = client(harness.paths)
    run = value(
        await first.post("/v1/agents/sales/runs", json={"message": "Remember this."})
    )
    await pending_interaction(first, "sales")
    await first.aclose()

    async with client(harness.paths) as second:
        interaction = await pending_interaction(second, "sales")
        answered = await second.post(
            f"/v1/agents/sales/interactions/{interaction['id']}",
            json={"decision": "approve"},
        )
        assert answered.status_code == 200
        finished = await finished_run(second, "sales", run["run_ref"])
        assert finished["state"] == "completed"
        assert value(await second.get("/v1/agents/sales/memory")) == NOTE


async def test_two_clients_see_the_same_events_and_the_first_answer_wins(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    other = client(harness.paths)
    try:
        async with harness.events("sales") as left, harness.events("sales") as right:
            left.http, right.http = harness.http, other
            run = await harness.submit(message="Remember this.")
            _seq, _event, opened = await left.next(
                "interaction", lambda data: data["op"] == "opened"
            )
            interaction = opened["interaction"]
            assert interaction["kind"] == "tool_approval"
            assert interaction["run_ref"] == run["run_ref"]
            assert interaction["request"]["reviewable"] is True
            assert NOTE in interaction["request"]["review_document"]

            url = f"/v1/agents/sales/interactions/{interaction['id']}"
            elsewhere = await other.post(
                f"/v1/agents/support/interactions/{interaction['id']}",
                json={"decision": "approve"},
            )
            assert error_code(elsewhere) == "already_resolved"
            approve = await other.post(url, json={"decision": "approve"})
            deny = await harness.http.post(url, json={"decision": "deny"})
            assert approve.status_code == 200
            assert (deny.status_code, error_code(deny)) == (409, "already_resolved")
            await left.next("run_state", lambda data: data["state"] == "completed")
            await right.next("run_state", lambda data: data["state"] == "completed")
        assert left.records == right.records
    finally:
        await other.aclose()
    assert value(await harness.http.get("/v1/agents/sales/memory")) == NOTE


async def test_an_expired_interaction_interrupts_the_run_and_never_denies(
    start_host: StartHost,
) -> None:
    harness = await start_host(
        model=_remembering("Saved."), interaction_expiry=timedelta(seconds=0.3)
    )
    await harness.create_agent()

    async with harness.events("sales") as events:
        run = await harness.submit(message="Remember this.")
        _seq, _event, expired = await events.next(
            "interaction", lambda data: data["op"] == "expired"
        )
        finished = await finished_run(harness.http, "sales", run["run_ref"])

    assert expired["interaction"]["state"] == "expired"
    assert expired["interaction"]["answer"] is None
    assert finished["state"] == "interrupted"
    assert finished["exit"] is None or finished["exit"].kind is LoopExitKind.INTERRUPTED
    assert not any(
        data.kind is AgentEventKind.APPROVAL_DECIDED
        and data.data.get("decision") == "deny"
        for _s, event, data in events.records
        if event == "agent_event"
    )
    assert value(await harness.http.get("/v1/agents/sales/memory")) == ""
    late = await harness.http.post(
        f"/v1/agents/sales/interactions/{expired['interaction']['id']}",
        json={"decision": "approve"},
    )
    assert error_code(late) == "already_resolved"


async def test_dismissing_an_interaction_interrupts_the_run(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    run = await harness.submit(message="Remember this.")
    interaction = await pending_interaction(harness.http, "sales")

    dismissed = value(
        await harness.http.post(
            f"/v1/agents/sales/interactions/{interaction['id']}/dismiss"
        )
    )

    assert dismissed["state"] == "dismissed"
    finished = await finished_run(harness.http, "sales", run["run_ref"])
    assert finished["state"] == "interrupted"
    assert value(await harness.http.get("/v1/agents/sales/memory")) == ""


async def test_one_active_run_per_conversation_and_cancel_is_explicit(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."))
    await harness.create_agent()
    first = await harness.submit(message="Remember this.")
    await pending_interaction(harness.http, "sales")
    running = value(await harness.http.get(f"/v1/agents/sales/runs/{first['run_ref']}"))
    assert running["state"] == "waiting_for_input"
    conversation_id = running["conversation_id"]

    busy = await harness.http.post(
        "/v1/agents/sales/runs",
        json={"message": "And this.", "conversation_id": conversation_id},
    )
    assert (busy.status_code, error_code(busy)) == (409, "conversation_busy")
    assert busy.json()["error"]["details"]["attributes"]["conversation_id"] == (
        conversation_id
    )
    active = value(await harness.http.get("/v1/agents/sales/runs?active=true"))
    assert [run["run_ref"] for run in active] == [first["run_ref"]]

    url = f"/v1/agents/sales/runs/{first['run_ref']}/cancel"
    assert value(await harness.http.post(url))["run_ref"] == first["run_ref"]
    assert value(await harness.http.post(url))["run_ref"] == first["run_ref"]
    cancelled = await finished_run(harness.http, "sales", first["run_ref"])
    assert cancelled["state"] == "cancelled"
    assert value(await harness.http.get("/v1/agents/sales/interactions")) == ()
    assert value(await harness.http.get("/v1/agents/sales/runs?active=true")) == ()

    # The cancelled run consumed the tool call; the next scripted turn answers.
    again = await harness.submit(
        message="Remember this.", conversation_id=conversation_id
    )
    assert (await finished_run(harness.http, "sales", again["run_ref"]))["state"] == (
        "completed"
    )


async def test_graceful_shutdown_lets_active_runs_finish_and_refuses_new_ones(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."), drain_seconds=20)
    await harness.create_agent()
    run = await harness.submit(message="Remember this.")
    interaction = await pending_interaction(harness.http, "sales")

    stopping = harness.host.request_shutdown("test")
    refused = await harness.http.post("/v1/agents/sales/runs", json={"message": "Hi"})
    assert (refused.status_code, error_code(refused)) == (503, "shutting_down")
    await harness.http.post(
        f"/v1/agents/sales/interactions/{interaction['id']}",
        json={"decision": "approve"},
    )
    await asyncio.wait_for(stopping, 20)

    assert harness.host.runs.get("sales", run["run_ref"]).state == "completed"
    await asyncio.wait_for(harness.task, 20)
    assert not harness.paths.socket.exists()
    assert not harness.paths.status.exists()


async def test_shutdown_cancels_runs_still_active_after_the_drain_window(
    start_host: StartHost,
) -> None:
    harness = await start_host(model=_remembering("Saved."), drain_seconds=0.2)
    await harness.create_agent()
    run = await harness.submit(message="Remember this.")
    await pending_interaction(harness.http, "sales")

    await asyncio.wait_for(harness.host.request_shutdown("test"), 20)

    assert harness.host.runs.get("sales", run["run_ref"]).state == "cancelled"
