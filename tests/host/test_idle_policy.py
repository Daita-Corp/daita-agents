"""The idle policy (§10.7): idle agents close; an auto-spawned host exits."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta

import pytest

from daita import connect
from tests.support.approval_learning import MockModelProvider
from tests.support.host import StartHost, bounded, measured, scheduled_routine

pytestmark = pytest.mark.integration

IDLE = 60.0


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float = 10.0) -> None:
        self.now += seconds

    def ticked(self, condition: Callable[[], bool]) -> bool:
        """Advance, then check; a bounded() probe that moves time each try."""

        self.tick()
        return condition()


async def test_an_unwatched_agent_without_background_work_closes_when_idle(
    start_host: StartHost,
) -> None:
    clock = Clock()
    model = MockModelProvider((measured("Seven."),))
    harness = await start_host(
        model=model, clock=clock, idle_seconds=IDLE, sweep_seconds=0.02
    )
    for name in ("sales", "ops", "billing"):
        await harness.create_agent(name)
    billing = await connect("billing", cwd=harness.work(), root=harness.root)
    try:
        await scheduled_routine(billing, model.provider_id, due_in=timedelta(days=1))
    finally:
        await billing.close()
    async with harness.events("ops"):
        await bounded(
            lambda: clock.ticked(
                lambda: "sales" not in harness.host.registry.open_names()
            )
        )
        clock.tick(IDLE)
        await asyncio.sleep(0.1)
        # Watched, or holding a routine: both stay open.
        assert harness.host.registry.open_names() == ("billing", "ops")
    await bounded(
        lambda: clock.ticked(lambda: harness.host.registry.open_names() == ("billing",))
    )


@pytest.mark.parametrize("installed", [False, True])
async def test_an_auto_spawned_host_exits_when_idle_unless_installed(
    start_host: StartHost, installed: bool
) -> None:
    clock = Clock()
    harness = await start_host(
        clock=clock,
        idle_seconds=IDLE,
        sweep_seconds=0.02,
        idle_exit=True,
        service_installed=lambda: installed,
    )
    await harness.create_agent()
    await harness.http.post("/v1/agents/sales/close")

    if installed:
        for _ in range(20):
            clock.tick()
            await asyncio.sleep(0.02)
        assert not harness.task.done()
    else:
        await bounded(lambda: clock.ticked(harness.task.done))
        assert not harness.paths.socket.exists()
