"""The keeper alone: a fake clock, scripted probes and an in-memory store."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from daita.adapters.models import SourceProbe
from daita.adapters.registry import SourceFreshness
from daita.catalog.models import CatalogCommitSummary, CatalogSourceState
from daita.hosting import keeper as keeper_module
from daita.hosting.keeper import CatalogKeeper, KeeperRead, KeptSource

_START = datetime(2026, 11, 7, 9, 0, tzinfo=UTC)
_SOURCE = KeptSource("src", "demo")
_FRESH = SourceFreshness(5.0, 40.0, 86_400.0)


class _Clock:
    def __init__(self) -> None:
        self.now = _START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class _Store:
    state: CatalogSourceState
    resources: int = 0
    described: int = 0
    saves: int = 0
    unreadable: list[str] = field(default_factory=list)

    async def load_source_state(self, agent_id, source_id):
        return self.state

    async def save_source_state(self, state):
        self.state = state
        self.saves += 1

    async def catalog_detail_counts(self, agent_id, source_id):
        return self.resources, self.described

    async def take_unreadable_catalog_sources(self, agent_id):
        taken, self.unreadable = tuple(self.unreadable), []
        return taken


class _Probe:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.tokens = {"main": "t1"}
        self.error: Exception | None = None
        self.hang = False
        self.calls = 0

    async def probe(self, *, agent_id, source_id, cursor):
        self.calls += 1
        if self.hang:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return SourceProbe(checked_at=self.clock.now, tokens=self.tokens)


class _World:
    def __init__(self, *, fresh: SourceFreshness = _FRESH, probing: bool = True):
        self.clock = _Clock()
        self.probe = _Probe(self.clock)
        self.store = _Store(
            CatalogSourceState(
                agent_id="a",
                source_id="src",
                current_sync_id="s1",
                probe_tokens={"main": "t1"},
                last_probe_at=_START - timedelta(seconds=60),
                last_full_check_at=_START,
            )
        )
        self.reads: list[KeeperRead] = []
        self.read_hook: Callable[[KeeperRead], Awaitable[None]] | None = None
        self.keeper = CatalogKeeper(
            agent_id="a",
            store=self.store,
            commit_lock=asyncio.Lock(),
            clock=self.clock,
            sources=self._sources,
            probe_backend=lambda adapter: self.probe if probing else None,
            freshness=lambda adapter: fresh,
            read=self._read,
            fill_resources=2,
        )

    async def _sources(self):
        return (_SOURCE,)

    async def _read(self, source_id, plan):
        self.reads.append(plan)
        if self.read_hook is not None:
            await self.read_hook(plan)
        state = self.store.state
        if plan.probe_tokens is not None:
            state = replace(
                state,
                probe_tokens=dict(plan.probe_tokens),
                last_probe_at=plan.probed_at,
            )
        if plan.fill:
            self.store.described = min(
                self.store.resources, self.store.described + plan.max_resources
            )
        self.store.state = state
        return CatalogCommitSummary(0, 0, 0)

    @property
    def mem(self):
        return self.keeper._mems[_SOURCE.source_id]

    @property
    def next_due(self) -> datetime:
        due = self.mem.next_due
        assert due is not None
        return due

    async def due(self) -> None:
        """Move the clock to the source's next probe and let the keeper work."""

        self.clock.now = max(self.clock.now, self.next_due)
        await self.keeper.tick()


async def test_the_probe_interval_doubles_when_nothing_changes_and_resets_on_a_change():
    world = _World()
    await world.keeper.tick()
    gaps = []
    for _ in range(5):
        gaps.append((world.next_due - world.clock.now).total_seconds())
        await world.due()
    assert gaps == [5, 10, 20, 40, 40]
    world.probe.tokens = {"main": "t2"}
    await world.due()
    assert world.reads and (world.next_due - world.clock.now).total_seconds() == 5


async def test_a_run_probes_only_when_the_last_probe_is_old():
    world = _World()
    await world.keeper.tick()
    calls = world.probe.calls
    world.keeper.use(["src"])
    await world.keeper.tick()
    assert world.probe.calls == calls
    world.clock.advance(31)
    world.keeper.use(["src"])
    await world.keeper.tick()
    assert world.probe.calls == calls + 1


async def test_a_lookup_miss_probes_after_five_seconds():
    world = _World()
    await world.keeper.tick()
    calls = world.probe.calls
    world.clock.advance(4)
    world.keeper.miss("src")
    await world.keeper.tick()
    assert world.probe.calls == calls
    world.clock.advance(2)
    world.keeper.miss("src")
    await world.keeper.tick()
    assert world.probe.calls == calls + 1


async def test_a_change_reads_only_the_scopes_that_moved_and_stores_the_tokens():
    world = _World()
    world.store.state = replace(
        world.store.state, probe_tokens={"a": "1", "b": "1", "c": "1"}
    )
    world.probe.tokens = {"a": "1", "b": "2", "c": "1"}
    await world.keeper.tick()
    (read,) = world.reads
    assert read.scopes == ("b",) and not read.full
    assert dict(read.probe_tokens or {}) == world.probe.tokens
    assert world.store.state.probe_tokens == world.probe.tokens


async def test_a_change_in_every_scope_or_a_removed_scope_reads_the_whole_source():
    world = _World()
    world.store.state = replace(world.store.state, probe_tokens={"a": "1", "b": "1"})
    world.probe.tokens = {"a": "2", "b": "2"}
    await world.keeper.tick()
    world.store.state = replace(world.store.state, probe_tokens={"a": "1", "b": "1"})
    world.probe.tokens = {"a": "1"}
    await world.due()
    assert [r.scopes for r in world.reads] == [(), ()]
    assert all(r.full for r in world.reads)


async def test_an_unchanged_probe_does_not_rewrite_the_state_row_inside_the_throttle():
    world = _World()
    await world.keeper.tick()
    for _ in range(3):
        await world.due()
    assert world.store.saves == 0
    world.clock.advance(601)
    await world.due()
    assert world.store.saves == 1
    assert world.store.state.last_probe_at == world.clock.now


async def test_two_failed_probes_make_a_source_unreachable_and_one_success_recovers():
    world = _World()
    world.probe.error = RuntimeError("password=hunter2 host=db.internal")
    await world.keeper.tick()
    assert not world.keeper.facts("src").unreachable
    await world.due()
    facts = world.keeper.facts("src")
    assert facts.unreachable and facts.error_code == "probe_failed"
    assert world.store.state.last_error_code == "probe_failed"
    world.probe.error = None
    await world.due()
    facts = world.keeper.facts("src")
    assert not facts.unreachable and facts.error_code is None
    assert world.store.state.last_error_code is None


async def test_a_failing_probe_leaks_nothing_into_the_log(caplog):
    world = _World()
    world.probe.error = RuntimeError("password=hunter2 host=db.internal")
    with caplog.at_level(logging.DEBUG, logger="daita.hosting.keeper"):
        await world.keeper.tick()
        await world.due()
    assert "RuntimeError" in caplog.text and "probe_failed" in caplog.text
    assert "hunter2" not in caplog.text and "db.internal" not in caplog.text
    assert "hunter2" not in repr(world.keeper.facts("src"))


async def test_a_hung_probe_times_out_and_is_a_failure_not_a_hang(monkeypatch):
    monkeypatch.setattr(keeper_module, "PROBE_TIMEOUT_SECONDS", 0.05)
    world = _World()
    world.probe.hang = True
    await asyncio.wait_for(world.keeper.tick(), timeout=10)
    assert world.keeper.facts("src").error_code == "probe_timeout"


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Wait for a condition; the timeout is only reached when something is broken."""

    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def test_close_is_prompt_with_a_hung_probe():
    world = _World()
    world.probe.hang = True
    world.keeper.start()
    try:
        await _until(lambda: world.probe.calls > 0)
    finally:
        await asyncio.wait_for(world.keeper.close(), timeout=10)


async def test_close_is_prompt_with_a_hung_read():
    world = _World()
    world.probe.tokens = {"main": "t2"}
    started = asyncio.Event()

    async def hang(plan):
        started.set()
        await asyncio.Event().wait()

    world.read_hook = hang
    world.keeper.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        await _until(lambda: world.keeper.facts("src").updating)
    finally:
        await asyncio.wait_for(world.keeper.close(), timeout=10)


async def test_a_person_cancels_a_background_read_and_keeps_the_keeper_out():
    world = _World()
    world.probe.tokens = {"main": "t2"}
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def hang(plan):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    world.read_hook = hang
    world.keeper.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        async with asyncio.timeout(10):
            async with world.keeper.person_work():
                assert cancelled.is_set()
                assert not world.keeper.facts("src").updating
                reads = len(world.reads)
                world.keeper.use(["src"])
                # Nothing may start while a person works; the window cannot
                # fail a correct run, only a keeper that ignores the person.
                await asyncio.sleep(0.05)
                assert len(world.reads) == reads
    finally:
        await asyncio.wait_for(world.keeper.close(), timeout=10)


async def test_a_clock_jump_catches_up_with_a_probe_and_the_daily_full_check(caplog):
    world = _World()
    await world.keeper.tick()
    world.clock.advance(3 * 86_400)
    with caplog.at_level(logging.WARNING, logger="daita.hosting.keeper"):

        async def read(source_id, plan):
            world.reads.append(plan)
            if plan.full and plan.probe_tokens is None:
                world.store.state = replace(
                    world.store.state, last_full_check_at=world.clock.now
                )
                return CatalogCommitSummary(2, 0, 1)
            return CatalogCommitSummary(0, 0, 0)

        world.keeper._read = read
        await world.keeper.tick()
    assert world.probe.calls == 2
    assert world.store.state.last_missed == 3
    assert "demo" in caplog.text and "added" in caplog.text


async def test_a_source_without_a_probe_is_only_fully_checked_on_its_schedule():
    world = _World(probing=False)
    await world.keeper.tick()
    assert world.reads == []
    world.clock.advance(86_401)
    await world.keeper.tick()
    assert len(world.reads) == 1 and world.reads[0].full
    assert world.store.state.last_missed is None


async def test_keep_current_off_stops_every_trigger():
    world = _World()
    world.store.state = replace(world.store.state, keep_current=False)
    world.store.resources = 5
    world.probe.tokens = {"main": "t2"}
    world.keeper.use(["src"])
    world.keeper.miss("src")
    await world.keeper.check("src", wait=1)
    world.clock.advance(3 * 86_400)
    await world.keeper.tick()
    assert world.probe.calls == 0 and world.reads == []


async def test_a_catalog_without_detail_is_filled_in_passes_until_complete():
    world = _World()
    world.store.resources, world.store.described = 5, 1
    busy = [await world.keeper.tick() for _ in range(4)]
    assert busy == [True, False, False, False]
    assert [r.fill for r in world.reads] == [True, True]
    assert world.store.described == 5
    assert not world.keeper.facts("src").updating


async def test_a_fill_that_makes_no_progress_stops_instead_of_looping():
    world = _World()
    world.store.resources, world.store.described = 5, 1

    async def nothing(source_id, plan):
        world.reads.append(plan)
        return CatalogCommitSummary(0, 0, 0)

    world.keeper._read = nothing
    assert await world.keeper.tick() is False
    assert len(world.reads) == 1
    await world.keeper.tick()
    assert len(world.reads) == 1


async def test_damaged_catalog_rows_cause_a_full_read_from_the_keeper_loop():
    world = _World()
    world.store.unreadable = ["src"]
    await world.keeper.tick()
    assert [r.full for r in world.reads] == [True]
    assert world.store.unreadable == []


async def test_a_check_that_outlasts_its_wait_keeps_going_without_error():
    world = _World()
    world.probe.tokens = {"main": "t2"}
    release = asyncio.Event()

    async def slow(plan):
        await release.wait()

    world.read_hook = slow
    try:
        await asyncio.wait_for(world.keeper.check("src", wait=0.05), timeout=10)
        # The read may not have begun when the wait ran out; the work continues.
        await _until(lambda: len(world.reads) == 1)
        await _until(lambda: world.keeper.facts("src").updating)
        release.set()
        await asyncio.wait_for(world.keeper._checks["src"], timeout=10)
        assert world.store.state.probe_tokens == {"main": "t2"}
    finally:
        release.set()
        await asyncio.wait_for(world.keeper.close(), timeout=10)


async def test_a_fill_a_person_cancelled_resumes_on_the_next_tick():
    world = _World()
    world.store.resources, world.store.described = 5, 1
    started = asyncio.Event()
    hold = True

    async def hang(plan):
        if hold:
            started.set()
            await asyncio.Event().wait()

    world.read_hook = hang
    ticking = asyncio.create_task(world.keeper.tick())
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        async with world.keeper.person_work():
            await asyncio.wait_for(ticking, timeout=10)
        hold = False
        assert world.keeper.facts("src").updating
        assert await asyncio.wait_for(world.keeper.tick(), timeout=10) is True
        assert world.store.described > 1
    finally:
        ticking.cancel()
